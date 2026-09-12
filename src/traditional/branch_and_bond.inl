#include <vector>
#include <iostream>
#include <span>
#include "core/reactor_util.hpp"

namespace reactor_optimizer::traditional {

template <size_t W, size_t H, ReactorConfig Config>
bool passes_thermal_bound(const ReactorState<W, H, Config>& reactor) {
    constexpr int max_vent_cooling = (Config.meltdown_temp / Config.vent_divisor) + Config.vent_base;
    constexpr int max_absorber_cooling = Config.absorber_cooling * 4;
    constexpr int max_theoretical_block_cooling = std::max(max_vent_cooling, max_absorber_cooling);

    long long current_heat_gen = 0;
    long long max_cooling_capacity = 0;

    constexpr std::array<std::pair<int, int>, 4> directions = {{{0, -1}, {0, 1}, {-1, 0}, {1, 0}}};

    for (int y = 0; y < static_cast<int>(H); y++) {
        for (int x = 0; x < static_cast<int>(W); x++) {
            BlockType block = reactor.get_block(x, y);
            
            if (block == BlockType::Empty) {
                max_cooling_capacity += max_theoretical_block_cooling;
            } 
            else if (block == BlockType::Absorber) { max_cooling_capacity += max_absorber_cooling; } 
            else if (block == BlockType::HeatVent) { max_cooling_capacity += max_vent_cooling; } 
            else if (block == BlockType::SingleRod || block == BlockType::DoubleRod || block == BlockType::QuadRod) {
                
                int internal_pulses = 0;
                int outbound_pulses = 0;
                if (block == BlockType::SingleRod) { internal_pulses = 1; outbound_pulses = 1; }
                else if (block == BlockType::DoubleRod) { internal_pulses = 4; outbound_pulses = 2; }
                else if (block == BlockType::QuadRod) { internal_pulses = 12; outbound_pulses = 4; }

                int pulses = internal_pulses; 
                
                for (const auto& dir : directions) {
                    int nx = x + dir.first;
                    int ny = y + dir.second;
                    if (reactor.is_valid_coordinate(nx, ny)) {
                        BlockType neighbor = reactor.get_block(nx, ny);
                        
                        // We do NOT count Empty blocks here to guarantee the minimum possible heat
                        if (neighbor == BlockType::Reflector) { pulses += outbound_pulses; }
                        else if (neighbor == BlockType::QuadRod) { pulses += 4; }
                        else if (neighbor == BlockType::DoubleRod) { pulses += 2; }
                        else if (neighbor == BlockType::SingleRod) { pulses += 1; }
                    }
                }
                current_heat_gen += (pulses / 2) * pulses + 4;
            }
        }
    }

    return current_heat_gen <= max_cooling_capacity;
}

template <size_t W, size_t H, ReactorConfig Config>
bool passes_rf_bound(const ReactorState<W, H, Config>& reactor, long long current_best_rf) {
    long long max_rf_potential = 0;
    
    constexpr std::array<std::pair<int, int>, 4> directions = {{{0, -1}, {0, 1}, {-1, 0}, {1, 0}}};
    constexpr long long max_quad_rf = 28LL * Config.rf_per_pulse;

    for (int y = 0; y < static_cast<int>(H); y++) {
        for (int x = 0; x < static_cast<int>(W); x++) {
            BlockType block = reactor.get_block(x, y);

            if (block == BlockType::Empty) {
                // The highest possible RF any single slot can generate is a perfectly surrounded QuadRod
                max_rf_potential += max_quad_rf;
            } 
            else if (block == BlockType::SingleRod || block == BlockType::DoubleRod || block == BlockType::QuadRod) {
                
                int internal_pulses = 0;
                int outbound_pulses = 0;
                if (block == BlockType::SingleRod) { internal_pulses = 1; outbound_pulses = 1; }
                else if (block == BlockType::DoubleRod) { internal_pulses = 4; outbound_pulses = 2; }
                else if (block == BlockType::QuadRod) { internal_pulses = 12; outbound_pulses = 4; }

                int pulses = internal_pulses;
                
                for (const auto& dir : directions) {
                    int nx = x + dir.first;
                    int ny = y + dir.second;
                    if (reactor.is_valid_coordinate(nx, ny)) {
                        BlockType neighbor = reactor.get_block(nx, ny);
                        
                        // Assume the best: this empty slot will become whatever yields max pulses (which is 4)
                        if (neighbor == BlockType::Empty) { pulses += 4; }
                        else if (neighbor == BlockType::Reflector) { pulses += outbound_pulses; }
                        else if (neighbor == BlockType::QuadRod) { pulses += 4; }
                        else if (neighbor == BlockType::DoubleRod) { pulses += 2; }
                        else if (neighbor == BlockType::SingleRod) { pulses += 1; }
                    }
                }
                max_rf_potential += static_cast<long long>(pulses) * Config.rf_per_pulse;
            }
        }
    }

    return max_rf_potential >= current_best_rf;
}

template <size_t W, size_t H, ReactorConfig Config, typename BatchManagerType>
void branch_and_bound(int depth, 
                      const std::vector<Coord>& placement_order,
                      ReactorState<W, H, Config>& current_reactor, 
                      BatchManagerType& batch,
                      std::vector<int>& current_path,
                      unsigned long long& nodes_evaluated,
                      std::span<const BlockType> allowed_blocks) {
    nodes_evaluated++;

    // Update the terminal every 50,000 nodes to prevent I/O slowdowns
    if (nodes_evaluated % 10'000'000 == 0) {
        double progress = 0.0;
        double divider = static_cast<double>(allowed_blocks.size());
        
        // Calculate exact fraction of the search space explored
        for (int i = 0; i < depth; i++) {
            progress += current_path[i] / divider;
            divider *= static_cast<double>(allowed_blocks.size());
        }
        
        int percentage = static_cast<int>(progress * 100.0);
        int bar_width = 50;
        int pos = (percentage * bar_width) / 100;

        std::cout << "\r[";
        for (int p = 0; p < bar_width; ++p) {
            if (p < pos) std::cout << "=";
            else if (p == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << percentage << "% | Nodes: " << nodes_evaluated 
                  << " | Best RF/t: " << batch.best_stats.rf << "    " << std::flush;
    }

    if (depth == W * H) {
        // Flatten the valid reactor into the GPU queue
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                batch.grids.push_back(static_cast<uint8_t>(current_reactor.get_block(x, y)));
            }
        }

        // If the queue is full, fire the GPU!
        if (batch.grids.size() >= batch.MAX_BATCH_SIZE * (W * H)) {
            batch.flush_to_gpu();
        }
        return; 
    }

    int target_x = placement_order[depth].x;
    int target_y = placement_order[depth].y;
    int choice_index = 0;

    // THE BLIND HANDOFF KNOB: Stop bounding math for the last 4 blocks.
    // For a 4x4 (16 blocks), depths 12, 13, 14, and 15 will skip the math.
    int blind_depth_threshold = (W * H) - 3; 
    bool is_blind_zone = (depth >= blind_depth_threshold);

    for (BlockType next_block : allowed_blocks) {
        current_path[depth] = choice_index++;
        current_reactor.set_block(target_x, target_y, next_block);

        // If we are in the blind zone, skip the bounds and just blast the tree!
        if (is_blind_zone || (passes_thermal_bound(current_reactor) && passes_rf_bound(current_reactor, batch.best_stats.rf))) {
            branch_and_bound(depth + 1, placement_order, current_reactor, batch, current_path, nodes_evaluated, allowed_blocks);
        }

        current_reactor.set_block(target_x, target_y, BlockType::Empty);
    }
}

}; // namespace reactor_optimizer::traditional