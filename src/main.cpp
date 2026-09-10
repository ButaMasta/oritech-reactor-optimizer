#include "../include/ReactorUtil.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <chrono>
#include <future>
#include <fstream>
#include <map>
#include <locale>
#include <torch/torch.h>

// Neat printing.
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"

constexpr size_t OPTIMIZE_W = 5;
constexpr size_t OPTIMIZE_H = 5;

constexpr std::array<BlockType, 4> ALLOWED_BLOCKS = {
    // BlockType::Empty,
    // BlockType::SingleRod,
    // BlockType::DoubleRod,
    BlockType::QuadRod,
    // BlockType::Reflector,
    BlockType::HeatPipe,
    BlockType::HeatVent,
    BlockType::Absorber
};

// Must match the struct in your .cu file exactly
struct KernelConfig {
    int rf_per_pulse;
    int meltdown_temp;
    int vent_divisor;
    int vent_base;
    int pipe_divisor;
    int pipe_base;
    int absorber_cooling; 
};

// Expose the CUDA wrapper function
extern void launch_gpu_batch_eval(
    const uint8_t* d_grids, 
    int batch_size, 
    KernelConfig config,
    long long* d_out_rf, 
    int* d_out_temp, 
    bool* d_out_melted
);

struct Coord { int x, y; };

// Dynamically generates a center-out spiral for ANY grid dimensions.
std::vector<Coord> generate_spiral_order(int w, int h) {
    std::vector<Coord> order;
    order.reserve(w * h);
    
    // Find the mathematical center of the grid.
    float cx = (w - 1) / 2.0f;
    float cy = (h - 1) / 2.0f;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            order.push_back({x, y});
        }
    }

    // Sort coordinates by squared distance to the center.
    std::sort(order.begin(), order.end(), [cx, cy](const Coord& a, const Coord& b) {
        float distA = (a.x - cx) * (a.x - cx) + (a.y - cy) * (a.y - cy);
        float distB = (b.x - cx) * (b.x - cx) + (b.y - cy) * (b.y - cy);
        return distA < distB; // Closest to center comes first.
    });

    return order;
}

template <size_t W, size_t H, ReactorConfig Config = ReactorConfig{}>
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

template <size_t W, size_t H, ReactorConfig Config = ReactorConfig{}>
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

// Parser to convert ASCII characters to BlockTypes.
BlockType char_to_block(char c) {
    switch (c) {
        case '1': return BlockType::SingleRod;
        case '2': return BlockType::DoubleRod;
        case '4': return BlockType::QuadRod;
        case 'R': return BlockType::Reflector;
        case 'P': return BlockType::HeatPipe;
        case 'V': return BlockType::HeatVent;
        case 'A': return BlockType::Absorber;
        case '.':
        default:  return BlockType::Empty;
    }
}

// Ingests the 2D string array and loads it into pre-allocated grid.
void loadLayout(ReactorState<OPTIMIZE_W, OPTIMIZE_H>& reactor, const std::vector<std::string>& layout) {
    for (size_t y = 0; y < layout.size(); ++y) {
        for (size_t x = 0; x < layout[y].size(); ++x) {
            reactor.set_block(x, y, char_to_block(layout[y][x]));
        }
    }
}

ReactorState<OPTIMIZE_W, OPTIMIZE_H> optimize_layout(int seed, bool show_progress = false) {
    ReactorState<OPTIMIZE_W, OPTIMIZE_H> current_reactor;
    ReactorState<OPTIMIZE_W, OPTIMIZE_H> best_reactor;

    std::random_device rd;

    // Initialize random number generators.
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> dist_type(0, ALLOWED_BLOCKS.size() - 1);
    std::uniform_int_distribution<int> dist_x(0, OPTIMIZE_W - 1);
    std::uniform_int_distribution<int> dist_y(0, OPTIMIZE_H - 1);
    std::uniform_real_distribution<double> dist_prob(0.0, 1.0);

    // Seed the initial state with completely random blocks.
    for (int y = 0; y < OPTIMIZE_H; ++y) {
        for (int x = 0; x < OPTIMIZE_W; ++x) {
            current_reactor.set_block(x, y, BlockType::Empty);
        }
    }
    // for (int y = 0; y < OPTIMIZE_H; ++y) {
    //     for (int x = 0; x < OPTIMIZE_W; ++x) {
    //         current_reactor.set_block(x, y, static_cast<BlockType>(dist_type(rng)));
    //     }
    // }
    
    // Evaluate baseline.
    auto initial_result = current_reactor.simulate_to_equilibrium();
    // Multiply by 1000 to ensure even minor meltdowns score worse than an empty 0 RF/t grid.
    long long current_score = initial_result.melted_down ? -static_cast<long long>(initial_result.max_temp) * 1000 : initial_result.rf_per_tick;
    long long best_score = current_score;
    best_reactor = current_reactor;

    // Hyperparameters.
    double start_temp = 100'000.0;
    double end_temp = 0.1;
    int max_iterations = 10'000'000;

    // Calculates the exact multiplier to reach end_temp at max_iterations.
    double cooling_rate = std::pow(end_temp / start_temp, 1.0 / max_iterations);
    double temperature = start_temp;

    if (show_progress) {
        std::cout << "Starting Simulated Annealing (" << max_iterations << " iterations)..." << std::endl;
    }
    
    // Throttle I/O updates to only 100 times total.
    int update_interval = max_iterations / 100;

    for (int i = 0; i < max_iterations; i++) {
        // Pick a random coordinate to mutate.
        int mx = dist_x(rng);
        int my = dist_y(rng);
        
        BlockType old_block = current_reactor.get_block(mx, my);
        BlockType new_block = ALLOWED_BLOCKS[dist_type(rng)];

        if (old_block == new_block) continue; // Skip redundant checks.

        // Apply mutation in-place.
        current_reactor.set_block(mx, my, new_block);

        // Score the mutated layout.
        auto new_result = current_reactor.simulate_to_equilibrium();
        long long new_score = new_result.melted_down ? -static_cast<long long>(new_result.max_temp) * 1000 : new_result.rf_per_tick;

        // Calculate delta.
        double delta = static_cast<double>(new_score - current_score);
        bool accept = false;

        // Stochastic Acceptance Logic.
        if (delta > 0) {
            accept = true; // Always accept improvements.
        } else {
            // Explore worse layouts based on current temperature.
            double p = std::exp(delta / temperature);
            if (dist_prob(rng) < p) {
                accept = true;
            }
        }

        if (accept) {
            current_score = new_score;
            if (current_score > best_score) {
                best_score = current_score;
                best_reactor = current_reactor;
            }
        } else {
            current_reactor.set_block(mx, my, old_block);
        }

        temperature *= cooling_rate;

        // Terminal Progress Bar (Proxy Thread Only).
        if (show_progress && (i % update_interval == 0 || i == max_iterations - 1)) {
            int percentage = (i * 100) / max_iterations;
            int bar_width = 50;
            int pos = (percentage * bar_width) / 100;

            std::cout << "\r[";
            for (int p = 0; p < bar_width; ++p) {
                if (p < pos) std::cout << "=";
                else if (p == pos) std::cout << ">";
                else std::cout << " ";
            }
            
            std::cout << "] " << percentage << "% | Temp: " << static_cast<int>(temperature) 
                      << " | Best RF/t: " << best_score << "   " << std::flush; 
        }
    }

    return best_reactor;
}

template <size_t W, size_t H>
int calculate_symmetry_score(const ReactorState<W, H>& reactor) {
    int score = 0;
    
    // 1. Horizontal Symmetry (Left mirrors Right)
    for (int y = 0; y < static_cast<int>(H); y++) {
        for (int x = 0; x < static_cast<int>(W) / 2; x++) {
            if (reactor.get_block(x, y) == reactor.get_block(W - 1 - x, y)) score++;
        }
    }
    
    // 2. Vertical Symmetry (Top mirrors Bottom)
    for (int y = 0; y < static_cast<int>(H) / 2; y++) {
        for (int x = 0; x < static_cast<int>(W); x++) {
            if (reactor.get_block(x, y) == reactor.get_block(x, H - 1 - y)) score++;
        }
    }
    
    if constexpr (W == H) {
        // 3. Diagonal Symmetries
        for (int y = 0; y < static_cast<int>(H); y++) {
            for (int x = y + 1; x < static_cast<int>(W); x++) {
                if (reactor.get_block(x, y) == reactor.get_block(y, x)) score++;
                if (reactor.get_block(x, y) == reactor.get_block(W - 1 - y, H - 1 - x)) score++;
            }
        }
        
        // 4. Rotational Symmetry (Tiling / Patterns)
        for (int y = 0; y < static_cast<int>(H); y++) {
            for (int x = 0; x < static_cast<int>(W); x++) {
                // 90-degree rotation check
                if (reactor.get_block(x, y) == reactor.get_block(H - 1 - y, x)) score++;
                // 180-degree rotation check
                if (reactor.get_block(x, y) == reactor.get_block(W - 1 - x, H - 1 - y)) score++;
            }
        }
    }
    
    return score;
}

struct BestStats {
    long long rf = 921'600;
    int temp = 999999;
    int sym_score = -1;
};

template <size_t W, size_t H>
struct BatchManager {
    std::vector<uint8_t> grids;
    BestStats& best_stats;
    ReactorState<W, H>& best_reactor;
    
    // 1. Increase batch size to 1,000,000 to saturate the RTX 5070
    const size_t MAX_BATCH_SIZE = 1'000'000; 
    
    // 2. Persistent GPU Tensors
    torch::Tensor t_out_rf;
    torch::Tensor t_out_temp;
    torch::Tensor t_out_melted;

    // 3. Constructor to allocate memory exactly ONCE
    BatchManager(BestStats& stats, ReactorState<W, H>& best) 
        : best_stats(stats), best_reactor(best) {
        
        // Pre-reserve CPU memory so the vector doesn't resize
        grids.reserve(MAX_BATCH_SIZE * W * H); 

        // Pre-allocate VRAM
        auto device = torch::kCUDA;
        t_out_rf = torch::empty({(long long)MAX_BATCH_SIZE}, torch::TensorOptions().dtype(torch::kInt64).device(device));
        t_out_temp = torch::empty({(long long)MAX_BATCH_SIZE}, torch::TensorOptions().dtype(torch::kInt32).device(device));
        t_out_melted = torch::empty({(long long)MAX_BATCH_SIZE}, torch::TensorOptions().dtype(torch::kBool).device(device));
    }
    
    void flush_to_gpu() {
        if (grids.empty()) return;
        
        int current_batch = grids.size() / (W * H);
        
        // Instantly map the vector to a CPU tensor, then ship to GPU
        auto options_grid = torch::TensorOptions().dtype(torch::kUInt8);
        torch::Tensor t_grids_cpu = torch::from_blob(grids.data(), {current_batch, W * H}, options_grid);
        
        // non_blocking=true allows the transfer to happen slightly faster
        torch::Tensor t_grids_gpu = t_grids_cpu.to(torch::kCUDA, /*non_blocking=*/true); 
        
        constexpr ReactorConfig cfg;
        KernelConfig k_cfg {
            cfg.rf_per_pulse, cfg.meltdown_temp, cfg.vent_divisor, 
            cfg.vent_base, cfg.pipe_divisor, cfg.pipe_base, cfg.absorber_cooling
        };
        
        // Fire the kernel using our pre-allocated persistent pointers
        launch_gpu_batch_eval(
            t_grids_gpu.data_ptr<uint8_t>(),
            current_batch,
            k_cfg,
            reinterpret_cast<long long*>(t_out_rf.data_ptr<int64_t>()),
            reinterpret_cast<int*>(t_out_temp.data_ptr<int32_t>()),
            t_out_melted.data_ptr<bool>()
        );
        
        // Slice the persistent tensors down to the current batch size before copying back to CPU
        torch::Tensor rf_cpu = t_out_rf.slice(0, 0, current_batch).to(torch::kCPU);
        torch::Tensor temp_cpu = t_out_temp.slice(0, 0, current_batch).to(torch::kCPU);
        torch::Tensor melted_cpu = t_out_melted.slice(0, 0, current_batch).to(torch::kCPU);
        
        auto* rf_ptr = rf_cpu.data_ptr<int64_t>();
        auto* temp_ptr = temp_cpu.data_ptr<int32_t>();
        auto* melted_ptr = melted_cpu.data_ptr<bool>();
        
        // 5. Evaluate the results
        for (int i = 0; i < current_batch; i++) {
            if (!melted_ptr[i]) {
                long long r = rf_ptr[i];
                int t = temp_ptr[i];
                
                bool is_new_best = false;
                
                // TIER 1: Raw Power Output
                if (r > best_stats.rf) {
                    is_new_best = true;
                } 
                // TIER 2: Thermal Efficiency
                else if (r == best_stats.rf) {
                    if (t < best_stats.temp) {
                        is_new_best = true;
                    } 
                    // TIER 3: Aesthetic Pattern
                    else if (t == best_stats.temp) {
                        // Lazy load the grid to calculate symmetry ONLY if it tied RF and Temp
                        ReactorState<W, H> temp_reactor;
                        for(int k=0; k < W*H; k++) {
                            temp_reactor.set_block(k%W, k/W, static_cast<BlockType>(grids[i*(W*H) + k]));
                        }
                        if (calculate_symmetry_score(temp_reactor) > best_stats.sym_score) {
                            is_new_best = true;
                        }
                    }
                }
                
                // Save and output if it passed any tier
                if (is_new_best) {
                    best_stats.rf = r;
                    best_stats.temp = t;
                    
                    ReactorState<W, H> temp_reactor;
                    for(int k=0; k < W*H; k++) {
                        temp_reactor.set_block(k%W, k/W, static_cast<BlockType>(grids[i*(W*H) + k]));
                    }
                    
                    best_stats.sym_score = calculate_symmetry_score(temp_reactor);
                    best_reactor = temp_reactor;
                    
                    std::cout << "\r\033[K[GPU] New Best -> RF/t: " << best_stats.rf 
                              << " | Temp: " << best_stats.temp << " C" 
                              << " | Sym: " << best_stats.sym_score << "       \n";
                }
            }
        }
        
        grids.clear();
    }
};

template <size_t W, size_t H>
void branch_and_bound(int depth, 
                      const std::vector<Coord>& placement_order,
                      ReactorState<W, H>& current_reactor, 
                      BatchManager<W, H>& batch,
                      std::vector<int>& current_path,
                      unsigned long long& nodes_evaluated) {
    nodes_evaluated++;

    // Update the terminal every 50,000 nodes to prevent I/O slowdowns
    if (nodes_evaluated % 10'000'000 == 0) {
        double progress = 0.0;
        double divider = static_cast<double>(ALLOWED_BLOCKS.size());
        
        // Calculate exact fraction of the search space explored
        for (int i = 0; i < depth; i++) {
            progress += current_path[i] / divider;
            divider *= static_cast<double>(ALLOWED_BLOCKS.size());
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

    for (BlockType next_block : ALLOWED_BLOCKS) {
        current_path[depth] = choice_index++;
        current_reactor.set_block(target_x, target_y, next_block);

        // If we are in the blind zone, skip the bounds and just blast the tree!
        if (is_blind_zone || (passes_thermal_bound(current_reactor) && passes_rf_bound(current_reactor, batch.best_stats.rf))) {
            branch_and_bound(depth + 1, placement_order, current_reactor, batch, current_path, nodes_evaluated);
        }

        current_reactor.set_block(target_x, target_y, BlockType::Empty);
    }
}

void test_layout() {
    // Define layout.
    // std::vector<std::string> test_layout = {
    //     "R2A12",
    //     "2AP2A",
    //     "AP4A2",
    //     "12A21",
    //     "2A21A"
    // };
    std::vector<std::string> test_layout = {
        "V4A2V",
        "2A214",
        "211VV",
        "V2A4V",
        "V22A2"
    };

    // Instantiate reactor simulation.
    ReactorState<OPTIMIZE_W, OPTIMIZE_H> reactor;
    
    // Load the visual representation into the grid.
    loadLayout(reactor, test_layout);

    // Run the simulation loop.
    std::cout << "Simulating layout to equilibrium..." << std::endl;
    auto result = reactor.simulate_to_equilibrium();

    // Output the results for verification.
    std::cout << "\n--- Simulation Results ---" << std::endl;
    std::cout << "Total RF/t: " << result.rf_per_tick << std::endl;
    std::cout << "Max Temp:   " << result.max_temp << " C" << std::endl;
    if (result.melted_down) {
        std::cout << "Meltdown:   " << RED << "YES (>4000 C)" << RESET << std::endl;
    } else {
        std::cout << "Meltdown:   " << GREEN << "NO" << RESET << std::endl;
    }
    if (result.stabilized) {
        std::cout << "Stabilized: " << GREEN << "YES" << RESET << std::endl;
    } else {
        std::cout << "Stabilized: " << RED << "NO" << RESET << std::endl;
    }
    std::cout << "Ticks Ran:  " << result.ticks_ran << std::endl;
}

void print_layout(ReactorState<OPTIMIZE_W, OPTIMIZE_H>& reactor) {
    for (int y = 0; y < OPTIMIZE_H; ++y) {
        for (int x = 0; x < OPTIMIZE_W; ++x) {
            BlockType block = reactor.get_block(x, y);
            char c = '.';
            switch (block) {
                case BlockType::SingleRod: c = '1'; break;
                case BlockType::DoubleRod: c = '2'; break;
                case BlockType::QuadRod:   c = '4'; break;
                case BlockType::Reflector: c = 'R'; break;
                case BlockType::HeatPipe:  c = 'P'; break;
                case BlockType::HeatVent:  c = 'V'; break;
                case BlockType::Absorber:  c = 'A'; break;
                case BlockType::Empty:
                default: break;
            }
            std::cout << "[ " << c << " ] ";
        }
        std::cout << std::endl;
    }
}

void export_building_gadgets(ReactorState<OPTIMIZE_W, OPTIMIZE_H>& reactor, const std::string& filename) {
    std::map<std::string, int> counts;
    std::string statelist = "";
    
    // Add +2 to X and Z to accommodate the outer encasing walls. Y is exactly 3 layers tall.
    int x_size = OPTIMIZE_W + 2;
    int z_size = OPTIMIZE_H + 2;
    int y_size = 3;

    bool first_block = true;
    
    auto add_block = [&](int id, const std::string& name) {
        if (!first_block) { statelist += ","; }
        statelist += std::to_string(id);
        first_block = false;
        if (!name.empty() && name != "minecraft:air") {
            counts[name]++;
        }
    };

    // Iterate how Building Gadgets parses volumetric data: Z (outer) -> Y (middle) -> X (inner).
    for (int z = 0; z < z_size; ++z) {
        for (int y = 0; y < y_size; ++y) {
            for (int x = 0; x < x_size; ++x) {
                // Edge logic applies to the X and Z axes for the casing.
                bool is_edge_xz = (x == 0 || x == x_size - 1 || z == 0 || z == z_size - 1);

                if (y == 0) {
                    // LAYER 0 (Floor): Completely solid reactor wall.
                    add_block(1, "oritech:reactor_wall");
                } 
                else if (y == 1) {
                    // LAYER 1 (Core): Edges are walls, inner blocks are the optimized layout.
                    if (is_edge_xz) {
                        add_block(1, "oritech:reactor_wall");
                    } else {
                        BlockType block = reactor.get_block(x - 1, z - 1);
                        switch (block) {
                            case BlockType::SingleRod: add_block(2, "oritech:reactor_rod"); break;
                            case BlockType::DoubleRod: add_block(3, "oritech:reactor_double_rod"); break;
                            case BlockType::QuadRod:   add_block(4, "oritech:reactor_quad_rod"); break;
                            case BlockType::Reflector: add_block(5, "oritech:reactor_reflector"); break;
                            case BlockType::Absorber:  add_block(6, "oritech:reactor_condenser"); break;
                            case BlockType::HeatPipe:  add_block(7, "oritech:reactor_heat_pipe"); break;
                            case BlockType::HeatVent:  add_block(8, "oritech:reactor_vent"); break;
                            case BlockType::Empty:
                            default: add_block(0, "minecraft:air"); break;
                        }
                    }
                } 
                else if (y == 2) {
                    // LAYER 2 (Roof): Edges are walls, inner blocks map ports to their required core components.
                    if (is_edge_xz) {
                        add_block(1, "oritech:reactor_wall");
                    } else {
                        BlockType block = reactor.get_block(x - 1, z - 1);
                        if (block == BlockType::SingleRod || block == BlockType::DoubleRod || block == BlockType::QuadRod) {
                            add_block(10, "oritech:reactor_fuel_port");
                        } else if (block == BlockType::Absorber) {
                            add_block(9, "oritech:reactor_absorber_port");
                        } else {
                            add_block(1, "oritech:reactor_wall");
                        }
                    }
                }
            }
        }
    }

    std::ofstream out(filename);
    out << "{\n";
    out << "  \"name\": \"Optimized Reactor " << OPTIMIZE_W << "x" << OPTIMIZE_H << "\",\n";
    
    // SNBT Palette array mapping IDs 0-9 to their precise NBT properties.
    std::string palette = "{Name:\\\"minecraft:air\\\"},"
                          "{Name:\\\"oritech:reactor_wall\\\"},"
                          "{Name:\\\"oritech:reactor_rod\\\",Properties:{lit:\\\"false\\\"}},"
                          "{Name:\\\"oritech:reactor_double_rod\\\",Properties:{lit:\\\"false\\\"}},"
                          "{Name:\\\"oritech:reactor_quad_rod\\\",Properties:{lit:\\\"false\\\"}},"
                          "{Name:\\\"oritech:reactor_reflector\\\"},"
                          "{Name:\\\"oritech:reactor_condenser\\\"},"
                          "{Name:\\\"oritech:reactor_heat_pipe\\\"},"
                          "{Name:\\\"oritech:reactor_vent\\\"},"
                          "{Name:\\\"oritech:reactor_absorber_port\\\"},"
                          "{Name:\\\"oritech:reactor_fuel_port\\\"}";
    
    // Write the 3D bounding box dimensions and flat integer state list.
    out << "  \"statePosArrayList\": \"{blockstatemap:[" << palette << "],endpos:{X:" << (x_size - 1) << ",Y:" << (y_size - 1) << ",Z:" << (z_size - 1) << "},startpos:{X:0,Y:0,Z:0},statelist:[I;" << statelist << "]}\",\n";
    
    // Write required materials block with correct unicode formatting.
    out << "  \"requiredItems\": {\n";
    bool first_item = true;
    for (const auto& pair : counts) {
        if (!first_item) out << ",\n";
        out << "    \"oritech:Reference{ResourceKey[minecraft:item / " << pair.first << "]\\u003d" << pair.first << "}\": " << pair.second;
        first_item = false;
    }
    out << "\n  }\n";
    out << "}\n";
    out.close();
}

int main() {

    std::cout.imbue(std::locale("en_US.UTF-8"));

    // test_layout();
    // return 0;

    // Branch and Bound.
    std::cout << "Generating Spiral Placement Order for " << OPTIMIZE_W << "x" << OPTIMIZE_H << "..." << std::endl;
    std::vector<Coord> spiral = generate_spiral_order(OPTIMIZE_W, OPTIMIZE_H);

    ReactorState<OPTIMIZE_W, OPTIMIZE_H> current_reactor;
    ReactorState<OPTIMIZE_W, OPTIMIZE_H> absolute_best_reactor;
    
    BestStats best_stats;
    BatchManager<OPTIMIZE_W, OPTIMIZE_H> batch_manager(best_stats, absolute_best_reactor);

    std::vector<int> current_path(OPTIMIZE_W * OPTIMIZE_H, 0);
    unsigned long long nodes_evaluated = 0;

    auto start_time = std::chrono::high_resolution_clock::now();
    std::cout << "Starting Hybrid CPU/GPU Branch and Bound..." << std::endl;
    
    branch_and_bound(0, spiral, current_reactor, batch_manager, current_path, nodes_evaluated);

    // CRITICAL: Flush the remaining queue!
    batch_manager.flush_to_gpu();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Final clean up of the console line
    std::cout << "\r\033[K--- Search Exhausted in " << duration.count() << "ms ---" << std::endl;
    std::cout << "Total Nodes Evaluated: " << nodes_evaluated << std::endl;
    std::cout << "Supposedly Proven Max RF/t: " << best_stats.rf << std::endl;
    std::cout << "Kept Reactor Temp: " << best_stats.temp << std::endl;
    std::cout << "Kept Reactor Symmetry Score: " << best_stats.sym_score << std::endl;
    std::cout << "\n--- Reactor Layout ---" << std::endl;
    print_layout(absolute_best_reactor);

    // Generate the Building Gadgets blueprint.
    std::string filename = "schematics/perfected_reactor_" + std::to_string(OPTIMIZE_W) + "x" + std::to_string(OPTIMIZE_H) + ".json";
    export_building_gadgets(absolute_best_reactor, filename);
    std::cout << "\nSchematic exported to: " << filename << std::endl;
    return 0;

    // Simulated Annealing.
    // int num_threads = 8;

    // auto start_time = std::chrono::high_resolution_clock::now();
    
    // std::cout << "Launching " << num_threads << " parallel optimization threads..." << std::endl;

    // // Launch multiple independent Simulated Annealing runs.
    // std::vector<std::future<ReactorState<OPTIMIZE_W, OPTIMIZE_H>>> futures;
    // for (int i = 0; i < num_threads; ++i) {
    //     bool is_proxy_thread = (i == 0);
    //     futures.push_back(std::async(std::launch::async, optimize_layout, 9481 + i, is_proxy_thread));
    // }

    // // Collect the results and find the absolute best layout.
    // ReactorState<OPTIMIZE_W, OPTIMIZE_H> absolute_best_reactor;
    // long long global_best_rf = -1;
    // SimResult global_best_stats;

    // for (int i = 0; i < num_threads; ++i) {
    //     ReactorState<OPTIMIZE_W, OPTIMIZE_H> thread_result = futures[i].get();
    //     auto stats = thread_result.simulate_to_equilibrium();
        
    //     if (!stats.melted_down && stats.rf_per_tick > global_best_rf) {
    //         global_best_rf = stats.rf_per_tick;
    //         absolute_best_reactor = thread_result;
    //         global_best_stats = stats;
    //     }
    // }
    
    // auto end_time = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // std::cout << "\n--- Optimization Complete in " << duration.count() << "ms ---" << std::endl;
    // std::cout << "Max Optimized RF/t: " << global_best_stats.rf_per_tick << std::endl;
    // std::cout << "Final Peak Temp:    " << global_best_stats.max_temp << " C" << std::endl;
    // std::cout << "\n--- Reactor Layout ---" << std::endl;
    // print_layout(absolute_best_reactor);

    // // Generate the Building Gadgets blueprint.
    // std::string filename = "schematics/optimized_reactor_" + std::to_string(OPTIMIZE_W) + "x" + std::to_string(OPTIMIZE_H) + ".json";
    // export_building_gadgets(absolute_best_reactor, filename);
    // std::cout << "\nSchematic exported to: " << filename << std::endl;

    // return 0;
}