#include <array>
#include <algorithm>

enum class BlockType {
    Empty,
    SingleRod,
    DoubleRod,
    QuadRod,
    Reflector,
    HeatPipe,
    HeatVent,
    Absorber
};

struct ReactorConfig {
    int rf_per_pulse = 9600;
    int meltdown_temp = 3000;
    int vent_divisor = 100;
    int vent_base = 4;
    int pipe_divisor = 4;
    int pipe_base = 10;
    int absorber_cooling = 16; 
};

struct SimResult {
        long long rf_per_tick;
        int max_temp;
        bool melted_down;
        bool stabilized;
        int ticks_ran;
};

template <size_t MAX_W = 62, size_t MAX_H = 62>
class ReactorState {
private:
    // Runtime bounding box for testing designs.
    int active_width;
    int active_height;
    ReactorConfig config;

    // Max size static arrays to work in.
    std::array<std::array<BlockType, MAX_W>, MAX_H> grid;
    std::array<std::array<int, MAX_W>, MAX_H> heat_map;

    // Helper functions for pulse mechanics.
    inline constexpr bool is_rod(BlockType type) const {
        return type == BlockType::SingleRod || 
               type == BlockType::DoubleRod || 
               type == BlockType::QuadRod;
    }

    inline constexpr int get_internal_pulses(BlockType type) const {
        switch (type) {
            case BlockType::SingleRod: return 1;
            case BlockType::DoubleRod: return 4;
            case BlockType::QuadRod: return 12;
            default: return 0;
        }
    }

    inline constexpr int get_outbound_pulses(BlockType type) const {
        switch (type) {
            case BlockType::SingleRod: return 1;
            case BlockType::DoubleRod: return 2;
            case BlockType::QuadRod: return 4;
            default: return 0;
        }
    }

public:
    // Initialize with requested bounds.
    ReactorState(int w, int h, const ReactorConfig& cfg = ReactorConfig{}) 
        : active_width(w), active_height(h), config(cfg) {
        // Clear grids.
        for (auto& row : grid) {
            row.fill(BlockType::Empty);
        }
        for (auto& row : heat_map) {
            row.fill(0);
        }
    }

    inline BlockType get_block(int x, int y) {
        return grid[y][x];
    }

    inline void set_block(int x, int y, BlockType type) {
        grid[y][x] = type;
    }

    inline bool is_valid_coordinate(int x, int y) const {
        return (x >= 0 && x < active_width && y >=0 && y < active_height);
    }

    SimResult simulate_to_equilibrium() {
        long long rf_per_tick = 0;

        for (auto& row : heat_map) {
            row.fill(0);
        }

        constexpr std::array<std::pair<int, int>, 4> directions = {{{0, -1}, {0, 1}, {-1, 0}, {1, 0}}};

        // Precompute Adjacency Lists.
        struct Coord { int x, y; };
        
        struct GeneratorData { int x, y, heat_gen; };
        GeneratorData generators[MAX_W * MAX_H];
        int num_generators = 0;

        struct ComponentData { 
            int x, y; 
            Coord neighbors[4]; 
            int num_neighbors; 
        };
        ComponentData absorbers[MAX_W * MAX_H];
        int num_absorbers = 0;
        ComponentData pipes[MAX_W * MAX_H];
        int num_pipes = 0;
        ComponentData vents[MAX_W * MAX_H];
        int num_vents = 0;
        
        Coord heat_holders[MAX_W * MAX_H]; 
        int num_heat_holders = 0;

        long long total_heat_generated = 0;

        // Precomputation.
        for (int y = 0; y < active_height; y++) {
            for (int x = 0; x < active_width; x++) {
                BlockType curr_block = get_block(x, y);

                if (curr_block == BlockType::Empty || curr_block == BlockType::Reflector) {
                    continue; // Skip non-active static blocks entirely.
                }

                if (is_rod(curr_block) || curr_block == BlockType::HeatPipe) {
                    heat_holders[num_heat_holders++] = {x, y};
                }

                // Gather valid, non-empty neighbors for this specific coordinate.
                Coord valid_neighbors[4];
                int n_count = 0;
                for (const auto& dir : directions) {
                    int nx = x + dir.first;
                    int ny = y + dir.second;
                    if (is_valid_coordinate(nx, ny) && get_block(nx, ny) != BlockType::Empty) {
                        valid_neighbors[n_count++] = {nx, ny};
                    }
                }

                // Catalog active cooling components with their neighbors.
                if (curr_block == BlockType::Absorber) {
                    absorbers[num_absorbers] = {x, y, {}, n_count};
                    std::copy(valid_neighbors, valid_neighbors + n_count, absorbers[num_absorbers].neighbors);
                    num_absorbers++;
                }
                else if (curr_block == BlockType::HeatPipe) {
                    pipes[num_pipes] = {x, y, {}, n_count};
                    std::copy(valid_neighbors, valid_neighbors + n_count, pipes[num_pipes].neighbors);
                    num_pipes++;
                }
                else if (curr_block == BlockType::HeatVent) {
                    vents[num_vents] = {x, y, {}, n_count};
                    std::copy(valid_neighbors, valid_neighbors + n_count, vents[num_vents].neighbors);
                    num_vents++;
                }
                
                // Catalog generators and calculate pulses.
                if (is_rod(curr_block)) {
                    int total_pulses = get_internal_pulses(curr_block);
                    int curr_outbound = get_outbound_pulses(curr_block);

                    // Rod pulse logic.
                    for (const auto& dir : directions) {
                        int nx = x + dir.first;
                        int ny = y + dir.second;
                        if (!is_valid_coordinate(nx, ny)) continue;

                        BlockType neighbor = get_block(nx, ny);
                        if (neighbor == BlockType::Reflector) {
                            total_pulses += curr_outbound;
                        } else if (is_rod(neighbor)) {
                            total_pulses += get_outbound_pulses(neighbor);
                        }
                    }

                    rf_per_tick += total_pulses * config.rf_per_pulse;

                    int heat_gen = (total_pulses / 2) * total_pulses + 4;
                    if (heat_gen > 0) {
                        generators[num_generators++] = {x, y, heat_gen}; 
                        total_heat_generated += heat_gen;
                    }
                }
            }
        }

        // If the layout generates more heat than the absolute max theoretical cooling capacity, fail.
        int max_vent_cooling = (config.meltdown_temp / config.vent_divisor) + config.vent_base;
        long long max_theoretical_cooling = (num_absorbers * config.absorber_cooling * 4) + (num_vents * max_vent_cooling);

        if (total_heat_generated > max_theoretical_cooling) {
            return {rf_per_tick, config.meltdown_temp + 1, true, false, 0};
        }

        // Track state history to help identify a stable state.
        constexpr int HISTORY_SIZE = 32;
        struct StateSnapshot {
            long long total_heat = -1;
            int max_temp = -1;
            unsigned int state_hash = 0;
        };
        std::array<StateSnapshot, HISTORY_SIZE> history;
        int history_idx = 0;

        bool stabilized = false;
        int max_temp = 0;
        int ticks = 0;

        while (!stabilized && ticks < 1500) {
            max_temp = 0;
            int intra_tick_max = 0;
            
            // Add heat.
            for (int i = 0; i < num_generators; i++) {
                int x = generators[i].x;
                int y = generators[i].y;
                heat_map[y][x] += generators[i].heat_gen;

                if (heat_map[y][x] > intra_tick_max) {
                    intra_tick_max = heat_map[y][x];
                }
            }

            if (intra_tick_max > config.meltdown_temp) {
                return {rf_per_tick, intra_tick_max, true, false, ticks};
            }

            // Tick all active heat moving or removing blocks in the reactor. 
            // Ordering is to assume worst case scenario for cooling since Oritech 
            // uses non-deterministic Iterators over HashMaps.

            // Absorbers.
            for (int i = 0; i < num_absorbers; i++) {
                for (int n = 0; n < absorbers[i].num_neighbors; n++) {
                    int nx = absorbers[i].neighbors[n].x;
                    int ny = absorbers[i].neighbors[n].y;
                    if (heat_map[ny][nx] > 0) {
                        heat_map[ny][nx] -= config.absorber_cooling;
                    }
                }
            }

            // Heat Pipes.
            for (int i = 0; i < num_pipes; i++) {
                int curr_heat = heat_map[pipes[i].y][pipes[i].x];
                for (int n = 0; n < pipes[i].num_neighbors; n++) {
                    int nx = pipes[i].neighbors[n].x;
                    int ny = pipes[i].neighbors[n].y;
                    int neighbor_heat = heat_map[ny][nx];
                    
                    if (neighbor_heat > curr_heat) {
                        int diff = neighbor_heat - curr_heat;
                        int gained = std::min(diff / config.pipe_divisor + config.pipe_base, diff);
                        heat_map[ny][nx] -= gained;
                        curr_heat += gained;
                    }
                }
                heat_map[pipes[i].y][pipes[i].x] = curr_heat;
            }

            // Heat Vents.
            for (int i = 0; i < num_vents; i++) {
                int max_neighbor_heat = 0;
                int hx = -1, hy = -1;
                for (int n = 0; n < vents[i].num_neighbors; n++) {
                    int nx = vents[i].neighbors[n].x;
                    int ny = vents[i].neighbors[n].y;
                    int neighbor_heat = heat_map[ny][nx];
                    
                    if (neighbor_heat > max_neighbor_heat) {
                        max_neighbor_heat = neighbor_heat;
                        hx = nx;
                        hy = ny;
                    }
                }
                if (hx != -1) {
                    int removed = std::min(max_neighbor_heat / config.vent_divisor + config.vent_base, max_neighbor_heat);
                    heat_map[hy][hx] -= removed;
                }
            }

            // Calculate Totals and the state hash.
            int curr_max_temp = 0;
            long long curr_total_heat = 0;
            unsigned int curr_hash = 0;
            
            for (int i = 0; i < num_heat_holders; i++) {
                int x = heat_holders[i].x;
                int y = heat_holders[i].y;
                int local_heat = heat_map[y][x];

                curr_total_heat += local_heat;
                if (local_heat > curr_max_temp) {
                    curr_max_temp = local_heat;
                }
                
                curr_hash ^= (static_cast<unsigned int>(local_heat) + i) * 2654435761u;
            }

            if (curr_max_temp > config.meltdown_temp) {
                return {rf_per_tick, curr_max_temp, true, stabilized, ticks};
            }

            // Cycle detection.
            for (int i = 0; i < HISTORY_SIZE; i++) {
                if (history[i].total_heat == curr_total_heat && 
                    history[i].max_temp == curr_max_temp && 
                    history[i].state_hash == curr_hash) {
                    stabilized = true;
                    break;
                }
            }

            // Record current state to history buffer.
            history[history_idx] = {curr_total_heat, curr_max_temp, curr_hash};
            history_idx = (history_idx + 1) % HISTORY_SIZE;
            
            max_temp = curr_max_temp;
            ticks++;
        }

        if (!stabilized) {
            return {rf_per_tick, max_temp, true, stabilized, ticks};
        }

        return {rf_per_tick, max_temp, false, stabilized, ticks};
    }
};