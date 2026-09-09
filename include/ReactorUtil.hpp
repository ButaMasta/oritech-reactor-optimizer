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
    int meltdown_temp = 4000;
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

        // Precompute the internal/external pulses and heat generation of the grid.
        std::array<std::array<int, MAX_W>, MAX_H> heat_gen_per_tick;
        for (auto& row : heat_gen_per_tick) {
            row.fill(0);
        }

        // Cardinal directions.
        constexpr std::array<std::pair<int, int>, 4> directions = {{{0, -1}, {0, 1}, {-1, 0}, {1, 0}}};

        // Precomputation.
        for (int y = 0; y < active_height; y++) {
            for (int x = 0; x < active_width; x++) {
                BlockType curr_block = get_block(x, y);

                // Check if it is a rod.
                if (!is_rod(curr_block)) {
                    continue;
                }

                int total_pulses = get_internal_pulses(curr_block);
                int curr_outbound = get_outbound_pulses(curr_block);

                for (const auto& dir : directions) {
                    int nx = x + dir.first;
                    int ny = y + dir.second;

                    if (!is_valid_coordinate(nx, ny)) {
                        continue;
                    }

                    BlockType neighbor = get_block(nx, ny);

                    if (neighbor == BlockType::Reflector) {
                        total_pulses += curr_outbound;
                    } else if (is_rod(neighbor)) {
                        total_pulses += get_outbound_pulses(neighbor);
                    }
                }

                // RF generation scales linearly with pulses.
                rf_per_tick += total_pulses * config.rf_per_pulse;

                // Calculate heat per tick. Accounting for integer division here.
                heat_gen_per_tick[y][x] = (total_pulses / 2) * total_pulses + 4;
            }
        }

        // Track state history to help identify a stable state.
        constexpr int HISTORY_SIZE = 32;
        struct StateSnapshot {
            long long total_heat = -1;
            int max_temp = -1;
        };
        std::array<StateSnapshot, HISTORY_SIZE> history;
        int history_idx = 0;

        // Run tick sim for heat.
        bool stabilized = false;
        int max_temp = 0;
        int ticks = 0;

        while (!stabilized && ticks < 1500) {
            max_temp = 0;
            int intra_tick_max = 0;

            // Add heat.
            for (int y = 0; y < active_height; y++) {
                for (int x = 0; x < active_width; x++) {                    
                    heat_map[y][x] += heat_gen_per_tick[y][x];

                    if (heat_map[y][x] > intra_tick_max) {
                        intra_tick_max = heat_map[y][x];
                    }
                }
            }

            // Instantly fail if any component ever passes meltdown temp.
            if (intra_tick_max > config.meltdown_temp) {
                return {rf_per_tick, intra_tick_max, true, false, ticks};
            }

            // Component interactions.
            for (int y = 0; y < active_height; y++) {
                for (int x = 0; x < active_width; x++) {
                    BlockType curr_block = get_block(x, y);
                    
                    switch (curr_block) {
                        case BlockType::HeatPipe: {
                            int curr_heat = heat_map[y][x];

                            for (const auto& dir : directions) {
                                int nx = x + dir.first;
                                int ny = y + dir.second;
                                if (is_valid_coordinate(nx, ny) && get_block(nx, ny) != BlockType::Empty) {
                                    int neighbor_heat = heat_map[ny][nx];

                                    if (neighbor_heat <= curr_heat) {
                                        continue;
                                    }

                                    int diff = neighbor_heat - curr_heat;
                                    int gained = std::min(diff / config.pipe_divisor + config.pipe_base, diff);

                                    // Mutate heat map during calculation (why oritech... why...).
                                    heat_map[ny][nx] -= gained;
                                    curr_heat += gained;
                                }
                            }
                            heat_map[y][x] = curr_heat;
                            break;
                        }
                        case BlockType::Absorber: {
                            for (const auto& dir : directions) {
                                int nx = x + dir.first;
                                int ny = y + dir.second;
                                if (is_valid_coordinate(nx, ny) && get_block(nx, ny) != BlockType::Empty) {
                                    int neighbor_heat = heat_map[ny][nx];

                                    if (neighbor_heat <= 0) {
                                        continue;
                                    }

                                    // Allows for negative temperatures.
                                    heat_map[ny][nx] -= config.absorber_cooling;
                                }
                            }
                            break;
                        }
                        case BlockType::HeatVent: {
                            int max_neighbor_heat = 0;
                            int hx = -1;
                            int hy = -1;

                            for (const auto& dir : directions) {
                                int nx = x + dir.first;
                                int ny = y + dir.second;
                                if (is_valid_coordinate(nx, ny) && get_block(nx, ny) != BlockType::Empty) {
                                    int neighbor_heat = heat_map[ny][nx];
                                    
                                    if (neighbor_heat <= max_neighbor_heat) {
                                        continue;
                                    }

                                    max_neighbor_heat = neighbor_heat;
                                    hx = nx;
                                    hy = ny;
                                }

                            }
                            if (max_neighbor_heat != 0 && hx != -1) {
                                int removed = std::min(max_neighbor_heat / config.vent_divisor + config.vent_base, max_neighbor_heat);
                                heat_map[hy][hx] -= removed;
                            }
                            break;
                        }
                        default:
                            // In case of an air block.
                            break;
                    }
                }
            }

            // Check limits and calculate total.
            int curr_max_temp = 0;
            long long curr_total_heat = 0;
            for (int y = 0; y < active_height; y++) {
                for (int x = 0; x < active_width; x++) {
                    int local_heat = heat_map[y][x];

                    curr_total_heat += local_heat;
                    
                    if (local_heat > curr_max_temp) {
                        curr_max_temp = local_heat;
                    }
                }
            }

            if (curr_max_temp > config.meltdown_temp) {
                return {rf_per_tick, curr_max_temp, true, stabilized, ticks};
            }

            // Cycle detection for a stable state.
            for (int i = 0; i < HISTORY_SIZE; i++) {
                if (history[i].total_heat == curr_total_heat && history[i].max_temp == curr_max_temp) {
                    stabilized = true;
                    break;
                }
            }

            // Record current state to history buffer.
            history[history_idx] = {curr_total_heat, curr_max_temp};
            history_idx = (history_idx + 1) % HISTORY_SIZE;
            
            max_temp = curr_max_temp;
            ticks++;
        }

        // If it has not stabilized by the time we hit max ticks then assume it will meltdown.
        if (!stabilized) {
            return {rf_per_tick, max_temp, true, stabilized, ticks};
        }

        return {rf_per_tick, max_temp, false, stabilized, ticks};
    }
};