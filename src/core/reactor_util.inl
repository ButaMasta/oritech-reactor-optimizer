namespace reactor_optimizer::core {

template <size_t W, size_t H, ReactorConfig Config>
ReactorState<W, H, Config>::ReactorState() {
    grid.fill(BlockType::Empty);
    heat_map.fill(0);
}

template <size_t W, size_t H, ReactorConfig Config>
SimResult ReactorState<W, H, Config>::simulate_to_equilibrium() {
    long long rf_per_tick = 0;
    heat_map.fill(0);

    constexpr std::array<std::pair<int, int>, 4> directions = {{{0, -1}, {0, 1}, {-1, 0}, {1, 0}}};

    struct GeneratorData { int flat_idx, heat_gen; };
    GeneratorData generators[GRID_SIZE];
    int num_generators = 0;

    struct ComponentData { 
        int flat_idx; 
        int neighbor_indices[4]; 
        int num_neighbors; 
    };
    ComponentData absorbers[GRID_SIZE];
    int num_absorbers = 0;
    ComponentData pipes[GRID_SIZE];
    int num_pipes = 0;
    ComponentData vents[GRID_SIZE];
    int num_vents = 0;
    
    int heat_holders[GRID_SIZE]; 
    int num_heat_holders = 0;

    long long total_heat_generated = 0;

    // Precomputation (Using 2D purely for logic, but storing 1D).
    for (int y = 0; y < static_cast<int>(H); y++) {
        for (int x = 0; x < static_cast<int>(W); x++) {
            int flat_idx = get_flat_idx(x, y);
            BlockType curr_block = grid[flat_idx];

            if (curr_block == BlockType::Empty || curr_block == BlockType::Reflector) {
                continue; 
            }

            if (is_rod(curr_block) || curr_block == BlockType::HeatPipe) {
                heat_holders[num_heat_holders++] = flat_idx;
            }

            // Gather flat indices of valid neighbors.
            int valid_neighbors[4];
            int n_count = 0;
            for (const auto& dir : directions) {
                int nx = x + dir.first;
                int ny = y + dir.second;
                if (is_valid_coordinate(nx, ny) && get_block(nx, ny) != BlockType::Empty) {
                    valid_neighbors[n_count++] = get_flat_idx(nx, ny);
                }
            }

            if (curr_block == BlockType::Absorber) {
                absorbers[num_absorbers] = {flat_idx, {}, n_count};
                std::copy(valid_neighbors, valid_neighbors + n_count, absorbers[num_absorbers].neighbor_indices);
                num_absorbers++;
            }
            else if (curr_block == BlockType::HeatPipe) {
                pipes[num_pipes] = {flat_idx, {}, n_count};
                std::copy(valid_neighbors, valid_neighbors + n_count, pipes[num_pipes].neighbor_indices);
                num_pipes++;
            }
            else if (curr_block == BlockType::HeatVent) {
                vents[num_vents] = {flat_idx, {}, n_count};
                std::copy(valid_neighbors, valid_neighbors + n_count, vents[num_vents].neighbor_indices);
                num_vents++;
            }
            
            if (is_rod(curr_block)) {
                int total_pulses = get_internal_pulses(curr_block);
                int curr_outbound = get_outbound_pulses(curr_block);

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

                rf_per_tick += total_pulses * Config.rf_per_pulse;

                int heat_gen = (total_pulses / 2) * total_pulses + 4;
                if (heat_gen > 0) {
                    generators[num_generators++] = {flat_idx, heat_gen}; 
                    total_heat_generated += heat_gen;
                }
            }
        }
    }

    int max_vent_cooling = (Config.meltdown_temp / Config.vent_divisor) + Config.vent_base;
    long long max_theoretical_cooling = (num_absorbers * Config.absorber_cooling * 4) + (num_vents * max_vent_cooling);

    if (total_heat_generated > max_theoretical_cooling) {
        return {rf_per_tick, Config.meltdown_temp + 1, true, false, 0};
    }

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
        
        for (int i = 0; i < num_generators; i++) {
            int f_idx = generators[i].flat_idx;
            heat_map[f_idx] += generators[i].heat_gen;

            if (heat_map[f_idx] > intra_tick_max) {
                intra_tick_max = heat_map[f_idx];
            }
        }

        if (intra_tick_max > Config.meltdown_temp) {
            return {rf_per_tick, intra_tick_max, true, false, ticks};
        }

        // Absorbers.
        for (int i = 0; i < num_absorbers; i++) {
            for (int n = 0; n < absorbers[i].num_neighbors; n++) {
                int n_idx = absorbers[i].neighbor_indices[n];
                heat_map[n_idx] = std::max(0, heat_map[n_idx] - Config.absorber_cooling);
            }
        }

        // Heat Pipes.
        for (int i = 0; i < num_pipes; i++) {
            int p_idx = pipes[i].flat_idx;
            int curr_heat = heat_map[p_idx];
            
            for (int n = 0; n < pipes[i].num_neighbors; n++) {
                int n_idx = pipes[i].neighbor_indices[n];
                int neighbor_heat = heat_map[n_idx];
                
                if (neighbor_heat > curr_heat) {
                    int diff = neighbor_heat - curr_heat;
                    int gained = std::min(diff / Config.pipe_divisor + Config.pipe_base, diff);
                    heat_map[n_idx] -= gained;
                    curr_heat += gained;
                }
            }
            heat_map[p_idx] = curr_heat;
        }

        // Heat vents.
        for (int i = 0; i < num_vents; i++) {
            int max_neighbor_heat = 0;
            int target_idx = -1;
            
            for (int n = 0; n < vents[i].num_neighbors; n++) {
                int n_idx = vents[i].neighbor_indices[n];
                int neighbor_heat = heat_map[n_idx];
                
                if (neighbor_heat > max_neighbor_heat) {
                    max_neighbor_heat = neighbor_heat;
                    target_idx = n_idx;
                }
            }
            if (target_idx != -1) {
                int removed = std::min(max_neighbor_heat / Config.vent_divisor + Config.vent_base, max_neighbor_heat);
                heat_map[target_idx] -= removed;
            }
        }

        int curr_max_temp = 0;
        long long curr_total_heat = 0;
        unsigned int curr_hash = 0;
        
        for (int i = 0; i < num_heat_holders; i++) {
            int h_idx = heat_holders[i];
            int local_heat = heat_map[h_idx];

            curr_total_heat += local_heat;
            if (local_heat > curr_max_temp) {
                curr_max_temp = local_heat;
            }
            
            curr_hash ^= (static_cast<unsigned int>(local_heat) + i) * 2654435761u;
        }

        if (curr_max_temp > Config.meltdown_temp) {
            return {rf_per_tick, curr_max_temp, true, stabilized, ticks};
        }

        for (int i = 0; i < HISTORY_SIZE; i++) {
            if (history[i].total_heat == curr_total_heat && 
                history[i].max_temp == curr_max_temp && 
                history[i].state_hash == curr_hash) {
                stabilized = true;
                break;
            }
        }

        history[history_idx] = {curr_total_heat, curr_max_temp, curr_hash};
        history_idx = (history_idx + 1) & (HISTORY_SIZE - 1);
        
        max_temp = curr_max_temp;
        ticks++;
    }

    return {rf_per_tick, max_temp, !stabilized, stabilized, ticks};
}

template <size_t W, size_t H, ReactorConfig Config>
std::vector<Coord> ReactorState<W, H, Config>::generate_spiral_order() {
    std::vector<Coord> order;
    order.reserve(W * H);
    
    // Find the mathematical center of the grid.
    float cx = (W - 1) / 2.0f;
    float cy = (H - 1) / 2.0f;

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
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

template <size_t W, size_t H, ReactorConfig Config>
int calculate_symmetry_score(const ReactorState<W, H, Config>& reactor) {
    int score = 0;
    
    // Horizontal Symmetry
    for (int y = 0; y < static_cast<int>(H); y++) {
        for (int x = 0; x < static_cast<int>(W) / 2; x++) {
            if (reactor.get_block(x, y) == reactor.get_block(W - 1 - x, y)) score++;
        }
    }
    
    // Vertical Symmetry
    for (int y = 0; y < static_cast<int>(H) / 2; y++) {
        for (int x = 0; x < static_cast<int>(W); x++) {
            if (reactor.get_block(x, y) == reactor.get_block(x, H - 1 - y)) score++;
        }
    }
    
    if constexpr (W == H) {
        // Diagonal Symmetries
        for (int y = 0; y < static_cast<int>(H); y++) {
            for (int x = y + 1; x < static_cast<int>(W); x++) {
                if (reactor.get_block(x, y) == reactor.get_block(y, x)) score++;
                if (reactor.get_block(x, y) == reactor.get_block(W - 1 - y, H - 1 - x)) score++;
            }
        }
        
        // Rotational Symmetry
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

}; // namespace reactor_optimizer::core