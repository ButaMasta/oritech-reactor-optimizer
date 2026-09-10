#include <cuda_runtime.h>
#include <stdint.h>

// Ensure this perfectly matches your C++ definition
enum class BlockType : uint8_t {
    Empty = 0, SingleRod, DoubleRod, QuadRod,
    Reflector, HeatPipe, HeatVent, Absorber
};

// Config struct passed by value (copied to constant memory automatically)
struct KernelConfig {
    int rf_per_pulse;
    int meltdown_temp;
    int vent_divisor;
    int vent_base;
    int pipe_divisor;
    int pipe_base;
    int absorber_cooling; 
};

template <size_t W, size_t H>
__device__ void simulate_single_reactor(
    const uint8_t* local_grid, 
    const KernelConfig config,
    long long& out_rf, 
    int& out_temp, 
    bool& out_melted) 
{
    constexpr int GRID_SIZE = W * H;
    int heat_map[GRID_SIZE];
    for(int i=0; i<GRID_SIZE; i++) heat_map[i] = 0;

    long long rf_per_tick = 0;
    int generators_idx[GRID_SIZE];
    int generators_heat[GRID_SIZE];
    int num_generators = 0;

    int absorbers_idx[GRID_SIZE];
    int absorbers_neighbors[GRID_SIZE][4];
    int absorbers_num_neighbors[GRID_SIZE];
    int num_absorbers = 0;

    int vents_idx[GRID_SIZE];
    int vents_neighbors[GRID_SIZE][4];
    int vents_num_neighbors[GRID_SIZE];
    int num_vents = 0;

    int pipes_idx[GRID_SIZE];
    int pipes_neighbors[GRID_SIZE][4];
    int pipes_num_neighbors[GRID_SIZE];
    int num_pipes = 0;

    int heat_holders[GRID_SIZE];
    int num_heat_holders = 0;
    long long total_heat_generated = 0;

    const int dx[4] = {0, 0, -1, 1};
    const int dy[4] = {-1, 1, 0, 0};

    // Precomputation (Flat Arrays only)
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int flat_idx = y * W + x;
            BlockType curr_block = static_cast<BlockType>(local_grid[flat_idx]);

            if (curr_block == BlockType::Empty || curr_block == BlockType::Reflector) continue;

            bool is_rod = (curr_block == BlockType::SingleRod || curr_block == BlockType::DoubleRod || curr_block == BlockType::QuadRod);

            if (is_rod || curr_block == BlockType::HeatPipe) {
                heat_holders[num_heat_holders++] = flat_idx;
            }

            int valid_neighbors[4];
            int n_count = 0;
            for (int d = 0; d < 4; d++) {
                int nx = x + dx[d];
                int ny = y + dy[d];
                if (nx >= 0 && nx < W && ny >= 0 && ny < H) {
                    int n_idx = ny * W + nx;
                    if (local_grid[n_idx] != static_cast<uint8_t>(BlockType::Empty)) {
                        valid_neighbors[n_count++] = n_idx;
                    }
                }
            }

            if (curr_block == BlockType::Absorber) {
                absorbers_idx[num_absorbers] = flat_idx;
                absorbers_num_neighbors[num_absorbers] = n_count;
                for(int i=0; i<n_count; i++) absorbers_neighbors[num_absorbers][i] = valid_neighbors[i];
                num_absorbers++;
            }
            else if (curr_block == BlockType::HeatVent) {
                vents_idx[num_vents] = flat_idx;
                vents_num_neighbors[num_vents] = n_count;
                for(int i=0; i<n_count; i++) vents_neighbors[num_vents][i] = valid_neighbors[i];
                num_vents++;
            }
            else if (curr_block == BlockType::HeatPipe) {
                pipes_idx[num_pipes] = flat_idx;
                pipes_num_neighbors[num_pipes] = n_count;
                for(int i=0; i<n_count; i++) pipes_neighbors[num_pipes][i] = valid_neighbors[i];
                num_pipes++;
            }

            if (is_rod) {
                int pulses = (curr_block == BlockType::SingleRod) ? 1 : (curr_block == BlockType::DoubleRod) ? 4 : 12;
                int outbound = (curr_block == BlockType::SingleRod) ? 1 : (curr_block == BlockType::DoubleRod) ? 2 : 4;

                for (int d = 0; d < 4; d++) {
                    int nx = x + dx[d];
                    int ny = y + dy[d];
                    if (nx >= 0 && nx < W && ny >= 0 && ny < H) {
                        BlockType neighbor = static_cast<BlockType>(local_grid[ny * W + nx]);
                        if (neighbor == BlockType::Reflector) pulses += outbound;
                        else if (neighbor == BlockType::QuadRod) pulses += 4;
                        else if (neighbor == BlockType::DoubleRod) pulses += 2;
                        else if (neighbor == BlockType::SingleRod) pulses += 1;
                    }
                }
                
                rf_per_tick += static_cast<long long>(pulses) * config.rf_per_pulse;
                int heat_gen = (pulses / 2) * pulses + 4;
                if (heat_gen > 0) {
                    generators_idx[num_generators] = flat_idx;
                    generators_heat[num_generators] = heat_gen;
                    num_generators++;
                    total_heat_generated += heat_gen;
                }
            }
        }
    }

    int max_vent_cooling = (config.meltdown_temp / config.vent_divisor) + config.vent_base;
    long long max_theoretical_cooling = (num_absorbers * config.absorber_cooling * 4) + (num_vents * max_vent_cooling);

    if (total_heat_generated > max_theoretical_cooling) {
        out_rf = rf_per_tick;
        out_temp = config.meltdown_temp + 1;
        out_melted = true;
        return;
    }

    // Shrinking the history size to 8 saves precious register memory 
    // while still accurately catching stable oscillating loops.
    constexpr int HISTORY_SIZE = 8;
    long long hist_heat[HISTORY_SIZE];
    int hist_temp[HISTORY_SIZE];
    unsigned int hist_hash[HISTORY_SIZE];
    for(int i=0; i<HISTORY_SIZE; i++) { hist_heat[i] = -1; hist_temp[i] = -1; hist_hash[i] = 0; }
    
    int history_idx = 0;
    bool stabilized = false;
    int max_temp = 0;
    int ticks = 0;

    while (!stabilized && ticks < 1500) {
        max_temp = 0;
        int intra_tick_max = 0;
        
        for (int i = 0; i < num_generators; i++) {
            int f_idx = generators_idx[i];
            heat_map[f_idx] += generators_heat[i];
            if (heat_map[f_idx] > intra_tick_max) intra_tick_max = heat_map[f_idx];
        }

        if (intra_tick_max > config.meltdown_temp) {
            out_rf = rf_per_tick; out_temp = intra_tick_max; out_melted = true; return;
        }

        // Absorbers
        for (int i = 0; i < num_absorbers; i++) {
            for (int n = 0; n < absorbers_num_neighbors[i]; n++) {
                int n_idx = absorbers_neighbors[i][n];
                int current_h = heat_map[n_idx];
                heat_map[n_idx] = (current_h - config.absorber_cooling > 0) ? current_h - config.absorber_cooling : 0;
            }
        }

        // Pipes
        for (int i = 0; i < num_pipes; i++) {
            int p_idx = pipes_idx[i];
            int curr_heat = heat_map[p_idx];
            for (int n = 0; n < pipes_num_neighbors[i]; n++) {
                int n_idx = pipes_neighbors[i][n];
                int neighbor_heat = heat_map[n_idx];
                if (neighbor_heat > curr_heat) {
                    int diff = neighbor_heat - curr_heat;
                    int gained = diff;
                    int calculated_gain = diff / config.pipe_divisor + config.pipe_base;
                    if (calculated_gain < diff) gained = calculated_gain;
                    
                    heat_map[n_idx] -= gained;
                    curr_heat += gained;
                }
            }
            heat_map[p_idx] = curr_heat;
        }

        // Vents
        for (int i = 0; i < num_vents; i++) {
            int max_neighbor_heat = 0;
            int target_idx = -1;
            for (int n = 0; n < vents_num_neighbors[i]; n++) {
                int n_idx = vents_neighbors[i][n];
                if (heat_map[n_idx] > max_neighbor_heat) {
                    max_neighbor_heat = heat_map[n_idx];
                    target_idx = n_idx;
                }
            }
            if (target_idx != -1) {
                int calculated = max_neighbor_heat / config.vent_divisor + config.vent_base;
                int removed = (calculated < max_neighbor_heat) ? calculated : max_neighbor_heat;
                heat_map[target_idx] -= removed;
            }
        }

        long long curr_total_heat = 0;
        unsigned int curr_hash = 0;
        int curr_max_temp = 0;

        for (int i = 0; i < num_heat_holders; i++) {
            int h_idx = heat_holders[i];
            int local_heat = heat_map[h_idx];
            curr_total_heat += local_heat;
            if (local_heat > curr_max_temp) curr_max_temp = local_heat;
            curr_hash ^= (static_cast<unsigned int>(local_heat) + i) * 2654435761u;
        }

        if (curr_max_temp > config.meltdown_temp) {
            out_rf = rf_per_tick; out_temp = curr_max_temp; out_melted = true; return;
        }

        for (int i = 0; i < HISTORY_SIZE; i++) {
            if (hist_heat[i] == curr_total_heat && hist_temp[i] == curr_max_temp && hist_hash[i] == curr_hash) {
                stabilized = true;
                break;
            }
        }

        hist_heat[history_idx] = curr_total_heat;
        hist_temp[history_idx] = curr_max_temp;
        hist_hash[history_idx] = curr_hash;
        history_idx = (history_idx + 1) & (HISTORY_SIZE - 1);
        
        max_temp = curr_max_temp;
        ticks++;
    }

    out_rf = rf_per_tick;
    out_temp = max_temp;
    out_melted = !stabilized;
}

// 3. The Global Dispatch Kernel
template <size_t W, size_t H>
__global__ void evaluate_reactor_batch_kernel(
    const uint8_t* grids,      // [batch_size, W*H]
    const KernelConfig config,
    const int batch_size,
    long long* out_rf,         // [batch_size]
    int* out_temp,             // [batch_size]
    bool* out_melted)          // [batch_size]
{
    // Find our unique thread ID in the massive batch
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Safety check to prevent out-of-bounds memory access
    if (idx >= batch_size) return;

    // Isolate the exact W*H segment of the flattened input array for this thread
    const uint8_t* my_grid = &grids[idx * (W * H)];

    long long thread_rf = 0;
    int thread_temp = 0;
    bool thread_melted = false;

    // Run the native simulation
    simulate_single_reactor<W, H>(my_grid, config, thread_rf, thread_temp, thread_melted);

    // Write results safely back to global VRAM
    out_rf[idx] = thread_rf;
    out_temp[idx] = thread_temp;
    out_melted[idx] = thread_melted;
}

// 4. The LibTorch C++ Wrapper
// This function acts as the bridge between your standard C++ vectors and the CUDA kernel.
void launch_gpu_batch_eval(
    const uint8_t* d_grids, 
    int batch_size, 
    KernelConfig config,
    long long* d_out_rf, 
    int* d_out_temp, 
    bool* d_out_melted) 
{
    // Standard block sizes for modern NVIDIA architectures
    int threads_per_block = 256;
    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;

    // Assumes a 4x4 grid. You can template this wrapper or branch it for 3x3, 4x4, 5x5
    evaluate_reactor_batch_kernel<5, 5><<<num_blocks, threads_per_block>>>(
        d_grids, config, batch_size, d_out_rf, d_out_temp, d_out_melted
    );

    // Wait for all 100,000 threads to finish simulating before letting C++ continue
    cudaDeviceSynchronize(); 
}