#include <iostream>
#include "core/cuda/reactor_kernel.cuh"

namespace reactor_optimizer::core {

template <size_t W, size_t H, ReactorConfig Config>
BatchManager<W, H, Config>::BatchManager(BestStats& stats, ReactorState<W, H, Config>& best) 
    : best_stats(stats), best_reactor(best) {
    
    grids.reserve(MAX_BATCH_SIZE * W * H); 

    // Pre-allocate VRAM
    auto device = torch::kCUDA;
    t_out_rf = torch::empty({(long long)MAX_BATCH_SIZE}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    t_out_temp = torch::empty({(long long)MAX_BATCH_SIZE}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    t_out_melted = torch::empty({(long long)MAX_BATCH_SIZE}, torch::TensorOptions().dtype(torch::kBool).device(device));
}
    
template <size_t W, size_t H, ReactorConfig Config>
void BatchManager<W, H, Config>::flush_to_gpu() {
    if (grids.empty()) return;
    
    int current_batch = grids.size() / (W * H);
    
    auto options_grid = torch::TensorOptions().dtype(torch::kUInt8);
    torch::Tensor t_grids_cpu = torch::from_blob(grids.data(), {current_batch, W * H}, options_grid);
    
    torch::Tensor t_grids_gpu = t_grids_cpu.to(torch::kCUDA, true); 
    
    KernelConfig k_cfg {
        Config.rf_per_pulse, Config.meltdown_temp, Config.vent_divisor, 
        Config.vent_base, Config.pipe_divisor, Config.pipe_base, Config.absorber_cooling
    };
    
    // Fire the kernel using our pre-allocated persistent pointers
    launch_gpu_batch_eval<W, H>(
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
                
                ReactorState<W, H, Config> temp_reactor;
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

}; // namespace reactor_optimizer::core