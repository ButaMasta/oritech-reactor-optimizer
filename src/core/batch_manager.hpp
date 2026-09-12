#pragma once
#include <vector>
#include <cstdint>
#include <torch/torch.h>
#include "core/reactor_util.hpp"

namespace reactor_optimizer::core {

struct BestStats {
    long long rf = -1;
    int temp = 999999;
    int sym_score = -1;
};

template <size_t W, size_t H, ReactorConfig Config = ReactorConfig{}>
struct BatchManager {
    std::vector<uint8_t> grids;
    BestStats& best_stats;
    ReactorState<W, H, Config>& best_reactor;
    
    const size_t MAX_BATCH_SIZE = 1'000'000; 
    
    torch::Tensor t_out_rf;
    torch::Tensor t_out_temp;
    torch::Tensor t_out_melted;

    // 3. Constructor to allocate memory exactly ONCE
    BatchManager(BestStats& stats, ReactorState<W, H, Config>& best);
    
    void flush_to_gpu();
};

}; // namespace reactor_optimizer::core

#include "batch_manager.inl"