#pragma once

#include <torch/torch.h>

namespace reactor_optimizer::rl {

class RolloutBuffer {
public:
    // Core PPO Storage Tensors: Shape -> [Num_Steps, Num_Envs, ...]
    torch::Tensor observations;
    torch::Tensor actions;
    torch::Tensor log_probs;
    torch::Tensor values;
    torch::Tensor rewards;
    torch::Tensor dones;
    
    int current_step;
    int max_steps;
    int num_envs;

    RolloutBuffer(int steps, int envs);

    // Inserts a batch of data from all parallel environments simultaneously
    void insert(torch::Tensor obs, torch::Tensor act, torch::Tensor log_p, 
                torch::Tensor val, torch::Tensor rew, torch::Tensor done);

    void reset() {
        current_step = 0;
    }
    
    bool is_full() const {
        return current_step >= max_steps;
    }
};

}; // namespace reactor_optimizer::rl