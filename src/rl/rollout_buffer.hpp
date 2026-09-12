#pragma once

#include <torch/torch.h>

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

    RolloutBuffer(int steps, int envs) : max_steps(steps), num_envs(envs), current_step(0) {
        // Pre-allocate everything on the CPU. 
        // We use contiguous blocks of memory so LibTorch doesn't have to reallocate.
        observations = torch::zeros({max_steps, num_envs, 8, 5, 5}, torch::kFloat32);
        actions      = torch::zeros({max_steps, num_envs}, torch::kInt64);
        log_probs    = torch::zeros({max_steps, num_envs}, torch::kFloat32);
        values       = torch::zeros({max_steps, num_envs}, torch::kFloat32);
        rewards      = torch::zeros({max_steps, num_envs}, torch::kFloat32);
        
        // We store 'dones' as floats (1.0 or 0.0) to make the Advantage math easier later
        dones        = torch::zeros({max_steps, num_envs}, torch::kFloat32); 
    }

    // Inserts a batch of data from all parallel environments simultaneously
    void insert(torch::Tensor obs, torch::Tensor act, torch::Tensor log_p, 
                torch::Tensor val, torch::Tensor rew, torch::Tensor done) {
        
        observations[current_step] = obs;
        actions[current_step]      = act;
        log_probs[current_step]    = log_p;
        values[current_step]       = val;
        rewards[current_step]      = rew;
        dones[current_step]        = done;
        
        current_step++;
    }

    void reset() {
        current_step = 0;
    }
    
    bool is_full() const {
        return current_step >= max_steps;
    }
};