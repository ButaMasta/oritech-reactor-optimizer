#include "rl/rollout_buffer.hpp"

namespace reactor_optimizer::rl {

RolloutBuffer::RolloutBuffer(int steps, int envs) 
    : max_steps(steps), num_envs(envs), current_step(0) {
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

void RolloutBuffer::insert(torch::Tensor obs, torch::Tensor act, torch::Tensor log_p, 
            torch::Tensor val, torch::Tensor rew, torch::Tensor done) {
    
    observations[current_step] = obs;
    actions[current_step]      = act;
    log_probs[current_step]    = log_p;
    values[current_step]       = val;
    rewards[current_step]      = rew;
    dones[current_step]        = done;
    
    current_step++;
}

}; // namespace reactor_optimizer::rl