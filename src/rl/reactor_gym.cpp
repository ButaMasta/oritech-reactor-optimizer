#include "reactor_gym.hpp"

namespace reactor_optimizer::rl {

torch::Tensor ReactorEnv::get_observation() {
    auto obs = torch::zeros({8, 5, 5}, torch::kFloat32);
    for (int y = 0; y < 5; ++y) {
        for (int x = 0; x < 5; ++x) {
            int block_idx = static_cast<int>(current_reactor.get_block(x, y));
            obs[block_idx][y][x] = 1.0f;
        }
    }
    return obs;
}

torch::Tensor ReactorEnv::reset() {
    for (int y = 0; y < 5; ++y) {
        for (int x = 0; x < 5; ++x) {
            current_reactor.set_block(x, y, BlockType::Empty);
        }
    }
    current_step = 0;
    return get_observation();
}

StepResult ReactorEnv::step(int action) {
    // Map the step number (0-24) to X, Y coordinates
    int x = current_step % 5;
    int y = current_step / 5;
    
    current_reactor.set_block(x, y, static_cast<BlockType>(action));
    current_step++;

    bool done = (current_step == 25);
    float reward = 0.0f;

    if (done) {
        // FIRE THE SIMULATOR (You can batch this for multiple environments later)
        auto result = current_reactor.simulate_to_equilibrium();
        
        if (result.melted_down) {
            // Penalize based on how badly it melted to give the agent a gradient
            reward = -5.0f - (result.max_temp / 1000.0f); 
        } else {
            // Scale the RF so the neural network doesn't deal with massive numbers
            reward = result.rf_per_tick / 1'000'000.0f; 
        }
    }

    return {get_observation(), reward, done};
}

torch::Tensor VectorEnvManager::reset_all() {
    std::vector<torch::Tensor> obs_list;
    obs_list.reserve(num_envs);
    
    for (int i = 0; i < num_envs; ++i) {
        obs_list.push_back(envs[i].reset());
    }
    
    // torch::stack combines the vector of individual tensors into a single batched tensor
    return torch::stack(obs_list);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> VectorEnvManager::step_all(torch::Tensor actions) {
    std::vector<torch::Tensor> obs_list;
    std::vector<float> rewards_list;
    std::vector<float> dones_list;
    
    obs_list.reserve(num_envs);
    rewards_list.reserve(num_envs);
    dones_list.reserve(num_envs);

    // Access the raw action integers
    auto actions_accessor = actions.accessor<int64_t, 1>();

    for (int i = 0; i < num_envs; ++i) {
        int action = actions_accessor[i];
        StepResult result = envs[i].step(action);
        
        obs_list.push_back(result.observation);
        rewards_list.push_back(result.reward);
        dones_list.push_back(result.done ? 1.0f : 0.0f);
    }

    // Convert the lists back into PyTorch Tensors for the RolloutBuffer
    torch::Tensor batched_obs = torch::stack(obs_list);
    torch::Tensor batched_rewards = torch::tensor(rewards_list, torch::kFloat32);
    torch::Tensor batched_dones = torch::tensor(dones_list, torch::kFloat32);

    return {batched_obs, batched_rewards, batched_dones};
}


}; // namespace reactor_optimizer::rl