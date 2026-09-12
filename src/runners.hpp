#pragma once

#include <chrono>
#include <iostream>
#include <future>
#include <thread>
#include <span>
#include <vector>

#include "core/reactor_util.hpp"
#include "core/batch_manager.hpp"
#include "traditional/branch_and_bound.hpp"
#include "traditional/simulated_annealing.hpp"
#include "rl/actor_critic.hpp"
#include "rl/reactor_gym.hpp"
#include "rl/rollout_buffer.hpp"

namespace reactor_optimizer::runners {

using core::BlockType;
using core::ReactorConfig;
using core::ReactorState;
using core::Coord;

template <size_t W, size_t H, ReactorConfig Config = ReactorConfig{}>
ReactorState<W, H, Config> run_branch_and_bound(std::span<const BlockType> allowed_blocks) {
    ReactorState<W, H, Config> current_reactor;
    ReactorState<W, H, Config> absolute_best_reactor;
    
    core::BestStats best_stats;
    core::BatchManager<W, H, Config> batch_manager(best_stats, absolute_best_reactor);

    std::cout << "Generating Spiral Placement Order for " << W << "x" << H << "..." << std::endl;
    std::vector<Coord> spiral = ReactorState<W, H, Config>::generate_spiral_order();
    
    std::vector<int> current_path(W * H, 0);
    unsigned long long nodes_evaluated = 0;

    auto start_time = std::chrono::high_resolution_clock::now();
    std::cout << "Starting Hybrid CPU/GPU Branch and Bound..." << std::endl;
    
    traditional::branch_and_bound(0, spiral, current_reactor, batch_manager, current_path, nodes_evaluated, allowed_blocks);
    batch_manager.flush_to_gpu(); // CRITICAL: Flush remaining queue

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "\r\033[K--- B&B Search Exhausted in " << duration.count() << "ms ---" << std::endl;
    std::cout << "Total Nodes Evaluated: " << nodes_evaluated << std::endl;
    std::cout << "Proven Max RF/t: " << best_stats.rf << std::endl;
    std::cout << "Kept Reactor Temp: " << best_stats.temp << " C\n" << std::endl;

    return absolute_best_reactor;
}

template <size_t W, size_t H, ReactorConfig Config = ReactorConfig{}>
ReactorState<W, H, Config> run_simulated_annealing(std::span<const BlockType> allowed_blocks, unsigned int threads = (std::thread::hardware_concurrency() - 4)) {
    unsigned int num_threads = threads;
    if (num_threads == 0) num_threads = 8; // Fallback

    std::cout << "Launching " << num_threads << " parallel Simulated Annealing threads..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<std::future<ReactorState<W, H, Config>>> futures;
    for (unsigned int i = 0; i < num_threads; ++i) {
        bool is_proxy_thread = (i == 0);
        int current_seed = 9481 + i;
        
        // Wrap the call in a lambda for a perfectly clean thread launch
        futures.push_back(std::async(std::launch::async, [current_seed, is_proxy_thread, allowed_blocks]() {
            return traditional::optimize_layout<W, H, Config>(current_seed, is_proxy_thread, allowed_blocks);
        }));
    }

    ReactorState<W, H, Config> absolute_best_reactor;
    long long global_best_rf = -1;

    for (unsigned int i = 0; i < num_threads; ++i) {
        ReactorState<W, H, Config> thread_result = futures[i].get();
        auto stats = thread_result.simulate_to_equilibrium();
        
        if (!stats.melted_down && stats.rf_per_tick > global_best_rf) {
            global_best_rf = stats.rf_per_tick;
            absolute_best_reactor = thread_result;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "\n--- SA Optimization Complete in " << duration.count() << "ms ---" << std::endl;
    std::cout << "Max Optimized RF/t: " << global_best_rf << "\n" << std::endl;

    return absolute_best_reactor;
}

inline void run_ppo_collection(int num_envs = 1000, int num_steps = 25) {
    std::cout << "Initializing PPO environment..." << std::endl;

    // 1. Initialize the Neural Network
    rl::ActorCritic model;
    
    // 2. Initialize the parallel environment manager
    rl::VectorEnvManager env_manager(num_envs);
    
    // 3. Initialize the rollout buffer
    rl::RolloutBuffer buffer(num_steps, num_envs);

    std::cout << "Starting rollout collection..." << std::endl;

    // Reset all environments to get the initial blank states
    torch::Tensor obs = env_manager.reset_all();

    // Play the game to fill the buffer
    for (int step = 0; step < num_steps; ++step) {
        torch::NoGradGuard no_grad;

        rl::ActorCriticOutput out = model->forward(obs);

        torch::Tensor probs = torch::softmax(out.action_logits, /*dim=*/-1);
        torch::Tensor actions = torch::multinomial(probs, /*num_samples=*/1).squeeze(-1);
        
        torch::Tensor action_probs = probs.gather(/*dim=*/1, actions.unsqueeze(-1)).squeeze(-1);
        torch::Tensor log_probs = torch::log(action_probs);

        auto [next_obs, rewards, dones] = env_manager.step_all(actions);

        buffer.insert(obs, actions, log_probs, out.state_value.squeeze(-1), rewards, dones);

        obs = next_obs;
    }

    std::cout << "Rollout buffer successfully filled!" << std::endl;
    std::cout << "Final Buffer States:" << std::endl;
    std::cout << "- Observations Shape: " << buffer.observations.sizes() << std::endl;
    std::cout << "- Actions Shape: " << buffer.actions.sizes() << std::endl;
    std::cout << "- Rewards Shape: " << buffer.rewards.sizes() << std::endl;
}

} // namespace reactor_optimizer::runners