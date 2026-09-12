#pragma once
#include "core/reactor_util.hpp"
#include <torch/torch.h>
#include <vector>

namespace reactor_optimizer::rl {

using core::ReactorState;
using core::BlockType;

struct StepResult {
    torch::Tensor observation;
    float reward;
    bool done;
};

class ReactorEnv {
private:
    ReactorState<5, 5> current_reactor;
    int current_step;
    
    // Convert the C++ grid into a [8, 5, 5] LibTorch Tensor
    torch::Tensor get_observation();

public:
    ReactorEnv() { reset(); }

    // Wipes the board clean and returns the blank state
    torch::Tensor reset();

    // Takes an action (0-7), updates the state, and returns the reward
    StepResult step(int action);
};

class VectorEnvManager {
private:
    std::vector<ReactorEnv> envs;
    int num_envs;

public:
    VectorEnvManager(int count) : num_envs(count) {
        envs.resize(num_envs);
    }

    // Wipes all environments and returns a stacked tensor of shape [Num_Envs, 8, 5, 5]
    torch::Tensor reset_all();

    // Takes a tensor of actions [Num_Envs] and returns stacked results
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> step_all(torch::Tensor actions);
};

}; // namespace reactor_optimizer::rl