#pragma once
#include <torch/torch.h>

namespace reactor_optimizer::rl {

struct ActorCriticOutput {
    torch::Tensor action_logits; // [Batch, 8]
    torch::Tensor state_value;   // [Batch, 1]
};

class ActorCriticImpl : public torch::nn::Module {
private:
    // Shared feature extraction
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::Linear shared_fc{nullptr};

    // Separate heads
    torch::nn::Linear actor_head{nullptr};
    torch::nn::Linear critic_head{nullptr};

public:
    ActorCriticImpl(int in_channels = 8, int grid_w = 5, int grid_h = 5, int num_actions = 8);

    ActorCriticOutput forward(torch::Tensor x);
};

TORCH_MODULE(ActorCritic);

}; // namespace reactor_optimizer::rl