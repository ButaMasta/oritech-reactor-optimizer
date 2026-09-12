#include "actor_critic.hpp"

namespace reactor_optimizer::rl {

ActorCriticImpl::ActorCriticImpl(int in_channels, int grid_w, int grid_h, int num_actions) {
    // Conv1: preserves 5x5 spatial dimensions (padding=1, kernel=3)
    conv1 = register_module("conv1", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(in_channels, 32, 3).padding(1)));
    
    // Conv2: deepens spatial pattern detection
    conv2 = register_module("conv2", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(32, 64, 3).padding(1)));

    int flattened_dim = 64 * grid_w * grid_h; // 64 * 5 * 5 = 1600
    shared_fc = register_module("shared_fc", torch::nn::Linear(flattened_dim, 128));

    actor_head = register_module("actor_head", torch::nn::Linear(128, num_actions));
    critic_head = register_module("critic_head", torch::nn::Linear(128, 1));
}

ActorCriticOutput ActorCriticImpl::forward(torch::Tensor x) {
    // Shared trunk
    x = torch::relu(conv1->forward(x));
    x = torch::relu(conv2->forward(x));
    x = x.flatten(1); // Flatten to [Batch, 1600]
    x = torch::relu(shared_fc->forward(x));

    // Heads
    torch::Tensor logits = actor_head->forward(x);
    torch::Tensor value = critic_head->forward(x);

    return {logits, value};
}

}; // namespace reactor_optimizer::rl