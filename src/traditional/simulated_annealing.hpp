#pragma once
#include <span>
#include "core/reactor_util.hpp"

namespace reactor_optimizer::traditional {

using core::BlockType;
using core::ReactorConfig;
using core::ReactorState;
using core::Coord;

template <size_t W, size_t H, ReactorConfig Config>
ReactorState<W, H, Config> optimize_layout(int seed, bool show_progress, std::span<const BlockType> allowed_blocks);

}; // namespace reactor_optimizer::traditional

#include "simulated_annealing.inl"