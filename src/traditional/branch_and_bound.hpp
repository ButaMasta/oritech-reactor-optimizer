#pragma once
#include <vector>
#include <span>
#include "core/reactor_util.hpp"

namespace reactor_optimizer::traditional {

using core::BlockType;
using core::ReactorConfig;
using core::ReactorState;
using core::Coord;

template <size_t W, size_t H, ReactorConfig Config = ReactorConfig{}>
bool passes_thermal_bound(const ReactorState<W, H, Config>& reactor);

template <size_t W, size_t H, ReactorConfig Config = ReactorConfig{}>
bool passes_rf_bound(const ReactorState<W, H, Config>& reactor, long long current_best_rf);

template <size_t W, size_t H, ReactorConfig Config = ReactorConfig{}, typename BatchManagerType>
void branch_and_bound(int depth, 
                      const std::vector<Coord>& placement_order,
                      ReactorState<W, H, Config>& current_reactor, 
                      BatchManagerType& batch,
                      std::vector<int>& current_path,
                      unsigned long long& nodes_evaluated,
                      std::span<const BlockType> allowed_blocks);

};

#include "branch_and_bond.inl"