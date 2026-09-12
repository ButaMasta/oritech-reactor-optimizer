#pragma once
#include <array>
#include <algorithm>
#include <vector>
#include <cstddef>
#include <cstdint>

namespace reactor_optimizer::core {

enum class BlockType {
    Empty,
    SingleRod,
    DoubleRod,
    QuadRod,
    Reflector,
    HeatPipe,
    HeatVent,
    Absorber
};

struct ReactorConfig {
    int rf_per_pulse = 64;
    int meltdown_temp = 2000;
    int vent_divisor = 100;
    int vent_base = 4;
    int pipe_divisor = 4;
    int pipe_base = 10;
    int absorber_cooling = 16; 
};

struct Coord { int x, y; };

struct SimResult {
    long long rf_per_tick;
    int max_temp;
    bool melted_down;
    bool stabilized;
    int ticks_ran;
};

template <size_t W, size_t H, ReactorConfig Config = ReactorConfig{}>
class ReactorState {
private:
    static constexpr size_t GRID_SIZE = W * H;

    // 1D Arrays sized with W and H.
    std::array<BlockType, GRID_SIZE> grid;
    std::array<int, GRID_SIZE> heat_map;

    // Helper to convert 2D coordinates to 1D flat index.
    inline constexpr int get_flat_idx(int x, int y) const {
        return y * W + x;
    }

    inline constexpr bool is_rod(BlockType type) const {
        return type == BlockType::SingleRod || 
               type == BlockType::DoubleRod || 
               type == BlockType::QuadRod;
    }

    inline constexpr int get_internal_pulses(BlockType type) const {
        switch (type) {
            case BlockType::SingleRod: return 1;
            case BlockType::DoubleRod: return 4;
            case BlockType::QuadRod: return 12;
            default: return 0;
        }
    }

    inline constexpr int get_outbound_pulses(BlockType type) const {
        switch (type) {
            case BlockType::SingleRod: return 1;
            case BlockType::DoubleRod: return 2;
            case BlockType::QuadRod: return 4;
            default: return 0;
        }
    }

public:
    ReactorState();

    inline BlockType get_block(int x, int y) const {
        return grid[get_flat_idx(x, y)];
    }

    inline void set_block(int x, int y, BlockType type) {
        grid[get_flat_idx(x, y)] = type;
    }

    inline bool is_valid_coordinate(int x, int y) const {
        return (x >= 0 && x < static_cast<int>(W) && y >= 0 && y < static_cast<int>(H));
    }

    SimResult simulate_to_equilibrium();

    static std::vector<Coord> generate_spiral_order();
};

template <size_t W, size_t H, ReactorConfig Config>
int calculate_symmetry_score(const ReactorState<W, H, Config>& reactor);

}; // namespace reactor_optimizer::core

#include "reactor_util.inl"