#include <random>
#include <cmath>
#include <span>
#include <iostream>
#include "core/reactor_util.hpp"

namespace reactor_optimizer::traditional {

template <size_t W, size_t H, ReactorConfig Config>
ReactorState<W, H, Config> optimize_layout(int seed, bool show_progress, std::span<const BlockType> allowed_blocks) {
    ReactorState<W, H, Config> current_reactor;
    ReactorState<W, H, Config> best_reactor;

    std::random_device rd;

    // Initialize random number generators.
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> dist_type(0, allowed_blocks.size() - 1);
    std::uniform_int_distribution<int> dist_x(0, W - 1);
    std::uniform_int_distribution<int> dist_y(0, H - 1);
    std::uniform_real_distribution<double> dist_prob(0.0, 1.0);

    // Seed the initial state with completely random blocks.
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            current_reactor.set_block(x, y, BlockType::Empty);
        }
    }
    
    // Evaluate baseline.
    auto initial_result = current_reactor.simulate_to_equilibrium();
    // Multiply by 1000 to ensure even minor meltdowns score worse than an empty 0 RF/t grid.
    long long current_score = initial_result.melted_down ? -static_cast<long long>(initial_result.max_temp) * 1000 : initial_result.rf_per_tick;
    long long best_score = current_score;
    best_reactor = current_reactor;

    // Hyperparameters.
    double start_temp = 100'000.0;
    double end_temp = 0.1;
    int max_iterations = 10'000'000;

    // Calculates the exact multiplier to reach end_temp at max_iterations.
    double cooling_rate = std::pow(end_temp / start_temp, 1.0 / max_iterations);
    double temperature = start_temp;

    if (show_progress) {
        std::cout << "Starting Simulated Annealing (" << max_iterations << " iterations)..." << std::endl;
    }
    
    // Throttle I/O updates to only 100 times total.
    int update_interval = max_iterations / 100;

    for (int i = 0; i < max_iterations; i++) {
        // Pick a random coordinate to mutate.
        int mx = dist_x(rng);
        int my = dist_y(rng);
        
        BlockType old_block = current_reactor.get_block(mx, my);
        BlockType new_block = allowed_blocks[dist_type(rng)];

        if (old_block == new_block) continue; // Skip redundant checks.

        // Apply mutation in-place.
        current_reactor.set_block(mx, my, new_block);

        // Score the mutated layout.
        auto new_result = current_reactor.simulate_to_equilibrium();
        long long new_score = new_result.melted_down ? -static_cast<long long>(new_result.max_temp) * 1000 : new_result.rf_per_tick;

        // Calculate delta.
        double delta = static_cast<double>(new_score - current_score);
        bool accept = false;

        // Stochastic Acceptance Logic.
        if (delta > 0) {
            accept = true; // Always accept improvements.
        } else {
            // Explore worse layouts based on current temperature.
            double p = std::exp(delta / temperature);
            if (dist_prob(rng) < p) {
                accept = true;
            }
        }

        if (accept) {
            current_score = new_score;
            if (current_score > best_score) {
                best_score = current_score;
                best_reactor = current_reactor;
            }
        } else {
            current_reactor.set_block(mx, my, old_block);
        }

        temperature *= cooling_rate;

        // Terminal Progress Bar (Proxy Thread Only).
        if (show_progress && (i % update_interval == 0 || i == max_iterations - 1)) {
            int percentage = (i * 100) / max_iterations;
            int bar_width = 50;
            int pos = (percentage * bar_width) / 100;

            std::cout << "\r[";
            for (int p = 0; p < bar_width; ++p) {
                if (p < pos) std::cout << "=";
                else if (p == pos) std::cout << ">";
                else std::cout << " ";
            }
            
            std::cout << "] " << percentage << "% | Temp: " << static_cast<int>(temperature) 
                      << " | Best RF/t: " << best_score << "   " << std::flush; 
        }
    }

    return best_reactor;
}

}; // namespace reactor_optimizer::traditional