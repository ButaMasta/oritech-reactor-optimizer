#include "../include/ReactorUtil.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <chrono>
#include <future>

// Neat printing.
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"

// Parser to convert ASCII characters to BlockTypes.
BlockType char_to_block(char c) {
    switch (c) {
        case '1': return BlockType::SingleRod;
        case '2': return BlockType::DoubleRod;
        case '4': return BlockType::QuadRod;
        case 'R': return BlockType::Reflector;
        case 'P': return BlockType::HeatPipe;
        case 'V': return BlockType::HeatVent;
        case 'A': return BlockType::Absorber;
        case '.':
        default:  return BlockType::Empty;
    }
}

// Ingests the 2D string array and loads it into pre-allocated grid.
void loadLayout(ReactorState<>& reactor, const std::vector<std::string>& layout) {
    for (size_t y = 0; y < layout.size(); ++y) {
        for (size_t x = 0; x < layout[y].size(); ++x) {
            reactor.set_block(x, y, char_to_block(layout[y][x]));
        }
    }
}

ReactorState<> optimize_layout(int width, int height, int seed, bool show_progress = false) {
    ReactorState<> current_reactor(width, height);
    ReactorState<> best_reactor(width, height);

    // Initialize random number generators.
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist_type(0, 7); 
    std::uniform_int_distribution<int> dist_x(0, width - 1);
    std::uniform_int_distribution<int> dist_y(0, height - 1);
    std::uniform_real_distribution<double> dist_prob(0.0, 1.0);

    // Seed the initial state with completely random blocks.
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            current_reactor.set_block(x, y, static_cast<BlockType>(dist_type(rng)));
        }
    }

    // Evaluate baseline.
    auto initial_result = current_reactor.simulate_to_equilibrium();
    // Multiply by 1000 to ensure even minor meltdowns score worse than an empty 0 RF/t grid
    long long current_score = initial_result.melted_down ? -static_cast<long long>(initial_result.max_temp) * 1000 : initial_result.rf_per_tick;
    long long best_score = current_score;
    best_reactor = current_reactor;

    // Hyperparameters.
    double start_temp = 100000.0;
    double end_temp = 0.1;
    int max_iterations = 2500000; // Bumped up to 2.5 million!

    // Calculates the exact multiplier to reach end_temp at max_iterations
    double cooling_rate = std::pow(end_temp / start_temp, 1.0 / max_iterations);
    double temperature = start_temp;

    std::cout << "Starting Simulated Annealing (" << max_iterations << " iterations)..." << std::endl;
    
    // Throttle I/O updates to only 100 times total.
    int update_interval = max_iterations / 100;

    for (int i = 0; i < max_iterations; i++) {
        // Pick a random coordinate to mutate.
        int mx = dist_x(rng);
        int my = dist_y(rng);
        
        BlockType old_block = current_reactor.get_block(mx, my);
        BlockType new_block = static_cast<BlockType>(dist_type(rng));

        if (old_block == new_block) continue; // Skip redundant checks.

        // Apply mutation IN-PLACE.
        current_reactor.set_block(mx, my, new_block);

        // Score the mutated layout
        auto new_result = current_reactor.simulate_to_equilibrium();
        long long new_score = new_result.melted_down ? -static_cast<long long>(new_result.max_temp) * 1000 : new_result.rf_per_tick;

        // Calculate delta
        double delta = static_cast<double>(new_score - current_score);
        bool accept = false;

        // Stochastic Acceptance Logic
        if (delta > 0) {
            accept = true; // Always accept improvements
        } else {
            // Explore worse layouts based on current temperature
            double p = std::exp(delta / temperature);
            if (dist_prob(rng) < p) {
                accept = true;
            }
        }

        if (accept) {
            current_score = new_score;
            // Did we find a new global maximum?
            if (current_score > best_score) {
                best_score = current_score;
                best_reactor = current_reactor; // C++ default assignment operator memcopies the state arrays
            }
        } else {
            // Rollback the mutation (Zero Allocation!)
            current_reactor.set_block(mx, my, old_block);
        }

        // Cool the system
        temperature *= cooling_rate;

        // Terminal Progress Bar (Proxy Thread Only)
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

void test_layout() {
    // Define layout.
    // std::vector<std::string> test_layout = {
    //     "R2A12",
    //     "2AP2A",
    //     "AP4A2",
    //     "12A21",
    //     "2A21A"
    // };
    std::vector<std::string> test_layout = {
        "P222V",
        "2AV2V",
        "R21A4",
        "2A22A",
        "V4V12"
    };

    size_t width = test_layout[0].size();
    size_t height = test_layout.size();

    // Instantiate reactor simulation.
    ReactorState<> reactor(width, height);
    
    // Load the visual representation into the grid.
    loadLayout(reactor, test_layout);

    // Run the simulation loop.
    std::cout << "Simulating layout to equilibrium..." << std::endl;
    auto result = reactor.simulate_to_equilibrium();

    // Output the results for verification.
    std::cout << "\n--- Simulation Results ---" << std::endl;
    std::cout << "Total RF/t: " << result.rf_per_tick << std::endl;
    std::cout << "Max Temp:   " << result.max_temp << " C" << std::endl;
    if (result.melted_down) {
        std::cout << "Meltdown:   " << RED << "YES (>4000 C)" << RESET << std::endl;
    } else {
        std::cout << "Meltdown:   " << GREEN << "NO" << RESET << std::endl;
    }
    if (result.stabilized) {
        std::cout << "Stabilized: " << GREEN << "YES" << RESET << std::endl;
    } else {
        std::cout << "Stabilized: " << RED << "NO" << RESET << std::endl;
    }
    std::cout << "Ticks Ran:  " << result.ticks_ran << std::endl;
}

void print_layout(ReactorState<>& reactor, int width, int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            BlockType block = reactor.get_block(x, y);
            char c = '.';
            switch (block) {
                case BlockType::SingleRod: c = '1'; break;
                case BlockType::DoubleRod: c = '2'; break;
                case BlockType::QuadRod:   c = '4'; break;
                case BlockType::Reflector: c = 'R'; break;
                case BlockType::HeatPipe:  c = 'P'; break;
                case BlockType::HeatVent:  c = 'V'; break;
                case BlockType::Absorber:  c = 'A'; break;
                case BlockType::Empty:
                default: break;
            }
            std::cout << "[ " << c << " ] ";
        }
        std::cout << std::endl;
    }
}

int main() {
    int width = 5;
    int height = 5;
    int num_threads = 16; // Leaves a few threads free so your PC stays responsive

    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "Launching " << num_threads << " parallel optimization threads..." << std::endl;

    // Launch multiple independent Simulated Annealing runs
    std::vector<std::future<ReactorState<>>> futures;
    for (int i = 0; i < num_threads; ++i) {
        // Pass 'i' as the seed, and true only for thread 0
        bool is_proxy_thread = (i == 0);
        futures.push_back(std::async(std::launch::async, optimize_layout, width, height, 1337 + i, is_proxy_thread));
    }

    // Collect the results and find the absolute best layout
    ReactorState<> absolute_best_reactor(width, height);
    long long global_best_rf = -1;
    SimResult global_best_stats;

    for (int i = 0; i < num_threads; ++i) {
        ReactorState<> thread_result = futures[i].get();
        auto stats = thread_result.simulate_to_equilibrium();
        
        if (!stats.melted_down && stats.rf_per_tick > global_best_rf) {
            global_best_rf = stats.rf_per_tick;
            absolute_best_reactor = thread_result;
            global_best_stats = stats;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "\n--- Optimization Complete in " << duration.count() << "ms ---" << std::endl;
    std::cout << "Max Optimized RF/t: " << global_best_stats.rf_per_tick << std::endl;
    std::cout << "Final Peak Temp:    " << global_best_stats.max_temp << " C" << std::endl;
    std::cout << "\n--- Reactor Layout ---" << std::endl;
    print_layout(absolute_best_reactor, width, height);
    // test_layout();
    return 0;
}