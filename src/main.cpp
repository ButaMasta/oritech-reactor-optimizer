#include "../include/ReactorUtil.hpp"
#include <iostream>
#include <vector>
#include <string>

// Lightweight parser to convert ASCII characters to BlockTypes
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

// Ingests the 2D string array and loads it into our pre-allocated grid
void loadLayout(ReactorState<>& reactor, const std::vector<std::string>& layout) {
    for (size_t y = 0; y < layout.size(); ++y) {
        for (size_t x = 0; x < layout[y].size(); ++x) {
            reactor.set_block(x, y, char_to_block(layout[y][x]));
        }
    }
}

int main() {
    // Visually define a 5x5 test layout.
    // Example: A core of quad rods surrounded by vents and pipes.
    std::vector<std::string> test_layout = {
        "R2A12",
        "2A22A",
        "A21A2",
        "12A21",
        "2A21A"
    };

    int width = test_layout[0].size();
    int height = test_layout.size();

    // Instantiate our zero-allocation reactor simulation
    ReactorState<> reactor(width, height);
    
    // Load the visual representation into the grid
    loadLayout(reactor, test_layout);

    // Run the simulation loop
    std::cout << "Simulating layout to equilibrium..." << std::endl;
    auto result = reactor.simulate_to_equilibrium();

    // Output the results for verification
    std::cout << "\n--- Simulation Results ---" << std::endl;
    std::cout << "Total RF/t: " << result.rf_per_tick << std::endl;
    std::cout << "Max Temp:   " << result.max_temp << " C" << std::endl;
    std::cout << "Meltdown:   " << (result.melted_down ? "YES (>4000 C)" : "NO") << std::endl; //[cite: 1]

    return 0;
}