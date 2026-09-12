#include "runners.hpp"
#include <iostream>
#include <string>
#include <locale>
#include <map>
#include <fstream>

// Neat printing.
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"

using namespace reactor_optimizer::core;
using namespace reactor_optimizer::runners;

constexpr size_t OPTIMIZE_W = 5;
constexpr size_t OPTIMIZE_H = 5;
constexpr ReactorConfig OPTIMIZE_CONFIG{
    .rf_per_pulse = 9600,
    .meltdown_temp = 4000
};

constexpr std::array<BlockType, 4> ALLOWED_BLOCKS = {
    // BlockType::Empty,
    // BlockType::SingleRod,
    // BlockType::DoubleRod,
    BlockType::QuadRod,
    // BlockType::Reflector,
    BlockType::HeatPipe,
    BlockType::HeatVent,
    BlockType::Absorber
};

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
void loadLayout(ReactorState<OPTIMIZE_W, OPTIMIZE_H, OPTIMIZE_CONFIG>& reactor, const std::vector<std::string>& layout) {
    for (size_t y = 0; y < layout.size(); ++y) {
        for (size_t x = 0; x < layout[y].size(); ++x) {
            reactor.set_block(x, y, char_to_block(layout[y][x]));
        }
    }
}

void print_layout(ReactorState<OPTIMIZE_W, OPTIMIZE_H, OPTIMIZE_CONFIG>& reactor) {
    for (int y = 0; y < OPTIMIZE_H; ++y) {
        for (int x = 0; x < OPTIMIZE_W; ++x) {
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

void export_building_gadgets(ReactorState<OPTIMIZE_W, OPTIMIZE_H, OPTIMIZE_CONFIG>& reactor, const std::string& filename) {
    std::map<std::string, int> counts;
    std::string statelist = "";
    
    // Add +2 to X and Z to accommodate the outer encasing walls. Y is exactly 3 layers tall.
    int x_size = OPTIMIZE_W + 2;
    int z_size = OPTIMIZE_H + 2;
    int y_size = 3;

    bool first_block = true;
    
    auto add_block = [&](int id, const std::string& name) {
        if (!first_block) { statelist += ","; }
        statelist += std::to_string(id);
        first_block = false;
        if (!name.empty() && name != "minecraft:air") {
            counts[name]++;
        }
    };

    // Iterate how Building Gadgets parses volumetric data: Z (outer) -> Y (middle) -> X (inner).
    for (int z = 0; z < z_size; ++z) {
        for (int y = 0; y < y_size; ++y) {
            for (int x = 0; x < x_size; ++x) {
                // Edge logic applies to the X and Z axes for the casing.
                bool is_edge_xz = (x == 0 || x == x_size - 1 || z == 0 || z == z_size - 1);

                if (y == 0) {
                    // LAYER 0 (Floor): Completely solid reactor wall.
                    add_block(1, "oritech:reactor_wall");
                } 
                else if (y == 1) {
                    // LAYER 1 (Core): Edges are walls, inner blocks are the optimized layout.
                    if (is_edge_xz) {
                        add_block(1, "oritech:reactor_wall");
                    } else {
                        BlockType block = reactor.get_block(x - 1, z - 1);
                        switch (block) {
                            case BlockType::SingleRod: add_block(2, "oritech:reactor_rod"); break;
                            case BlockType::DoubleRod: add_block(3, "oritech:reactor_double_rod"); break;
                            case BlockType::QuadRod:   add_block(4, "oritech:reactor_quad_rod"); break;
                            case BlockType::Reflector: add_block(5, "oritech:reactor_reflector"); break;
                            case BlockType::Absorber:  add_block(6, "oritech:reactor_condenser"); break;
                            case BlockType::HeatPipe:  add_block(7, "oritech:reactor_heat_pipe"); break;
                            case BlockType::HeatVent:  add_block(8, "oritech:reactor_vent"); break;
                            case BlockType::Empty:
                            default: add_block(0, "minecraft:air"); break;
                        }
                    }
                } 
                else if (y == 2) {
                    // LAYER 2 (Roof): Edges are walls, inner blocks map ports to their required core components.
                    if (is_edge_xz) {
                        add_block(1, "oritech:reactor_wall");
                    } else {
                        BlockType block = reactor.get_block(x - 1, z - 1);
                        if (block == BlockType::SingleRod || block == BlockType::DoubleRod || block == BlockType::QuadRod) {
                            add_block(10, "oritech:reactor_fuel_port");
                        } else if (block == BlockType::Absorber) {
                            add_block(9, "oritech:reactor_absorber_port");
                        } else {
                            add_block(1, "oritech:reactor_wall");
                        }
                    }
                }
            }
        }
    }

    std::ofstream out(filename);
    out << "{\n";
    out << "  \"name\": \"Optimized Reactor " << OPTIMIZE_W << "x" << OPTIMIZE_H << "\",\n";
    
    // SNBT Palette array mapping IDs 0-9 to their precise NBT properties.
    std::string palette = "{Name:\\\"minecraft:air\\\"},"
                          "{Name:\\\"oritech:reactor_wall\\\"},"
                          "{Name:\\\"oritech:reactor_rod\\\",Properties:{lit:\\\"false\\\"}},"
                          "{Name:\\\"oritech:reactor_double_rod\\\",Properties:{lit:\\\"false\\\"}},"
                          "{Name:\\\"oritech:reactor_quad_rod\\\",Properties:{lit:\\\"false\\\"}},"
                          "{Name:\\\"oritech:reactor_reflector\\\"},"
                          "{Name:\\\"oritech:reactor_condenser\\\"},"
                          "{Name:\\\"oritech:reactor_heat_pipe\\\"},"
                          "{Name:\\\"oritech:reactor_vent\\\"},"
                          "{Name:\\\"oritech:reactor_absorber_port\\\"},"
                          "{Name:\\\"oritech:reactor_fuel_port\\\"}";
    
    // Write the 3D bounding box dimensions and flat integer state list.
    out << "  \"statePosArrayList\": \"{blockstatemap:[" << palette << "],endpos:{X:" << (x_size - 1) << ",Y:" << (y_size - 1) << ",Z:" << (z_size - 1) << "},startpos:{X:0,Y:0,Z:0},statelist:[I;" << statelist << "]}\",\n";
    
    // Write required materials block with correct unicode formatting.
    out << "  \"requiredItems\": {\n";
    bool first_item = true;
    for (const auto& pair : counts) {
        if (!first_item) out << ",\n";
        out << "    \"oritech:Reference{ResourceKey[minecraft:item / " << pair.first << "]\\u003d" << pair.first << "}\": " << pair.second;
        first_item = false;
    }
    out << "\n  }\n";
    out << "}\n";
    out.close();
}

int main() {
    std::cout.imbue(std::locale("en_US.UTF-8"));

    
    enum class Mode { BranchAndBound, SimulatedAnnealing, PPO };
    
    // Easily toggle between algorithms here
    Mode current_mode = Mode::PPO; 
    
    ReactorState<OPTIMIZE_W, OPTIMIZE_H, OPTIMIZE_CONFIG> final_reactor;

    switch (current_mode) {
        case Mode::BranchAndBound: {
            final_reactor = run_branch_and_bound<OPTIMIZE_W, OPTIMIZE_H, OPTIMIZE_CONFIG>(ALLOWED_BLOCKS);
            break;
        }
        case Mode::SimulatedAnnealing: {
            final_reactor = run_simulated_annealing<OPTIMIZE_W, OPTIMIZE_H, OPTIMIZE_CONFIG>(ALLOWED_BLOCKS);
            break;
        }
        case Mode::PPO: {
            run_ppo_collection();
            return 0;
        }
    }

    std::cout << "--- Final Reactor Layout ---" << std::endl;
    print_layout(final_reactor);

    // Generate the Building Gadgets blueprint.
    std::string filename = "schematics/optimized_reactor_" + std::to_string(OPTIMIZE_W) + "x" + std::to_string(OPTIMIZE_H) + ".json";
    export_building_gadgets(final_reactor, filename);
    std::cout << "\nSchematic exported to: " << filename << std::endl;
    return 0;
}