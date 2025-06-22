#pragma once
#include <string>

    
enum class MethodType {
    BASE,
    CUTOFF,
    CELL,
    NEIGHBOUR
};

struct SimulationConfig {
    std::string input_file;
    std::string output_dir;
    float dt;
    int num_steps;
    float sigma;
    float epsilon;
    int output_freq;
    bool benchmark;
    float box_size[3];                              // x, y, z = [0, 1, 2]
    float rcut = 0.0f;       
    MethodType method = MethodType::BASE;
    float particle_radius;                      // Default particle radius
    float stiffness;                            // Spring constant
    float damping;                              // Damping coefficient
    float bounce_coeff;                         // Bounce coefficient
    float gravity = 9.8;                        // default 9.8, but can be changed anytime in command line
};

void parse_command_line_args(int argc, char** argv, SimulationConfig& config);