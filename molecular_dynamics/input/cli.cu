#include "cli.cuh"
#include <cstring>
#include <iostream>

void parse_command_line_args(int argc, char** argv, SimulationConfig& config) {

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--dt") == 0 && i + 1 < argc)
            config.dt = std::stof(argv[++i]);
        else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc)
            config.num_steps = std::stoi(argv[++i]);
        else if (strcmp(argv[i], "--sigma") == 0 && i + 1 < argc)
            config.sigma = std::stof(argv[++i]);
        else if (strcmp(argv[i], "--epsilon") == 0 && i + 1 < argc)
            config.epsilon = std::stof(argv[++i]);
        else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc)
            config.input_file = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            config.output_dir = argv[++i];
        else if (strcmp(argv[i], "--freq") == 0 && i + 1 < argc)
            config.output_freq = std::stoi(argv[++i]);
        else if (strcmp(argv[i], "--benchmark") == 0)
            config.benchmark = true;
        else if (strcmp(argv[i], "--test") == 0 && i + 1 < argc) {
            std::string val = argv[++i];
            if (val == "stable") 
                config.test_case = TestCaseType::STABLE;
            else if (val == "repulsive") 
                config.test_case = TestCaseType::REPULSIVE;
            else if (val == "attractive") 
                config.test_case = TestCaseType::ATTRACTIVE;
        }
        else if (strcmp(argv[i], "--box") == 0 && i + 3 < argc) {
            config.box_size[0] = std::stof(argv[++i]);
            config.box_size[1] = std::stof(argv[++i]);
            config.box_size[2] = std::stof(argv[++i]);
        }
        else if (strcmp(argv[i], "--rcut") == 0 && i + 1 < argc) {
            config.rcut = std::stof(argv[++i]);
        }
        else if (strcmp(argv[i], "--method") == 0 && i + 1 < argc) {
            std::string m = argv[++i];
            if (m == "base")        config.method = MethodType::BASE;
            else if (m == "cutoff") config.method = MethodType::CUTOFF;
            else if (m == "cell")   config.method = MethodType::CELL;
            else if (m == "neighbour") config.method = MethodType::NEIGHBOUR;
        }
        else if (strcmp(argv[i], "--particle_radius") == 0 && i + 1 < argc) {
            config.particle_radius = std::stof(argv[++i]);
        }
        else if (strcmp(argv[i], "--stiffness") == 0 && i + 1 < argc) {
            config.stiffness = std::stof(argv[++i]);
        }
        else if (strcmp(argv[i], "--damping") == 0 && i + 1 < argc) {
            config.damping = std::stof(argv[++i]);
        }
        else if (strcmp(argv[i], "--bounce_coeff") == 0 && i + 1 < argc) {
            config.bounce_coeff = std::stof(argv[++i]);
        }

        else if (strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage:\n"
                      << "--dt <float>           Time step\n"
                      << "--steps <int>          Number of steps\n"
                      << "--sigma <float>        LJ sigma\n"
                      << "--epsilon <float>      LJ epsilon\n"
                      << "--input <path>         Input file\n"
                      << "--output <dir>         Output directory\n"
                      << "--freq <int>           VTK output frequency\n"
                      << "--benchmark            Enable performance logging\n"
                      << "--test <case>          Run predefined test case: stable | repel | attract\n"
                      << "--box <float 3>        Dimensions of the box boundary x, y and z\n"
                      << "--rcut <float>         cutoff radius\n"
                      << "--method               accelaration techniques\n";
                      
            exit(0);
        } else {
            std::cerr << "Unknown option: " << argv[i] << "\n";
        }
    }
}
