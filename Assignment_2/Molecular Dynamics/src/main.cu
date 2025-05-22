#include "io.h"
#include "md_kernel.cuh"
#include "vtk_writer.h"
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <string>

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " input.txt dt nsteps sigma epsilon" << std::endl;
        return 1;
    }
    std::string input_file = argv[1];
    float dt = std::stof(argv[2]);
    int nsteps = std::stoi(argv[3]);
    float sigma = std::stof(argv[4]);
    float epsilon = std::stof(argv[5]);

    std::vector<Particle> particles;
    if (!read_particles(input_file, particles)) return 1;
    int N = particles.size();

    // Allocate device memory
    Particle* d_particles;
    cudaMalloc(&d_particles, N * sizeof(Particle));
    cudaMemcpy(d_particles, particles.data(), N * sizeof(Particle), cudaMemcpyHostToDevice);

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < nsteps; ++step) {
        launch_compute_forces(d_particles, N, sigma, epsilon);
        launch_integrate(d_particles, N, dt);

        // Output VTK every 100 steps
        if (step % 100 == 0) {
            cudaMemcpy(particles.data(), d_particles, N * sizeof(Particle), cudaMemcpyDeviceToHost);
            std::string vtkfile = "output/step_" + std::to_string(step) + ".vtk";
            write_vtk(vtkfile, particles, step);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "Total simulation time: " << elapsed << " s\n";
    std::cout << "Average time per step: " << (elapsed / nsteps) << " s\n";

    cudaMemcpy(particles.data(), d_particles, N * sizeof(Particle), cudaMemcpyDeviceToHost);
    cudaFree(d_particles);

    void print_particles(const std::vector<Particle>& particles, int max_print);
    print_particles(particles, 5);
    return 0;
}