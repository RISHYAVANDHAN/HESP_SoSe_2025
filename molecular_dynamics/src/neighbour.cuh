#pragma once
#include "binning.cuh"

struct DeviceBinningData;
struct Grid;
struct Particle;

struct DeviceNeighborData {
    int* neighbors;       // Flattened neighbor lists
    int* num_neighbors;   // Number of neighbors per particle
    float* rcut_sq;       // Squared cutoff distance
    int max_neighbors;    // Max neighbors per particle
    int num_particles;    // Number of particles
};

void build_neighbor_list(DeviceNeighborData& nb_data, const Particle* d_particles, const DeviceBinningData& bin_data, const Grid& grid, float rcut);

void free_neighbor_data(DeviceNeighborData& nb_data);

__global__ void kernel_build_neighbor_list(
    const Particle* particles,
    const DeviceBinningData bin_data,
    const Grid grid,
    DeviceNeighborData nb_data,
    float rcut_sq
);