// neighbor.cu
#include "neighbour.cuh"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

// Define max_particles_per_cell if not defined elsewhere
#ifndef max_particles_per_cell
#define max_particles_per_cell 64
#endif

extern __constant__ float d_box_size[3];  // Add this at the top

void build_neighbor_list(DeviceNeighborData& nb_data, const Particle* d_particles, const DeviceBinningData& bin_data, const Grid& grid, float rcut) {
    int max_neighbors = 27 * max_particles_per_cell;
    if (nb_data.neighbors == nullptr) {
        cudaMalloc(&nb_data.neighbors, bin_data.num_particles * max_neighbors * sizeof(int));
        cudaMalloc(&nb_data.num_neighbors, bin_data.num_particles * sizeof(int));
        nb_data.max_neighbors = max_neighbors;
    }
    float rcut_sq = rcut * rcut;
    //float box_size_arr[3] = {box_size[0], box_size[1], box_size[2]};
    int blockSize = 256;
    int gridSize = (bin_data.num_particles + blockSize - 1) / blockSize;
    kernel_build_neighbor_list<<<gridSize, blockSize>>>(
        d_particles, bin_data, grid, nb_data, rcut_sq
    );
    cudaDeviceSynchronize();
}

__global__ void kernel_build_neighbor_list(
    const Particle* particles,
    const DeviceBinningData bin_data,
    const Grid grid,
    DeviceNeighborData nb_data,
    float *box_size,
    float rcut_sq
) {
    int i_sorted = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_sorted >= bin_data.num_particles) return;

    int original_i = bin_data.particle_indices[i_sorted];
    const Particle* pi = &particles[original_i];
    int* neighbor_list = &nb_data.neighbors[original_i * nb_data.max_neighbors];
    int count = 0;

    // Get cell coordinates
    int cell_idx = bin_data.cell_indices[i_sorted];
    int cz = cell_idx / (grid.dims.x * grid.dims.y);
    int cy = (cell_idx % (grid.dims.x * grid.dims.y)) / grid.dims.x;
    int cx = cell_idx % grid.dims.x;

    // Check neighboring cells (3x3x3)
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int3 nb_coord = make_int3(cx + dx, cy + dy, cz + dz);
                
                // Apply PBC to grid
                if (nb_coord.x < 0) nb_coord.x += grid.dims.x;
                else if (nb_coord.x >= grid.dims.x) nb_coord.x -= grid.dims.x;
                if (nb_coord.y < 0) nb_coord.y += grid.dims.y;
                else if (nb_coord.y >= grid.dims.y) nb_coord.y -= grid.dims.y;
                if (nb_coord.z < 0) nb_coord.z += grid.dims.z;
                else if (nb_coord.z >= grid.dims.z) nb_coord.z -= grid.dims.z;

                int nb_idx = nb_coord.x + nb_coord.y * grid.dims.x + 
                            nb_coord.z * grid.dims.x * grid.dims.y;
                
                int start = bin_data.cell_offsets[nb_idx];
                int end = bin_data.cell_offsets[nb_idx + 1];
                
                for (int j_sorted = start; j_sorted < end; j_sorted++) {
                    int original_j = bin_data.particle_indices[j_sorted];
                    if (original_i == original_j) continue;
                    
                    // Distance calculation with MIC
                    Vector3 rij = particles[original_j].position - pi->position;
                    for (int d = 0; d < 3; d++) {
                        if (rij[d] >  0.5f * d_box_size[d]) rij[d] -= d_box_size[d];
                        else if (rij[d] < -0.5f * d_box_size[d]) rij[d] += d_box_size[d];
                    }
                    
                    float r2 = rij.squaredNorm();
                    if (r2 < rcut_sq && count < nb_data.max_neighbors) {
                        neighbor_list[count++] = original_j;
                    }
                }
            }
        }
    }
    nb_data.num_neighbors[original_i] = count;
}