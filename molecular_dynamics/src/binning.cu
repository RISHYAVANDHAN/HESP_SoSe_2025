// binning.cu
#include "binning.cuh"
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

Grid compute_grid(const AABB& domain, float cutoff) {
    Grid grid;
    grid.origin = domain.min;
    grid.cell_size = cutoff;
    grid.dims = make_int3(
        static_cast<int>(ceil((domain.max.x - domain.min.x) / cutoff)),
        static_cast<int>(ceil((domain.max.y - domain.min.y) / cutoff)),
        static_cast<int>(ceil((domain.max.z - domain.min.z) / cutoff))
    );
    return grid;
}

__host__ __device__ int compute_cell_index(const float3& pos, const Grid& grid) {
    int cx = static_cast<int>((pos.x - grid.origin.x) / grid.cell_size);
    int cy = static_cast<int>((pos.y - grid.origin.y) / grid.cell_size);
    int cz = static_cast<int>((pos.z - grid.origin.z) / grid.cell_size);

    cx = max(0, min(cx, grid.dims.x - 1));
    cy = max(0, min(cy, grid.dims.y - 1));
    cz = max(0, min(cz, grid.dims.z - 1));

    return cx + cy * grid.dims.x + cz * grid.dims.x * grid.dims.y;
}

__global__ void kernel_assign_cell_indices(
    const Particle* particles, int* cell_indices, int N, Grid grid
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float3 pos = make_float3(particles[i].position.x, 
                                particles[i].position.y, 
                                particles[i].position.z);
        cell_indices[i] = compute_cell_index(pos, grid);
    }
}

// Kernel: Mark the start of each cell in the sorted cell_indices array
__global__ void kernel_mark_cell_starts(
    const int* sorted_cell_indices, int* cell_offsets, int num_particles, int num_cells
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    
    int cell = sorted_cell_indices[i];
    // Only mark the first occurrence of each cell
    if (i == 0 || cell != sorted_cell_indices[i - 1]) {
        cell_offsets[cell] = i;
    }
    // Always set the end offset for the last cell
    if (i == num_particles - 1) {
        cell_offsets[num_cells] = num_particles;
    }
    // protect going out-of-bounds in corner
    if (cell < num_cells)
    cell_offsets[cell] = i;

}

void build_binning(DeviceBinningData& bin_data, const Particle* d_particles, const Grid& grid) {
    int num_particles = bin_data.num_particles;
    
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // SAFETY: Validate particle count
    if (num_particles <= 0) {
        std::cerr << "Error: Invalid particle count: " << num_particles << std::endl;
        return;
    }

    // SAFETY: Choose safe block size
    int blockSize = 256;
    if (blockSize > prop.maxThreadsPerBlock) {
        blockSize = prop.maxThreadsPerBlock;
        std::cout << "Adjusted block size to " << blockSize << " (max threads per block)\n";
    }

    // SAFETY: Calculate grid size with overflow protection
    int gridSize = (num_particles + blockSize - 1) / blockSize;
    
    // SAFETY: Clamp grid size to CUDA limits
    if (gridSize > prop.maxGridSize[0]) {
        gridSize = prop.maxGridSize[0];
        std::cerr << "Warning: Grid size clamped to " << gridSize 
                  << " (max allowed: " << prop.maxGridSize[0] << ")\n";
    }

    // SAFETY: Check for zero grid size
    if (gridSize == 0) {
        std::cerr << "Error: Invalid grid size calculation (particles: " 
                  << num_particles << ", block: " << blockSize << ")\n";
        return;
    }

    std::cout << "Launching kernel: grid=" << gridSize << ", block=" << blockSize 
              << ", particles=" << num_particles << "\n";
    
    // Launch kernel with enhanced error checking
    kernel_assign_cell_indices<<<gridSize, blockSize>>>(d_particles, bin_data.cell_indices, num_particles, grid);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) 
                  << " (code " << err << ")\n";
        std::cerr << "Grid: " << gridSize << " blocks, Block: " << blockSize << " threads\n";
        std::cerr << "Particles: " << num_particles << "\n";
        exit(1);
    }

    // Note: cell_offsets[c] gives the start index in sorted particle_indices for cell c,
    // and cell_offsets[c+1] gives the end index (exclusive).
}

void free_binning_data(DeviceBinningData& bin_data) {
    cudaFree(bin_data.cell_indices);
    cudaFree(bin_data.particle_indices);
    cudaFree(bin_data.cell_offsets);
    bin_data = {nullptr, nullptr, nullptr, 0, 0};
}