#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include "particle.cuh"
#include "binning.cuh"
#include "../input/cli.cuh"
#include "neighbour.cuh"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

static DeviceBinningData bin_data = {nullptr, nullptr, nullptr, 0, 0};
static DeviceNeighborData nb_data = {nullptr, nullptr, 0, 0};
static Grid grid;
static bool first_run = true;

__constant__ float d_box_size[3];

__constant__ float d_gravity[3] = {0.0f, -9.81f, 0.0f};

__constant__ DEMParams d_dem_params;

void load_particles_from_file(const std::string& filename, Particle*& particles, int& num_particles) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Cannot open input file " + filename);
    }

    std::string line;
    std::getline(file, line);  // Skip header line

    std::vector<Particle> particle_p;
    float x, y, z, vx, vy, vz, m;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;  // Skip empty/comment lines
        std::istringstream iss(line);
        if (!(iss >> x >> y >> z >> vx >> vy >> vz >> m)) continue;

        Particle p;
        p.position = Vector3(x, y, z);
        p.velocity = Vector3(vx, vy, vz);
        p.force = Vector3(0.0f, 0.0f, 0.0f);
        p.mass = m;
        particle_p.push_back(p);
    }

    file.close();

    num_particles = particle_p.size();
    particles = new Particle[num_particles];
    for (int i = 0; i < num_particles; ++i) {
        particles[i] = particle_p[i];
    }
}


__host__ void initialize_particles(Particle* particles, int num_particles, float spacing) {
    int n = std::ceil(std::cbrt(num_particles));
    int idx = 0;
    for (int x = 0; x < n; ++x) {
        for (int y = 0; y < n; ++y) {
            for (int z = 0; z < n; ++z) {
                if (idx < num_particles) {
                    particles[idx].position = Vector3(0.2f + x * spacing, 0.2f + y * spacing, 0.2f + z * spacing);
                    particles[idx].velocity = Vector3(0.0f, 0.0f, 0.0f);
                    particles[idx].force = Vector3(0.0f, 0.0f, 0.0f);
                    particles[idx].mass = 1.0f;
                    idx++;
                }
            }
        }
    }
}

__global__ void clear_forces(Particle* particles, int num_particles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_particles) {
        particles[idx].force = Vector3(0.0f, 0.0f, 0.0f);
    }
}

__global__ void apply_gravity(Particle* particles, int num_particles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_particles) {
        Vector3 gravity(d_gravity[0], d_gravity[1], d_gravity[2]);
        particles[idx].force += gravity * particles[idx].mass;
    }
}

__host__ void print_particles(const Particle* particles, int num_particles) {
    for (int i = 0; i < num_particles; ++i) {
        const auto& p = particles[i];
        std::cout << "Particle " << i
                  << " | Pos: (" << p.position.x << ", " << p.position.y << ", " << p.position.z << ")"
                  << " | Vel: (" << p.velocity.x << ", " << p.velocity.y << ", " << p.velocity.z << ")"
                  << " | Force: (" << p.force.x << ", " << p.force.y << ", " << p.force.z << ")"
                  << " | Mass: " << p.mass << ", "
                  << " | Radius: "<< p.radius << '\n' ;
    }
    std::cout<<std::endl;
}


__global__ void velocity_verlet_step1(Particle* particles, int num_particles, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_particles) {
        auto& p = particles[i];
        p.acceleration = (p.force / p.mass);
        p.position += p.velocity * dt + p.acceleration * (0.5f * dt * dt);
        
        // Reflexive boundary conditions
        for (int d = 0; d < 3; ++d) {
            if (p.position[d] < p.radius) {
                p.position[d] = 2.0f * p.radius - p.position[d];
                p.velocity[d] = -p.velocity[d] * d_dem_params.bounce_coeff;
            }
            else if (p.position[d] > d_box_size[d] - p.radius) {
                p.position[d] = 2.0f * (d_box_size[d] - p.radius) - p.position[d];
                p.velocity[d] = -p.velocity[d] * d_dem_params.bounce_coeff;
            }
        }
    }
}

__global__ void velocity_verlet_step2(Particle* particles, int num_particles, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_particles) {
        auto& p = particles[i];

        Vector3 a_new = p.force / p.mass;
        
        // Update velocity: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
        p.velocity += (p.acceleration + a_new) * (0.5f * dt);
        
        // Store new acceleration for next step
        p.acceleration = a_new;
    }
}

__device__ bool in_grid_bounds(int3 coord, int3 dims) {
    return coord.x >= 0 && coord.x < dims.x &&
           coord.y >= 0 && coord.y < dims.y &&
           coord.z >= 0 && coord.z < dims.z;
}

__device__ void compute_contact_force(Particle* pi, Particle* pj, Vector3& force) {
    Vector3 rij = pj->position - pi->position;
    float dist = rij.norm();
    float r_sum = pi->radius + pj->radius;
    float overlap = r_sum - dist;
    
    if (overlap > 0) {
        Vector3 normal = rij / dist;
        Vector3 v_rel = pj->velocity - pi->velocity;
        float v_normal = v_rel.dot(normal);
        float f_contact = d_dem_params.stiffness * overlap;
        float f_damp = d_dem_params.damping * v_normal;
        force = normal * (f_contact - f_damp);
    }
}

__global__ void compute_dem_contacts(Particle* particles, int num_particles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    Particle& pi = particles[i];
    Vector3 total_force(0.0f, 0.0f, 0.0f);

    for (int j = 0; j < num_particles; ++j) {
        if (i == j) continue;
        Particle& pj = particles[j];
        Vector3 contact_force;
        compute_contact_force(&pi, &pj, contact_force);
        total_force += contact_force;
    }
    pi.force += total_force;
}

__global__ void compute_dem_contacts_binned(Particle* particles, int num_particles, const DeviceBinningData bin_data, const Grid grid) {
    int i_sorted = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_sorted >= num_particles) return;

    int original_i = bin_data.particle_indices[i_sorted];
    Particle* pi = &particles[original_i];
    Vector3 total_force(0.0f, 0.0f, 0.0f);
    
    int cell_idx = bin_data.cell_indices[i_sorted];
    int cz = cell_idx / (grid.dims.x * grid.dims.y);
    int cy = (cell_idx % (grid.dims.x * grid.dims.y)) / grid.dims.x;
    int cx = cell_idx % grid.dims.x;

    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int3 neighbor_coord = make_int3(cx + dx, cy + dy, cz + dz);
                
                // Skip out-of-bound cells for DEM
                if (neighbor_coord.x < 0 || neighbor_coord.x >= grid.dims.x ||
                    neighbor_coord.y < 0 || neighbor_coord.y >= grid.dims.y ||
                    neighbor_coord.z < 0 || neighbor_coord.z >= grid.dims.z) {
                    continue;
                }
                
                int neighbor_idx =  neighbor_coord.x + 
                                    neighbor_coord.y * grid.dims.x + 
                                    neighbor_coord.z * grid.dims.x * grid.dims.y;
                
                int start = bin_data.cell_offsets[neighbor_idx];
                int end = bin_data.cell_offsets[neighbor_idx + 1];
                
                for (int j_sorted = start; j_sorted < end; j_sorted++) {
                    int original_j = bin_data.particle_indices[j_sorted];
                    if (original_i == original_j) continue;
                    
                    Particle* pj = &particles[original_j];
                    Vector3 contact_force;
                    compute_contact_force(pi, pj, contact_force);
                    total_force += contact_force;
                }
            }
        }
    }
    pi->force += total_force;
}

__global__ void compute_dem_contacts_neighbor(Particle* particles, int num_particles, const DeviceNeighborData nb_data) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    Particle& pi = particles[i];
    Vector3 total_force(0.0f, 0.0f, 0.0f);

    int neighbor_count = nb_data.num_neighbors[i];
    int* neighbor_list = &nb_data.neighbors[i * nb_data.max_neighbors];

    for (int n = 0; n < neighbor_count; ++n) {
        int j = neighbor_list[n];
        Particle& pj = particles[j];
        Vector3 contact_force;
        compute_contact_force(&pi, &pj, contact_force);
        total_force += contact_force;
    }
    pi.force += total_force;
}


__host__ void run_simulation(Particle* particles, int num_particles, float dt, float sigma, float epsilon, float rcut, const float box_size[3], MethodType method, const DEMParams& dem_params) 
{
    Particle* d_particles;
    size_t size = num_particles * sizeof(Particle);
    cudaMalloc(&d_particles, size);
    cudaMemcpy(d_particles, particles, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (num_particles + blockSize - 1) / blockSize;
    float max_radius = 0.0f;

    // Prepare box_size array for device kernels
    float box_size_arr[3] = {box_size[0], box_size[1], box_size[2]};
    cudaMemcpyToSymbol(d_box_size, box_size_arr, 3 * sizeof(float));

    // Initialize binning structures on first run
    if (first_run && rcut > 0.0f) {
        // Define simulation domain
        AABB domain;
        domain.min = make_float3(0, 0, 0);
        domain.max = make_float3(box_size[0], box_size[1], box_size[2]);
        
        // Compute grid dimensions
        grid = compute_grid(domain, rcut);
        bin_data.num_particles = num_particles;
        bin_data.num_cells = grid.dims.x * grid.dims.y * grid.dims.z;
        
        // Allocate memory for binning data
        cudaMalloc(&bin_data.cell_indices, num_particles * sizeof(int));
        cudaMalloc(&bin_data.particle_indices, num_particles * sizeof(int));
        cudaMalloc(&bin_data.cell_offsets, (bin_data.num_cells + 1) * sizeof(int));
        
        // Initialize particle indices
        thrust::device_ptr<int> d_ptr(bin_data.particle_indices);
        thrust::sequence(d_ptr, d_ptr + num_particles);
        
        first_run = false;
    }

    cudaMemcpyToSymbol(d_dem_params, &dem_params, sizeof(DEMParams));

    // Step 1: Position update
    if (dt > 0.0f) {
        velocity_verlet_step1<<<gridSize, blockSize>>>(d_particles, num_particles, dt);
        cudaDeviceSynchronize();
    }

    // Clear forces and apply gravity
    clear_forces<<<gridSize, blockSize>>>(d_particles, num_particles);
    apply_gravity<<<gridSize, blockSize>>>(d_particles, num_particles);
    cudaDeviceSynchronize();

    // Compute DEM contacts based on method
    switch (method) {
        case MethodType::BASE:
            compute_dem_contacts<<<gridSize, blockSize>>>(d_particles, num_particles);
            break;
            
        case MethodType::CELL:
            build_binning(bin_data, d_particles, grid);
            compute_dem_contacts_binned<<<gridSize, blockSize>>>(d_particles, num_particles, bin_data, grid);
            break;
            
        case MethodType::NEIGHBOUR:
            if (nb_data.neighbors == nullptr) {
                int max_neighbors = 64 * num_particles;
                cudaMalloc(&nb_data.neighbors, num_particles * max_neighbors * sizeof(int));
                cudaMalloc(&nb_data.num_neighbors, num_particles * sizeof(int));
                nb_data.max_neighbors = max_neighbors;
                nb_data.num_particles = num_particles;
            }
            build_binning(bin_data, d_particles, grid);
            // Set cutoff to max particle diameter * 1.2
            for (int i = 0; i < num_particles; i++) {
                if (particles[i].radius > max_radius) max_radius = particles[i].radius;
            }
            build_neighbor_list(nb_data, d_particles, bin_data, grid, 2.4f * max_radius);
            compute_dem_contacts_neighbor<<<gridSize, blockSize>>>(d_particles, num_particles, nb_data);
            break;
            
        default: // CUTOFF not used in DEM
            compute_dem_contacts<<<gridSize, blockSize>>>(d_particles, num_particles);
            break;
    }
    cudaDeviceSynchronize();

    // Step 2: Velocity update
    if (dt > 0.0f) {
        velocity_verlet_step2<<<gridSize, blockSize>>>(d_particles, num_particles, dt);
        cudaDeviceSynchronize();
    }

    // Copy results back to host
    cudaMemcpy(particles, d_particles, size, cudaMemcpyDeviceToHost);
    cudaFree(d_particles);
}



__host__ void print_diagnostics(const Particle* particles, int num_particles) {
    Vector3 total_momentum(0.0f, 0.0f, 0.0f);
    float total_kinetic_energy = 0.0f;
    float max_velocity = 0.0f;
    float max_position = 0.0f;

    for (int i = 0; i < num_particles; ++i) {
        const Particle& p = particles[i];

        total_momentum += p.velocity * p.mass;
        total_kinetic_energy += 0.5f * p.mass * p.velocity.squaredNorm();

        float vel_mag = p.velocity.norm();
        float pos_mag = p.position.norm();

        if (vel_mag > max_velocity) max_velocity = vel_mag;
        if (pos_mag > max_position) max_position = pos_mag;
    }

    std::cout << "[Diagnostics] "
              << "Total KE: " << total_kinetic_energy << " | "
              << "Momentum: (" << total_momentum.x << ", " << total_momentum.y << ", " << total_momentum.z << ") | "
              << "Max Vel: " << max_velocity << " | "
              << "Max Pos: " << max_position << '\n';
}

void cleanup_simulation() {
    if (bin_data.cell_indices) cudaFree(bin_data.cell_indices);
    if (bin_data.particle_indices) cudaFree(bin_data.particle_indices);
    if (bin_data.cell_offsets) cudaFree(bin_data.cell_offsets);
    bin_data = {nullptr, nullptr, nullptr, 0, 0};
    first_run = true;

    if (nb_data.neighbors) cudaFree(nb_data.neighbors);
    if (nb_data.num_neighbors) cudaFree(nb_data.num_neighbors);
    nb_data = {nullptr, nullptr, 0, 0};
}









