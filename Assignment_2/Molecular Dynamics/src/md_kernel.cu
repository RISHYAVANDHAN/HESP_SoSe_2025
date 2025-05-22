#include "io.h"
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <iostream>

// Lennard-Jones force calculation kernel
__global__ void compute_forces_kernel(Particle* particles, int N, float sigma, float epsilon) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float3 force = make_float3(0, 0, 0);
    float3 pi = particles[i].pos;

    for (int j = 0; j < N; ++j) {
        if (i == j) continue;
        float3 pj = particles[j].pos;
        float3 rij = {pj.x - pi.x, pj.y - pi.y, pj.z - pi.z};
        float r2 = rij.x*rij.x + rij.y*rij.y + rij.z*rij.z + 1e-12f;
        float r6 = r2 * r2 * r2;
        float r12 = r6 * r6;
        float sig6 = sigma*sigma*sigma*sigma*sigma*sigma;
        float sig12 = sig6 * sig6;
        float fmag = 24 * epsilon * (2 * sig12 / r12 - sig6 / r6) / r2;
        force.x += fmag * rij.x;
        force.y += fmag * rij.y;
        force.z += fmag * rij.z;
    }
    particles[i].acc.x = force.x / particles[i].mass;
    particles[i].acc.y = force.y / particles[i].mass;
    particles[i].acc.z = force.z / particles[i].mass;
}

void launch_compute_forces(Particle* d_particles, int N, float sigma, float epsilon) {
    int block = 128;
    int grid = (N + block - 1) / block;
    compute_forces_kernel<<<grid, block>>>(d_particles, N, sigma, epsilon);
}

// Velocity Verlet integration kernel
__global__ void integrate_kernel(Particle* particles, int N, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    particles[i].vel.x += 0.5f * particles[i].acc.x * dt;
    particles[i].vel.y += 0.5f * particles[i].acc.y * dt;
    particles[i].vel.z += 0.5f * particles[i].acc.z * dt;

    particles[i].pos.x += particles[i].vel.x * dt;
    particles[i].pos.y += particles[i].vel.y * dt;
    particles[i].pos.z += particles[i].vel.z * dt;
}

void launch_integrate(Particle* d_particles, int N, float dt) {
    int block = 128;
    int grid = (N + block - 1) / block;
    integrate_kernel<<<grid, block>>>(d_particles, N, dt);
}

void print_particles(const std::vector<Particle>& particles, int max_print) {
    for (int i = 0; i < std::min((int)particles.size(), max_print); ++i) {
        const auto& p = particles[i];
        std::cout << "Particle " << i << ": pos=(" << p.pos.x << "," << p.pos.y << "," << p.pos.z
                  << ") vel=(" << p.vel.x << "," << p.vel.y << "," << p.vel.z
                  << ") mass=" << p.mass << std::endl;
    }
}

void check_cuda_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error after " << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}