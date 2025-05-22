#pragma once
#include <vector>
#include <string>
#include <cuda_runtime.h>

struct Particle {
    float3 pos;
    float3 vel;
    float3 acc;
    float mass;
};

bool read_particles(const std::string& filename, std::vector<Particle>& particles);
