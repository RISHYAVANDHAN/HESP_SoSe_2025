#pragma once
#include "io.h"

void launch_compute_forces(Particle* d_particles, int N, float sigma, float epsilon);
void launch_integrate(Particle* d_particles, int N, float dt);