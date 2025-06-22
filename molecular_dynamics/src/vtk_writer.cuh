// vtk_writer.cuh
#ifndef VTK_WRITER_CUH
#define VTK_WRITER_CUH

#include "particle.cuh"

void write_vtk(const Particle* particles, int num_particles, int step, const std::string& output_dir);
void write_paraview_script(const std::string& output_dir, int num_particles);

#endif