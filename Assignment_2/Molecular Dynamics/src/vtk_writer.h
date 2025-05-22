#pragma once
#include <vector>
#include <string>
#include "io.h"

bool write_vtk(const std::string& filename, const std::vector<Particle>& particles, int step);