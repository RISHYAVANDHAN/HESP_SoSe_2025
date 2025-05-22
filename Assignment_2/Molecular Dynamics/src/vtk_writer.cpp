#include "vtk_writer.h"
#include <fstream>
#include <iomanip>

bool write_vtk(const std::string& filename, const std::vector<Particle>& particles, int step) {
    std::ofstream ofs(filename);
    if (!ofs) return false;

    ofs << "# vtk DataFile Version 4.0\n";
    ofs << "hesp visualization file\n";
    ofs << "ASCII\n";
    ofs << "DATASET UNSTRUCTURED_GRID\n";
    ofs << "POINTS " << particles.size() << " double\n";
    for (const auto& p : particles) {
        ofs << std::fixed << std::setprecision(6)
            << p.pos.x << " " << p.pos.y << " " << p.pos.z << "\n";
    }
    ofs << "CELLS 0 0\n";
    ofs << "CELL_TYPES 0\n";
    ofs << "POINT_DATA " << particles.size() << "\n";
    ofs << "SCALARS m double\n";
    ofs << "LOOKUP_TABLE default\n";
    for (const auto& p : particles) {
        ofs << p.mass << "\n";
    }
    ofs << "VECTORS v double\n";
    for (const auto& p : particles) {
        ofs << p.vel.x << " " << p.vel.y << " " << p.vel.z << "\n";
    }
    return true;
}