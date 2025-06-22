#include <fstream>
#include <iomanip>
#include <filesystem>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include "particle.cuh"
#include "vtk_writer.cuh"

__host__ void write_vtk(const Particle* particles, int num_particles, int step, const std::string& output_dir) {
    // Create output directory if it doesn't exist
    struct stat st = {0};
    if (stat(output_dir.c_str(), &st) == -1) {
        mkdir(output_dir.c_str(), 0755);
    }
    
    std::ostringstream filename;
    filename << output_dir << "/particles_" 
             << std::setw(4) << std::setfill('0') << step << ".vtp";  // Use .vtp for better ParaView compatibility
    
    std::ofstream vtk_file(filename.str());
    if (!vtk_file.is_open()) {
        std::cerr << "Error: Failed to create VTK file: " << filename.str() << "\n";
        return;
    }

    // VTK PolyData header (XML format)
    vtk_file << "<?xml version=\"1.0\"?>\n";
    vtk_file << "<VTKFile type=\"PolyData\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    vtk_file << "  <PolyData>\n";
    vtk_file << "    <Piece NumberOfPoints=\"" << num_particles << "\" NumberOfVerts=\"0\" NumberOfLines=\"0\" "
             << "NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n";
    
    // Point data (positions)
    vtk_file << "      <Points>\n";
    vtk_file << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int i = 0; i < num_particles; ++i) {
        const auto& p = particles[i];
        vtk_file << p.position.x << " " << p.position.y << " " << p.position.z << "\n";
    }
    vtk_file << "        </DataArray>\n";
    vtk_file << "      </Points>\n";
    
    // Point data attributes
    vtk_file << "      <PointData>\n";
    
    // Particle radius (essential for sphere visualization)
    vtk_file << "        <DataArray type=\"Float32\" Name=\"radius\" format=\"ascii\">\n";
    for (int i = 0; i < num_particles; ++i) {
        vtk_file << particles[i].radius << "\n";
    }
    vtk_file << "        </DataArray>\n";
    
    // Velocity vectors
    vtk_file << "        <DataArray type=\"Float32\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int i = 0; i < num_particles; ++i) {
        const auto& v = particles[i].velocity;
        vtk_file << v.x << " " << v.y << " " << v.z << "\n";
    }
    vtk_file << "        </DataArray>\n";
    
    // Force vectors
    vtk_file << "        <DataArray type=\"Float32\" Name=\"force\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int i = 0; i < num_particles; ++i) {
        const auto& f = particles[i].force;
        vtk_file << f.x << " " << f.y << " " << f.z << "\n";
    }
    vtk_file << "        </DataArray>\n";
    
    // Mass
    vtk_file << "        <DataArray type=\"Float32\" Name=\"mass\" format=\"ascii\">\n";
    for (int i = 0; i < num_particles; ++i) {
        vtk_file << particles[i].mass << "\n";
    }
    vtk_file << "        </DataArray>\n";
    
    // Additional useful data
    vtk_file << "        <DataArray type=\"Int32\" Name=\"particle_id\" format=\"ascii\">\n";
    for (int i = 0; i < num_particles; ++i) {
        vtk_file << i << "\n";
    }
    vtk_file << "        </DataArray>\n";
    
    vtk_file << "      </PointData>\n";
    
    // Cell data (empty but required structure)
    vtk_file << "      <Verts></Verts>\n";
    vtk_file << "      <Lines></Lines>\n";
    vtk_file << "      <Strips></Strips>\n";
    vtk_file << "      <Polys></Polys>\n";
    
    // Footer
    vtk_file << "    </Piece>\n";
    vtk_file << "  </PolyData>\n";
    vtk_file << "</VTKFile>\n";
    
    vtk_file.close();
    std::cout << "Wrote VTK file: " << filename.str() << "\n";
}

__host__ void write_paraview_script(const std::string& output_dir, int num_particles) {
    std::string script_path = output_dir + "/visualize.py";
    std::ofstream script_file(script_path);
    
    if (!script_file.is_open()) {
        std::cerr << "Failed to create ParaView script: " << script_path << "\n";
        return;
    }

    script_file << R"(
from paraview.simple import *
import glob
import os

# Find all particle files
files = sorted(glob.glob(")" << output_dir << R"(/particles_*.vtp"))
if not files:
    print("Error: No VTK files found in ", ")" << output_dir << R"(")
    exit(1)

# Create pipeline
particles = XMLPolyDataReader(FileName=files)
particles.PointArrayStatus = ['radius', 'velocity', 'force', 'mass', 'particle_id']

# Create spherical glyphs
sphere_glyph = Glyph(Input=particles)
sphere_glyph.GlyphType = 'Sphere'
sphere_glyph.ScaleArray = ['POINTS', 'radius']
sphere_glyph.ScaleFactor = 1.0

# Show and color by velocity magnitude
display = Show(sphere_glyph)
ColorBy(display, ('POINTS', 'velocity', 'Magnitude'))

# Create scalar bar (modern API)
view = GetActiveViewOrCreate('RenderView')
scalar_bar = CreateScalarBar(Title='Velocity Magnitude', 
                             TitleFontSize=12,
                             LabelFontSize=12,
                             Position=[0.9, 0.05])
display.SetScalarBarVisibility(view, True)  # Correct 2-argument version

# Set view properties
view.Background = [1, 1, 1]  # White background
view.ResetCamera()

# Save screenshot
output_image = os.path.join(")" << output_dir << R"(", "screenshot.png")
SaveScreenshot(output_image, view, ImageResolution=[1920, 1080])

print(f"Saved screenshot to {output_image}")
print("Close ParaView window to exit...")

Render()
)";

    script_file.close();
    std::cout << "Wrote ParaView script: " << script_path << "\n";
}