import os
import argparse
import math
import numpy as np

def generate_particles(count, box_size, radius=0.5):
    """
    Generate non-overlapping particles in a 3D grid with proper spacing
    box_size: [width, height, depth] of simulation domain
    radius: particle radius
    """
    # Calculate grid dimensions
    spacing = 1.75 * radius
    grid_size = math.ceil(count ** (1/3))
    
    # Calculate starting positions to center grid in box
    start_x = (box_size[0] - (grid_size-1)*spacing) / 2
    start_y = (box_size[1] - (grid_size-1)*spacing) / 2
    start_z = (box_size[2] - (grid_size-1)*spacing) / 2

    particles = []
    generated = 0
    
    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                if generated >= count:
                    return particles
                
                px = start_x + x * spacing
                py = start_y + y * spacing
                pz = start_z + z * spacing
                
                # Ensure particles stay within bounds
                px = max(radius, min(px, box_size[0] - radius))
                py = max(radius, min(py, box_size[1] - radius))
                pz = max(radius, min(pz, box_size[2] - radius))
                
                particles.append(f"{px:.5f} {py:.5f} {pz:.5f} 0.0 0.0 0.0 1.0")
                generated += 1
    return particles

def generate_collision_clusters(count_per_cluster, box_size, radius=0.5):
    """
    Generate two separated clusters with safe boundary distances
    """
    spacing = 2.0 * radius * 1.01
    cluster_size = math.ceil(count_per_cluster ** (1/3)) * spacing
    
    # Calculate safe positions
    padding = 2 * radius
    cluster1_pos = [padding, box_size[1]/2, box_size[2]/2]
    cluster2_pos = [box_size[0] - padding - cluster_size, box_size[1]/2, box_size[2]/2]
    
    particles = []
    
    # Generate cluster 1
    start_offset = [
        cluster1_pos[0],
        cluster1_pos[1] - cluster_size/2,
        cluster1_pos[2] - cluster_size/2
    ]
    for x in range(int(math.ceil(count_per_cluster ** (1/3)))):
        for y in range(int(math.ceil(count_per_cluster ** (1/3)))):
            for z in range(int(math.ceil(count_per_cluster ** (1/3)))):
                if len(particles) >= count_per_cluster:
                    break
                px = start_offset[0] + x * spacing
                py = start_offset[1] + y * spacing
                pz = start_offset[2] + z * spacing
                particles.append(f"{px:.5f} {py:.5f} {pz:.5f} 0.0 0.0 0.0 1.0")
    
    # Generate cluster 2
    start_offset = [
        cluster2_pos[0],
        cluster2_pos[1] - cluster_size/2,
        cluster2_pos[2] - cluster_size/2
    ]
    for x in range(int(math.ceil(count_per_cluster ** (1/3)))):
        for y in range(int(math.ceil(count_per_cluster ** (1/3)))):
            for z in range(int(math.ceil(count_per_cluster ** (1/3)))):
                if len(particles) >= 2 * count_per_cluster:
                    break
                px = start_offset[0] + x * spacing
                py = start_offset[1] + y * spacing
                pz = start_offset[2] + z * spacing
                particles.append(f"{px:.5f} {py:.5f} {pz:.5f} 0.0 0.0 0.0 1.0")
    
    return particles

def generate_repulsive_shell(points_on_shell, radius=3.0, particle_radius=0.5):
    """
    Generate particles on a sphere surface with minimum arc distance
    """
    # Calculate minimum angular separation
    min_angle = 2 * math.asin(particle_radius / radius)
    
    particles = []
    # Fibonacci sphere algorithm
    phi = math.pi * (3.0 - math.sqrt(5.0))  # Golden angle
    for i in range(points_on_shell):
        y = 1 - (i / float(points_on_shell - 1)) * 2  # y from 1 to -1
        radius_y = math.sqrt(1 - y*y)  # radius at y
        theta = phi * i
        x = math.cos(theta) * radius_y
        z = math.sin(theta) * radius_y
        particles.append(f"{x*radius:.5f} {y*radius:.5f} {z*radius:.5f} 0.0 0.0 0.0 1.0")
    return particles

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, help="Generate cubic lattice particles")
    parser.add_argument("--case", type=str, choices=["collision_clusters", "repulsive_shell"],
                        help="Generate special test case")
    parser.add_argument("--output", default="particles")
    parser.add_argument("--box_size", nargs=3, type=float, default=[10.0, 10.0, 10.0],
                        help="Simulation box dimensions (x y z)")
    parser.add_argument("--radius", type=float, default=0.5, help="Particle radius")
    parser.add_argument("--count_per_cluster", type=int, default=100,
                        help="Particles per cluster for collision_clusters")
    parser.add_argument("--points_on_shell", type=int, default=200,
                        help="Points on shell for repulsive_shell")
    parser.add_argument("--shell_radius", type=float, default=3.0,
                        help="Radius of shell (repulsive_shell)")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    box_size = args.box_size

    if args.case == "collision_clusters":
        particles = generate_collision_clusters(
            count_per_cluster=args.count_per_cluster,
            box_size=box_size,
            radius=args.radius
        )
        filename = f"collision_clusters_{len(particles)}.txt"
        
    elif args.case == "repulsive_shell":
        particles = generate_repulsive_shell(
            points_on_shell=args.points_on_shell,
            radius=args.shell_radius,
            particle_radius=args.radius
        )
        filename = f"repulsive_shell_{len(particles)}.txt"
        
    else:  # Default cubic lattice
        if not args.count:
            raise ValueError("Must specify --count for cubic lattice")
        particles = generate_particles(
            count=args.count,
            box_size=box_size,
            radius=args.radius
        )
        filename = f"particles_{args.count}.txt"

    filepath = os.path.join(args.output, filename)
    with open(filepath, "w") as f:
        f.write("# x y z    vx vy vz    mass\n")
        f.write("\n".join(particles))
    print(f"Generated {len(particles)} particles at {filepath}")