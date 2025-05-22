#!/bin/bash
set -e

echo "========= Molecular Dynamics Simulation ========="

# Build the project
make

# Create output directory if it doesn't exist
mkdir -p output

# Simulation parameters
INPUT_FILE="src/input/particles_test1.txt"
DT=0.001
NSTEPS=1000
SIGMA=1.0
EPSILON=1.0

# Run the simulation
./build/md_sim $INPUT_FILE $DT $NSTEPS $SIGMA $EPSILON

echo "Done! Check the output directory for results."