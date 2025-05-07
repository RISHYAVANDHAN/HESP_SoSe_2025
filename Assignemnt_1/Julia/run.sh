#!/bin/bash
set -e

echo "========= Julia Set Generator ========="

# Create output directory if it doesn't exist
mkdir -p output

# === CONFIG ===
CXX=g++
NVCC=nvcc
OUTPUT_NAME=julia_simple
CUDA_PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/2025/cuda"
CUDA_LIB="$CUDA_PATH/lib64"
CUDA_INCLUDE="$CUDA_PATH/include"

# For RTX 4050, the compute capability is 8.9 (Ada Lovelace architecture)
NVCC_FLAGS="--std=c++17 -arch=sm_89 -Xcompiler=-std=c++20,-fPIC -Wno-deprecated-gpu-targets -I$CUDA_INCLUDE"

# === BUILD ===
echo "[1/3] Compiling lodepng.cpp..."
$CXX -std=c++20 -O2 -c lodepng.cpp -o lodepng.o

echo "[2/3] Compiling julia.cu..."
$NVCC $NVCC_FLAGS -c julia.cu -o julia.o

echo "[3/3] Linking..."
$CXX -o $OUTPUT_NAME julia.o lodepng.o -L/opt/nvidia/hpc_sdk/Linux_x86_64/2025/cuda/lib64 -lcudart

# === RUN ===
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/2025/cuda/lib64:$LD_LIBRARY_PATH

# Define Julia set configurations
configs=(
    "-0.8 0.2 output/julia_classic.png"
    "0.285 0.01 output/julia_spiral1.png"
    "-0.4 0.6 output/julia_swirl.png"
    "-0.70176 -0.3842 output/julia_dense.png"
    "0.355 0.355 output/julia_eye.png"
    "-0.75 0.11 output/julia_nebula.png"
)

# Process each configuration
echo "========= Processing Configurations ========="
for config in "${configs[@]}"; do
    echo "============================================"
    ./$OUTPUT_NAME $config
    echo "============================================"
    echo ""
done

echo "Done! Check your output directory for the generated images."