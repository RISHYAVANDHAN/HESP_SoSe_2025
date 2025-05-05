#include <chrono>
#include <iostream>
#include <cuda_runtime.h>
#include "../util.h"
#include "stream-util.h"

// CUDA kernel for the stream operation
__global__ void streamKernel(size_t nx, const double *src, double *dest) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nx) {
        dest[i] = src[i] + 1;
    }
}

inline void streamCUDA(size_t nx, const double *src, double *dest, int blockSize) {
    double *d_src, *d_dest;

    // Allocate memory on the device
    cudaMalloc(&d_src, nx * sizeof(double));
    cudaMalloc(&d_dest, nx * sizeof(double));

    // Copy data from host to device
    cudaMemcpy(d_src, src, nx * sizeof(double), cudaMemcpyHostToDevice);

    // Launch the kernel
    int gridSize = (nx + blockSize - 1) / blockSize; // Calculate grid size
    streamKernel<<<gridSize, blockSize>>>(nx, d_src, d_dest);

    // Copy results back to host
    cudaMemcpy(dest, d_dest, nx * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_src);
    cudaFree(d_dest);
}

int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);

    auto src = new double[nx];
    auto dest = new double[nx];

    // Initialize the source array
    initStream(src, nx);

    // Warm-up
    for (int i = 0; i < nItWarmUp; ++i) {
        streamCUDA(nx, src, dest, 256); // Use a block size of 256
        std::swap(src, dest);
    }

    // Measurement
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < nIt; ++i) {
        streamCUDA(nx, src, dest, 256); // Use a block size of 256
        std::swap(src, dest);
    }

    auto end = std::chrono::steady_clock::now();

    // Print performance statistics
    printStats(end - start, nx, nIt, streamNumReads, streamNumWrites);

    // Check the solution
    checkSolutionStream(src, nx, nIt + nItWarmUp);

    delete[] src;
    delete[] dest;

    return 0;
}