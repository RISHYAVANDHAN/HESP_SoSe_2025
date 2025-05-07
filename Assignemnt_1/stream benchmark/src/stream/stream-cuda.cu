#include <chrono>

#include "../util.h"
#include "stream-util.h"

// CUDA kernel
__global__ void stream(size_t nx, const double *__restrict__ src, double *__restrict__ dest) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nx)
        dest[i] = src[i] + 1;
}

int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);

    auto src = new double[nx];
    auto dest = new double[nx];

    // init
    initStream(src, nx);

    // Allocate device memory
    double *d_src, *d_dest;
    cudaMalloc(&d_src, nx * sizeof(double));
    cudaMalloc(&d_dest, nx * sizeof(double));
    cudaMemcpy(d_src, src, nx * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (nx + blockSize - 1) / blockSize;

    // warm-up
    for (int i = 0; i < nItWarmUp; ++i) {
        stream<<<numBlocks, blockSize>>>(nx, d_src, d_dest);
        cudaDeviceSynchronize();
        std::swap(d_src, d_dest);
    }

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < nIt; ++i) {
        stream<<<numBlocks, blockSize>>>(nx, d_src, d_dest);
        cudaDeviceSynchronize();
        std::swap(d_src, d_dest);
    }
    // measure time

    auto end = std::chrono::steady_clock::now();

    // Copy result back to host
    cudaMemcpy(src, d_src, nx * sizeof(double), cudaMemcpyDeviceToHost);

    printStats(end - start, nx, nIt, streamNumReads, streamNumWrites);

    // check solution
    checkSolutionStream(src, nx, nIt + nItWarmUp);

    delete[] src;
    delete[] dest;
    cudaFree(d_src);
    cudaFree(d_dest);

    return 0;
}