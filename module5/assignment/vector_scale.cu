// vector_scale.cu — Five CUDA memory types, one vector scale operation

#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#define SCALE 2.5f
#define REPEAT 100

// ============================================================================
// Constant memory
// ============================================================================

__constant__ float d_scale;

// ============================================================================
// Kernels
// ============================================================================

// Host memory — CPU-only baseline, all data in malloc'd memory
void host_scale(float *out, const float *in, int n) {
    for (int i = 0; i < n; i++) {
        float tmp = in[i];
        for (int r = 0; r < REPEAT; r++) { tmp *= SCALE; }
        out[i] = tmp;
    }
}

// Global memory — read/write directly from cudaMalloc'd arrays
__global__ void global_scale(float *out, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float tmp = in[i];
        for (int r = 0; r < REPEAT; r++) { tmp *= SCALE; }
        out[i] = tmp;
    }
}

// Register memory — load into local variable, operate, store back
// Compile with nvcc -Xptxas -v to verify register allocation
__global__ void register_scale(float *out, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float tmp = in[i];
        for (int r = 0; r < REPEAT; r++) { tmp *= SCALE; }
        out[i] = tmp;
    }
}

// Constant memory — scale factor broadcast via __constant__ cache
__global__ void constant_scale(float *out, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float tmp = in[i];
        for (int r = 0; r < REPEAT; r++) { tmp *= d_scale; }
        out[i] = tmp;
    }
}

// Shared memory — tile input into __shared__, sync, then scale
__global__ void shared_scale(float *out, const float *in, int n) {
    extern __shared__ float tile[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;

    if (i < n) { tile[lid] = in[i]; }
    __syncthreads();
    if (i < n) {
        float tmp = tile[lid];
        for (int r = 0; r < REPEAT; r++) { tmp *= SCALE; }
        out[i] = tmp;
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
    int totalThreads = (1 << 20);
    int blockSize = 256;
    if (argc >= 2) { totalThreads = atoi(argv[1]); }
    if (argc >= 3) { blockSize = atoi(argv[2]); }

    int numBlocks = totalThreads / blockSize;
    if (totalThreads % blockSize != 0) {
        ++numBlocks;
        totalThreads = numBlocks * blockSize;
        printf("Warning: rounded up to %d threads\n", totalThreads);
    }

    int n = totalThreads;
    size_t bytes = n * sizeof(float);

    printf("Vector Scale: n=%d, %d blocks x %d threads, scale=%.1f\n\n",
           n, numBlocks, blockSize, SCALE);

    // Allocate host memory
    float *h_in  = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    for (int i = 0; i < n; i++) { h_in[i] = (float)(i % 1000) * 0.001f; }

    // Allocate device memory
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // Copy scale factor to constant memory
    float h_scale = SCALE;
    cudaMemcpyToSymbol(d_scale, &h_scale, sizeof(float));

    size_t smem = blockSize * sizeof(float);

    // Warmup
    global_scale<<<numBlocks, blockSize>>>(d_out, d_in, n);
    cudaDeviceSynchronize();

    printf("%-20s %12s\n", "Memory Type", "Time (us)");
    printf("%-20s %12s\n", "--------------------", "------------");

    // Host memory
    auto t0 = std::chrono::high_resolution_clock::now();
    host_scale(h_out, h_in, n);
    auto t1 = std::chrono::high_resolution_clock::now();
    long long host_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    printf("%-20s %12lld\n", "Host (CPU)", host_ns / 1000);

    // Global memory
    t0 = std::chrono::high_resolution_clock::now();
    global_scale<<<numBlocks, blockSize>>>(d_out, d_in, n);
    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
    long long global_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    printf("%-20s %12lld\n", "Global", global_ns / 1000);

    // Register memory
    t0 = std::chrono::high_resolution_clock::now();
    register_scale<<<numBlocks, blockSize>>>(d_out, d_in, n);
    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
    long long reg_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    printf("%-20s %12lld\n", "Register", reg_ns / 1000);

    // Constant memory
    t0 = std::chrono::high_resolution_clock::now();
    constant_scale<<<numBlocks, blockSize>>>(d_out, d_in, n);
    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
    long long const_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    printf("%-20s %12lld\n", "Constant", const_ns / 1000);

    // Shared memory
    t0 = std::chrono::high_resolution_clock::now();
    shared_scale<<<numBlocks, blockSize, smem>>>(d_out, d_in, n);
    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
    long long shared_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    printf("%-20s %12lld\n", "Shared", shared_ns / 1000);

    printf("\n");

    // Append to CSV if requested via argv[3]
    if (argc >= 4) {
        FILE *csv = fopen(argv[3], "a");
        if (csv) {
            fprintf(csv, "%d,%d,%lld,%lld,%lld,%lld,%lld\n",
                    n, blockSize, host_ns, global_ns,
                    reg_ns, const_ns, shared_ns);
            fclose(csv);
        }
    }

    // Cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    return 0;
}
