#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <chrono>

// ============================================================================
// File I/O
// ============================================================================

// Image names from preprocess.py
const char* IMAGE_NAMES[] = {
    "astronaut", "brick", "camera", "cat", "checkerboard", "chelsea", "clock",
    "coffee", "coins", "colorwheel", "grass", "gravel", "hubble_deep_field",
    "immunohistochemistry", "logo", "moon", "page", "retina", "rocket",
    "shepp_logan_phantom", "text"
};

const int NUM_IMAGES = 21;

// Load image from binary file.
// Binary format: [width (4B), height (4B), pixels (n bytes)]
int* loadImage(const char* filename, int* width, int* height) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        return NULL;
    }
    
    fread(width, sizeof(int), 1, f);
    fread(height, sizeof(int), 1, f);
    
    int n = (*width) * (*height);
    
    // Read pixel data
    unsigned char* bytes = (unsigned char*)malloc(n);
    fread(bytes, 1, n, f);
    fclose(f);
    
    // Convert pixels to int array
    int* pixels = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        pixels[i] = bytes[i];
    }
    free(bytes);
    
    return pixels;
}

// Save processed image to binary file.
// Binary format: [width (4B), height (4B), pixels (n bytes)]
void saveImage(const char* filename, const int* pixels, int width, int height) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        printf("Failed to save: %s\n", filename);
        return;
    }
    
    fwrite(&width, sizeof(int), 1, f);
    fwrite(&height, sizeof(int), 1, f);
    
    int n = width * height;
    unsigned char* bytes = (unsigned char*)malloc(n);
    for (int i = 0; i < n; i++) {
        bytes[i] = (unsigned char)pixels[i];
    }
    fwrite(bytes, 1, n, f);
    free(bytes);
    fclose(f);
}

// ============================================================================
// GPU Kernels
// ============================================================================

// Contrast adjustment threshold (fixed at midpoint of 8-bit range)
const int THRESHOLD = 128;

// GPU Kernel - With Branching
// Uses conditionals
__global__ void contrastGPU_branching(const int* in, int* out, int offset, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int sign;
        if (in[idx] > THRESHOLD) {
            sign = 1;
        } else {
            sign = -1;
        }
        int result = in[idx] + sign * offset;
        out[idx] = result < 0 ? 0 : (result > 255 ? 255 : result);
    }
}

// GPU Kernel - No Branching
// Uses bit manipulation to compute sign without branching
__global__ void contrastGPU_noBranching(const int* in, int* out, int offset, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Compute sign using bit manipulation
        // diff is negative if in[idx] <= 128, non-negative otherwise
        int diff = in[idx] - (THRESHOLD + 1);
        // Right-shift by 31 gives 0xFFFFFFFF (-1) for negative, 0 for non-negative
        // So sign = 1 + (-1 or 0) * 2 = -1 or 1
        int sign = 1 + (diff >> 31) * 2;
        int result = in[idx] + sign * offset;
        out[idx] = max(0, min(255, result));
    }
}

// ============================================================================
// CPU Functions
// ============================================================================

// CPU Function - With Branching
// Same algorithm as GPU branching kernel for fair comparison.
void contrastCPU_branching(const int* in, int* out, int offset, int n) {
    for (int i = 0; i < n; i++) {
        int sign;
        if (in[i] > THRESHOLD) {
            sign = 1;
        } else {
            sign = -1;
        }
        int result = in[i] + sign * offset;
        out[i] = result < 0 ? 0 : (result > 255 ? 255 : result);
    }
}

// CPU Function - No Branching
// Uses bit manipulation to compute sign without branching
void contrastCPU_noBranching(const int* in, int* out, int offset, int n) {
    for (int i = 0; i < n; i++) {
        // Compute sign using bit manipulation similar to GPU no-branching kernel
        int diff = in[i] - (THRESHOLD + 1);
        int sign = 1 + (diff >> 31) * 2;
        int result = in[i] + sign * offset;
        out[i] = std::max(0, std::min(255, result));
    }
}

int main(int argc, char** argv)
{
    // Default kernel configuration
    // Usage: ./assignment.exe <image_index> <block_size>
    // image_index: 0 = all images, 1-21 = specific image
    // block_size: threads per block (default 256)
    int imageIndex = 0;
    int blockSize = 256;
    int contrastOffset = 25;
    
    // Parse command line arguments
    if (argc >= 2) imageIndex = atoi(argv[1]);
    if (argc >= 3) blockSize = atoi(argv[2]);
    
    // Validate image index
    if (imageIndex < 0 || imageIndex > NUM_IMAGES) {
        printf("Error: image_index must be 0-%d (0=all)\n", NUM_IMAGES);
        return 1;
    }
    
    printf("Contrast Adjustment: blockSize=%d, offset=%d\n", blockSize, contrastOffset);
    
    // Open CSV file for performance results
    FILE* csvFile = fopen("performance.csv", "w");
    if (csvFile) {
        fprintf(csvFile, "image,pixels,threads,block_size,gpu_branch_ns,gpu_nobranch_ns,cpu_branch_ns,cpu_nobranch_ns\n");
    }
    
    // Determine which images to process
    int startIdx = (imageIndex == 0) ? 0 : imageIndex - 1;
    int endIdx = (imageIndex == 0) ? NUM_IMAGES : imageIndex;
    
    printf("Processing %s\n\n", imageIndex == 0 ? "all images" : IMAGE_NAMES[startIdx]);
    printf("%-25s %10s %12s %12s %12s %12s\n", 
           "Image", "Pixels", "GPU Branch", "GPU NoBranch", "CPU Branch", "CPU NoBranch");
    
    for (int i = startIdx; i < endIdx; i++) {
        char filepath[256];
        snprintf(filepath, sizeof(filepath), "images/%s.bin", IMAGE_NAMES[i]);
        
        int width, height;
        int* pixels = loadImage(filepath, &width, &height);
        
        if (!pixels) {
            printf("  %-25s FAILED TO LOAD\n", IMAGE_NAMES[i]);
            continue;
        }
        
        int n = width * height;
        
        // Allocate host output buffer for CPU functions
        int* outCPU = (int*)malloc(n * sizeof(int));
        
        // Allocate device memory
        int *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(int));
        cudaMalloc(&d_out, n * sizeof(int));
        
        // Copy input data to device
        cudaMemcpy(d_in, pixels, n * sizeof(int), cudaMemcpyHostToDevice);
        
        // Calculate grid size
        int gridSize = (n + blockSize - 1) / blockSize;
        int totalThreads = gridSize * blockSize;
        
        // Warmup kernel (not timed)
        contrastGPU_branching<<<gridSize, blockSize>>>(d_in, d_out, contrastOffset, n);
        cudaDeviceSynchronize();
        
        // Time GPU branching kernel
        auto start = std::chrono::high_resolution_clock::now();
        contrastGPU_branching<<<gridSize, blockSize>>>(d_in, d_out, contrastOffset, n);
        cudaDeviceSynchronize();
        auto stop = std::chrono::high_resolution_clock::now();
        long long gpu_branch_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        
        // Time GPU non-branching kernel
        start = std::chrono::high_resolution_clock::now();
        contrastGPU_noBranching<<<gridSize, blockSize>>>(d_in, d_out, contrastOffset, n);
        cudaDeviceSynchronize();
        stop = std::chrono::high_resolution_clock::now();
        long long gpu_nobranch_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        
        // Time CPU branching function
        start = std::chrono::high_resolution_clock::now();
        contrastCPU_branching(pixels, outCPU, contrastOffset, n);
        stop = std::chrono::high_resolution_clock::now();
        long long cpu_branch_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        
        // Time CPU non-branching function
        start = std::chrono::high_resolution_clock::now();
        contrastCPU_noBranching(pixels, outCPU, contrastOffset, n);
        stop = std::chrono::high_resolution_clock::now();
        long long cpu_nobranch_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        
        // Copy GPU result back and save output image
        int* outGPU = (int*)malloc(n * sizeof(int));
        cudaMemcpy(outGPU, d_out, n * sizeof(int), cudaMemcpyDeviceToHost);
        
        char outpath[256];
        snprintf(outpath, sizeof(outpath), "images/%s_output.bin", IMAGE_NAMES[i]);
        saveImage(outpath, outGPU, width, height);
        free(outGPU);
        
        // Print results
        printf("%-25s %10d %10lld us %10lld us %10lld us %10lld us\n", 
               IMAGE_NAMES[i], n, 
               gpu_branch_ns/1000, gpu_nobranch_ns/1000,
               cpu_branch_ns/1000, cpu_nobranch_ns/1000);
        
        // Write to CSV
        if (csvFile) {
            fprintf(csvFile, "%s,%d,%d,%d,%lld,%lld,%lld,%lld\n",
                    IMAGE_NAMES[i], n, totalThreads, blockSize,
                    gpu_branch_ns, gpu_nobranch_ns, cpu_branch_ns, cpu_nobranch_ns);
        }
        cudaFree(d_in);
        cudaFree(d_out);
        free(pixels);
        free(outCPU);
    }
    
    // Close CSV file
    if (csvFile) {
        fclose(csvFile);
        printf("\nSaved: performance.csv\n");
    }
    
    printf("\nDone.\n");
    return 0;
}
