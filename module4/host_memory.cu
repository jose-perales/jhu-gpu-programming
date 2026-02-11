#include <stdio.h>
#include <stdlib.h>
#include <chrono>

// From https://devblogs.nvidia.com/parallelforall/easy-introduction-cuda-c-and-c/

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(int argc, char *argv[])
{
  int N = (argc > 1) ? atoi(argv[1]) : 1<<20;

  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  cudaMalloc(&d_x, N*sizeof(float));
  cudaMalloc(&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Warmup (not timed)
  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  // Time H→D
  auto start = std::chrono::high_resolution_clock::now();
  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  auto stop = std::chrono::high_resolution_clock::now();
  long long h2d_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
  // Perform SAXPY on 1M elements
  // Time kernel
  start = std::chrono::high_resolution_clock::now();
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
  cudaDeviceSynchronize();
  stop = std::chrono::high_resolution_clock::now();
  long long kernel_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

  // Time D→H
  start = std::chrono::high_resolution_clock::now();
  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  stop = std::chrono::high_resolution_clock::now();
  long long d2h_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

  // Verify
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i] - 4.0f));

  // Print CSV: N,kernel_ns,h2d_ns,d2h_ns,max_error
  printf("%d,%lld,%lld,%lld,%f\n", N, kernel_ns, h2d_ns, d2h_ns, maxError);

  // Cleanup
  cudaFree(d_x); cudaFree(d_y);
  free(x); free(y);
  return 0;
}
