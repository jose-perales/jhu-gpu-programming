/*
 * Global Memory: Interleaved (AoS) vs Non-interleaved (SoA)
 * Compares coalesced vs non-coalesced access patterns on GPU and CPU.
 */
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#define NUM_ELEMENTS 4096

typedef struct {
	unsigned int a;
	unsigned int b;
	unsigned int c;
	unsigned int d;
} INTERLEAVED_T;

// CPU functions

void add_interleaved_cpu(INTERLEAVED_T *dst, const INTERLEAVED_T *src,
                         unsigned int iter, unsigned int n) {
	for (unsigned int tid = 0; tid < n; tid++)
		for (unsigned int i = 0; i < iter; i++) {
			dst[tid].a += src[tid].a;
			dst[tid].b += src[tid].b;
			dst[tid].c += src[tid].c;
			dst[tid].d += src[tid].d;
		}
}

void add_non_interleaved_cpu(unsigned int *da, unsigned int *db, unsigned int *dc, unsigned int *dd,
                             const unsigned int *sa, const unsigned int *sb,
                             const unsigned int *sc, const unsigned int *sd,
                             unsigned int iter, unsigned int n) {
	for (unsigned int tid = 0; tid < n; tid++)
		for (unsigned int i = 0; i < iter; i++) {
			da[tid] += sa[tid];
			db[tid] += sb[tid];
			dc[tid] += sc[tid];
			dd[tid] += sd[tid];
		}
}

// GPU kernels

__global__ void add_interleaved_gpu(INTERLEAVED_T *dst, const INTERLEAVED_T *src,
                                    unsigned int iter, unsigned int n) {
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n)
		for (unsigned int i = 0; i < iter; i++) {
			dst[tid].a += src[tid].a;
			dst[tid].b += src[tid].b;
			dst[tid].c += src[tid].c;
			dst[tid].d += src[tid].d;
		}
}

__global__ void add_non_interleaved_gpu(unsigned int *da, unsigned int *db,
                                        unsigned int *dc, unsigned int *dd,
                                        const unsigned int *sa, const unsigned int *sb,
                                        const unsigned int *sc, const unsigned int *sd,
                                        unsigned int iter, unsigned int n) {
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n)
		for (unsigned int i = 0; i < iter; i++) {
			da[tid] += sa[tid];
			db[tid] += sb[tid];
			dc[tid] += sc[tid];
			dd[tid] += sd[tid];
		}
}

int main(int argc, char *argv[])
{
	unsigned int N = (argc > 1) ? atoi(argv[1]) : NUM_ELEMENTS;
	unsigned int iter = 4;
	unsigned int threads = 256;
	unsigned int blocks = (N + threads - 1) / threads;

	// Allocate host data
	size_t aos_bytes = N * sizeof(INTERLEAVED_T);
	size_t arr_bytes = N * sizeof(unsigned int);

	INTERLEAVED_T *h_il_src = (INTERLEAVED_T *)calloc(N, sizeof(INTERLEAVED_T));
	INTERLEAVED_T *h_il_dst = (INTERLEAVED_T *)calloc(N, sizeof(INTERLEAVED_T));
	unsigned int *h_a = (unsigned int *)calloc(N, sizeof(unsigned int));
	unsigned int *h_b = (unsigned int *)calloc(N, sizeof(unsigned int));
	unsigned int *h_c = (unsigned int *)calloc(N, sizeof(unsigned int));
	unsigned int *h_d = (unsigned int *)calloc(N, sizeof(unsigned int));
	unsigned int *h_da = (unsigned int *)calloc(N, sizeof(unsigned int));
	unsigned int *h_db = (unsigned int *)calloc(N, sizeof(unsigned int));
	unsigned int *h_dc = (unsigned int *)calloc(N, sizeof(unsigned int));
	unsigned int *h_dd = (unsigned int *)calloc(N, sizeof(unsigned int));

	for (unsigned int i = 0; i < N; i++) {
		h_il_src[i].a = h_a[i] = 1;
		h_il_src[i].b = h_b[i] = 2;
		h_il_src[i].c = h_c[i] = 3;
		h_il_src[i].d = h_d[i] = 4;
	}

	// GPU allocations
	INTERLEAVED_T *d_il_src, *d_il_dst;
	cudaMalloc(&d_il_src, aos_bytes);
	cudaMalloc(&d_il_dst, aos_bytes);

	unsigned int *d_a, *d_b, *d_c, *d_d, *d_da, *d_db, *d_dc, *d_dd;
	cudaMalloc(&d_a, arr_bytes); cudaMalloc(&d_b, arr_bytes);
	cudaMalloc(&d_c, arr_bytes); cudaMalloc(&d_d, arr_bytes);
	cudaMalloc(&d_da, arr_bytes); cudaMalloc(&d_db, arr_bytes);
	cudaMalloc(&d_dc, arr_bytes); cudaMalloc(&d_dd, arr_bytes);

	cudaMemcpy(d_il_src, h_il_src, aos_bytes, cudaMemcpyHostToDevice);
	cudaMemset(d_il_dst, 0, aos_bytes);
	cudaMemcpy(d_a, h_a, arr_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, arr_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, h_c, arr_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_d, h_d, arr_bytes, cudaMemcpyHostToDevice);
	cudaMemset(d_da, 0, arr_bytes); cudaMemset(d_db, 0, arr_bytes);
	cudaMemset(d_dc, 0, arr_bytes); cudaMemset(d_dd, 0, arr_bytes);

	// Warmup
	add_interleaved_gpu<<<blocks, threads>>>(d_il_dst, d_il_src, 1, N);
	cudaDeviceSynchronize();
	cudaMemset(d_il_dst, 0, aos_bytes);

	// Time CPU interleaved
	auto start = std::chrono::high_resolution_clock::now();
	add_interleaved_cpu(h_il_dst, h_il_src, iter, N);
	auto stop = std::chrono::high_resolution_clock::now();
	long long cpu_il_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

	// Time CPU non-interleaved
	start = std::chrono::high_resolution_clock::now();
	add_non_interleaved_cpu(h_da, h_db, h_dc, h_dd, h_a, h_b, h_c, h_d, iter, N);
	stop = std::chrono::high_resolution_clock::now();
	long long cpu_ni_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

	// Time GPU interleaved
	start = std::chrono::high_resolution_clock::now();
	add_interleaved_gpu<<<blocks, threads>>>(d_il_dst, d_il_src, iter, N);
	cudaDeviceSynchronize();
	stop = std::chrono::high_resolution_clock::now();
	long long gpu_il_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

	// Time GPU non-interleaved
	start = std::chrono::high_resolution_clock::now();
	add_non_interleaved_gpu<<<blocks, threads>>>(d_da, d_db, d_dc, d_dd, d_a, d_b, d_c, d_d, iter, N);
	cudaDeviceSynchronize();
	stop = std::chrono::high_resolution_clock::now();
	long long gpu_ni_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

	// CSV: N,cpu_il_ns,cpu_ni_ns,gpu_il_ns,gpu_ni_ns
	printf("%u,%lld,%lld,%lld,%lld\n", N, cpu_il_ns, cpu_ni_ns, gpu_il_ns, gpu_ni_ns);

	// Cleanup
	free(h_il_src); free(h_il_dst);
	free(h_a); free(h_b); free(h_c); free(h_d);
	free(h_da); free(h_db); free(h_dc); free(h_dd);
	cudaFree(d_il_src); cudaFree(d_il_dst);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_d);
	cudaFree(d_da); cudaFree(d_db); cudaFree(d_dc); cudaFree(d_dd);
	return 0;
}
