// CUDA Blocks Example
// Demonstrates 1D block/thread organization
// To compile: nvcc blocks.cu -L /usr/local/cuda/lib -lcudart -o blocks
//
// Usage: ./blocks [total_threads] [block_size]

#include <stdio.h>
#include <stdlib.h>

__global__
void what_is_my_id(unsigned int * block, unsigned int * thread)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	block[thread_idx] = blockIdx.x;
	thread[thread_idx] = threadIdx.x;
}

int main(int argc, char *argv[])
{
	/* Default values */
	int total_threads = 256;
	int block_size = 16;

	/* Parse command line arguments */
	if (argc >= 2) {
		total_threads = atoi(argv[1]);
	}
	if (argc >= 3) {
		block_size = atoi(argv[2]);
	}

	int num_blocks = total_threads / block_size;
	size_t array_size_bytes = sizeof(unsigned int) * total_threads;

	/* Allocate host memory */
	unsigned int *cpu_block = (unsigned int *)malloc(array_size_bytes);
	unsigned int *cpu_thread = (unsigned int *)malloc(array_size_bytes);

	/* Declare pointers for GPU based params */
	unsigned int *gpu_block;
	unsigned int *gpu_thread;

	cudaMalloc((void **)&gpu_block, array_size_bytes);
	cudaMalloc((void **)&gpu_thread, array_size_bytes);
	cudaMemcpy(gpu_block, cpu_block, array_size_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_thread, cpu_thread, array_size_bytes, cudaMemcpyHostToDevice);

	/* Execute our kernel */
	what_is_my_id<<<num_blocks, block_size>>>(gpu_block, gpu_thread);

	/* Copy results back to host */
	cudaMemcpy(cpu_block, gpu_block, array_size_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_thread, gpu_thread, array_size_bytes, cudaMemcpyDeviceToHost);

	/* Free GPU memory */
	cudaFree(gpu_block);
	cudaFree(gpu_thread);

	/* Print configuration */
	printf("Configuration: %d total threads, %d threads/block, %d blocks\n\n",
		total_threads, block_size, num_blocks);

	/* Iterate through the arrays and print */
	for(int i = 0; i < total_threads; i++)
	{
		printf("GlobalIdx: %3d - Thread: %2u - Block: %2u\n", i, cpu_thread[i], cpu_block[i]);
	}

	/* Free host memory */
	free(cpu_block);
	free(cpu_thread);

	return EXIT_SUCCESS;
}
