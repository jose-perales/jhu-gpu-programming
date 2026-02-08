// Modification of Ingemar Ragnemalm "Real Hello World!" program
// To compile execute below:
// nvcc hello-world.cu -L /usr/local/cuda/lib -lcudart -o hello-world
//
// Usage: ./hello-world [total_threads] [block_size]

#include <stdio.h>
#include <stdlib.h>

__global__ 
void hello(int * block)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	block[thread_idx] = threadIdx.x;
}

int main(int argc, char *argv[])
{
	/* Default values */
	int total_threads = 16;
	int block_size = 16;

	/* Parse command line arguments */
	if (argc >= 2) {
		total_threads = atoi(argv[1]);
	}
	if (argc >= 3) {
		block_size = atoi(argv[2]);
	}

	int num_blocks = total_threads / block_size;
	size_t array_size_bytes = sizeof(int) * total_threads;

	/* Allocate host memory */
	int *cpu_block = (int *)malloc(array_size_bytes);

	/* Declare pointers for GPU based params */
	int *gpu_block;

	cudaMalloc((void **)&gpu_block, array_size_bytes);
	cudaMemcpy(gpu_block, cpu_block, array_size_bytes, cudaMemcpyHostToDevice);

	/* Execute our kernel */
	hello<<<num_blocks, block_size>>>(gpu_block);

	/* Free the arrays on the GPU as now we're done with them */
	cudaMemcpy( cpu_block, gpu_block, array_size_bytes, cudaMemcpyDeviceToHost );
	cudaFree(gpu_block);

	/* Iterate through the arrays and print */
	for(int i = 0; i < total_threads; i++)
	{
		printf("Calculated Thread: - Block: %2u\n",cpu_block[i]);
	}

	free(cpu_block);
	return EXIT_SUCCESS;
}
