// CUDA 2D Grids Example
// To compile: nvcc grids.cu -L /usr/local/cuda/lib -lcudart -o grids
//
// Usage: ./grids [grid_x] [grid_y] [block_x] [block_y]

#include <stdio.h>
#include <stdlib.h>

__global__ void what_is_my_id_2d_A(
				unsigned int * const block_x,
				unsigned int * const block_y,
				unsigned int * const thread,
				unsigned int * const calc_thread,
				unsigned int * const x_thread,
				unsigned int * const y_thread,
				unsigned int * const grid_dimx,
				unsigned int * const block_dimx,
				unsigned int * const grid_dimy,
				unsigned int * const block_dimy)
{
	const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
	const unsigned int thread_idx = ((gridDim.x * blockDim.x) * idy) + idx;

	block_x[thread_idx] = blockIdx.x;
	block_y[thread_idx] = blockIdx.y;
	thread[thread_idx] = threadIdx.x;
	calc_thread[thread_idx] = thread_idx;
	x_thread[thread_idx] = idx;
	y_thread[thread_idx] = idy;
	grid_dimx[thread_idx] = gridDim.x;
	block_dimx[thread_idx] = blockDim.x;
	grid_dimy[thread_idx] = gridDim.y;
	block_dimy[thread_idx] = blockDim.y;
}

int main(int argc, char *argv[])
{
	/* Default values: 1x4 grid, 32x4 block = 512 threads */
	int grid_x = 1, grid_y = 4;
	int block_x = 32, block_y = 4;

	/* Parse command line arguments */
	if (argc >= 2) grid_x = atoi(argv[1]);
	if (argc >= 3) grid_y = atoi(argv[2]);
	if (argc >= 4) block_x = atoi(argv[3]);
	if (argc >= 5) block_y = atoi(argv[4]);

	const dim3 threads(block_x, block_y);
	const dim3 blocks(grid_x, grid_y);

	int array_size_x = grid_x * block_x;
	int array_size_y = grid_y * block_y;
	int total_threads = array_size_x * array_size_y;
	size_t array_size_bytes = total_threads * sizeof(unsigned int);

	/* Allocate host arrays */
	unsigned int *cpu_block_x = (unsigned int *)malloc(array_size_bytes);
	unsigned int *cpu_block_y = (unsigned int *)malloc(array_size_bytes);
	unsigned int *cpu_thread = (unsigned int *)malloc(array_size_bytes);
	unsigned int *cpu_calc_thread = (unsigned int *)malloc(array_size_bytes);
	unsigned int *cpu_xthread = (unsigned int *)malloc(array_size_bytes);
	unsigned int *cpu_ythread = (unsigned int *)malloc(array_size_bytes);
	unsigned int *cpu_grid_dimx = (unsigned int *)malloc(array_size_bytes);
	unsigned int *cpu_block_dimx = (unsigned int *)malloc(array_size_bytes);
	unsigned int *cpu_grid_dimy = (unsigned int *)malloc(array_size_bytes);
	unsigned int *cpu_block_dimy = (unsigned int *)malloc(array_size_bytes);

	/* Declare pointers for GPU based params */
	unsigned int * gpu_block_x;
	unsigned int * gpu_block_y;
	unsigned int * gpu_thread;
	unsigned int * gpu_calc_thread;
	unsigned int * gpu_xthread;
	unsigned int * gpu_ythread;
	unsigned int * gpu_grid_dimx;
	unsigned int * gpu_block_dimx;
	unsigned int * gpu_grid_dimy;
	unsigned int * gpu_block_dimy;

	/* Allocate arrays on the GPU */
	cudaMalloc((void **)&gpu_block_x, array_size_bytes);
	cudaMalloc((void **)&gpu_block_y, array_size_bytes);
	cudaMalloc((void **)&gpu_thread, array_size_bytes);
	cudaMalloc((void **)&gpu_calc_thread, array_size_bytes);
	cudaMalloc((void **)&gpu_xthread, array_size_bytes);
	cudaMalloc((void **)&gpu_ythread, array_size_bytes);
	cudaMalloc((void **)&gpu_grid_dimx, array_size_bytes);
	cudaMalloc((void **)&gpu_block_dimx, array_size_bytes);
	cudaMalloc((void **)&gpu_grid_dimy, array_size_bytes);
	cudaMalloc((void **)&gpu_block_dimy, array_size_bytes);

	/* Execute our kernel */
	what_is_my_id_2d_A<<<blocks, threads>>>(gpu_block_x, gpu_block_y,
		gpu_thread, gpu_calc_thread, gpu_xthread, gpu_ythread, gpu_grid_dimx, gpu_block_dimx,
		gpu_grid_dimy, gpu_block_dimy);

	/* Copy back the gpu results to the CPU */
	cudaMemcpy(cpu_block_x, gpu_block_x, array_size_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_block_y, gpu_block_y, array_size_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_thread, gpu_thread, array_size_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_calc_thread, gpu_calc_thread, array_size_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_xthread, gpu_xthread, array_size_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_ythread, gpu_ythread, array_size_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_grid_dimx, gpu_grid_dimx, array_size_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_block_dimx, gpu_block_dimx, array_size_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_grid_dimy, gpu_grid_dimy, array_size_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_block_dimy, gpu_block_dimy, array_size_bytes, cudaMemcpyDeviceToHost);

	printf("Grid: (%d x %d), Block: (%d x %d), Total: %d threads\n\n",
		grid_x, grid_y, block_x, block_y, total_threads);

	/* Iterate through the arrays and print */
	for(int y = 0; y < array_size_y; y++)
	{
		for(int x = 0; x < array_size_x; x++)
		{
			int idx = y * array_size_x + x;
			printf("CT: %2u BKX: %1u BKY: %1u TID: %2u YTID: %2u XTID: %2u GDX: %1u BDX: %1u GDY: %1u BDY: %1u\n",
					cpu_calc_thread[idx], cpu_block_x[idx], cpu_block_y[idx], cpu_thread[idx], cpu_ythread[idx],
					cpu_xthread[idx], cpu_grid_dimx[idx], cpu_block_dimx[idx], cpu_grid_dimy[idx], cpu_block_dimy[idx]);
		}
	}

	/* Free the arrays on the GPU as now we're done with them */
	cudaFree(gpu_block_x);
	cudaFree(gpu_block_y);
	cudaFree(gpu_thread);
	cudaFree(gpu_calc_thread);
	cudaFree(gpu_xthread);
	cudaFree(gpu_ythread);
	cudaFree(gpu_grid_dimx);
	cudaFree(gpu_block_dimx);
	cudaFree(gpu_grid_dimy);
	cudaFree(gpu_block_dimy);

	/* Free host memory */
	free(cpu_block_x);
	free(cpu_block_y);
	free(cpu_thread);
	free(cpu_calc_thread);
	free(cpu_xthread);
	free(cpu_ythread);
	free(cpu_grid_dimx);
	free(cpu_block_dimx);
	free(cpu_grid_dimy);
	free(cpu_block_dimy);

	return EXIT_SUCCESS;
}
