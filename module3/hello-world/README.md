# CUDA Hello World

Modify the hello-world.cu CUDA code to execute 5 different numbers of threads, with various block sizes and numbers of blocks.

## Configurations

| Config | Threads | Block Size | Blocks | Output               |
|--------|---------|------------|--------|----------------------|
| run1   | 16      | 16         | 1      | Thread IDs 0-15      |
| run2   | 32      | 16         | 2      | IDs 0-15 × 2 blocks  |
| run3   | 256     | 64         | 4      | IDs 0-63 × 4 blocks  |
| run4   | 1024    | 128        | 8      | IDs 0-127 × 8 blocks |
| run5   | 512     | 256        | 2      | IDs 0-255 × 2 blocks |

Run with `make run1`, `make run2`, etc., or `make run-all` for all configurations.

## CUDA Blocks & Threads Concepts

CUDA organizes parallel threads into a hierarchy:

- **Grid**: Contains all blocks
- **Block**: A group of threads (max 1024 threads per block)
- **Thread**: Individual unit of execution

Each thread knows its position:

- `threadIdx.x` - Thread's index within its block (resets to 0 in each block)
- `blockIdx.x` - Which block the thread belongs to
- `blockDim.x` - Number of threads per block

Global thread index: `globalIdx = blockIdx.x * blockDim.x + threadIdx.x`

Kernel launch syntax: `kernel<<<num_blocks, block_size>>>(args)`
