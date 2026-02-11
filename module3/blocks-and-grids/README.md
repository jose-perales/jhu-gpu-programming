# CUDA Blocks and Grids

Modify the blocks.cu and grids.cu CUDA code to execute at least 5 different numbers of threads, with various block sizes, numbers of blocks, and grid dimensions.

## Blocks.cu Configurations (1D)

| Config | Threads | Block Size | Blocks | Command |
| -------- | --------- | ------------ | -------- | --------- |
| blocks-run1 | 64 | 16 | 4 | `make blocks-run1` |
| blocks-run2 | 128 | 32 | 4 | `make blocks-run2` |
| blocks-run3 | 256 | 64 | 4 | `make blocks-run3` |
| blocks-run4 | 512 | 128 | 4 | `make blocks-run4` |
| blocks-run5 | 1024 | 256 | 4 | `make blocks-run5` |

Run with `make blocks-run1`, etc., or `make blocks-run-all` for all configurations.

## Grids.cu Configurations (2D)

| Config | Grid Dims | Block Dims | Total Threads | Command |
| -------- | ----------- | ------------ | --------------- | --------- |
| grids-run1 | 1×4 | 32×4 | 512 | `make grids-run1` |
| grids-run2 | 2×2 | 16×8 | 512 | `make grids-run2` |
| grids-run3 | 4×2 | 8×8 | 512 | `make grids-run3` |
| grids-run4 | 2×4 | 16×4 | 512 | `make grids-run4` |
| grids-run5 | 1×8 | 32×2 | 512 | `make grids-run5` |

Run with `make grids-run1`, etc., or `make grids-run-all` for all configurations.

## Usage

```bash
# Build both programs
make

# Run individual configurations
make blocks-run1    # or blocks-run2, blocks-run3, etc.
make grids-run1     # or grids-run2, grids-run3, etc.

# Run all configurations
make blocks-run-all  # All blocks.cu configs
make grids-run-all   # All grids.cu configs
make run-all         # Everything

# Manual execution
./blocks [total_threads] [block_size]
./grids [grid_x] [grid_y] [block_x] [block_y]
```

## CUDA Blocks & Grids Concepts

CUDA organizes parallel threads into a hierarchy:

- **Grid**: Contains all blocks (can be 1D, 2D, or 3D)
- **Block**: A group of threads (max 1024 threads per block)
- **Thread**: Individual unit of execution

### 1D Organization (blocks.cu)

- `threadIdx.x` - Thread's index within its block
- `blockIdx.x` - Which block the thread belongs to
- `blockDim.x` - Number of threads per block
- Global thread index: `globalIdx = blockIdx.x * blockDim.x + threadIdx.x`

### 2D Organization (grids.cu)

- `threadIdx.x/y` - Thread's 2D index within its block
- `blockIdx.x/y` - 2D block position in the grid
- `blockDim.x/y` - Block dimensions
- `gridDim.x/y` - Grid dimensions

Thread indexing in 2D:

```cpp
idx = blockIdx.x * blockDim.x + threadIdx.x
idy = blockIdx.y * blockDim.y + threadIdx.y
global_idx = (gridDim.x * blockDim.x) * idy + idx
```

Kernel launch syntax: `kernel<<<grid_dim, block_dim>>>(args)`
