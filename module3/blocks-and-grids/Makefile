# CUDA Compiler
NVCC = nvcc
NVCC_FLAGS = -L /usr/local/cuda/lib -lcudart

# Targets
all: blocks grids

blocks: blocks.cu
	$(NVCC) blocks.cu $(NVCC_FLAGS) -o blocks

grids: grids.cu
	$(NVCC) grids.cu $(NVCC_FLAGS) -o grids

# ========================================
# BLOCKS.CU CONFIGURATIONS (1D)
# ========================================

blocks-run1: blocks
	@echo "\n>>> Blocks Config 1: 64 threads, 16/block, 4 blocks"
	./blocks 64 16

blocks-run2: blocks
	@echo "\n>>> Blocks Config 2: 128 threads, 32/block, 4 blocks"
	./blocks 128 32

blocks-run3: blocks
	@echo "\n>>> Blocks Config 3: 256 threads, 64/block, 4 blocks"
	./blocks 256 64

blocks-run4: blocks
	@echo "\n>>> Blocks Config 4: 512 threads, 128/block, 4 blocks"
	./blocks 512 128

blocks-run5: blocks
	@echo "\n>>> Blocks Config 5: 1024 threads, 256/block, 4 blocks"
	./blocks 1024 256

blocks-run-all: blocks
	@echo "\n========================================"
	@echo "Running all 5 BLOCKS configurations"
	@echo "========================================"
	$(MAKE) blocks-run1
	@echo "\n----------------------------------------"
	$(MAKE) blocks-run2
	@echo "\n----------------------------------------"
	$(MAKE) blocks-run3
	@echo "\n----------------------------------------"
	$(MAKE) blocks-run4
	@echo "\n----------------------------------------"
	$(MAKE) blocks-run5

# ========================================
# GRIDS.CU CONFIGURATIONS (2D)
# ========================================

grids-run1: grids
	@echo "\n>>> Grids Config 1: 1x4 grid, 32x4 block (512 threads, rectangular)"
	./grids 1 4 32 4

grids-run2: grids
	@echo "\n>>> Grids Config 2: 2x2 grid, 16x8 block (512 threads, square-ish)"
	./grids 2 2 16 8

grids-run3: grids
	@echo "\n>>> Grids Config 3: 4x2 grid, 8x8 block (512 threads)"
	./grids 4 2 8 8

grids-run4: grids
	@echo "\n>>> Grids Config 4: 2x4 grid, 16x4 block (512 threads)"
	./grids 2 4 16 4

grids-run5: grids
	@echo "\n>>> Grids Config 5: 1x8 grid, 32x2 block (512 threads, tall grid)"
	./grids 1 8 32 2

grids-run-all: grids
	@echo "\n========================================"
	@echo "Running all 5 GRIDS configurations"
	@echo "========================================"
	$(MAKE) grids-run1
	@echo "\n----------------------------------------"
	$(MAKE) grids-run2
	@echo "\n----------------------------------------"
	$(MAKE) grids-run3
	@echo "\n----------------------------------------"
	$(MAKE) grids-run4
	@echo "\n----------------------------------------"
	$(MAKE) grids-run5

# Run all configurations for both programs
run-all: blocks-run-all grids-run-all

clean:
	rm -f blocks grids

.PHONY: all clean blocks-run1 blocks-run2 blocks-run3 blocks-run4 blocks-run5 blocks-run-all grids-run1 grids-run2 grids-run3 grids-run4 grids-run5 grids-run-all run-all
