# CUDA Compiler
NVCC = nvcc
NVCC_FLAGS = -L /usr/local/cuda/lib -lcudart

# Targets
all: hello-world

hello-world: hello-world.cu
	$(NVCC) hello-world.cu $(NVCC_FLAGS) -o hello-world

run1: hello-world
	@echo "\n>>> Config 1: 16 threads, 16/block, 1 block"
	./hello-world 16 16

run2: hello-world
	@echo "\n>>> Config 2: 32 threads, 16/block, 2 blocks"
	./hello-world 32 16

run3: hello-world
	@echo "\n>>> Config 3: 256 threads, 64/block, 4 blocks"
	./hello-world 256 64

run4: hello-world
	@echo "\n>>> Config 4: 1024 threads, 128/block, 8 blocks"
	./hello-world 1024 128

run5: hello-world
	@echo "\n>>> Config 5: 512 threads, 256/block, 2 blocks"
	./hello-world 512 256

# Run all configurations
run-all: hello-world
	@echo "\n========================================"
	@echo "Running all 5 thread configurations"
	@echo "========================================"
	$(MAKE) run1
	@echo "\n----------------------------------------"
	$(MAKE) run2
	@echo "\n----------------------------------------"
	$(MAKE) run3
	@echo "\n----------------------------------------"
	$(MAKE) run4
	@echo "\n----------------------------------------"
	$(MAKE) run5

clean:
	rm -f hello-world

.PHONY: all clean run1 run2 run3 run4 run5 run-all
