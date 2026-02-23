THREADS = 1024 65536 1048576
BLOCKS  = 64 128 256 512

.PHONY: all build run clean

all: build run

build: vector_scale.cu
	mkdir -p build
	nvcc -Xptxas -v $< -o build/vector_scale

run:
	@echo "n,block_size,host_ns,global_ns,register_ns,constant_ns,shared_ns" > performance.csv
	@for t in $(THREADS); do for b in $(BLOCKS); do \
		echo "=== threads=$$t blocks=$$b ===" && ./build/vector_scale $$t $$b performance.csv; \
	done; done
	@echo "Saved: performance.csv"
	@uv run postprocess.py

clean:
	rm -rf build performance.csv charts
