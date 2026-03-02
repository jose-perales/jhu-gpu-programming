THREADS = 1024 65536 1048576
BLOCKS  = 64 128 256 512

.PHONY: all build run clean clean-all

all: build run

build: assignment.cu
	mkdir -p build
	nvcc $< -o build/assignment

run: build
	mkdir -p data
	@uv run collect.py --output data/posts.txt --num-posts 10000
	@echo "threads,block_size,num_streams,sequential_ms,concurrent_ms,speedup,data_bytes" > data/performance.csv
	@for t in $(THREADS); do for b in $(BLOCKS); do \
		echo "=== threads=$$t block_size=$$b ===" && \
		./build/assignment $$t $$b --file data/posts.txt \
			--csv data/performance.csv; \
	done; done
	@echo ""
	@echo "Saved: data/performance.csv"
	@uv run postprocess.py

clean:
	rm -rf build data charts

clean-all: clean
	rm -rf .venv __pycache__ uv.lock
