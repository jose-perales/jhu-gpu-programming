NVCC = nvcc
UV   = uv

SIZES = 256 1024 4096 16384 65536 262144 1048576 4194304 16777216 67108864

all: host_memory.exe global_memory.exe

host_memory.exe: host_memory.cu
	$(NVCC) host_memory.cu -o host_memory.exe

global_memory.exe: global_memory.cu
	$(NVCC) global_memory.cu -o global_memory.exe

.venv: pyproject.toml
	$(UV) sync
	@touch .venv

run-host: host_memory.exe
	@mkdir -p data
	@echo "N,kernel_ns,h2d_ns,d2h_ns,max_error" > data/host_memory.csv
	@for N in $(SIZES); do ./host_memory.exe $$N >> data/host_memory.csv; done
	@echo "Results saved to data/host_memory.csv"
	@cat data/host_memory.csv

run-global: global_memory.exe
	@mkdir -p data
	@echo "N,cpu_il_ns,cpu_ni_ns,gpu_il_ns,gpu_ni_ns" > data/global_memory.csv
	@for N in $(SIZES); do ./global_memory.exe $$N >> data/global_memory.csv; done
	@echo "Results saved to data/global_memory.csv"
	@cat data/global_memory.csv

run: run-host run-global

charts: run .venv
	$(UV) run python host_memory.py
	$(UV) run python global_memory.py

clean:
	rm -f *.exe
	rm -rf data/ charts/ .venv

clean-results:
	rm -rf data/ charts/

.PHONY: all run run-host run-global charts clean clean-results
