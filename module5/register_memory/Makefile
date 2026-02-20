all: register

register: register.cu
	mkdir -p build
	nvcc -Xptxas -v register.cu -o build/register

run: register
	bash run_experiments.sh

clean:
	rm -rf build
