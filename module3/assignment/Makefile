all: preprocess build run postprocess

build: assignment.cu
	nvcc assignment.cu -o assignment.exe

run: assignment.exe
	./assignment.exe 0 256

preprocess:
	uv run preprocess.py

postprocess:
	uv run postprocess.py

clean:
	rm -f assignment.exe performance.csv
	rm -rf images/ charts/

clean-all: clean
	rm -rf .venv __pycache__
