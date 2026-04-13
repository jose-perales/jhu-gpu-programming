CXX      = g++
CXXFLAGS = -std=c++11 -O2 -Wall -Wextra
LDFLAGS  = -lOpenCL -lm

.PHONY: all build run images video clean clean-all

all: build run

build: mandelbrot.cpp mandelbrot.cl ../info.hpp
	$(CXX) $(CXXFLAGS) -I.. -o mandelbrot mandelbrot.cpp $(LDFLAGS)

images: build
	mkdir -p output
	./mandelbrot --width 1920 --height 1080 --iter 256 \
		--output output/mandelbrot_default.ppm --cpu
	./mandelbrot --width 1920 --height 1080 --iter 512 \
		--zoom 50 --cx -0.7453 --cy 0.1127 \
		--output output/mandelbrot_seahorse.ppm
	ffmpeg -y -i output/mandelbrot_default.ppm output/mandelbrot_default.png 2>/dev/null
	ffmpeg -y -i output/mandelbrot_seahorse.ppm output/mandelbrot_seahorse.png 2>/dev/null
	rm -f output/*.ppm mandelbrot_cpu.ppm

video: build
	rm -rf frames
	./mandelbrot --width 1280 --height 720 --iter 512 \
		--cx -0.7453 --cy 0.1127 --frames 600 --zoom-end 5000
	./make_mp4.sh
	mkdir -p output
	mv mandelbrot_zoom.mp4 output/
	rm -rf frames

run: images video

clean:
	rm -f mandelbrot *.ppm
	rm -rf frames *.mp4

clean-all: clean
	rm -rf output
