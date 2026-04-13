#!/bin/bash
# make_mp4.sh - Encode Mandelbrot frames into an MP4 video
# Requires: frames/ directory with PPM files from
#   ./mandelbrot --frames N --zoom-end Z ...
#
# Usage: ./make_mp4.sh [--fps N] [--output FILE]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

FPS=30
OUTPUT="mandelbrot_zoom.mp4"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fps)
            FPS="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        *)
            echo "Usage: $0 [--fps N] [--output FILE]"
            exit 1
            ;;
    esac
done

if [[ ! -d frames ]] || \
   [[ -z "$(ls frames/*.ppm 2>/dev/null)" ]]; then
    echo "Error: no frames found in frames/"
    echo "Generate them first:"
    echo "  ./mandelbrot --frames 120 --zoom-end 2000 \\"
    echo "    --cx -0.7453 --cy 0.1127 --iter 512"
    exit 1
fi

NFRAMES=$(ls frames/*.ppm | wc -l)
echo "Encoding $NFRAMES frames at $FPS fps..."

ffmpeg -y -framerate "$FPS" \
    -i frames/frame_%05d.ppm \
    -c:v libx264 -pix_fmt yuv420p \
    -crf 18 -preset medium \
    "$OUTPUT" 2>&1 | tail -5

echo ""
ls -lh "$OUTPUT"
