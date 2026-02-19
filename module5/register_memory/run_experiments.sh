#!/usr/bin/env bash
# ---------------------------------------------------------------
# register.cu — Practical Activity: Understanding GPU Registers
# ---------------------------------------------------------------
# This script compiles and runs register.cu with different
# parameters, using -Xptxas -v to reveal register allocation
# and (optionally) ncu to profile occupancy.
# ---------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p build

echo "register.cu — practical activity"
echo ""

# Step 1: compile with -Xptxas -v so the PTX assembler tells us
# how many registers each kernel uses.

echo "--- compiling baseline (KERNEL_LOOP=2048, KERNEL_SIZE=128) ---"
echo ""
nvcc -Xptxas -v register.cu -o build/register_baseline 2>&1 | grep -E "registers|spill"
echo ""
echo "The kernel's local variable d_tmp lives in a register."
echo "Let's verify the output (first & last 3 lines):"
echo ""
TMP=$(mktemp)
./build/register_baseline > "$TMP"
head -3 "$TMP"
echo "  ..."
tail -3 "$TMP"
rm -f "$TMP"

# Step 2: change the data size. Since registers are per-thread
# (not per-problem), the count shouldn't change.

echo ""
echo "--- now let's change the data size and see if register count changes ---"
echo ""

for LOOP in 256 2048 65536; do
    printf "KERNEL_LOOP=%-6d → " "$LOOP"
    nvcc -Xptxas -v -DKERNEL_LOOP=$LOOP register.cu -o build/reg_loop_$LOOP 2>&1 | \
        grep -oP 'Used \d+ registers'
done

# Step 3: change the block size. This only affects the launch
# config (how threads are grouped), not the kernel code itself.

echo ""
echo "Same register count every time. Makes sense — we didn't change the kernel."
echo ""
echo "--- what about changing the block size? ---"
echo ""

for SIZE in 32 128 512 1024; do
    BLOCKS=$(( (16384 + SIZE - 1) / SIZE ))
    printf "KERNEL_SIZE=%-5d (grid: %4d blocks × %4d threads) → " "$SIZE" "$BLOCKS" "$SIZE"
    nvcc -Xptxas -v -DKERNEL_LOOP=16384 -DKERNEL_SIZE=$SIZE register.cu \
        -o build/reg_size_$SIZE 2>&1 | grep -oP 'Used \d+ registers'
done

echo ""
echo "Still the same. Block size changes the grid layout, not the kernel code."

# Step 4: try Nsight Compute for runtime register/occupancy info.

echo ""
echo "--- nsight compute profiling ---"

if command -v ncu &>/dev/null; then
    echo ""
    NCU_OUT=$(sudo ncu --metrics \
        launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active \
        ./build/register_baseline 2>&1) || true

    if echo "$NCU_OUT" | grep -q "No kernels were profiled"; then
        echo "ncu couldn't profile. Check permissions or GPU availability."
    else
        echo "$NCU_OUT" | grep -E "registers_per_thread|warps_active|Kernel"
    fi
    echo ""
else
    echo "ncu not found. The -Xptxas -v output above already gives us register counts,"
    echo "but ncu would also show occupancy (active warps vs. SM capacity)."
fi

echo ""
echo "---"
echo ""
echo "So what did we learn?"
echo ""
echo "  - d_tmp and other local vars in the kernel live in registers,"
echo "    the fastest memory on the GPU."
echo ""
echo "  - Register count depends on what the kernel DOES, not on how"
echo "    much data you throw at it or how you configure the grid."
echo ""
echo "  - Use -Xptxas -v when compiling to check register usage."
echo "    Use ncu at runtime to also see occupancy."
echo ""
