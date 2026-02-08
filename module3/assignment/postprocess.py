#!/usr/bin/env python3
"""
Postprocess CUDA assignment output.
Generates performance charts and converts output images to PNG.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def load_csv(filepath: Path) -> pd.DataFrame:
    """Load performance data from CSV."""
    return pd.read_csv(filepath)


def create_performance_chart(df: pd.DataFrame, output_path: Path):
    """Create GPU vs CPU performance comparison chart."""
    
    df = df.sort_values('pixels').copy()
    df['size'] = df['pixels'].apply(lambda x: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1000:.0f}K")
    
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(df))
    w = 0.2
    
    ax.bar(x - 1.5*w, df['gpu_branch_ns']/1e6, w, label='GPU Branch', color='#3498db')
    ax.bar(x - 0.5*w, df['gpu_nobranch_ns']/1e6, w, label='GPU No-Branch', color='#2ecc71')
    ax.bar(x + 0.5*w, df['cpu_branch_ns']/1e6, w, label='CPU Branch', color='#e74c3c')
    ax.bar(x + 1.5*w, df['cpu_nobranch_ns']/1e6, w, label='CPU No-Branch', color='#f39c12')
    
    ax.set_xlabel('Data Size (Pixels)')
    ax.set_ylabel('Time (ms)')
    ax.set_title('GPU vs CPU Performance: Contrast Adjustment')
    ax.set_xticks(x)
    ax.set_xticklabels(df['size'], rotation=45, ha='right')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def create_overhead_chart(df: pd.DataFrame, output_path: Path):
    """Create branching overhead comparison chart."""
    
    gpu_overhead = ((df['gpu_branch_ns'] - df['gpu_nobranch_ns']) / df['gpu_nobranch_ns'] * 100).mean()
    cpu_overhead = ((df['cpu_branch_ns'] - df['cpu_nobranch_ns']) / df['cpu_nobranch_ns'] * 100).mean()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bars = ax.bar(['GPU', 'CPU'], [gpu_overhead, cpu_overhead], 
                  color=['#3498db', '#e74c3c'], width=0.5)
    
    ax.set_ylabel('Branching Overhead (%)')
    ax.set_title('Average Branching Overhead\n(Positive = Slower, Negative = Faster)')
    ax.axhline(y=0, color='black', linewidth=1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        h = bar.get_height()
        ax.bar_label(bars, fmt='%+.1f%%', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def convert_output_images(images_dir: Path):
    """Convert output .bin files to PNG."""
    
    for bin_path in images_dir.glob("*_output.bin"):
        with open(bin_path, "rb") as f:
            width = int.from_bytes(f.read(4), "little")
            height = int.from_bytes(f.read(4), "little")
            pixels = np.frombuffer(f.read(width * height), dtype=np.uint8)
        
        img = Image.fromarray(pixels.reshape((height, width)), mode="L")
        png_path = images_dir / f"{bin_path.stem.replace('_output', '_contrast')}.png"
        img.save(png_path)
        print(f"  {png_path.name}")


def main():
    script_dir = Path(__file__).parent
    csv_path = script_dir / "performance.csv"
    charts_dir = script_dir / "charts"
    images_dir = script_dir / "images"
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run assignment first.")
        return 1
    
    charts_dir.mkdir(exist_ok=True)
    df = load_csv(csv_path)
    
    print("\nGenerating charts...")
    create_performance_chart(df, charts_dir / "gpu_vs_cpu_performance.png")
    print(f"  gpu_vs_cpu_performance.png")
    create_overhead_chart(df, charts_dir / "branching_effect.png")
    print(f"  branching_effect.png")
    
    print("\nConverting output images...")
    convert_output_images(images_dir)
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
