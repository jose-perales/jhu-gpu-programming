#!/usr/bin/env python3
"""Generate timing charts from vector_scale CSV output."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    "host": "#e74c3c",
    "global": "#3498db",
    "register": "#2ecc71",
    "constant": "#f39c12",
    "shared": "#9b59b6",
}
GPU_TYPES = ["global", "register", "constant", "shared"]
ALL_TYPES = ["host"] + GPU_TYPES


def load_data(path="performance.csv"):
    df = pd.read_csv(path)
    for m in ALL_TYPES:
        df[f"{m}_us"] = df[f"{m}_ns"] / 1000
    return df


def chart_memory_type_bars(df, charts_dir):
    """Per-n: log-scale comparison by memory type + GPU-only linear zoom."""
    for n, g in df.groupby("n"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        x = np.arange(len(g))
        labels = g["block_size"].values

        # Left: all 5 types, log scale
        w = 0.15
        for i, m in enumerate(ALL_TYPES):
            ax1.bar(
                x + (i - 2) * w,
                g[f"{m}_us"].values,
                w,
                label=m.capitalize(),
                color=COLORS[m],
            )
        ax1.set_yscale("log")
        ax1.set_xlabel("Block Size")
        ax1.set_ylabel("Time (us, log scale)")
        ax1.set_title("All Memory Types")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.legend(fontsize=8)
        ax1.grid(axis="y", linestyle="--", alpha=0.4)

        # Right: GPU only, linear scale (zoom)
        w2 = 0.2
        for i, m in enumerate(GPU_TYPES):
            ax2.bar(
                x + (i - 1.5) * w2,
                g[f"{m}_us"].values,
                w2,
                label=m.capitalize(),
                color=COLORS[m],
            )
        ax2.set_xlabel("Block Size")
        ax2.set_ylabel("Time (us)")
        ax2.set_title("GPU Only (linear zoom)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.legend(fontsize=8)
        ax2.grid(axis="y", linestyle="--", alpha=0.4)

        fig.suptitle(
            f"Memory Type Comparison (n={n:,})",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()
        out = charts_dir / f"comparison_n{n}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  {out}")


def chart_scaling(df, charts_dir):
    """Lines across problem sizes, averaged over block sizes."""
    avg = df.groupby("n").mean(numeric_only=True).reset_index()
    ns = avg["n"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: all types log scale
    for m in ALL_TYPES:
        ax1.plot(
            ns, avg[f"{m}_us"], "o-",
            label=m.capitalize(), color=COLORS[m],
        )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Problem Size (n)")
    ax1.set_ylabel("Time (us, log)")
    ax1.set_title("All Memory Types")
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Right: GPU only, linear
    for m in GPU_TYPES:
        ax2.plot(
            ns, avg[f"{m}_us"], "o-",
            label=m.capitalize(), color=COLORS[m],
        )
    ax2.set_xscale("log")
    ax2.set_xlabel("Problem Size (n)")
    ax2.set_ylabel("Time (us)")
    ax2.set_title("GPU Only (linear zoom)")
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        "Scaling Across Problem Sizes (avg over block sizes)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    out = charts_dir / "scaling.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  {out}")


def main():
    df = load_data()
    charts_dir = Path("charts")
    charts_dir.mkdir(exist_ok=True)

    chart_memory_type_bars(df, charts_dir)
    chart_scaling(df, charts_dir)
    print("Done.")


if __name__ == "__main__":
    main()
