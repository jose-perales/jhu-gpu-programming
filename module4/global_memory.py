#!/usr/bin/env python3
"""Plot Global Memory benchmark: Interleaved (AoS) vs Non-interleaved (SoA)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    "cpu_il": "#4c72b0", "cpu_ni": "#55a868",
    "gpu_il": "#dd8452", "gpu_ni": "#c44e52",
}


def load_csv(filepath: Path) -> pd.DataFrame:
    """Load benchmark CSV and convert ns columns to ms."""
    df = pd.read_csv(filepath)
    for col in ["cpu_il_ns", "cpu_ni_ns", "gpu_il_ns", "gpu_ni_ns"]:
        df[col.replace("_ns", "_ms")] = df[col] / 1e6
    return df


def fmt_n(n: int) -> str:
    """Format element count as human-readable label (e.g. 4K, 16M)."""
    if n >= 1e6: return f"{n/1e6:.0f}M"
    if n >= 1e3: return f"{n/1e3:.0f}K"
    return str(n)


def create_grouped_bar_chart(ax: plt.Axes, df: pd.DataFrame):
    """Grouped bar chart comparing all four combinations (log scale)."""
    x = np.arange(len(df))
    w = 0.2
    labels = [fmt_n(n) for n in df["N"]]

    ax.bar(x - 1.5*w, df["cpu_il_ms"], w, color=COLORS["cpu_il"], label="CPU AoS")
    ax.bar(x - 0.5*w, df["cpu_ni_ms"], w, color=COLORS["cpu_ni"], label="CPU SoA")
    ax.bar(x + 0.5*w, df["gpu_il_ms"], w, color=COLORS["gpu_il"], label="GPU AoS")
    ax.bar(x + 1.5*w, df["gpu_ni_ms"], w, color=COLORS["gpu_ni"], label="GPU SoA")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Time (ms)")
    ax.set_yscale("log")
    ax.set_title("All Combinations (log scale)")
    ax.legend(fontsize=8)


def create_scaling_chart(ax: plt.Axes, df: pd.DataFrame):
    """Log-log scaling chart of each variant."""
    ax.plot(df["N"], df["cpu_il_ms"], "o-", color=COLORS["cpu_il"], label="CPU AoS", markersize=4)
    ax.plot(df["N"], df["cpu_ni_ms"], "s-", color=COLORS["cpu_ni"], label="CPU SoA", markersize=4)
    ax.plot(df["N"], df["gpu_il_ms"], "^-", color=COLORS["gpu_il"], label="GPU AoS", markersize=4)
    ax.plot(df["N"], df["gpu_ni_ms"], "D-", color=COLORS["gpu_ni"], label="GPU SoA", markersize=4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (elements)")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Scaling (log–log)")
    ax.legend(fontsize=8)
    ax.grid(True, which="major", ls="-", alpha=0.3)
    ax.grid(True, which="minor", ls=":", alpha=0.15)


def main():
    script_dir = Path(__file__).parent
    csv_path = script_dir / "data" / "global_memory.csv"
    charts_dir = script_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

    df = load_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Global Memory — Interleaved (AoS) vs Non-interleaved (SoA)",
                 fontsize=13, fontweight="bold")
    create_grouped_bar_chart(axes[0], df)
    create_scaling_chart(axes[1], df)

    plt.tight_layout()
    out_path = charts_dir / "global_memory_benchmark.png"
    plt.savefig(out_path, dpi=150)
    print(f"Chart saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
