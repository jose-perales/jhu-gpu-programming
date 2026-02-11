#!/usr/bin/env python3
"""Plot SAXPY host-memory benchmark results from CSV."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = ["#4c72b0", "#dd8452", "#55a868"]  # H2D, Kernel, D2H


def load_csv(filepath: Path) -> pd.DataFrame:
    """Load benchmark CSV and compute derived columns."""
    df = pd.read_csv(filepath)
    df["kernel_ms"] = df["kernel_ns"] / 1e6
    df["h2d_ms"]    = df["h2d_ns"]    / 1e6
    df["d2h_ms"]    = df["d2h_ns"]    / 1e6
    df["total_ms"]  = df["kernel_ms"] + df["h2d_ms"] + df["d2h_ms"]
    df["h2d_pct"]    = df["h2d_ms"]    / df["total_ms"] * 100
    df["kernel_pct"] = df["kernel_ms"] / df["total_ms"] * 100
    df["d2h_pct"]    = df["d2h_ms"]    / df["total_ms"] * 100
    return df


def fmt_n(n: int) -> str:
    """Format element count as human-readable label (e.g. 4K, 16M)."""
    if n >= 1e6: return f"{n/1e6:.0f}M"
    if n >= 1e3: return f"{n/1e3:.0f}K"
    return str(n)


def create_breakdown_chart(ax: plt.Axes, df: pd.DataFrame):
    """100% stacked bar chart showing proportional time breakdown."""
    x = np.arange(len(df))
    labels = [fmt_n(n) for n in df["N"]]

    ax.bar(x, df["h2d_pct"],    color=COLORS[0], label="H→D")
    ax.bar(x, df["kernel_pct"], color=COLORS[1], label="Kernel", bottom=df["h2d_pct"])
    ax.bar(x, df["d2h_pct"],    color=COLORS[2], label="D→H",
           bottom=df["h2d_pct"] + df["kernel_pct"])

    for i, pct in enumerate(df["kernel_pct"]):
        y_pos = df["h2d_pct"].iloc[i] + pct / 2
        if pct > 3:
            ax.text(i, y_pos, f"{pct:.1f}%",
                    ha="center", va="center", fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Share of Total Time (%)")
    ax.set_title("Time Breakdown (% of total)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=8)


def create_scaling_chart(ax: plt.Axes, df: pd.DataFrame):
    """Log-log scaling chart of each timing component."""
    ax.plot(df["N"], df["h2d_ms"],    "s-", color=COLORS[0], label="H→D",    markersize=5)
    ax.plot(df["N"], df["kernel_ms"], "o-", color=COLORS[1], label="Kernel",  markersize=5)
    ax.plot(df["N"], df["d2h_ms"],    "^-", color=COLORS[2], label="D→H",    markersize=5)
    ax.plot(df["N"], df["total_ms"],  "D-", color="#333333",  label="Total",
            markersize=5, linewidth=2)

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
    csv_path = script_dir / "data" / "host_memory.csv"
    charts_dir = script_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

    df = load_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("SAXPY Host Memory Benchmark", fontsize=13, fontweight="bold")
    create_breakdown_chart(axes[0], df)
    create_scaling_chart(axes[1], df)

    plt.tight_layout()
    out_path = charts_dir / "host_memory_benchmark.png"
    plt.savefig(out_path, dpi=150)
    print(f"Chart saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
