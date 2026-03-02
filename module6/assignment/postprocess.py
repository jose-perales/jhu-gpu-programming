#!/usr/bin/env python3
"""Generate charts from CUDA streams pipeline CSV output."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    "sequential": "#e74c3c",
    "concurrent": "#3498db",
    "speedup": "#2ecc71",
}


def load_data(path="data/performance.csv"):
    return pd.read_csv(path)


def config_label(row):
    """Short label for a config row."""
    return (
        f"T={row.threads}\n"
        f"B={row.block_size}\n"
        f"S={row.num_streams}"
    )


def chart_seq_vs_conc(df, charts_dir):
    """Bar chart: sequential vs concurrent per config."""
    labels = [config_label(r) for _, r in df.iterrows()]
    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(
        x - w / 2, df["sequential_ms"], w,
        label="Sequential (1 stream)",
        color=COLORS["sequential"],
    )
    ax.bar(
        x + w / 2, df["concurrent_ms"], w,
        label="Concurrent (N streams)",
        color=COLORS["concurrent"],
    )
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Pipeline Time (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(
        "Sequential vs Concurrent Pipeline Time",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    out = charts_dir / "seq_vs_concurrent.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  {out}")


def chart_speedup(df, charts_dir):
    """Bar chart: speedup per configuration."""
    labels = [config_label(r) for _, r in df.iterrows()]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, df["speedup"], 0.5, color=COLORS["speedup"])
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Speedup (sequential / concurrent)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(
        "Speedup from CUDA Streams",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    out = charts_dir / "speedup.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  {out}")


def chart_throughput(df, charts_dir):
    """Throughput (MB/s) for sequential and concurrent."""
    labels = [config_label(r) for _, r in df.iterrows()]
    x = np.arange(len(labels))
    w = 0.35

    seq_mbps = df["data_bytes"] / df["sequential_ms"] / 1e3
    conc_mbps = df["data_bytes"] / df["concurrent_ms"] / 1e3

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(
        x - w / 2, seq_mbps, w,
        label="Sequential",
        color=COLORS["sequential"],
    )
    ax.bar(
        x + w / 2, conc_mbps, w,
        label="Concurrent",
        color=COLORS["concurrent"],
    )
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Throughput (MB/s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(
        "Throughput Comparison",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    out = charts_dir / "throughput.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  {out}")


def main():
    df = load_data()
    charts_dir = Path("charts")
    charts_dir.mkdir(exist_ok=True)

    chart_seq_vs_conc(df, charts_dir)
    chart_speedup(df, charts_dir)
    chart_throughput(df, charts_dir)
    print("Done.")


if __name__ == "__main__":
    main()
