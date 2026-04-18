"""Main figures for the report.

fig_cross_vs_same.png : the headline plot — Recall@10 vs bits/dim, split into
                       cross-modal tasks (T1, T2) and same-modal tasks (T3, T4),
                       one subplot per compressor.

fig_memory_tradeoff.png : Recall@10 vs index bytes, one marker per (method, bits).

Reads results/main_results.csv (from experiment.py) and results/profiles.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_cross_vs_same(results_csv: Path, out_path: Path):
    df = pd.read_csv(results_csv)
    cross = {"T1_text2image", "T2_image2text"}
    df["modality"] = df["task"].apply(lambda t: "cross-modal" if t in cross else "same-modal")

    agg = (
        df.groupby(["method", "bits", "modality"])["recall_at_10"]
        .agg(mean="mean", std="std")
        .reset_index()
    )

    methods = sorted(df["method"].unique())
    fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 4), sharey=True)
    if len(methods) == 1:
        axes = [axes]
    for ax, m in zip(axes, methods):
        sub = agg[agg["method"] == m]
        for mod in ("cross-modal", "same-modal"):
            s = sub[sub["modality"] == mod].sort_values("bits")
            ax.errorbar(s["bits"], s["mean"], yerr=s["std"], marker="o", label=mod, capsize=3)
        ax.set_title(m)
        ax.set_xlabel("bits per dim")
        ax.set_xscale("log", base=2)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Recall@10")
    axes[-1].legend()
    fig.suptitle("Cross-modal vs same-modal retrieval under compression")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[plots] wrote {out_path}")


def plot_memory_tradeoff(results_csv: Path, profiles_csv: Path, out_path: Path):
    results = pd.read_csv(results_csv)
    profiles = pd.read_csv(profiles_csv)

    recall = results.groupby(["method", "bits"], as_index=False)["recall_at_10"].mean()
    prof = profiles.groupby(["method", "bits"], as_index=False)["bytes_per_vector"].mean()
    merged = recall.merge(prof, on=["method", "bits"])

    fig, ax = plt.subplots(figsize=(7, 5))
    for method, sub in merged.groupby("method"):
        s = sub.sort_values("bytes_per_vector")
        ax.plot(s["bytes_per_vector"], s["recall_at_10"], marker="o", label=method)
    ax.set_xlabel("bytes per vector (log)")
    ax.set_ylabel("mean Recall@10 across 4 tasks")
    ax.set_xscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.suptitle("Accuracy vs memory tradeoff")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[plots] wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()
    rdir = Path(args.results_dir)
    fig_dir = rdir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_cross_vs_same(rdir / "main_results.csv", fig_dir / "fig_cross_vs_same.png")
    plot_memory_tradeoff(rdir / "main_results.csv", rdir / "profiles.csv",
                         fig_dir / "fig_memory_tradeoff.png")


if __name__ == "__main__":
    main()
