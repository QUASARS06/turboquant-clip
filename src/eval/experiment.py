"""The full bit-width sweep. Produces results/main_results.csv.

One row per (method, bits_per_dim, task, seed), with Recall@{1,5,10},
memory, compression time, and query latency.

Usage
-----
    python -m src.eval.experiment \\
        --data-dir data \\
        --out-dir results \\
        --seeds 0,1,2,3,4 \\
        --max-queries 1000

Skip subsets while iterating:
    --methods qjl,turboquant
    --bits 3,4
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ..compressors import FaissPQ, PolarQuant, QJL, TurboQuant, Uncompressed
from .profiler import profile
from .retrieval import evaluate_all_tasks

BITS_DEFAULT = (1, 2, 3, 4, 8, 32)
METHODS_DEFAULT = ("qjl", "polarquant", "turboquant", "faiss_pq", "uncompressed")
FAISS_SUPPORTED_BITS = {1, 2, 4, 8}


def _build(method: str, d: int, bits: int, seed: int):
    if method == "uncompressed":
        return Uncompressed(d)
    if method == "qjl":
        return QJL(d, bits_per_dim=float(bits), seed=seed)
    if method == "polarquant":
        return PolarQuant(d, angle_bits=int(bits), seed=seed)
    if method == "turboquant":
        # Convention: report TurboQuant at target bits/dim `bits`, using
        # angle_bits = bits - 1 and a 1-bit QJL residual. For bits=1 the
        # budget is spent entirely on QJL.
        if bits == 1:
            return QJL(d, bits_per_dim=1.0, seed=seed)
        return TurboQuant(d, angle_bits=int(bits) - 1, qjl_bits_per_dim=1.0, seed=seed)
    if method == "faiss_pq":
        return FaissPQ(d, bits_per_dim=int(bits), seed=seed)
    raise ValueError(method)


def _applicable(method: str, bits: int) -> bool:
    if method == "uncompressed":
        return bits == 32
    if method == "faiss_pq":
        return bits in FAISS_SUPPORTED_BITS
    return bits in (1, 2, 3, 4, 8)


def run(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = np.load(data_dir / "image_embeddings.npy")
    txts = np.load(data_dir / "text_embeddings.npy")
    print(f"[sweep] images {imgs.shape}, texts {txts.shape}")
    d = imgs.shape[1]

    methods = tuple(args.methods.split(",")) if args.methods else METHODS_DEFAULT
    bits_list = tuple(int(b) for b in args.bits.split(",")) if args.bits else BITS_DEFAULT
    seeds = tuple(int(s) for s in args.seeds.split(","))

    results_path = out_dir / "main_results.csv"
    profile_path = out_dir / "profiles.csv"
    new_run = not results_path.exists()

    with results_path.open("a", newline="") as f_main, profile_path.open("a", newline="") as f_prof:
        w_main = csv.writer(f_main)
        w_prof = csv.writer(f_prof)
        if new_run:
            w_main.writerow(["method", "bits", "seed", "task", "recall_at_1", "recall_at_5", "recall_at_10", "n_queries"])
            w_prof.writerow(["method", "bits", "seed", "bytes_per_vector", "index_bytes",
                             "compress_seconds", "query_latency_ms"])

        for method in methods:
            for bits in bits_list:
                if not _applicable(method, bits):
                    continue
                for seed in seeds:
                    comp = _build(method, d, bits, seed)
                    print(f"[sweep] method={method} bits={bits} seed={seed}")

                    # Recall evaluation (fits/encodes inside)
                    results = evaluate_all_tasks(comp, imgs, txts, max_queries=args.max_queries, seed=seed)
                    for r in results:
                        w_main.writerow([
                            method, bits, seed, r.task,
                            r.recall[1], r.recall[5], r.recall[10], r.n_queries,
                        ])

                    # Profile on the image DB only (one representative number)
                    prof_comp = _build(method, d, bits, seed)
                    prof = profile(prof_comp, imgs, imgs[:200])
                    w_prof.writerow([
                        method, bits, seed, prof.bytes_per_vector,
                        prof.index_bytes, prof.compress_seconds, prof.query_latency_ms,
                    ])
                    f_main.flush()
                    f_prof.flush()

    print(f"[sweep] done. results -> {results_path}, profiles -> {profile_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--bits", default=None, help="comma list of bit-widths; default 1,2,3,4,8,32")
    p.add_argument("--methods", default=None, help="comma list of method names")
    p.add_argument("--max-queries", type=int, default=1000)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
