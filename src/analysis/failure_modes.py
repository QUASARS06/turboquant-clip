"""Analysis C — which queries fail most under aggressive compression?

Runs TurboQuant at 2 bits, finds the 50 text queries whose Recall@10 (text→image)
degrades the most vs. float32, and dumps them to a CSV for manual inspection.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ..compressors import TurboQuant


def run(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = np.load(data_dir / "image_embeddings.npy")
    txts = np.load(data_dir / "text_embeddings.npy")
    captions = json.loads((data_dir / "captions.json").read_text())
    d = imgs.shape[1]

    comp = TurboQuant(d, angle_bits=1, qjl_bits_per_dim=1.0, seed=0)
    comp.fit(np.concatenate([imgs, txts], axis=0))
    img_code = comp.encode(imgs)

    rng = np.random.default_rng(0)
    q_idx = rng.choice(txts.shape[0], args.n_queries, replace=False)
    queries = txts[q_idx]

    scores_full = queries @ imgs.T
    scores_comp = comp.ip_estimate(queries, img_code)

    def rank_of_gt(scores, gt_imgs):
        order = np.argsort(-scores, axis=1)
        ranks = np.empty(scores.shape[0], dtype=np.int64)
        for i, g in enumerate(gt_imgs):
            ranks[i] = int(np.where(order[i] == g)[0][0])
        return ranks

    gt_imgs = q_idx // 5
    r_full = rank_of_gt(scores_full, gt_imgs)
    r_comp = rank_of_gt(scores_comp, gt_imgs)
    delta = r_comp - r_full

    worst = np.argsort(-delta)[:50]
    out_csv = out_dir / "worst_queries.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["caption", "float32_rank", "turboquant2_rank", "rank_delta"])
        for wi in worst:
            w.writerow([captions[int(q_idx[wi])], int(r_full[wi]), int(r_comp[wi]), int(delta[wi])])
    print(f"[failure_modes] wrote {out_csv}  (higher rank_delta = worse degradation)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="results/figures")
    p.add_argument("--n-queries", type=int, default=1000)
    run(p.parse_args())


if __name__ == "__main__":
    main()
