"""Analysis A — embedding geometry before and after the random rotation.

Checks whether CLIP coordinates post-rotation match the near-iid Beta/Gaussian
assumption that TurboQuant's theory needs. If they don't for image embeddings
specifically, that's the mechanistic explanation for cross-modal degradation.

Usage
-----
    python -m src.analysis.geometry --data-dir data --out-dir results/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def coordinate_stats(X: np.ndarray) -> dict:
    flat = X.ravel()
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "skew": float(((flat - flat.mean()) ** 3).mean() / (flat.std() ** 3 + 1e-12)),
        "kurtosis": float(((flat - flat.mean()) ** 4).mean() / (flat.std() ** 4 + 1e-12) - 3.0),
    }


def modality_gap(image_emb: np.ndarray, text_emb: np.ndarray) -> dict:
    mu_i = image_emb.mean(axis=0)
    mu_t = text_emb.mean(axis=0)
    gap_vec = mu_i - mu_t
    return {
        "gap_norm": float(np.linalg.norm(gap_vec)),
        "cos_mu_i_mu_t": float((mu_i @ mu_t) / (np.linalg.norm(mu_i) * np.linalg.norm(mu_t) + 1e-12)),
    }


def run(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = np.load(data_dir / "image_embeddings.npy")
    txts = np.load(data_dir / "text_embeddings.npy")

    rng = np.random.default_rng(0)
    d = imgs.shape[1]
    R = rng.standard_normal((d, d))
    R, _ = np.linalg.qr(R)

    imgs_rot = imgs @ R.T
    txts_rot = txts @ R.T

    stats = {
        "image_raw": coordinate_stats(imgs),
        "image_rot": coordinate_stats(imgs_rot),
        "text_raw": coordinate_stats(txts),
        "text_rot": coordinate_stats(txts_rot),
        "modality_gap_raw": modality_gap(imgs, txts),
        "modality_gap_rot": modality_gap(imgs_rot, txts_rot),
    }
    print(stats)
    (out_dir / "geometry_stats.txt").write_text(str(stats))

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    bins = np.linspace(-0.3, 0.3, 80)
    ax[0, 0].hist(imgs.ravel(), bins=bins, density=True, alpha=0.7)
    ax[0, 0].set_title("Image coords (raw)")
    ax[0, 1].hist(imgs_rot.ravel(), bins=bins, density=True, alpha=0.7)
    ax[0, 1].set_title("Image coords (rotated)")
    ax[1, 0].hist(txts.ravel(), bins=bins, density=True, alpha=0.7)
    ax[1, 0].set_title("Text coords (raw)")
    ax[1, 1].hist(txts_rot.ravel(), bins=bins, density=True, alpha=0.7)
    ax[1, 1].set_title("Text coords (rotated)")
    for a in ax.ravel():
        a.set_xlabel("coordinate value")
        a.set_ylabel("density")
    fig.suptitle("CLIP embedding coordinate distributions")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_geometry.png", dpi=150)
    print(f"[geometry] wrote {out_dir / 'fig_geometry.png'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="results/figures")
    run(p.parse_args())


if __name__ == "__main__":
    main()
