# TurboQuant-family Compression on CLIP Multimodal Embeddings

CS 639, Spring 2026 — Team of 7.

Evaluates QJL, PolarQuant, TurboQuant, and FAISS PQ compression of CLIP
ViT-B/32 image+text embeddings on Flickr30k, across bit-widths {1,2,3,4,8,32}
and four retrieval tasks.

## What's in here

```
src/
  compressors/        QJL, PolarQuant, TurboQuant, FAISS PQ, float32 — all same API
  embed.py            Resumable CLIP encoding (only step needing GPU)
  eval/
    retrieval.py      4 tasks, Recall@{1,5,10}
    profiler.py       memory + latency
    experiment.py     full bit-width sweep orchestrator
  analysis/
    geometry.py       coordinate distributions (Analysis A)
    failure_modes.py  worst queries under 2-bit TurboQuant (Analysis C)
    plots.py          main figures (Analysis B + memory tradeoff)
tests/                pytest IP-correlation gate
```

All compressors are implemented from the papers, not ported from the
CUDA/Triton reference repos. The QJL math is in `src/compressors/qjl.py`,
PolarQuant follows Algorithm 1 of arXiv:2502.02617, TurboQuant follows
Algorithm 2 of arXiv:2504.19874.

## Free-tier compute plan

The project is designed to run end-to-end on **free Colab or Kaggle**:

| Step | Where | GPU? | Time |
|------|-------|------|------|
| 1. Encode embeddings | Colab/Kaggle (once) | T4 free tier | ~25 min |
| 2–8. Everything else | Your laptop or Colab CPU | No | ~2 hours total |

If the Colab session disconnects during Step 1, **just re-run the same
command** — chunks are saved atomically and `encode` resumes from the last
completed batch.

## Setup

```bash
git clone <repo-url> turboquant-clip
cd turboquant-clip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Sanity check:

```python
import torch, faiss, transformers
print(torch.cuda.is_available(), faiss.__version__)
```

## Sequence to run (what to do, in order)

### Step 1 — Precompute CLIP embeddings on a GPU (once)

Open `notebooks/01_encode_clip.ipynb` on Colab with a T4 GPU, **or** run:

```bash
# Mount Google Drive so chunks survive disconnects:
python -m src.embed encode --out-dir /content/drive/MyDrive/cs639/data
# When the session dies, re-run the same command. It will skip finished chunks.
python -m src.embed finalize --out-dir /content/drive/MyDrive/cs639/data
```

After this, you have:
- `data/image_embeddings.npy`  (30014, 512) L2-normalized
- `data/text_embeddings.npy`   (150070, 512) L2-normalized
- `data/captions.json`, `data/image_ids.json`

From this point on, **no GPU is needed**.

### Step 2 — Verify compressors

```bash
pytest tests/ -v
```

Every compressor must pass the IP-correlation gate (r > 0.92 at ≥4 bits).
If one fails, debug the estimator math before moving on.

### Step 3 — Run the full bit-width sweep

```bash
python -m src.eval.experiment \
    --data-dir data \
    --out-dir results \
    --seeds 0,1,2,3,4 \
    --max-queries 1000
```

Outputs:
- `results/main_results.csv` — one row per (method, bits, seed, task)
- `results/profiles.csv` — memory, compression time, query latency

Incremental debugging: pass `--methods qjl --bits 2,4 --seeds 0` to smoke-test
one configuration first.

### Step 4 — Generate figures

```bash
python -m src.analysis.geometry    --data-dir data --out-dir results/figures
python -m src.analysis.failure_modes --data-dir data --out-dir results/figures
python -m src.analysis.plots       --results-dir results
```

This produces:
- `results/figures/fig_geometry.png`     — Analysis A
- `results/figures/worst_queries.csv`    — Analysis C
- `results/figures/fig_cross_vs_same.png`— Analysis B (headline plot)
- `results/figures/fig_memory_tradeoff.png`

### Step 5 — Write the report

The raw numbers and figures are everything you need. See the proposal for
the 5-page structure (Intro / Related Work / Methodology / Results / Conclusion).

## Team task assignments (7 members)

| Role | Primary files |
|------|--------------|
| Coordinator / integrator | README, CI, PR reviews, final report |
| Data team (×2) | `src/embed.py`, `notebooks/01_encode_clip.ipynb`; post-Phase-2 → Analysis B |
| QJL + TurboQuant team (×2) | `src/compressors/qjl.py`, `turboquant.py`, `tests/test_compressors.py`; post-Phase-3 → Analysis A |
| PolarQuant + PQ team (×2) | `src/compressors/polarquant.py`, `faiss_pq.py`, `src/eval/profiler.py`; post-Phase-3 → Analysis C |

Everyone opens their own GitHub issue and PRs into the same `Compressor`
interface (`src/compressors/base.py`).

## Expected headline result (proposal hypothesis)

At 3 bits/dim, TurboQuant should retain Recall@10 within ~3 points of
float32 on all 4 tasks. At 2 bits, cross-modal tasks (text→image,
image→text) should degrade noticeably faster than same-modal tasks — if
so, the hypothesis is confirmed. See `fig_cross_vs_same.png`.
