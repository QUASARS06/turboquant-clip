"""Retrieval evaluation. Four tasks, Recall@{1, 5, 10}.

Ground truth (Flickr30k convention): caption index `j` maps to image
`j // 5`. Any of the 5 captions of image `i` counts as a correct match
for image `i`.

Tasks
-----
    T1  text  -> image   (cross-modal; primary hypothesis target)
    T2  image -> text    (cross-modal)
    T3  text  -> text    (same-modal; 4 paraphrases are the ground truth)
    T4  image -> image   (same-modal; trivial self-retrieval is excluded)

Asymmetric search: queries stay at full float32; the database is
compressed. The compressor object is responsible for the inner-product
estimator — `ip_estimate(Q, code) -> (nq, N)` scores.

We cap the number of queries per task at `max_queries` (default 1000) so
the full sweep finishes in ~2 hours on CPU. Use -1 to run on everything.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


CAPTIONS_PER_IMAGE = 5


@dataclass
class TaskResult:
    task: str
    recall: dict[int, float]       # {1: 0.82, 5: 0.94, 10: 0.97}
    n_queries: int


def _recall_at_ks(
    scores: np.ndarray,           # (nq, N)
    gt_members: list[set[int]],   # for query i, the set of valid target indices
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[int, float]:
    topk = np.argpartition(-scores, max(ks), axis=1)[:, : max(ks)]
    # Re-sort the top-K partition by actual score so smaller k's are correct.
    row_idx = np.arange(topk.shape[0])[:, None]
    topk_sorted = topk[row_idx, np.argsort(-scores[row_idx, topk], axis=1)]
    hits = {k: 0 for k in ks}
    for i, gt in enumerate(gt_members):
        for k in ks:
            if any(int(idx) in gt for idx in topk_sorted[i, :k]):
                hits[k] += 1
    return {k: hits[k] / max(1, len(gt_members)) for k in ks}


def _scores(
    compressor,
    code_db,
    queries: np.ndarray,
    chunk: int = 200,
) -> np.ndarray:
    """Chunked IP estimation so we never materialize a (nq, N) matrix all at once.

    Returns (nq, N). Callers can reduce further (top-k) if needed.
    """
    parts = []
    for start in range(0, queries.shape[0], chunk):
        parts.append(compressor.ip_estimate(queries[start:start + chunk], code_db))
    return np.concatenate(parts, axis=0)


def evaluate_all_tasks(
    compressor,
    image_db: np.ndarray,
    text_db: np.ndarray,
    image_queries: np.ndarray | None = None,
    text_queries: np.ndarray | None = None,
    max_queries: int = 1000,
    seed: int = 0,
) -> list[TaskResult]:
    """Compress image_db and text_db once, then run the 4 tasks.

    `compressor.fit` is called on the concatenated image+text set so codebooks
    see both modalities. Queries stay at full precision.
    """
    rng = np.random.default_rng(seed)
    n_images = image_db.shape[0]
    n_texts = text_db.shape[0]
    assert n_texts == n_images * CAPTIONS_PER_IMAGE, (
        f"Expected {n_images * CAPTIONS_PER_IMAGE} texts, got {n_texts}."
    )

    all_vecs = np.concatenate([image_db, text_db], axis=0)
    compressor.fit(all_vecs)
    image_code = compressor.encode(image_db)
    text_code = compressor.encode(text_db)

    iq = image_db if image_queries is None else image_queries
    tq = text_db if text_queries is None else text_queries

    def _subsample(n: int) -> np.ndarray:
        if max_queries < 0 or n <= max_queries:
            return np.arange(n)
        return rng.choice(n, max_queries, replace=False)

    img_q_idx = _subsample(iq.shape[0])
    txt_q_idx = _subsample(tq.shape[0])

    results: list[TaskResult] = []

    # T1: text -> image. GT: caption j maps to image j // 5.
    q = tq[txt_q_idx]
    gt = [{int(j) // CAPTIONS_PER_IMAGE} for j in txt_q_idx]
    scores = _scores(compressor, image_code, q)
    results.append(TaskResult("T1_text2image", _recall_at_ks(scores, gt), len(gt)))

    # T2: image -> text. GT: image i matches any of captions {5i, 5i+1, ..., 5i+4}.
    q = iq[img_q_idx]
    gt = [set(range(int(i) * CAPTIONS_PER_IMAGE, (int(i) + 1) * CAPTIONS_PER_IMAGE))
          for i in img_q_idx]
    scores = _scores(compressor, text_code, q)
    results.append(TaskResult("T2_image2text", _recall_at_ks(scores, gt), len(gt)))

    # T3: text -> text. GT: caption j matches the OTHER 4 captions of image j//5,
    # excluding j itself.
    q = tq[txt_q_idx]
    gt = []
    for j in txt_q_idx:
        img = int(j) // CAPTIONS_PER_IMAGE
        sibs = set(range(img * CAPTIONS_PER_IMAGE, (img + 1) * CAPTIONS_PER_IMAGE))
        sibs.discard(int(j))
        gt.append(sibs)
    scores = _scores(compressor, text_code, q)
    # Mask self-match so a caption cannot retrieve itself.
    for row, j in enumerate(txt_q_idx):
        scores[row, int(j)] = -np.inf
    results.append(TaskResult("T3_text2text", _recall_at_ks(scores, gt), len(gt)))

    # T4: image -> image. Flickr30k has no intra-image duplicates, so we use a
    # "near-duplicate" proxy: the top-1 match in the float32 space at query
    # time. This needs the UNCOMPRESSED database as reference gold.
    q = iq[img_q_idx]
    gold_scores = q @ image_db.T
    for row, i in enumerate(img_q_idx):
        gold_scores[row, int(i)] = -np.inf
    gold_top1 = gold_scores.argmax(axis=1)
    gt = [{int(g)} for g in gold_top1]
    scores = _scores(compressor, image_code, q)
    for row, i in enumerate(img_q_idx):
        scores[row, int(i)] = -np.inf
    results.append(TaskResult("T4_image2image", _recall_at_ks(scores, gt), len(gt)))

    return results
