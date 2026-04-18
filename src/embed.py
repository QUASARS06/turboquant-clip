"""Precompute CLIP ViT-B/32 embeddings for Flickr30k. Resumable + chunked.

Run this ONCE on a free Colab/Kaggle GPU (T4 is plenty — total <30 min).
All downstream work is CPU-only.

Resumability
------------
Embeddings are saved as small chunk files in `data/chunks/images/` and
`data/chunks/texts/`. On each run, the script scans which chunk indices
already exist and resumes from the next unfinished batch. A separate
`finalize` step concatenates chunks into two single `.npy` files:

    data/image_embeddings.npy      # (30000, 512)  L2-normalized float32
    data/text_embeddings.npy       # (150000, 512) L2-normalized float32
    data/captions.json             # list of 150000 caption strings
    data/image_ids.json            # list of 30000 stable image IDs

Ground truth for retrieval: caption i ↔ image (i // 5). We preserve the
exact ordering by sorting the dataset rows by image id before iterating.

Usage
-----
    # On Colab/Kaggle with GPU runtime:
    python -m src.embed encode --out-dir /content/drive/MyDrive/cs639/data
    # If the session dies, re-run the same command — it will pick up where it left off.

    # After all chunks are saved:
    python -m src.embed finalize --out-dir /content/drive/MyDrive/cs639/data

Free tier tip: point --out-dir at your mounted Google Drive so chunks
survive a Colab disconnect. If Drive mounting is painful on Kaggle, use
`/kaggle/working` and download the files before the session ends.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = "openai/clip-vit-base-patch32"
IMAGE_BATCH = 64
TEXT_BATCH = 256
CAPTIONS_PER_IMAGE = 5


def _chunk_dir(out_dir: Path, which: str) -> Path:
    d = out_dir / "chunks" / which
    d.mkdir(parents=True, exist_ok=True)
    return d


_CHUNK_RE = re.compile(r"^chunk_(\d+)\.npy$")


def _existing_chunks(chunk_dir: Path) -> set[int]:
    """Return indices of completed chunks. Cleans up stale .tmp files from
    crashed prior runs. pathlib's `glob("chunk_*.npy")` also matches
    `chunk_*.npy.tmp` on some filesystems (Google Drive especially), so we
    iterate with an explicit regex instead."""
    out: set[int] = set()
    for p in chunk_dir.iterdir():
        if p.name.endswith(".npy.tmp"):
            try:
                p.unlink()
            except OSError:
                pass
            continue
        m = _CHUNK_RE.match(p.name)
        if m:
            out.add(int(m.group(1)))
    return out


def _save_chunk(chunk_dir: Path, idx: int, arr: np.ndarray) -> None:
    tmp = chunk_dir / f"chunk_{idx:05d}.npy.tmp"
    final = chunk_dir / f"chunk_{idx:05d}.npy"
    # np.save auto-appends ".npy" unless given a file handle, which would
    # turn "chunk_00000.npy.tmp" into "chunk_00000.npy.tmp.npy". Write via
    # a handle so the on-disk name matches `tmp` exactly.
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, final)  # atomic so a crash mid-save never leaves a corrupt chunk


def _load_dataset_sorted():
    """Load nlphuji/flickr30k via the auto-converted parquet revision.

    `datasets>=3.0` refuses to execute loading scripts, but every HF dataset
    has an auto-generated parquet snapshot on the hidden `refs/convert/parquet`
    branch. We load that branch, so no script execution happens.

    The parquet branch names the split 'TEST' (uppercase). We sort by 'img_id'
    to make iteration order deterministic so resuming produces bit-identical chunks.
    """
    try:
        ds = load_dataset(
            "nlphuji/flickr30k",
            split="TEST",
            revision="refs/convert/parquet",
        )
    except Exception:
        # Fallback: lower-case split name in case upstream renames it.
        ds = load_dataset(
            "nlphuji/flickr30k",
            split="test",
            revision="refs/convert/parquet",
        )
    sort_key = "img_id" if "img_id" in ds.column_names else ds.column_names[0]
    return ds.sort(sort_key)


def _load_clip(device: str):
    model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    return model, processor


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (x / norms).astype(np.float32)


@torch.no_grad()
def _encode_image_batch(model, processor, pil_images, device):
    # Some transformers versions have altered `get_image_features`'s return
    # type (tensor vs BaseModelOutputWithPooling). Call the sub-modules directly
    # so behavior is stable across versions.
    inputs = processor(images=pil_images, return_tensors="pt").to(device)
    vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
    pooled = vision_out.pooler_output if hasattr(vision_out, "pooler_output") else vision_out[1]
    feats = model.visual_projection(pooled)
    return _l2_normalize(feats.cpu().numpy().astype(np.float32))


@torch.no_grad()
def _encode_text_batch(model, processor, texts, device):
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    text_out = model.text_model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
    )
    pooled = text_out.pooler_output if hasattr(text_out, "pooler_output") else text_out[1]
    feats = model.text_projection(pooled)
    return _l2_normalize(feats.cpu().numpy().astype(np.float32))


def encode(out_dir: Path) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[embed] device: {device}")
    if device == "cpu":
        print("[embed] WARNING: no GPU detected. Image encoding will be very slow.")

    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = _chunk_dir(out_dir, "images")
    txt_dir = _chunk_dir(out_dir, "texts")

    ds = _load_dataset_sorted()
    n_images = len(ds)
    print(f"[embed] {n_images} images, {n_images * CAPTIONS_PER_IMAGE} captions")

    # Save the caption list + image id list now so downstream code can use them
    # regardless of embedding completion state.
    meta_path = out_dir / "captions.json"
    ids_path = out_dir / "image_ids.json"
    if not meta_path.exists() or not ids_path.exists():
        captions: list[str] = []
        image_ids: list[str] = []
        for row in ds:
            image_ids.append(str(row["img_id"]))
            caps = row["caption"]
            if len(caps) < CAPTIONS_PER_IMAGE:
                caps = caps + [caps[-1]] * (CAPTIONS_PER_IMAGE - len(caps))
            captions.extend(caps[:CAPTIONS_PER_IMAGE])
        meta_path.write_text(json.dumps(captions))
        ids_path.write_text(json.dumps(image_ids))
        print(f"[embed] wrote {meta_path.name} and {ids_path.name}")

    model, processor = _load_clip(device)

    # ---- images ----
    n_img_batches = (n_images + IMAGE_BATCH - 1) // IMAGE_BATCH
    done_img = _existing_chunks(img_dir)
    pending_img = [i for i in range(n_img_batches) if i not in done_img]
    print(f"[embed] images: {len(done_img)}/{n_img_batches} chunks already done")

    for bi in tqdm(pending_img, desc="images"):
        start = bi * IMAGE_BATCH
        stop = min(start + IMAGE_BATCH, n_images)
        pil = [ds[i]["image"].convert("RGB") if not isinstance(ds[i]["image"], Image.Image)
               else ds[i]["image"].convert("RGB") for i in range(start, stop)]
        feats = _encode_image_batch(model, processor, pil, device)
        _save_chunk(img_dir, bi, feats)

    # ---- text ----
    captions = json.loads(meta_path.read_text())
    n_txt = len(captions)
    n_txt_batches = (n_txt + TEXT_BATCH - 1) // TEXT_BATCH
    done_txt = _existing_chunks(txt_dir)
    pending_txt = [i for i in range(n_txt_batches) if i not in done_txt]
    print(f"[embed] texts: {len(done_txt)}/{n_txt_batches} chunks already done")

    for bi in tqdm(pending_txt, desc="texts"):
        start = bi * TEXT_BATCH
        stop = min(start + TEXT_BATCH, n_txt)
        feats = _encode_text_batch(model, processor, captions[start:stop], device)
        _save_chunk(txt_dir, bi, feats)

    print("[embed] all chunks written. Run `finalize` to concatenate.")


def finalize(out_dir: Path) -> None:
    img_dir = out_dir / "chunks" / "images"
    txt_dir = out_dir / "chunks" / "texts"

    def _concat(d: Path, name: str) -> np.ndarray:
        files = sorted(d.glob("chunk_*.npy"))
        if not files:
            raise FileNotFoundError(f"No chunks in {d}. Run `encode` first.")
        arrs = [np.load(f) for f in files]
        arr = np.concatenate(arrs, axis=0).astype(np.float32)
        # Paranoid renormalize (shouldn't change anything if chunks were normalized).
        arr = arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)
        out_path = out_dir / name
        np.save(out_path, arr)
        print(f"[finalize] wrote {out_path} shape={arr.shape}")
        return arr

    imgs = _concat(img_dir, "image_embeddings.npy")
    txts = _concat(txt_dir, "text_embeddings.npy")

    if txts.shape[0] != imgs.shape[0] * CAPTIONS_PER_IMAGE:
        print(
            f"[finalize] WARNING: {txts.shape[0]} captions != "
            f"{imgs.shape[0]} * {CAPTIONS_PER_IMAGE}. Re-check dataset ordering."
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["encode", "finalize"])
    p.add_argument("--out-dir", default="data", type=Path)
    args = p.parse_args()
    if args.action == "encode":
        encode(args.out_dir)
    else:
        finalize(args.out_dir)


if __name__ == "__main__":
    main()
