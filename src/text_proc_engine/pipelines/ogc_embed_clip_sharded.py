import os
import json
import argparse
from text_proc_engine.config import OUT_DIR_OGC

import numpy as np
from datasets import load_dataset
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor
from text_proc_engine.models.clip_encoder import CLIPDualEncoder

# As per usual, check for CUDA

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

# config
# check output dir
print(OUT_DIR_OGC)
    
def process_shard(
    shard_id: int,
    shard_size: int,
    batch_size: int,
    out_dir: str = None,
):
    ds = load_dataset(
        "racineai/OGC_MEGA_MultiDomain_DocRetrieval",
        split="train",
        streaming=True,
    )

    start = shard_id * shard_size
    end = start + shard_size

    if out_dir is None:
        if OUT_DIR_OGC is None:
            raise ValueError("OUT_DIR_OGC is not set in .env or config.py")
        out_dir = OUT_DIR_OGC

    # if out_dir is None:
    #     base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    #     out_dir = os.path.join(base_dir, "data", "ogc_clip_shards")

    os.makedirs(out_dir, exist_ok=True)

    encoder = CLIPDualEncoder()

    all_text_embs = []
    all_img_embs = []
    metas = []

    texts_batch = []
    imgs_batch = []

    def flush_batch():
        nonlocal all_text_embs, all_img_embs, texts_batch, imgs_batch
        if not texts_batch:
            return
        t_embs, i_embs = encoder.embed_batch(texts_batch, imgs_batch)
        all_text_embs.append(t_embs)
        all_img_embs.append(i_embs)
        texts_batch = []
        imgs_batch = []

    for i, row in enumerate(ds):
        if i < start:
            continue
        if i >= end:
            break

        query = row["query"]
        img = row["image"]

        # Defensive: ensure image is RGB
        if not isinstance(img, Image.Image):
            # In streaming mode it should be PIL, but just in case:
            img = Image.open(img).convert("RGB")
        else:
            img = img.convert("RGB")

        texts_batch.append(query)
        imgs_batch.append(img)

        metas.append(
            {
                "global_idx": i,
                "id": row["id"],
                "language": row["language"],
                "num_negatives": int(row["num_negatives"]),
            }
        )

        if len(texts_batch) >= batch_size:
            print(f"[shard {shard_id}] embedding batch at global row {i}")
            flush_batch()

    # Flush any remaining
    flush_batch()

    if not all_text_embs:
        print(f"[shard {shard_id}] No rows processed (probably beyond dataset size).")
        return

    # Concatenate all batches
    text_embs = np.concatenate(all_text_embs, axis=0)
    img_embs = np.concatenate(all_img_embs, axis=0)

    # Save shard
    text_out = os.path.join(out_dir, f"queries_shard{shard_id}.npy")
    img_out = os.path.join(out_dir, f"pages_shard{shard_id}.npy")
    meta_out = os.path.join(out_dir, f"meta_shard{shard_id}.jsonl")

    np.save(text_out, text_embs)
    np.save(img_out, img_embs)

    with open(meta_out, "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m) + "\n")

    print(
        f"[shard {shard_id}] Saved {len(metas)} rows "
        f"-> {text_out}, {img_out}, {meta_out}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--shard-size", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()
    process_shard(
        shard_id=args.shard_id,
        shard_size=args.shard_size,
        batch_size=args.batch_size,
    )





