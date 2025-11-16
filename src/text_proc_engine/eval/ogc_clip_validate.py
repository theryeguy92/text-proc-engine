# src/text_proc_engine/eval/ogc_clip_validate.py

import argparse
import math
from typing import Iterator, Tuple, List

from datasets import load_dataset
from PIL import Image

import numpy as np

from text_proc_engine.models.clip_encoder import CLIPDualEncoder


def iter_eval_examples(
    num_examples: int = 200,
    max_negatives: int = 8,
) -> Iterator[Tuple[str, str, Image.Image, List[Image.Image]]]:
    ds = load_dataset(
        "racineai/OGC_MEGA_MultiDomain_DocRetrieval",
        split="train",
        streaming=True,
    )

    count = 0
    for row in ds:
        query = row["query"]
        pos_img = row["image"]

        # Ensure PIL RGB
        if not isinstance(pos_img, Image.Image):
            pos_img = Image.open(pos_img).convert("RGB")
        else:
            pos_img = pos_img.convert("RGB")

        negs: List[Image.Image] = []
        for k in range(16):
            key = f"negative_image_{k}"
            img = row.get(key, None)
            if img is None:
                continue
            if not isinstance(img, Image.Image):
                img = Image.open(img).convert("RGB")
            else:
                img = img.convert("RGB")
            negs.append(img)

        # Skip rows with no negatives
        if not negs:
            continue

        if max_negatives is not None and len(negs) > max_negatives:
            negs = negs[:max_negatives]

        yield row["id"], query, pos_img, negs

        count += 1
        if count >= num_examples:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-examples", type=int, default=200)
    parser.add_argument("--max-negatives", type=int, default=8)
    args = parser.parse_args()

    encoder = CLIPDualEncoder()

    total = 0
    hits1 = 0
    hits3 = 0
    mrr_sum = 0.0

    print(
        f"[val] Evaluating on {args.num_examples} examples "
        f"(up to {args.max_negatives} negatives per query)"
    )

    for ex_id, query, pos_img, neg_imgs in iter_eval_examples(
        num_examples=args.num_examples,
        max_negatives=args.max_negatives,
    ):
        total += 1

        # 1) embed text
        text_vec = encoder.embed_text(query)  # shape (1, D)
        text_vec = text_vec[0]  # (D,)

        # 2) embed images: [positive] + negatives
        all_imgs = [pos_img] + neg_imgs
        img_vecs = encoder.embed_images(all_imgs)  # (1 + Nneg, D)

        # 3) cosine similarity via dot product (already normalized)
        scores = img_vecs @ text_vec  # (1 + Nneg,)
        scores = scores.tolist()

        # positive is index 0 by construction
        pos_score = scores[0]
        sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        rank = sorted_idx.index(0) + 1  # 1-based rank

        if rank == 1:
            hits1 += 1
        if rank <= 3:
            hits3 += 1
        mrr_sum += 1.0 / rank

        if total % 10 == 0:
            print(
                f"[val] example {total} | rank={rank} | "
                f"Hit@1={hits1/total:.3f} | Hit@3={hits3/total:.3f} | MRR={mrr_sum/total:.3f}"
            )

    if total == 0:
        print("[val] No examples evaluated (streaming issue)")
        return

    hit1 = hits1 / total
    hit3 = hits3 / total
    mrr = mrr_sum / total

    print("\n========== OGC CLIP VALIDATION ==========")
    print(f"Examples: {total}")
    print(f"Hit@1: {hit1:.4f}")
    print(f"Hit@3: {hit3:.4f}")
    print(f"MRR:   {mrr:.4f}")
    print("=========================================")


if __name__ == "__main__":
    main()
