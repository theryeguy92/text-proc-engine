# src/train_ogc_clip.py
import math
from typing import List

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from transformers import CLIPModel, CLIPProcessor

from ogc_torch_dataset import OGCIterableDataset


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def collate_fn(batch: List[dict]):
    """
    Build a batch out of a list of items from OGCIterableDataset.

    Strategy: one positive image per query, ignore negatives for first pass
    (we can integrate negatives later into the loss).
    """
    queries = [item["query"] for item in batch]
    images = [item["pos_image"] for item in batch]
    return queries, images


def main():
    device = get_device()
    print(f"[train] Using device: {device}")

    # 1) dataset & dataloader
    dataset = OGCIterableDataset(max_negatives=0)  # ignore negatives for now
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        collate_fn=collate_fn,
    )

    # 2) model & processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)

    # Option: freeze most layers and only train logit_scale or final layer.
    # For now, we'll train the whole model (might be heavy).
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    # simple constant LR to start; you can add scheduler later.

    # 3) training loop
    steps = 0
    max_steps = 1000  # <-- start small, just to see if loss decreases

    for queries, images in dataloader:
        steps += 1
        if steps > max_steps:
            break

        inputs = processor(
            text=list(queries),
            images=list(images),
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            return_loss=True,  # CLIP built-in contrastive loss
        )

        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if steps % 10 == 0:
            print(f"[train] step {steps}/{max_steps} | loss={loss.item():.4f}")

    # save checkpoint
    save_path = "./models/ogc_clip_finetuned"
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(f"[train] Saved finetuned model to {save_path}")


if __name__ == "__main__":
    main()
