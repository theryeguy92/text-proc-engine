# src/ogc_torch_dataset.py
from typing import Iterator, Dict, Any, List
import random

from datasets import load_dataset
from torch.utils.data import IterableDataset
from PIL import Image


class OGCIterableDataset(IterableDataset):

    def __init__(self, shuffle_negatives: bool = True, max_negatives: int = 4):
        super().__init__()
        self.shuffle_negatives = shuffle_negatives
        self.max_negatives = max_negatives

    def _get_stream(self):
        ds = load_dataset(
            "racineai/OGC_MEGA_MultiDomain_DocRetrieval",
            split="train",
            streaming=True,
        )
        return ds

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        ds_iter = self._get_stream()
        for row in ds_iter:
            query = row["query"]
            pos_img = row["image"]

            # make sure it's a PIL.Image
            if not isinstance(pos_img, Image.Image):
                pos_img = Image.open(pos_img).convert("RGB")
            else:
                pos_img = pos_img.convert("RGB")

            # collect available negatives
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

            if self.shuffle_negatives and negs:
                random.shuffle(negs)
            if self.max_negatives is not None and len(negs) > self.max_negatives:
                negs = negs[: self.max_negatives]

            yield {
                "query": query,
                "pos_image": pos_img,
                "neg_images": negs,
            }
