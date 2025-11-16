import os
import json
import numpy as np

import torch
from PIL import Image
from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor

from text_proc_engine.config import OUT_DIR_OGC


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class CLIPDualEncoder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = get_device()
        print(f"[clip] Loading {model_name} on {self.device}")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed_text(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.processor(
            text=texts,
            images=None,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        outputs = self.model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        # normalize
        outputs = outputs / outputs.norm(dim=-1, keepdim=True)
        return outputs.cpu().numpy()  # (B, D)


def load_shard(shard_id: int):
    out_dir = OUT_DIR_OGC
    if out_dir is None:
        raise ValueError("OUT_DIR_OGC is not set in config/.env")

    q_path = os.path.join(out_dir, f"queries_shard{shard_id}.npy")
    p_path = os.path.join(out_dir, f"pages_shard{shard_id}.npy")
    m_path = os.path.join(out_dir, f"meta_shard{shard_id}.jsonl")

    print(f"[load] {q_path}")
    query_embs = np.load(q_path)
    print(f"[load] {p_path}")
    page_embs = np.load(p_path)

    meta = []
    with open(m_path, "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))

    # sanity check
    assert query_embs.shape[0] == page_embs.shape[0] == len(meta)

    # normalize pages (should already be normalized, but just in case)
    norms = np.linalg.norm(page_embs, axis=1, keepdims=True) + 1e-8
    page_embs = page_embs / norms

    return query_embs, page_embs, meta


def test_retrieval_on_shard(shard_id: int = 0, top_k: int = 5):
    # 1) load shard
    query_embs, page_embs, meta = load_shard(shard_id)

    # 2) build CLIP encoder for new text queries
    encoder = CLIPDualEncoder()

    # 3) define some test queries (you can change these to your own)
    test_queries = [
        "optimization techniques for the Bellman-Ford algorithm",
        "equilibrium constant example with sulfur dioxide and oxygen",
        "publication date for a book about transnationalism",
    ]

    for q in test_queries:
        print("\n==================================")
        print(f"Query: {q}")
        # embed the query
        q_vec = encoder.embed_text(q)[0]  # (D,)
        # cosine similarity with all pages
        scores = page_embs @ q_vec  # (N,)
        idxs = np.argsort(scores)[::-1][:top_k]

        for rank, idx in enumerate(idxs, start=1):
            s = float(scores[idx])
            m = meta[idx]
            print(
                f"  #{rank} | score={s:.4f} "
                f"| id={m['id']} | lang={m['language']} | global_idx={m['global_idx']}"
            )

    print("\nDone.")


if __name__ == "__main__":
    test_retrieval_on_shard(shard_id=0, top_k=5)
