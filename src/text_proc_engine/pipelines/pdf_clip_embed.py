import os
import json
from typing import List, Dict

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

import torch
from transformers import CLIPModel, CLIPProcessor

from text_proc_engine.config import OUT_DIR_PDF  # path to your PDFs dir


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
    def embed_images(self, images: List[Image.Image]) -> np.ndarray:
        inputs = self.processor(
            text=None,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        feats = self.model.get_image_features(pixel_values=inputs["pixel_values"])
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()  # (B, D)


def render_pdf_to_images(pdf_path: str, dpi: int = 150) -> List[Dict]:
    """
    Render each page of a PDF to a PIL Image and return list of dicts:
    {
        "doc_id": str,
        "page_number": int,
        "image": PIL.Image.Image,
        "pdf_path": str
    }
    """
    doc = fitz.open(pdf_path)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    results = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        mat = fitz.Matrix(dpi / 72, dpi / 72)  # scale from 72dpi
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        results.append(
            {
                "doc_id": base_name,
                "page_number": page_idx + 1,
                "image": img,
                "pdf_path": os.path.abspath(pdf_path),
            }
        )

    doc.close()
    return results


def embed_pdf_folder(pdf_folder: str, out_dir: str):
    if not pdf_folder or not os.path.isdir(pdf_folder):
        raise ValueError(f"PDF folder does not exist or is not set: {pdf_folder}")

    os.makedirs(out_dir, exist_ok=True)

    encoder = CLIPDualEncoder()

    all_embs = []
    metas = []

    for root, _, files in os.walk(pdf_folder):
        for fname in files:
            if not fname.lower().endswith(".pdf"):
                continue

            pdf_path = os.path.join(root, fname)
            print(f"[pdf_clip] Processing {pdf_path}")
            pages = render_pdf_to_images(pdf_path)

            batch_size = 8
            for i in range(0, len(pages), batch_size):
                batch = pages[i : i + batch_size]
                imgs = [p["image"] for p in batch]
                embs = encoder.embed_images(imgs)
                all_embs.append(embs)

                for p in batch:
                    metas.append(
                        {
                            "doc_id": p["doc_id"],
                            "page_number": p["page_number"],
                            "pdf_path": p["pdf_path"],
                        }
                    )

    if not all_embs:
        print("[pdf_clip] No PDFs found.")
        return

    embs = np.concatenate(all_embs, axis=0)

    emb_path = os.path.join(out_dir, "pdf_page_clip_embeddings.npy")
    meta_path = os.path.join(out_dir, "pdf_page_clip_metadata.jsonl")

    np.save(emb_path, embs)
    print(f"[pdf_clip] Saved embeddings to {emb_path}")

    with open(meta_path, "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m) + "\n")
    print(f"[pdf_clip] Saved metadata to {meta_path}")


if __name__ == "__main__":
    # IN: your PDFs folder from .env
    pdf_folder = OUT_DIR_PDF

    # OUT: index directory for embeddings + metadata
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, "data", "pdf_clip_index")

    print(f"[pdf_clip] Using PDF folder: {pdf_folder}")
    print(f"[pdf_clip] Output index dir: {out_dir}")

    embed_pdf_folder(pdf_folder, out_dir)
