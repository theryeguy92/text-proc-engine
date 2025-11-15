import os
import numpy as np

from data_loader import load_pet, extract_texts
from inference import XLMRobertaEmbedder


def main():
    # Output directory
    out_dir = os.path.join("..", "data")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading PET dataset...")
    ds = load_pet()
    texts = extract_texts(ds)
    print(f"Loaded {len(texts)} documents.")

    embedder = XLMRobertaEmbedder()

    all_embeddings = []
    for i, text in enumerate(texts):
        emb = embedder.embed(text)
        all_embeddings.append(emb)

        if (i + 1) % 5 == 0 or i == len(texts) - 1:
            print(f"Embedded {i + 1}/{len(texts)} documents")

    embeddings = np.stack(all_embeddings, axis=0)
    print(f"Embeddings shape: {embeddings.shape}")

    # Save embeddings and the raw texts for reference
    emb_path = os.path.join(out_dir, "pet_embeddings.npy")
    txt_path = os.path.join(out_dir, "pet_texts.txt")

    np.save(emb_path, embeddings)
    print(f"Saved embeddings to {emb_path}")

    with open(txt_path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t.replace("\n", " ") + "\n")
    print(f"Saved texts to {txt_path}")


if __name__ == "__main__":
    main()
