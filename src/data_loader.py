from datasets import load_dataset

def load_pet(split="test"):
    # Loads the PET dataset (Process Extraction from Text)
    # Huggingface dataset
    dataset = load_dataset("patriziobellan/Pet", split=split)
    return dataset

def extract_texts(dataset):
    """
    Extracts a single text string per instance by joining the token list.
    """
    texts = []
    for row in dataset:
        if "tokens" in row:
            text = " ".join(row["tokens"])
        else:
            # fallback in case something changes in future
            text = str(row)
        texts.append(text)
    return texts


# Shows number of sampels that are in pet, and what each entry looks like
if __name__ == "__main__":
    ds = load_pet()
    print(ds)
    print(ds[0])