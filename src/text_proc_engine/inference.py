import torch
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from text_proc_engine.config import MODEL_DIR, DEVICE


class XLMRobertaEmbedder:
    def __init__(self):
        print(f"Loading model from local dir: {MODEL_DIR}")

        self.tokenizer = XLMRobertaTokenizer.from_pretrained(
            MODEL_DIR,
            local_files_only=True,  # important: do NOT treat as HF repo id
        )

        self.model = XLMRobertaModel.from_pretrained(
            MODEL_DIR,
            local_files_only=True,
        )

        self.device = DEVICE
        self.model.to(self.device)
        self.model.eval()

    def embed(self, text: str):
        if text is None:
            text = ""
        if not isinstance(text, str):
            text = str(text)

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)
            last_hidden = outputs.last_hidden_state
            embedding = last_hidden.mean(dim=1).squeeze(0)

        return embedding.cpu().numpy()