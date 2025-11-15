import torch
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from config import MODEL_DIR, DEVICE


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














# import torch
# from transformers import XLMRobertaTokenizer, XLMRobertaModel
# from config import MODEL_DIR, DEVICE


# class XLMRobertaEmbedder:
#     def __init__(self):
#         # Load model from MODEL_DIR
#         print(f"Loading model from: {MODEL_DIR}")

#         self.tokenizer = XLMRobertaTokenizer.from_pretrained(
#             MODEL_DIR,
#             local_files_only=True,
#         )

#         self.model = XLMRobertaModel.from_pretrained(
#             MODEL_DIR,
#             local_files_only=True,
#         )

#         # Device setup
#         self.device = DEVICE  # "cpu" or "cuda"
#         self.model.to(self.device)
#         self.model.eval()

#     def embed(self, text: str):
#         """
#         Returns a 1D numpy vector for a single piece of text.
#         """
#         # Handle empty/None safely
#         if text is None:
#             text = ""
#         if not isinstance(text, str):
#             text = str(text)

#         # Tokenize
#         encoded = self.tokenizer(
#             text,
#             return_tensors="pt",
#             truncation=True,
#             padding=True,
#             max_length=256,
#         )

#         # Move tensors to device
#         encoded = {k: v.to(self.device) for k, v in encoded.items()}

#         # Inference
#         with torch.no_grad():
#             outputs = self.model(**encoded)
#             # Mean pool over sequence length
#             last_hidden = outputs.last_hidden_state  # (batch, seq, hidden)
#             embedding = last_hidden.mean(dim=1).squeeze(0)  # (hidden,)

#         return embedding.cpu().numpy()


















# # import torch
# # from transformers import XLMRobertaTokenizer, XLMRobertaModel
# # from config import MODEL_DIR, DEVICE

# # class XLMRobertaEmbedder:
# #     def __init__(self):
# #         #Load model from MODEL_DIR
# #         def __init__(self):
# #             print(f"Loading model from: {MODEL_DIR}")
# #             self.tokenizer = XLMRobertaTokenizer.from_pretrained(
# #             MODEL_DIR,
# #             local_files_only=True
# #         )
# #             self.model = XLMRobertaModel.from_pretrained(
# #             MODEL_DIR,
# #             local_files_only=True
# #         )

# #         # self.tokenizer = XLMRobertaTokenizer.from_pretrained(
# #         #     MODEL_DIR,
# #         #     local_files_only=True
# #         # )
# #         # self.model = XLMRobertaModel.from_pretrained(
# #         #     MODEL_DIR,
# #         #     local_files_only=True
# #         # )
# #         self.device = DEVICE
# #         self.model.to(self.device)
# #         self.model.eval()

# #     def embed(self, text:str):
# #         if text is None:
# #             text = ""
# #         if not isinstance(text,str):
# #             text = str(text)

# #         encoded = self.tokenizer(
# #             text,
# #             return_tensors = "pt",
# #             truncation=True,
# #             padding=True,
# #             max_length=256
# #         )

# #         # Tensors moved to device

# #         encoded = {k: v.to(self.device) for k, v in encoded.items()}

# #         with torch.no_grand():
# #             outputs = self.model(**encoded)
# #             # For now, we will just do mean pool over sequence length
# #             last_hidden = outputs.last_hidden_state
# #             embedding = last_hidden.mean(dim=1).squeeze(0)

# #         return embedding.cpu().numpy()