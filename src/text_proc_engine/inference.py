# text_proc_engine/inference.py

from text_proc_engine.models.roberta_embedder import XLMRobertaEmbedder

# create a singleton embedder for quick use
_embedder = XLMRobertaEmbedder()

def embed_text(text: str):
    """
    Convenience function: embed a single piece of text with XLM-R.
    """
    return _embedder.embed(text)
