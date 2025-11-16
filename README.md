# text-proc-engine

A document-centric retrieval and embedding engine that combines:

- **CLIP** for cross-modal retrieval (text ↔ PDF page images)
- **XLM-RoBERTa** for high-quality text embeddings and (future) MLM adaptation for corporate/trade jargon
- **Hugging Face OGC_MEGA_MultiDomain_DocRetrieval** as a large-scale training/eval source for page-level retrieval.

**Objective:**

Given a PDF, relevant document pages can be found for a given user query (**CLIP**), or relative qualitative data (such as ticket descriptions, notes, emails, etc.) from various data sources (**XLM-RoBERTa**).

---

## High-Level Architecture

**Two main model families**:

1. **CLIP (vision–text retriever)**  
   - Fine-tuned on `racineai/OGC_MEGA_MultiDomain_DocRetrieval`
   - Learns to align **queries** and **page images** in a shared space
   - Used for:
     - Query → relevant PDF pages
     - Page-image → similar pages

2. **XLM-RoBERTa (text encoder, future MLM)**  
   - Loaded locally from disk (no external calls at runtime)
   - Used to embed raw text (from PDFs, emails, procedures, tickets, etc.)
   - Future: fine-tuned with **Masked Language Modeling (MLM)** on internal corpora to learn **corporate/trade jargon** and domain-specific semantics

**Data flow (target design):**

1. User query or PDF → CLIP retrieves top-k relevant pages  
2. Extract text from those pages (OCR or text layer)  
3. Domain-adapted RoBERTa encodes that text for:
   - clustering
   - similarity search
   - classification / tagging
   - linking issues ⇄ procedures

---

## Repo Layout

**Structure subject to change**

```text
text-proc-engine\
├─ models\
│  ├─ xlm-roberta-base\          # Local RoBERTa/XLM-R weights
│  └─ ogc_clip_finetuned\        # (After training) fine-tuned CLIP model
├─ data\
│  ├─ ogc_clip_shards\           # Precomputed CLIP embeddings for OGC pages/queries
│  ├─ pdfs\                      # Local PDF corpus
│  └─ pdf_clip_index\            # CLIP embeddings + metadata for PDFs
├─ src\
│  └─ text_proc_engine\
│     ├─ __init__.py
│     ├─ config.py               # Central config + .env loading
│     ├─ datasets\
│     │  ├─ ogc_hf_stream.py     # Streaming OGC from Hugging Face
│     │  └─ ogc_torch_dataset.py # IterableDataset wrapper for OGC
│     ├─ models\
│     │  ├─ clip_encoder.py      # CLIPDualEncoder (text + image embeddings)
│     │  └─ roberta_embedder.py  # XLMRobertaEmbedder (text embeddings)
│     ├─ pipelines\
│     │  ├─ ogc_embed_clip_sharded.py  # Embed OGC queries/pages into shards
│     │  ├─ ogc_retrieval_test_shard0.py  # Sanity-check retrieval on a shard
│     │  └─ pdf_clip_embed.py    # Embed local PDFs with CLIP
│     ├─ training\
│     │  └─ train_ogc_clip.py    # Fine-tune CLIP on OGC
│     ├─ eval\
│     │  └─ ogc_clip_validate.py # Retrieval metrics (Hit@k, MRR) on OGC
│     └─ inference.py            # Thin wrapper over XLMRobertaEmbedder
├─ .env
├─ requirements.txt
└─ README.md
```
---
## Installation & Setup

### 1. Create and activate a virtual environment
```bash
cd C:\dev\text-proc-engine
python -m venv venv
venv\Scripts\activate

```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
If using a specific CUDA build for PyTorch, **ensure requirements.txt or your manual install matches your GPU/driver setup.**

### 3. Environment Variables (```.env```)
Create a ```.env``` at the project root:

```env
# XLM-RoBERTa (local)
MODEL_NAME=xlm-roberta-base
MODEL_DIR= your/path/here

# CLIP (can be a HF model ID or a local finetuned dir)
CLIP_MODEL_DIR=openai/clip-vit-base-patch32
# later: your/path/here

# Device preference
DEVICE=cuda
EMBEDDING_BATCH_SIZE=8
LOG_LEVEL=INFO

# OGC shard output
OUT_DIR_OGC=./text-proc-engine/data/ogc_clip_shards

# Local PDFs
OUT_DIR_PDF=./text-proc-engine/data/pdfs
```

### 4. Make sure ```src``` is on  ```PYTHONPATH```
Powershell:
```powershell
$env:PYTHONPATH = "/your/path/to/text-proc-engine/src"
```

## Roadmap

### Near term:
- Build corpus construction pipeline for RoBERTa MLM:
    - PDFs → text chunks
    - tickets/emails/text data → text
    - docs → text

- Impliment ```train_roberta_mlm.py```
    - load local XLM-R from ```MODEL_DIR```
    - use HF ```datasets``` and ```DataCollatorForLanguageModeling```
    - fine-tune with MLM on combined corpus
    - save to ```models/roberta_corp_mlm```
    - Update ```XLMRobertaEmbedder``` to optionally load ```roberta_corp_mlm```
 