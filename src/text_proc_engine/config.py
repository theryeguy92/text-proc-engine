import os
from dotenv import load_dotenv
import pathlib

# Project root:
BASE_DIR = pathlib.Path(__file__).resolve().parents[2]


# Load .env from project root
env_path = BASE_DIR / ".env"
load_dotenv(env_path)

MODEL_NAME = os.getenv("MODEL_NAME", "xlm-roberta-base")
DEVICE = os.getenv("DEVICE", "cpu")
OUT_DIR_OGC = os.getenv("OUT_DIR_OGC")
MODEL_DIR = os.getenv("MODEL_DIR")
OUT_DIR_PDF = os.getenv("OUT_DIR_PDF")

print(f"[config] BASE_DIR   = {BASE_DIR}")
print(f"[config] PDF_DIR    = {OUT_DIR_PDF}")
print(f"[config] OUT_DIR_OGC = {OUT_DIR_OGC}")
print(f"[config] MODEL_DIR  = {MODEL_DIR}")
print(f"[config] DEVICE     = {DEVICE}")
