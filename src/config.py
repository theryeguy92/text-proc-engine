import os
from dotenv import load_dotenv
import pathlib

# Project root: C:\dev\roberta_app
BASE_DIR = pathlib.Path(__file__).resolve().parents[1]

# Load .env from project root
env_path = BASE_DIR / ".env"
load_dotenv(env_path)

MODEL_NAME = os.getenv("MODEL_NAME", "xlm-roberta-base")
MODEL_DIR_RAW = os.getenv("MODEL_DIR", "C:/dev/roberta_app/models/xlm-roberta-base")
DEVICE = os.getenv("DEVICE", "cpu")

# Normalize path (in case it's relative)
MODEL_DIR = os.path.abspath(MODEL_DIR_RAW)

print(f"[config] BASE_DIR   = {BASE_DIR}")
print(f"[config] MODEL_DIR  = {MODEL_DIR}")
print(f"[config] DEVICE     = {DEVICE}")


# import os
# from dotenv import load_dotenv

# load_dotenv()

# MODEL_NAME = os.getenv("MODEL_NAME", "xlm-roberta-base")
# MODEL_DIR = os.getenv("MODEL_DIR", "./models/xlm-roberta-base")
# DEVICE = os.getenv("DEVICE", "cpu")