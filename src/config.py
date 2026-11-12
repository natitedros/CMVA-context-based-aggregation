"""Shared paths and settings. Every script imports from here so Colab and any
other machine only differ by the PROJECT_ROOT environment variable."""
import os

# Your Google Drive project folder (set PROJECT_ROOT in the notebook)
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/content/drive/MyDrive/nlp-moral-lora")

DATA_DIR = f"{PROJECT_ROOT}/data"
MIC_PATH = f"{DATA_DIR}/MIC.csv"
SPLITS_DIR = f"{DATA_DIR}/splits"
MODELS_DIR = f"{PROJECT_ROOT}/models"
RESULTS_DIR = f"{PROJECT_ROOT}/results"

# Fast local disk for trainer checkpoints (writing these to Drive is slow)
SCRATCH_DIR = os.environ.get("SCRATCH_DIR", "/tmp/moral_lora")

# The five values used in Dognin et al. (2024)
VALUES = ["care", "fairness", "loyalty", "authority", "sanctity"]

BASE_LM = "Qwen/Qwen2.5-0.5B-Instruct"
REWARD_BACKBONE = "distilbert-base-uncased"

SEED = 42

for d in (SPLITS_DIR, MODELS_DIR, RESULTS_DIR, SCRATCH_DIR):
    os.makedirs(d, exist_ok=True)
