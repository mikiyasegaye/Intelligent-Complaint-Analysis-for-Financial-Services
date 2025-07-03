"""Configuration settings for the project."""

import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Vector store directory
VECTOR_STORE_DIR = ROOT_DIR / "vector_store"

# Model settings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Product categories mapping
# Map our target categories to the actual categories in the dataset
PRODUCT_MAPPING = {
    "Credit Cards": ["Credit card", "Credit card or prepaid card"],
    "Personal Loans": ["Personal loan", "Consumer loan", "Payday loan"],
    "Buy Now, Pay Later (BNPL)": ["Buy now, pay later"],
    "Savings Accounts": ["Checking or savings account", "Bank account or service"],
    "Money Transfers": ["Money transfer, virtual currency, or money service"]
}

# Flattened list of all possible product names in the dataset
PRODUCT_CATEGORIES = [cat for cats in PRODUCT_MAPPING.values() for cat in cats]

# RAG settings
TOP_K_CHUNKS = 5  # Number of chunks to retrieve for each query

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, VECTOR_STORE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
