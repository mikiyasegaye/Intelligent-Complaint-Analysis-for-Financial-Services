"""Configuration settings for the complaints analysis system."""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"

# Model settings
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 1024))

# Product categories and mappings
PRODUCT_CATEGORIES = [
    "Credit card",
    "Checking or savings account",
    "Mortgage",
    "Personal loan",
    "Student loan"
]

PRODUCT_MAPPING = {
    "Credit card": [
        "Credit card",
        "Credit Card",
        "CREDIT CARD"
    ],
    "Checking or savings account": [
        "Checking account",
        "Savings account",
        "Bank account",
        "Checking or savings account"
    ],
    "Mortgage": [
        "Mortgage",
        "Home loan",
        "House loan"
    ],
    "Personal loan": [
        "Personal loan",
        "Consumer loan",
        "Personal finance"
    ],
    "Student loan": [
        "Student loan",
        "Education loan",
        "Student debt"
    ]
}

# Chunk settings for text processing
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Vector store settings
VECTOR_STORE_DISTANCE_FUNC = "cosine"  # or "euclidean", "dot_product"
VECTOR_STORE_N_RESULTS = 5
