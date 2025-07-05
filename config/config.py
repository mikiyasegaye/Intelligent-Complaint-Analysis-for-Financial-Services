"""Configuration management for the project."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not found in environment variables")

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"

# Model configuration
# Using Llama-2 for better stability
MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/phi-2")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.4"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "500"))

# Data processing
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

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

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
