"""Functions for preprocessing CFPB complaint data."""

from config.config import PRODUCT_CATEGORIES, PRODUCT_MAPPING, RAW_DATA_DIR, PROCESSED_DATA_DIR
import pandas as pd
import numpy as np
from typing import List, Dict
import re
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))


def load_complaints_data(file_path: Path = RAW_DATA_DIR / "complaints.csv") -> pd.DataFrame:
    """
    Load the CFPB complaints dataset.

    Args:
        file_path: Path to the complaints CSV file

    Returns:
        DataFrame containing the complaints data
    """
    df = pd.read_csv(file_path, low_memory=False)
    print(f"\nLoaded {len(df):,} records with {len(df.columns)} columns")
    return df


def filter_products(df: pd.DataFrame, categories: List[str] = PRODUCT_CATEGORIES) -> pd.DataFrame:
    """
    Filter complaints to include only specified product categories.

    Args:
        df: Input DataFrame
        categories: List of product categories to include

    Returns:
        Filtered DataFrame
    """
    # Find the product column (case-sensitive since we know it's 'Product')
    if 'Product' not in df.columns:
        raise KeyError(
            f"Column 'Product' not found. Available columns: {', '.join(df.columns)}")

    print("\nFiltering products...")
    print("Looking for the following categories:")
    for target, variations in PRODUCT_MAPPING.items():
        print(f"\n{target}:")
        for var in variations:
            print(f"  - {var}")

    # Create the filter mask
    mask = df['Product'].isin(categories)
    filtered_df = df[mask].copy()

    # Show matching statistics
    print("\nMatched products distribution:")
    product_counts = filtered_df['Product'].value_counts()
    for product in product_counts.index:
        count = product_counts[product]
        percentage = (count / len(df)) * 100
        print(f"{product}: {count:,} ({percentage:.1f}%)")

    print(
        f"\nTotal filtered complaints: {len(filtered_df):,} ({len(filtered_df)/len(df)*100:.1f}% of total)")

    return filtered_df


def clean_narrative(text: str) -> str:
    """
    Clean complaint narrative text.

    Args:
        text: Input text

    Returns:
        Cleaned text
    """
    if pd.isna(text):
        return ""

    # Convert to string if not already
    text = str(text)

    # Remove common boilerplate text
    text = re.sub(r"XX+", "REDACTED", text)  # Replace XX with REDACTED
    text = re.sub(r"I am writing to file a complaint", "", text)
    text = re.sub(r"I am writing to complain", "", text)

    # Basic cleaning
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text


def preprocess_complaints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the complaints DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        Preprocessed DataFrame
    """
    # Print column names for debugging
    print("\nAvailable columns in DataFrame:")
    print(df.columns.tolist())

    # Select relevant columns (using exact column names from the dataset)
    relevant_columns = [
        'Date received',
        'Product',
        'Sub-product',
        'Issue',
        'Sub-issue',
        'Consumer complaint narrative',
        'Company public response',
        'Company',
        'State',
        'ZIP code',
        'Complaint ID'
    ]

    # Check which columns are actually available
    available_columns = [col for col in relevant_columns if col in df.columns]
    if not available_columns:
        raise ValueError("None of the expected columns found in DataFrame")

    df_processed = df[available_columns].copy()
    print(f"Selected {len(available_columns)} columns for processing")

    # Clean narratives if the column exists
    narrative_col = 'Consumer complaint narrative'
    if narrative_col in df_processed.columns:
        print("\nCleaning complaint narratives...")
        df_processed['clean_narrative'] = df_processed[narrative_col].apply(
            clean_narrative)
        df_processed = df_processed[df_processed['clean_narrative'].str.len(
        ) > 0]
        print(
            f"Retained {len(df_processed):,} complaints with non-empty narratives")

    # Convert date column
    date_col = 'Date received'
    if date_col in df_processed.columns:
        df_processed[date_col] = pd.to_datetime(df_processed[date_col])

    return df_processed


def save_processed_data(df: pd.DataFrame, file_name: str = "processed_complaints.csv") -> None:
    """
    Save the processed DataFrame.

    Args:
        df: Processed DataFrame
        file_name: Name of the output file
    """
    output_path = PROCESSED_DATA_DIR / file_name
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} processed complaints to {output_path}")
