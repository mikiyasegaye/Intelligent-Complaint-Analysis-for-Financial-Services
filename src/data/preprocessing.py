"""
Preprocessing module for financial complaints data.

This module provides functions for cleaning and preprocessing customer complaint
narratives and associated metadata. It handles text normalization, standardization
of categorical variables, and preparation of data for the RAG pipeline.

Functions:
    clean_text: Cleans and normalizes text data
    preprocess_text: Tokenizes and preprocesses text for analysis
    standardize_categories: Standardizes categorical variables
"""

from config.config import PRODUCT_CATEGORIES, PRODUCT_MAPPING, RAW_DATA_DIR, PROCESSED_DATA_DIR
import pandas as pd
import numpy as np
from typing import List, Dict
import re
from pathlib import Path
import sys
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


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


def clean_text(text: str) -> str:
    """
    Clean and normalize text data.

    This function performs several cleaning operations:
    1. Converts text to lowercase
    2. Replaces XXXX patterns with [REDACTED]
    3. Removes special characters and digits
    4. Normalizes whitespace

    Args:
        text: The input text to clean

    Returns:
        str: The cleaned text

    Examples:
        >>> clean_text("This is a SAMPLE text with XXXX!")
        "this is a sample text with [REDACTED]"
    """
    if pd.isna(text):
        return ""

    # Convert to string if not already
    text = str(text)

    # Remove 'b' prefix and quotes if present (from byte string)
    text = re.sub(r"^b'|'$", '', text)

    # Convert to lowercase
    text = text.lower()

    # Replace XXXX with [REDACTED]
    text = re.sub(r'x{2,}', '[REDACTED]', text)

    # Remove special characters and digits, but keep [REDACTED]
    text = re.sub(r'[^a-zA-Z\s\[\]REDACTED]', ' ', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text


def preprocess_text(text: str) -> list:
    """
    Preprocess text data for analysis.

    This function:
    1. Tokenizes the text
    2. Removes stopwords
    3. Removes short tokens
    4. Removes [REDACTED] tokens

    Args:
        text: The input text to preprocess

    Returns:
        list: List of preprocessed tokens

    Examples:
        >>> preprocess_text("This is a sample text")
        ['sample', 'text']
    """
    if pd.isna(text):
        return []

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]

    # Remove short tokens and [REDACTED]
    tokens = [token for token in tokens if len(
        token) > 2 and token != '[REDACTED]']

    return tokens


def standardize_categories(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Standardize categorical variables in the dataset.

    This function standardizes categories by:
    1. Converting to lowercase
    2. Removing extra whitespace
    3. Replacing common variations with standard forms
    4. Handling specific mappings for products and companies

    Args:
        df: Input DataFrame
        column: Name of the column to standardize

    Returns:
        pd.Series: Standardized category column

    Examples:
        >>> df = pd.DataFrame({'product': ['Credit Cards', 'CREDIT-CARD']})
        >>> standardize_categories(df, 'product')
        0    credit card
        1    credit card
        Name: product, dtype: object
    """
    if df[column].dtype != 'object':
        return df[column]

    # Convert to lowercase and strip whitespace
    result = df[column].str.lower().str.strip()

    # Common replacements for financial products
    product_replacements = {
        'credit card': ['credit cards', 'creditcard', 'credit-card'],
        'checking account': ['checking', 'checking acct', 'check account'],
        'savings account': ['savings', 'savings acct', 'save account'],
        'mortgage': ['home loan', 'house loan', 'mortgage loan'],
        'personal loan': ['personal loans', 'consumer loan'],
        'student loan': ['student loans', 'education loan'],
        'vehicle loan': ['auto loan', 'car loan', 'automobile loan']
    }

    # Common replacements for companies
    company_replacements = {
        'wells fargo': ['wellsfargo', 'wells fargo bank', 'wf bank'],
        'bank of america': ['bofa', 'bank of america na', 'bankofamerica'],
        'jpmorgan chase': ['chase', 'chase bank', 'jp morgan'],
        'citibank': ['citi', 'citigroup', 'citi bank']
    }

    # Apply replacements based on column name
    if 'product' in column.lower():
        replacements = product_replacements
    elif 'company' in column.lower():
        replacements = company_replacements
    else:
        return result

    # Apply standardization
    for standard, variations in replacements.items():
        result = result.replace(variations, standard)

    return result


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
