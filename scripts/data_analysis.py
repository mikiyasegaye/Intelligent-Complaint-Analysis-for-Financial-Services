"""
Script to analyze data filtering steps and verify data integrity.
"""

from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, PRODUCT_MAPPING
import pandas as pd
import os
import sys
from pathlib import Path

# Get the absolute path to the project root and add it to Python path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now we can import our config


def analyze_datasets():
    """Analyze and compare raw and filtered datasets."""
    # First check if the files exist
    raw_file = RAW_DATA_DIR / "complaints.csv"
    filtered_file = PROCESSED_DATA_DIR / "filtered_complaints.csv"

    if not raw_file.exists():
        print(f"Error: Raw data file not found at {raw_file}")
        return
    if not filtered_file.exists():
        print(f"Error: Filtered data file not found at {filtered_file}")
        return

    print("Loading raw data...")
    raw_df = pd.read_csv(raw_file, low_memory=False)
    print(f"\nRaw data shape: {raw_df.shape}")
    print(
        f"Raw data size: {raw_df.memory_usage(deep=True).sum() / (1024**3):.2f} GB")

    print("\nLoading filtered data...")
    filtered_df = pd.read_csv(filtered_file)
    print(f"Filtered data shape: {filtered_df.shape}")
    print(
        f"Filtered data size: {filtered_df.memory_usage(deep=True).sum() / (1024**3):.2f} GB")

    print("\nAnalyzing filtering steps:")

    # Check product distribution
    print("\nProduct distribution in raw data:")
    raw_products = raw_df['Product'].value_counts()
    print(raw_products)

    # Create a list of all valid product variations
    valid_products = []
    for category, variations in PRODUCT_MAPPING.items():
        valid_products.extend(variations)
        print(f"\n{category} matches in raw data:")
        for var in variations:
            count = raw_products.get(var, 0)
            print(f"  - {var}: {count:,}")

    # Check our target products
    target_products = raw_df[raw_df['Product'].isin(valid_products)]
    print(f"\nComplaints with target products: {len(target_products):,}")

    # Check narratives
    narratives_mask = target_products['Consumer complaint narrative'].notna()
    target_with_narratives = target_products[narratives_mask]
    print(f"Target products with narratives: {len(target_with_narratives):,}")

    # Compare with filtered dataset
    print(f"\nFiltered dataset size: {len(filtered_df):,}")

    # Check for any missing complaint_ids
    raw_ids = set(target_with_narratives['Complaint ID'])
    filtered_ids = set(filtered_df['complaint_id'])

    missing_ids = raw_ids - filtered_ids
    print(f"\nMissing complaint IDs: {len(missing_ids)}")

    if missing_ids:
        print("\nSample of missing complaints:")
        sample_missing = target_with_narratives[
            target_with_narratives['Complaint ID'].isin(list(missing_ids)[:5])
        ]
        print(
            sample_missing[['Product', 'Consumer complaint narrative']].head())

    # Calculate reduction percentages
    total_reduction = (1 - len(filtered_df)/len(raw_df)) * 100
    print(f"\nData reduction summary:")
    print(f"Raw data rows: {len(raw_df):,}")
    print(f"After product filtering: {len(target_products):,}")
    print(f"With narratives: {len(target_with_narratives):,}")
    print(f"Final filtered: {len(filtered_df):,}")
    print(f"Total reduction: {total_reduction:.1f}%")


if __name__ == "__main__":
    analyze_datasets()
