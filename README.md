# Intelligent Complaint Analysis for Financial Services

A RAG-powered chatbot that transforms customer feedback into actionable insights for CrediTrust Financial's internal teams.

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) based chatbot that helps internal teams at CrediTrust Financial analyze and understand customer complaints across five major product categories:

- Credit Cards
- Personal Loans
- Buy Now, Pay Later (BNPL)
- Savings Accounts
- Money Transfers

### Key Features

- Natural language querying of customer complaints
- Semantic search powered by vector database
- AI-generated insights from complaint narratives
- Multi-product analysis capabilities
- User-friendly chat interface

## Project Structure

```
.
├── data/               # Data files
│   ├── raw/           # Original, immutable data
│   └── processed/     # Cleaned and processed data
├── notebooks/         # Jupyter notebooks for EDA and experiments
├── src/              # Source code
│   ├── data/         # Data processing scripts
│   ├── models/       # Model-related code
│   ├── utils/        # Utility functions
│   └── web/          # Web interface code
├── tests/            # Test files
├── vector_store/     # Vector database files
├── config/           # Configuration files
├── docs/             # Documentation
└── requirements.txt  # Project dependencies
```

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Task 1: Data Analysis and Preprocessing

### Completed Components

1. **Exploratory Data Analysis** (`notebooks/01_exploratory_data_analysis.ipynb`)

   - Product distribution analysis
   - Temporal trends analysis
   - Geographic distribution
   - Text statistics and patterns
   - Missing value analysis

2. **Data Cleaning Pipeline** (`src/data/preprocessing.py`)

   - Text normalization and standardization
   - Handling of redacted information (XXXX patterns)
   - Categorical variable standardization
   - Missing value handling
   - Date format standardization

3. **Data Quality Validation** (`tests/test_data_processing.py`)
   - Unit tests for cleaning functions
   - Data quality checks
   - Validation of cleaning results

### Tools and Libraries Used

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical operations
- **nltk**: Text processing and tokenization
- **matplotlib/seaborn**: Data visualization
- **re**: Regular expressions for text cleaning
- **unittest**: Testing framework

### Exported Datasets

1. **Complete Cleaned Dataset**

   - Location: `data/processed/cleaned_complaints.csv`
   - Contains all columns with standardized formats
   - Includes derived features and cleaned text

2. **RAG-Optimized Dataset**
   - Location: `data/processed/complaints_rag.csv`
   - Selected columns for RAG pipeline:
     - complaint_narrative (cleaned)
     - product/sub-product
     - issue/sub-issue
     - company
     - state
     - date_received
     - complaint_id

### Key Findings

1. **Data Quality**

   - Identified and handled missing values
   - Standardized product and company names
   - Cleaned and normalized complaint narratives
   - Removed duplicates and near-duplicates

2. **Text Characteristics**
   - Average complaint length
   - Common phrases and patterns
   - Redacted information handling
   - Text quality metrics

### How to Run

1. **Run EDA and Cleaning**

   ```bash
   jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
   ```

2. **Run Tests**

   ```bash
   python -m unittest tests/test_data_processing.py
   ```

3. **Use Preprocessing Module**
   ```python
   from src.data.preprocessing import clean_text, preprocess_text, standardize_categories
   ```

## Development

This project is organized into four main tasks:

1. ✅ Exploratory Data Analysis and Data Preprocessing
2. Text Chunking, Embedding, and Vector Store Indexing
3. Building the RAG Core Logic and Evaluation
4. Creating an Interactive Chat Interface
