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

## Usage

[To be added as development progresses]

## Development

This project is organized into four main tasks:

1. Exploratory Data Analysis and Data Preprocessing
2. Text Chunking, Embedding, and Vector Store Indexing
3. Building the RAG Core Logic and Evaluation
4. Creating an Interactive Chat Interface
