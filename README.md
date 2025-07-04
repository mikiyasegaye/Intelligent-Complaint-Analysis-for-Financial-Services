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
├── notebooks/         # Jupyter notebooks
│   ├── 01_exploratory_data_analysis.ipynb    # EDA and cleaning
│   └── 02_text_chunking_and_embedding.ipynb  # Text processing
├── src/              # Source code
│   ├── data/         # Data processing scripts
│   │   └── preprocessing.py
│   ├── models/       # Model-related code
│   │   ├── text_processor.py
│   │   └── vector_store.py
│   ├── utils/        # Utility functions
│   └── web/          # Web interface code
├── tests/            # Test files
│   ├── test_data_processing.py
│   ├── test_text_processor.py
│   └── test_vector_store.py
├── .cache/           # Cached computations
├── vector_store/     # Vector database files
├── config/           # Configuration files
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
   - Location: `data/processed/filtered_complaints.csv`
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

## Task 2: Text Processing and Vector Storage

### Completed Components

1. **Text Processing Pipeline** (`src/models/text_processor.py`)

   - Implemented RecursiveCharacterTextSplitter
   - Optimized chunk size (500 chars) and overlap (50 chars)
   - Preserved metadata for each chunk
   - Added validation for minimum chunk size
   - Efficient batch processing with progress tracking

2. **Embedding Generation System** (`src/models/text_processor.py`)

   - Using all-MiniLM-L6-v2 model
   - Hardware acceleration support (MPS/CUDA)
   - Efficient batch processing
   - Progress tracking and performance metrics
   - Caching mechanism for embeddings

3. **Vector Store Integration** (`src/models/vector_store.py`)
   - ChromaDB implementation
   - Persistent storage configuration
   - Batched document addition
   - Flexible query interface
   - Metadata preservation

### Tools and Libraries Used

- **sentence-transformers**: Embedding generation and similarity search
- **langchain**: Text chunking and document handling
- **chromadb**: Vector storage and querying
- **torch**: ML operations with hardware acceleration
- **joblib**: Caching and parallel processing
- **numpy**: Numerical operations
- **tqdm**: Progress tracking

### Generated Assets

1. **Vector Store Database**

   - Location: `vector_store/`
   - Contains: Embeddings, metadata, and indices
   - Format: ChromaDB persistent storage
   - Size: ~2GB for 1.3M chunks

2. **Cached Embeddings**
   - Location: `.cache/cached_embeddings.pkl`
   - Contains: Pre-computed embeddings
   - Format: Joblib compressed pickle
   - Purpose: Speed up repeated runs

### Performance Metrics

1. **Processing Statistics**

   - Total Documents: 443,472
   - Total Chunks: 1,336,041
   - Average Chunks per Document: 3.01

2. **Speed Metrics**
   - Processing Speed: 664.2 documents/second
   - Total Processing Time: ~33.5 minutes
   - Average Query Time: < 100ms

### How to Run

1. **Run Text Processing and Embedding**

   ```bash
   jupyter notebook notebooks/02_text_chunking_and_embedding.ipynb
   ```

2. **Run Tests**

   ```bash
   python -m unittest tests/test_text_processor.py tests/test_vector_store.py
   ```

3. **Use Core Modules**

   ```python
   from src.models.text_processor import TextProcessor
   from src.models.vector_store import VectorStore

   # Process documents
   processor = TextProcessor()
   chunks = processor.create_chunks(text, metadata)
   ids, metadatas, embeddings = processor.generate_embeddings(chunks)

   # Query vector store
   store = VectorStore(persist_directory="vector_store")
   results = store.query(query_embedding, n_results=5)
   ```

## Development Status

This project is organized into four main tasks:

1. ✅ Exploratory Data Analysis and Data Preprocessing
2. ✅ Text Chunking, Embedding, and Vector Store Indexing
3. Building the RAG Core Logic and Evaluation
4. Creating an Interactive Chat Interface
