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
├── config/                 # Configuration files
│   └── config.py          # Project-wide configuration settings
├── data/                  # Data files
│   ├── processed/         # Cleaned and processed data
│   │   ├── cleaned_complaints.csv    # Initial cleaned data (1.9GB)
│   │   ├── processed_complaints.csv  # Intermediate processing (1.1GB)
│   │   └── filtered_complaints.csv   # Final RAG dataset (1.0GB)
│   └── raw/               # Original, immutable data
│       └── complaints.csv            # Raw complaints data (5.6GB)
├── notebooks/             # Jupyter notebooks
│   ├── 01_exploratory_data_analysis.ipynb    # EDA and data cleaning
│   └── 02_text_chunking_and_embedding.ipynb  # Text processing and embeddings
├── scripts/               # Analysis and utility scripts
│   └── data_analysis.py  # Data filtering analysis script
├── src/                  # Source code
│   ├── data/            # Data processing modules
│   │   └── preprocessing.py          # Data cleaning pipeline
│   ├── models/          # Model-related code
│   │   ├── text_processor.py         # Text chunking and embedding
│   │   └── vector_store.py           # Vector database operations
│   ├── utils/           # Utility functions
│   └── web/             # Web interface code
├── tests/               # Test files
│   ├── test_data_processing.py       # Data pipeline tests
│   ├── test_text_processor.py        # Text processing tests
│   └── test_vector_store.py          # Vector store tests
├── vector_store/        # Vector database files
└── requirements.txt     # Project dependencies
```

## Data Pipeline Overview

The project processes the CFPB complaints dataset through several stages:

1. **Raw Data** (`data/raw/complaints.csv`)

   - Size: 5.6GB
   - Records: 9.6M complaints
   - Format: Original CFPB CSV format
   - All financial products and complaints

2. **Cleaned Data** (`data/processed/cleaned_complaints.csv`)

   - Size: 1.9GB
   - Initial cleaning and standardization
   - All records with basic preprocessing
   - Normalized text and categories

3. **Processed Data** (`data/processed/processed_complaints.csv`)

   - Size: 1.1GB
   - Filtered to target product categories
   - Enhanced text preprocessing
   - Standardized formats

4. **RAG Dataset** (`data/processed/filtered_complaints.csv`)
   - Size: 1.0GB
   - Records: 443K complaints
   - Only complaints with narratives
   - Focus on 5 product categories:
     - Credit Cards (433K complaints)
     - Personal Loans (5.5K complaints)
     - Savings/Bank Accounts (377K complaints)
     - Money Transfers (145K complaints)
     - Buy Now, Pay Later (BNPL)

## Setup and Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd Intelligent-Complaint-Analysis-for-Financial-Services
```

2. Create and activate virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download the data:
   - Request access to the CFPB complaints dataset
   - Place the complaints.csv file in `data/raw/`
   - Run the preprocessing pipeline:
     ```bash
     python -m src.data.preprocessing
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
   - Size: 1.9 GB
   - Contains all columns with standardized formats
   - Includes derived features and cleaned text

2. **Intermediate Processing Dataset**

   - Location: `data/processed/processed_complaints.csv`
   - Size: 1.1 GB
   - Contains filtered products and standardized formats
   - Intermediate step before final RAG preparation

3. **RAG-Optimized Dataset**
   - Location: `data/processed/filtered_complaints.csv`
   - Size: 1.0 GB
   - Selected columns for RAG pipeline:
     - complaint_narrative (cleaned)
     - product/sub-product
     - issue/sub-issue
     - company
     - state
     - date_received
     - complaint_id
   - Focused on 5 target product categories
   - Contains only complaints with valid narratives

### Key Findings

1. **Data Quality**

   - Identified and handled missing values
   - Standardized product and company names
   - Cleaned and normalized complaint narratives
   - Removed duplicates and near-duplicates

2. **Data Filtering Process**

   - Raw dataset: 9.6M complaints (5.6 GB on disk)
   - Filtered to target products: 961K complaints
   - Final RAG dataset: 443K complaints (1.20 GB)
   - Filtering criteria:
     - Focus on 5 product categories (Credit Cards, Personal Loans, BNPL, Savings Accounts, Money Transfers)
     - Require non-empty complaint narratives
     - Validated zero data loss during filtering process

3. **Text Characteristics**
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

## Task 3: Building the RAG Core Logic and Evaluation

### Completed Components

1. **RAG Pipeline Implementation** (`src/models/rag_pipeline.py`)
   - Integrated retriever with similarity search
   - Robust prompt template with validation
   - Response generation with LLM
   - Comprehensive evaluation framework
   - Average quality score: 3/5

### Supporting Modules

1. **LLM Interface** (`src/models/llm_interface.py`)

   - Handles model interactions
   - Manages generation parameters
   - Error handling and retries

2. **Text Processing** (`src/models/text_processor.py`)

   - Document chunking
   - Embedding generation
   - Text cleaning and validation

3. **Vector Store** (`src/models/vector_store.py`)
   - Similarity search implementation
   - Vector storage management
   - Query optimization

### Performance Metrics

1. **Processing Statistics**

   - Total Documents: 443,472
   - Total Chunks: 1,336,041
   - Average Chunks per Document: 3.01

2. **Speed Metrics**
   - Processing Speed: ~664 documents/second
   - Average Query Time: < 100ms

### How to Run

1. **Run Evaluation**

   ```bash
   python scripts/evaluate_rag.py
   ```

2. **Use RAG Pipeline**

   ```python
   from src.models.rag_pipeline import RAGPipeline
   from src.models.vector_store import VectorStore
   from src.models.text_processor import TextProcessor

   # Initialize components
   vector_store = VectorStore(persist_directory="vector_store")
   text_processor = TextProcessor()
   pipeline = RAGPipeline(vector_store, text_processor)

   # Run query
   response, chunks = pipeline.query("What are common credit card issues?")
   ```

## Development Status

This project is organized into four main tasks:

1. ✅ Exploratory Data Analysis and Data Preprocessing
2. ✅ Text Chunking, Embedding, and Vector Store Indexing
3. ✅ Building the RAG Core Logic and Evaluation
   - Implemented retriever with similarity search
   - Designed robust prompt template
   - Created generator with validation
   - Completed evaluation with 5 test questions
   - Average quality score: 3/5
4. Creating an Interactive Chat Interface
