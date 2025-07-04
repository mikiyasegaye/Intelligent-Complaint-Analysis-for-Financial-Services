"""
Text processing and embedding generation module for financial complaints analysis.
"""

import torch
from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np
from tqdm import tqdm
import time

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from sentence_transformers import SentenceTransformer


class TextProcessor:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = None):
        """
        Initialize the text processor with specified model and device.

        Args:
            model_name (str): Name of the sentence transformer model
            device (str): Device to use for computation ('cpu', 'cuda', or 'mps')
        """
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.device = device
        self.model_name = model_name
        self.embedding_model = SentenceTransformer(model_name, device=device)

    def create_chunks(
        self,
        text: str,
        metadata: Dict,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_chunk_size: int = 50
    ) -> List[Document]:
        """
        Split text into chunks while preserving context.

        Args:
            text (str): Text to split into chunks
            metadata (Dict): Metadata to attach to each chunk
            chunk_size (int): Maximum size of each chunk
            chunk_overlap (int): Number of characters to overlap between chunks
            min_chunk_size (int): Minimum size for a chunk to be included

        Returns:
            List[Document]: List of Document objects containing chunks and metadata
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        texts = splitter.split_text(text)
        texts = [text for text in texts if len(text.strip()) > min_chunk_size]

        return [
            Document(
                page_content=text,
                metadata={**metadata, "chunk_index": i,
                          "total_chunks": len(texts)}
            )
            for i, text in enumerate(texts)
        ]

    def generate_embeddings(
        self,
        documents: List[Document],
        batch_size: int = 64,
        show_progress: bool = True
    ) -> Tuple[List[str], List[Dict], List[np.ndarray]]:
        """
        Generate embeddings for a list of documents.

        Args:
            documents (List[Document]): List of documents to generate embeddings for
            batch_size (int): Number of documents to process in each batch
            show_progress (bool): Whether to show progress bar

        Returns:
            Tuple[List[str], List[Dict], List[np.ndarray]]: Tuple containing:
                - List of document IDs
                - List of document metadata
                - List of document embeddings
        """
        ids = [f"chunk_{doc.metadata['complaint_id']}_{doc.metadata['chunk_index']}"
               for doc in documents]
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        embeddings = []
        total_time = 0
        n_batches = (len(texts) + batch_size - 1) // batch_size

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=n_batches,
                            desc="Generating embeddings")

        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            start = time.time()
            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=batch_size
            )
            embeddings.extend(batch_embeddings)
            total_time += time.time() - start

        if show_progress:
            print(f"Total time: {total_time:.2f}s | "
                  f"Avg speed: {len(texts)/total_time:.1f} docs/s")

        return ids, metadatas, embeddings
