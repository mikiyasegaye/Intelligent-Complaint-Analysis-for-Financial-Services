"""
Vector store operations for managing and querying document embeddings.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection


class VectorStore:
    def __init__(
        self,
        persist_directory: str,
        collection_name: str = "financial_complaints",
        collection_metadata: Optional[Dict] = None
    ):
        """
        Initialize the vector store.

        Args:
            persist_directory (str): Directory to persist the vector store
            collection_name (str): Name of the collection to use
            collection_metadata (Dict, optional): Metadata for the collection
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name

        # Ensure persist directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            persist_directory=str(self.persist_directory),
            is_persistent=True
        ))

        # Get or create collection
        try:
            self.collection = self.client.get_collection(collection_name)
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata=collection_metadata or {
                    "description": "Financial complaints embeddings"}
            )

    def add_documents(
        self,
        ids: List[str],
        embeddings: List[np.ndarray],
        metadatas: List[Dict],
        documents: List[str],
        batch_size: int = 1000
    ) -> None:
        """
        Add documents to the vector store in batches.

        Args:
            ids (List[str]): List of document IDs
            embeddings (List[np.ndarray]): List of document embeddings
            metadatas (List[Dict]): List of document metadata
            documents (List[str]): List of document texts
            batch_size (int): Number of documents to add in each batch
        """
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            batch_documents = documents[i:i + batch_size]

            self.collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                documents=batch_documents
            )

    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None,
        include: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Query the vector store for similar documents.

        Args:
            query_embedding (np.ndarray): Query embedding vector
            n_results (int): Number of results to return
            where (Dict, optional): Metadata filters
            where_document (Dict, optional): Document content filters
            include (List[str], optional): What to include in results

        Returns:
            Dict[str, Any]: Query results containing documents, metadata, and distances
        """
        if include is None:
            include = ['documents', 'metadatas', 'distances']

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=include
        )

    @property
    def count(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()

    def peek(self, n: int = 5) -> Dict[str, Any]:
        """
        Get a sample of documents from the collection.

        Args:
            n (int): Number of documents to retrieve

        Returns:
            Dict[str, Any]: Sample documents with their metadata
        """
        return self.collection.peek(limit=n)
