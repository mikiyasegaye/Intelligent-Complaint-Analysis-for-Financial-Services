"""
RAG Pipeline implementation for financial complaints analysis.

This module implements the core RAG (Retrieval Augmented Generation) logic including:
- Question embedding and retrieval
- Prompt engineering
- Response generation
"""

from src.models.llm_interface import LLMInterface
from src.models.vector_store import VectorStore
from src.models.text_processor import TextProcessor
from config.config import TOP_K_CHUNKS
from pathlib import Path
import sys
from typing import List, Dict, Any, Tuple
from langchain.schema import Document

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))


class RAGPipeline:
    """Implements the RAG pipeline for financial complaints analysis."""

    # Base prompt template
    PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints.
    
Use ONLY the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain enough information to fully answer the question, clearly state what information is missing.

Context:
{context}

Question: {question}

Instructions:
1. Base your answer ONLY on the provided context
2. If the context is insufficient, say so
3. If you cite specific complaints, reference them by their ID
4. Be concise but thorough
5. If there are multiple relevant complaints, summarize the common themes

Answer:"""

    def __init__(
        self,
        vector_store_path: str = "vector_store",
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: str = "auto"
    ):
        """Initialize the RAG pipeline.

        Args:
            vector_store_path: Path to the vector store directory
            model_name: Name of the LLM to use
            device: Device to run LLM on
        """
        self.text_processor = TextProcessor()
        self.vector_store = VectorStore(persist_directory=vector_store_path)
        self.llm = LLMInterface(
            model_name=model_name,
            device=device
        )

    def retrieve(self, question: str, k: int = TOP_K_CHUNKS) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a given question.

        Args:
            question: The user's question
            k: Number of chunks to retrieve

        Returns:
            List of retrieved chunks with their metadata
        """
        # Generate embedding for the question
        question_doc = Document(page_content=question, metadata={
                                "complaint_id": "query", "chunk_index": 0})
        _, _, embeddings = self.text_processor.generate_embeddings(
            [question_doc],
            show_progress=False
        )
        question_embedding = embeddings[0]

        # Retrieve similar chunks
        results = self.vector_store.query(
            query_embedding=question_embedding,
            n_results=k
        )

        # Format results into list of dictionaries
        formatted_results = []
        # ChromaDB returns nested lists
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })

        return formatted_results

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into context string.

        Args:
            chunks: List of retrieved chunks with metadata

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            # Format metadata
            metadata = chunk.get('metadata', {})
            complaint_id = metadata.get('complaint_id', 'Unknown')
            product = metadata.get('product', 'Unknown Product')
            date = metadata.get('date_received', 'Unknown Date')

            # Format chunk text
            text = chunk.get('document', '').strip()
            distance = chunk.get('distance', 0.0)

            # Combine into a formatted string
            chunk_text = f"[Complaint {i}]\nID: {complaint_id}\nProduct: {product}\nDate: {date}\nRelevance: {1.0 - distance:.2f}\nContent: {text}\n"
            context_parts.append(chunk_text)

        return "\n\n".join(context_parts)

    def generate_response(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        """Generate a response using the LLM.

        Args:
            question: The user's question
            chunks: Retrieved relevant chunks

        Returns:
            Generated response
        """
        # Format the context
        context = self._format_context(chunks)

        # Create the full prompt
        prompt = self.PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )

        # Generate response using LLM
        response = self.llm.generate(
            prompt,
            max_new_tokens=512,  # Reasonable length for answers
            temperature=0.7      # Balanced between creativity and consistency
        )

        return response

    def query(self, question: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Process a question through the full RAG pipeline.

        Args:
            question: The user's question

        Returns:
            Tuple of (generated response, retrieved chunks)
        """
        # Retrieve relevant chunks
        chunks = self.retrieve(question)

        # Generate response
        response = self.generate_response(question, chunks)

        return response, chunks
