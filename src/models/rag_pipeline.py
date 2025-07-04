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

    # Configuration
    MIN_RELEVANCE_SCORE = 0.3
    MAX_RESPONSE_LENGTH = 300
    TEMPERATURE = 0.2

    # Base prompt template
    PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints using ONLY the provided context.

Context:
{context}

Question: {question}

Instructions:
1. Base your answer ONLY on the provided context - do not make assumptions or add external knowledge
2. If the context doesn't contain enough information, clearly state what's missing
3. For each claim, quote the relevant text and cite its complaint ID
4. Keep your response under 300 words
5. Use bullet points for better readability
6. Focus on complaints with higher relevance scores (ignore scores below 0.3)
7. If you notice contradictions or inconsistencies, point them out
8. If multiple complaints show a pattern, summarize it with supporting quotes

Example Good Response:
• Based on complaint ID 12345 (relevance 0.8): "customers reported frequent billing errors" showing issues with transaction processing
• Limited information about resolution times - only one complaint (ID 67890, relevance 0.7) mentions "resolved within 24 hours"
• Cannot determine overall trends as context only covers 2022-2023

Example Bad Response:
• Most customers are satisfied (not supported by context)
• Banks typically resolve issues quickly (contradicts available evidence)
• Response times vary between 1-5 days (making up specific numbers)

Remember: It's better to acknowledge limited information than to make unsupported claims.

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

        # Retrieve more chunks than needed to filter by relevance
        results = self.vector_store.query(
            query_embedding=question_embedding,
            n_results=k * 2  # Get more chunks to filter
        )

        # Format and filter results
        formatted_results = []
        seen_content = set()  # For deduplication

        for i in range(len(results['ids'][0])):
            distance = results['distances'][0][i]
            relevance = 1.0 - distance

            # Skip low relevance chunks
            if relevance < self.MIN_RELEVANCE_SCORE:
                continue

            content = results['documents'][0][i]

            # Skip duplicate content
            content_hash = hash(content)
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)

            formatted_results.append({
                'document': content,
                'metadata': results['metadatas'][0][i],
                'distance': distance,
                'relevance': relevance
            })

        # Return top k unique, relevant results
        return formatted_results[:k]

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

    def _validate_response(self, response: str, chunks: List[Dict[str, Any]]) -> bool:
        """Validate that response only contains information from chunks.

        Args:
            response: Generated response
            chunks: Retrieved chunks used for generation

        Returns:
            bool: Whether response is valid
        """
        # Extract all complaint IDs mentioned in response
        import re
        cited_ids = set(re.findall(r'ID: (\d+)', response))

        # Get actual chunk IDs
        chunk_ids = set(
            chunk.get('metadata', {}).get('complaint_id', '')
            for chunk in chunks
        )

        # Check if response cites non-existent chunks
        valid_ids = all(cid in chunk_ids for cid in cited_ids)

        # Check if response length is within limits
        valid_length = len(response.split()) <= self.MAX_RESPONSE_LENGTH

        # Check for quoted content
        quotes = re.findall(r'"([^"]*)"', response)
        valid_quotes = all(
            any(quote in chunk.get('document', '') for chunk in chunks)
            for quote in quotes
        )

        return valid_ids and valid_length and valid_quotes

    def generate_response(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        """Generate a response using the LLM.

        Args:
            question: The user's question
            chunks: Retrieved relevant chunks

        Returns:
            Generated response
        """
        # Sort chunks by relevance
        chunks = sorted(chunks, key=lambda x: x.get('distance', 1.0))

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
            max_new_tokens=self.MAX_RESPONSE_LENGTH,
            temperature=self.TEMPERATURE
        )

        # Validate response
        if not self._validate_response(response, chunks):
            # If validation fails, regenerate with stricter parameters
            response = self.llm.generate(
                prompt,
                max_new_tokens=self.MAX_RESPONSE_LENGTH,
                temperature=self.TEMPERATURE * 0.5  # Even more conservative
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
