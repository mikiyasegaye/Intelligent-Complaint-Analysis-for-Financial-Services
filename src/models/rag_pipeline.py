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
from config.config import MODEL_NAME, TEMPERATURE, MAX_TOKENS
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
    BASE_RELEVANCE_THRESHOLD = 0.2
    MIN_CHUNKS = 3
    MAX_RESPONSE_LENGTH = MAX_TOKENS
    TEMPERATURE = TEMPERATURE

    # Simplified prompt template
    PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints using ONLY the provided context.

Context:
{context}

Question: {question}

Instructions:
1. Base your answer ONLY on the provided context - do not make assumptions
2. For each claim, quote the exact text and cite the complaint ID
3. Keep your response under 300 words
4. Use bullet points for better readability
5. Focus on complaints with higher relevance scores

Response Format:
• Main Findings:
  - Finding 1 supported by quote from ID X (relevance Y)
  - Finding 2 supported by quote from ID X (relevance Y)

• Data Limitations:
  - What information is missing
  - What time period is covered

Remember: Only make claims that are directly supported by quotes from the context.

Answer:"""

    def __init__(
        self,
        vector_store: VectorStore,
        text_processor: TextProcessor,
        model_name: str = MODEL_NAME
    ):
        """Initialize the RAG pipeline.

        Args:
            vector_store: Vector store for retrieving relevant chunks
            text_processor: Text processor for handling documents
            model_name: Name of the LLM to use
        """
        self.vector_store = vector_store
        self.text_processor = text_processor
        self.llm = LLMInterface(model_name=model_name)

    def retrieve(self, question: str, k: int = 10) -> List[Dict[str, Any]]:
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

        # Start with base threshold
        threshold = self.BASE_RELEVANCE_THRESHOLD
        min_chunks_found = False

        while not min_chunks_found:
            # Retrieve chunks
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
                if relevance < threshold:
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

            # Check if we have enough chunks
            if len(formatted_results) >= self.MIN_CHUNKS or threshold <= 0.1:
                min_chunks_found = True
            else:
                # Lower threshold and try again
                threshold -= 0.05

        # Return top k unique, relevant results
        return formatted_results[:k]

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format chunks into context string.

        Args:
            chunks: List of chunks with their metadata

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            # Get metadata
            metadata = chunk.get('metadata', {})
            complaint_id = metadata.get('complaint_id', 'unknown')
            product = metadata.get('product', 'unknown')
            date = metadata.get('date', 'unknown')
            relevance = chunk.get('relevance', 0.0)

            # Format chunk
            context_parts.append(
                f"[Complaint {i}]\n"
                f"ID: {complaint_id}\n"
                f"Product: {product}\n"
                f"Date: {date}\n"
                f"Relevance: {relevance:.2f}\n"
                f"Content: {chunk.get('document', '')}\n"
            )

        return "\n\n".join(context_parts)

    def _validate_response(self, response: str, chunks: List[Dict[str, Any]]) -> bool:
        """Validate that response only contains information from chunks and follows template.

        Args:
            response: Generated response
            chunks: Retrieved chunks used for generation

        Returns:
            bool: Whether response is valid
        """
        # Extract all complaint IDs mentioned in response
        import re
        cited_ids = set(re.findall(r'ID (\d+)', response))

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

        # Check for proper quote citation format
        quote_patterns = [
            # "quote" from ID X (relevance Y)
            r'"[^"]+" from ID \d+ \(relevance \d+\.\d+\)',
            # "quote" (ID X, relevance Y)
            r'"[^"]+" \(ID \d+, relevance \d+\.\d+\)',
        ]
        has_proper_citations = any(
            re.search(pattern, response)
            for pattern in quote_patterns
        )

        # Check for template sections
        required_sections = [
            "Main Findings:",
            "Data Limitations:"
        ]
        has_template = all(
            section in response for section in required_sections)

        # Check bullet point formatting
        # At least one bullet per section
        has_bullets = response.count("•") >= 2

        # Check for unsupported claims
        unsupported_phrases = [
            "most customers",
            "typically",
            "usually",
            "generally",
            "always",
            "never",
            "all customers",
            "no customers",
            "many customers",
            "few customers",
            "several customers",
            "multiple customers"
        ]
        no_unsupported = not any(phrase in response.lower()
                                 for phrase in unsupported_phrases)

        # Check for repetition (same sentence appearing multiple times)
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        sentence_counts = {}
        for s in sentences:
            if s in sentence_counts:
                sentence_counts[s] += 1
            else:
                sentence_counts[s] = 1
        no_repetition = all(count == 1 for count in sentence_counts.values())

        # Check for proper finding format
        finding_pattern = r'Finding \d+ supported by quote'
        has_proper_findings = bool(re.search(finding_pattern, response))

        # All checks must pass
        return (valid_ids and valid_length and valid_quotes and no_repetition
                and has_template and has_bullets and no_unsupported
                and has_proper_citations and has_proper_findings)

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
                temperature=self.TEMPERATURE * 0.5  # More conservative
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
