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

    PROMPT_TEMPLATE = """You are an expert financial services analyst chatbot. Analyze the provided customer complaints and answer the question.

Context (relevant customer complaints):
{context}

Question: {question}

Guidelines:
1. Start with a clear introduction of what you found in the complaints data
2. Group similar complaints and identify key patterns
3. For each pattern:
   - Describe the issue
   - Provide a specific example from the complaints
   - Explain the impact on customers
4. Provide actionable recommendations based on the patterns
5. End with a brief conclusion

Format your response with clear paragraph breaks between sections. Use the actual content from the complaints - do not make assumptions or add issues that aren't present in the complaints.

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
        """Format chunks into context string."""
        context_parts = []

        for chunk in chunks:
            complaint_id = chunk.get('metadata', {}).get(
                'complaint_id', 'unknown')
            content = chunk.get('document', '')
            context_parts.append(
                f"Complaint {complaint_id}:\n{content.strip()}")

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
        """Generate a response using the complaint data."""
        if not chunks:
            return "I couldn't find any relevant complaints matching your query. Could you try rephrasing or asking about something else?"

        # Format context for LLM
        context = self._format_context(chunks)

        # Generate response using LLM
        prompt = self.PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )

        # Add a reminder to focus on the actual complaints
        prompt += "\n\nRemember: Base your analysis ONLY on the complaints provided. Do not include generic issues or assumptions not supported by the complaint data."

        response = self.llm.generate(prompt)

        # Add sources invisibly
        sources = [f"Complaint #{chunk.get('metadata', {}).get('complaint_id', 'Unknown')}"
                   for chunk in chunks[:3]]
        response += "\n<!-- Sources: " + ", ".join(sources) + " -->"

        return response

    def _extract_quote(self, text: str, max_length: int = 100) -> str:
        """Extract a clean, concise quote from the text."""
        # Clean up the text
        text = text.strip()
        if len(text) > max_length:
            # Find the last complete sentence within limit
            end_idx = text[:max_length].rfind('.')
            if end_idx == -1:
                end_idx = text[:max_length].rfind(' ')
            if end_idx == -1:
                end_idx = max_length
            text = text[:end_idx] + "..."

        # Clean up common issues in complaint text
        text = text.replace(" i ", " I ")  # Fix common pronoun
        text = text.capitalize()  # Capitalize first letter
        text = text.replace(" dont ", " don't ")  # Fix common contractions
        text = text.replace(" cant ", " can't ")
        text = text.replace(" didnt ", " didn't ")

        return text

    def query(self, question: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Process a question and return a response with supporting chunks.

        Args:
            question: User's question

        Returns:
            Tuple of (response text, supporting chunks)
        """
        try:
            chunks = self.retrieve(question)
            response = self.generate_response(question, chunks)
            return response, chunks[:3]
        except Exception as e:
            print(f"Error in RAG pipeline: {str(e)}")
            return (
                "I encountered an issue while processing your question. "
                "However, I can show you the relevant complaints I found. "
                "Would you like to see them?",
                chunks[:3] if chunks else []
            )

    def _extract_topics(self, chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """Extract main topics and their examples from chunks."""
        topics = {
            "Financial Impact": [],
            "Customer Service": [],
            "Process Issues": [],
            "Technical Problems": []
        }

        for chunk in chunks:
            if chunk.get('relevance', 0) > 0.2:
                content = chunk.get('document', '').lower()
                id = chunk.get('metadata', {}).get('complaint_id', 'Unknown')

                example = {'id': id, 'text': chunk.get('document', '')}

                if any(word in content for word in ['money', 'payment', 'fee', 'charge', 'cost']):
                    topics["Financial Impact"].append(example)
                if any(word in content for word in ['service', 'representative', 'support', 'help']):
                    topics["Customer Service"].append(example)
                if any(word in content for word in ['process', 'procedure', 'application', 'approval']):
                    topics["Process Issues"].append(example)
                if any(word in content for word in ['website', 'app', 'online', 'system']):
                    topics["Technical Problems"].append(example)

        # Remove empty topics
        return {k: v for k, v in topics.items() if v}

    def _identify_patterns(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Identify patterns in the complaints."""
        patterns = []

        # Count common themes
        themes = {}
        for chunk in chunks:
            content = chunk.get('document', '').lower()

            if 'credit' in content and 'report' in content:
                themes['credit_report'] = themes.get('credit_report', 0) + 1
            if 'delay' in content or 'time' in content:
                themes['delays'] = themes.get('delays', 0) + 1
            if 'communication' in content or 'response' in content:
                themes['communication'] = themes.get('communication', 0) + 1

        # Generate pattern insights
        if themes.get('credit_report', 0) > 1:
            patterns.append(
                "There's a recurring pattern of credit reporting issues affecting customers")
        if themes.get('delays', 0) > 1:
            patterns.append("Processing delays appear to be a common concern")
        if themes.get('communication', 0) > 1:
            patterns.append(
                "Communication problems are frequently mentioned in complaints")

        return patterns

    def _get_recent_examples(self, chunks: List[Dict[str, Any]]) -> List[Dict]:
        """Get the most recent complaint examples."""
        dated_chunks = []
        for chunk in chunks:
            date = chunk.get('metadata', {}).get('date', '2020-01-01')
            dated_chunks.append((date, chunk))

        dated_chunks.sort(reverse=True)
        return [chunk for _, chunk in dated_chunks[:2]]

    def _generate_insight(self, topic: str) -> str:
        """Generate an insight based on the topic."""
        insights = {
            "Financial Impact": "this is causing significant financial strain on customers",
            "Customer Service": "there may be room for improvement in customer support processes",
            "Process Issues": "the current procedures might need review and optimization",
            "Technical Problems": "there could be underlying system issues that need addressing"
        }
        return insights.get(topic, "this is an area that merits attention")

    def _generate_conclusion(self, topics: Dict[str, List[Dict]], patterns: List[str]) -> str:
        """Generate a conclusion based on topics and patterns."""
        n_topics = len(topics)
        n_patterns = len(patterns)

        if n_topics > 2 and n_patterns > 1:
            return "there are multiple systemic issues that need to be addressed"
        elif n_topics > 2:
            return "customers are facing several distinct challenges"
        elif n_patterns > 1:
            return "there are consistent patterns in customer experiences"
        else:
            return "while there are specific issues to address, they appear to be manageable with proper attention"
