"""
Script to evaluate the RAG pipeline performance.

This script:
1. Defines a set of representative test questions
2. Runs each through the RAG pipeline
3. Generates an evaluation report
"""

from datetime import datetime
import json
from typing import List, Dict, Any
import pandas as pd
import sys
from pathlib import Path
import time

# Add the parent directory to Python path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


try:
    from src.models.rag_pipeline import RAGPipeline
    from src.models.vector_store import VectorStore
    from src.models.text_processor import TextProcessor
    from config.config import VECTOR_STORE_DIR
except ImportError as e:
    print(f"Error importing RAGPipeline: {e}")
    print(f"Python path: {sys.path}")
    sys.exit(1)


def load_test_questions() -> List[Dict[str, Any]]:
    """Load test questions for evaluation.

    Returns:
        List of test questions with metadata
    """
    return [
        {
            "question_id": "Q1",
            "category": "Product Issues",
            "question": "What are the most common issues reported with credit cards?",
            "expected_themes": ["fees", "interest rates", "billing disputes"]
        },
        {
            "question_id": "Q2",
            "category": "Temporal Analysis",
            "question": "Have there been any noticeable trends in savings account complaints over time?",
            "expected_themes": ["temporal patterns", "issue frequency"]
        },
        {
            "question_id": "Q3",
            "category": "Company Comparison",
            "question": "How do complaint patterns differ between major credit card companies?",
            "expected_themes": ["company comparison", "issue types"]
        },
        {
            "question_id": "Q4",
            "category": "Customer Impact",
            "question": "What are the typical financial impacts reported in personal loan complaints?",
            "expected_themes": ["monetary impact", "customer hardship"]
        },
        {
            "question_id": "Q5",
            "category": "Resolution Analysis",
            "question": "How effective are banks at resolving money transfer complaints?",
            "expected_themes": ["resolution time", "customer satisfaction"]
        }
    ]


def evaluate_pipeline(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Evaluate RAG pipeline on test questions.

    Args:
        questions: List of test questions

    Returns:
        Evaluation results for each question
    """
    # Initialize components
    vector_store = VectorStore(persist_directory=str(VECTOR_STORE_DIR))
    text_processor = TextProcessor()
    pipeline = RAGPipeline(vector_store, text_processor)

    results = []

    for question in questions:
        print(
            f"\nProcessing question {question['question_id']}: {question['question']}")

        # Get response
        chunks = pipeline.retrieve(question["question"])
        print(f"Retrieved {len(chunks)} chunks")

        response = pipeline.generate_response(
            question["question"],
            chunks
        )

        # Store results
        result = {
            "question_id": question["question_id"],
            "category": question["category"],
            "question": question["question"],
            "response": response,
            "chunks": [
                {
                    "id": chunk.get("metadata", {}).get("complaint_id", "unknown"),
                    "text": chunk.get("document", "")[:100] + "..."
                }
                for chunk in chunks[:2]  # Only store first 2 chunks
            ],
            "expected_themes": question["expected_themes"],
            "quality_score": None,  # To be filled manually
            "comments": None  # To be filled manually
        }
        results.append(result)

    return results


def save_results(results: List[Dict[str, Any]]) -> None:
    """Save evaluation results to files.

    Args:
        results: Evaluation results to save
    """
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory
    output_dir = Path("evaluations/rag")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON results
    json_path = output_dir / f"rag_evaluation_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save markdown summary
    md_path = output_dir / f"rag_evaluation_{timestamp}.md"
    with open(md_path, "w") as f:
        f.write("# RAG Evaluation Results\n\n")
        f.write("## Test Questions and Results\n\n")

        # Create results table
        f.write(
            "| ID | Category | Question | Sample Retrieved Content | Quality Score | Comments |\n")
        f.write(
            "|----|-----------|-----------|-----------------------|---------------|----------|\n")

        for result in results:
            chunks_text = "<br>".join([
                f"**{chunk['id']}**: {chunk['text']}"
                for chunk in result["chunks"]
            ])
            f.write(
                f"| {result['question_id']} "
                f"| {result['category']} "
                f"| {result['question']} "
                f"| {chunks_text} "
                f"| TBD "
                f"| TBD |\n"
            )

        # Add analysis sections
        f.write("\n## Analysis\n\n")
        f.write("### What Worked Well\n\n- TBD\n\n")
        f.write("### Areas for Improvement\n\n- TBD\n\n")
        f.write("### Recommendations\n\n- TBD\n")

    print("\nResults saved to:")
    print(f"- JSON: {json_path}")
    print(f"- Markdown: {md_path}")


def main():
    """Run the evaluation."""
    questions = load_test_questions()
    results = evaluate_pipeline(questions)
    save_results(results)


if __name__ == "__main__":
    main()
