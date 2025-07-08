"""
Evaluation script for the RAG pipeline.
Tests the system's performance on a set of predefined questions.
"""

from config.config import VECTOR_STORE_DIR
from src.models.text_processor import TextProcessor
from src.models.vector_store import VectorStore
from src.models.rag_pipeline import RAGPipeline
import sys
from pathlib import Path
import json
from datetime import datetime
import time
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


# Test questions covering different aspects
TEST_QUESTIONS = [
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


def evaluate_pipeline(pipeline: RAGPipeline, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Evaluate the RAG pipeline on a set of test questions.

    Args:
        pipeline: Initialized RAG pipeline
        questions: List of test questions with metadata

    Returns:
        List of evaluation results
    """
    results = []

    for q in questions:
        print(f"\nEvaluating {q['question_id']}: {q['question']}")

        try:
            # Get response and chunks
            response, chunks = pipeline.query(q['question'])

            # Add to results
            result = {
                **q,  # Include original question data
                "response": response,
                "chunks": [
                    {
                        "id": chunk.get('metadata', {}).get('complaint_id', 'unknown'),
                        # First 200 chars
                        "text": chunk.get('document', '')[:200] + "..."
                    }
                    for chunk in chunks[:2]  # Include top 2 chunks
                ],
                "quality_score": None,  # To be filled manually
                "comments": None  # To be filled manually
            }
            results.append(result)

        except Exception as e:
            print(f"Error evaluating question {q['question_id']}: {str(e)}")
            results.append({
                **q,
                "response": f"Error: {str(e)}",
                "chunks": [],
                "quality_score": 0,
                "comments": f"Failed with error: {str(e)}"
            })

    return results


def save_results(results: List[Dict[str, Any]], output_dir: Path) -> None:
    """
    Save evaluation results to JSON and Markdown files.

    Args:
        results: Evaluation results
        output_dir: Directory to save results
    """
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_dir / f"rag_evaluation_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save Markdown
    md_path = output_dir / f"rag_evaluation_{timestamp}.md"
    with open(md_path, 'w') as f:
        f.write("# RAG System Evaluation Results\n\n")
        f.write(
            f"Evaluation performed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for result in results:
            f.write(f"## {result['question_id']}: {result['category']}\n\n")
            f.write(f"**Question:** {result['question']}\n\n")
            f.write(
                f"**Expected Themes:** {', '.join(result['expected_themes'])}\n\n")
            f.write(f"**Response:**\n{result['response']}\n\n")

            f.write("**Retrieved Sources:**\n")
            for chunk in result['chunks']:
                f.write(f"- ID {chunk['id']}: {chunk['text']}\n")
            f.write("\n")

            if result['quality_score'] is not None:
                f.write(f"**Quality Score:** {result['quality_score']}/5\n\n")

            if result['comments']:
                f.write(f"**Comments:** {result['comments']}\n\n")

            f.write("---\n\n")

    print(f"\nResults saved to:")
    print(f"- JSON: {json_path}")
    print(f"- Markdown: {md_path}")


def main():
    """Run the evaluation."""
    print("Initializing RAG pipeline...")
    vector_store = VectorStore(persist_directory=str(VECTOR_STORE_DIR))
    text_processor = TextProcessor()
    pipeline = RAGPipeline(vector_store, text_processor)

    print("\nRunning evaluation...")
    results = evaluate_pipeline(pipeline, TEST_QUESTIONS)

    print("\nSaving results...")
    save_results(results, output_dir=project_root / "evaluations" / "rag")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
