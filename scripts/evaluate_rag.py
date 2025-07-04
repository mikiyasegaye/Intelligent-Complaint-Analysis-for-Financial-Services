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

# Add the parent directory to Python path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


try:
    from src.models.rag_pipeline import RAGPipeline
except ImportError as e:
    print(f"Error importing RAGPipeline: {e}")
    print(f"Python path: {sys.path}")
    sys.exit(1)

# Test questions covering different aspects of complaints analysis
TEST_QUESTIONS = [
    {
        "id": "Q1",
        "category": "Product Issues",
        "question": "What are the most common issues reported with credit cards?",
        "expected_themes": ["fees", "interest rates", "billing disputes"]
    },
    {
        "id": "Q2",
        "category": "Temporal Analysis",
        "question": "Have there been any noticeable trends in savings account complaints over time?",
        "expected_themes": ["temporal patterns", "issue frequency"]
    },
    {
        "id": "Q3",
        "category": "Company Comparison",
        "question": "How do complaint patterns differ between major credit card companies?",
        "expected_themes": ["company comparison", "issue types"]
    },
    {
        "id": "Q4",
        "category": "Customer Impact",
        "question": "What are the typical financial impacts reported in personal loan complaints?",
        "expected_themes": ["monetary impact", "customer hardship"]
    },
    {
        "id": "Q5",
        "category": "Resolution Analysis",
        "question": "How effective are banks at resolving money transfer complaints?",
        "expected_themes": ["resolution time", "customer satisfaction"]
    }
]


def evaluate_response(question: Dict[str, Any], response: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate a single RAG response.

    Args:
        question: Question dictionary with metadata
        response: Generated response
        chunks: Retrieved chunks

    Returns:
        Evaluation dictionary
    """
    # Extract key information from chunks
    chunk_info = []
    for chunk in chunks[:2]:  # Show top 2 chunks
        chunk_info.append({
            "id": chunk.get("metadata", {}).get("complaint_id", "Unknown"),
            "text": chunk.get("document", "")[:100] + "..."  # First 100 chars
        })

    return {
        "question_id": question["id"],
        "category": question["category"],
        "question": question["question"],
        "response": response,
        "chunks": chunk_info,
        "expected_themes": question["expected_themes"],
        # These would be filled in manually during analysis
        "quality_score": None,  # 1-5
        "comments": None
    }


def run_evaluation() -> List[Dict[str, Any]]:
    """Run the evaluation on all test questions.

    Returns:
        List of evaluation results
    """
    # Initialize RAG pipeline
    rag = RAGPipeline()

    # Run evaluation for each question
    results = []
    for question in TEST_QUESTIONS:
        print(
            f"\nProcessing question {question['id']}: {question['question']}")

        # Run RAG pipeline
        response, chunks = rag.query(question["question"])

        # Evaluate response
        eval_result = evaluate_response(question, response, chunks)
        results.append(eval_result)

        print(f"Retrieved {len(chunks)} chunks")

    return results


def save_results(results: List[Dict[str, Any]], output_dir: Path = None):
    """Save evaluation results.

    Args:
        results: List of evaluation results
        output_dir: Directory to save results (defaults to evaluations/rag)
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "evaluations" / "rag"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw results as JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"rag_evaluation_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Create markdown table
    markdown = ["# RAG Evaluation Results\n"]
    markdown.append("## Test Questions and Results\n")
    markdown.append(
        "| ID | Category | Question | Sample Retrieved Content | Quality Score | Comments |")
    markdown.append(
        "|----|-----------|-----------|-----------------------|---------------|----------|")

    for result in results:
        # Format chunk info
        chunks_text = "<br>".join([
            f"**{c['id']}**: {c['text']}"
            for c in result["chunks"]
        ])

        row = [
            result["question_id"],
            result["category"],
            result["question"],
            chunks_text,
            "TBD",  # Quality score to be filled manually
            "TBD"   # Comments to be filled manually
        ]
        markdown.append(f"| {' | '.join(row)} |")

    # Add analysis sections
    markdown.extend([
        "\n## Analysis\n",
        "### What Worked Well\n",
        "- TBD\n",
        "\n### Areas for Improvement\n",
        "- TBD\n",
        "\n### Recommendations\n",
        "- TBD\n"
    ])

    # Save markdown
    md_path = output_dir / f"rag_evaluation_{timestamp}.md"
    with open(md_path, "w") as f:
        f.write("\n".join(markdown))

    print(f"\nResults saved to:")
    print(f"- JSON: {json_path}")
    print(f"- Markdown: {md_path}")


if __name__ == "__main__":
    # Run evaluation
    results = run_evaluation()

    # Save results
    save_results(results)
