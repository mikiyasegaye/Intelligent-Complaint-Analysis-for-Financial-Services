"""
Gradio interface for the Financial Complaints RAG system.
Provides an interactive chat interface with source citation.
"""

from ..models.text_processor import TextProcessor
from ..models.vector_store import VectorStore
from ..models.rag_pipeline import RAGPipeline
from config.config import VECTOR_STORE_DIR
import time
import torch
import gradio as gr
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))


# Configure device and environment
device = "cpu"  # Force CPU usage for stability
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid deadlocks
os.environ['USE_CPU'] = '1' if device == "cpu" else '0'

# Initialize components
vector_store = VectorStore(persist_directory=str(VECTOR_STORE_DIR))
text_processor = TextProcessor()

# Initialize RAG Pipeline with required components
rag_pipeline = RAGPipeline(vector_store=vector_store,
                           text_processor=text_processor)


def analyze_complaints(message, history):
    """Process the question and return analysis from RAG pipeline with streaming"""
    # Get the response from RAG pipeline
    response, chunks = rag_pipeline.query(message)

    # Split response and sources
    main_response = response.split("<!-- Sources:")[0].strip()
    sources = response.split(
        "<!-- Sources:")[1].strip(" -->") if "<!-- Sources:" in response else ""

    # Stream the response word by word
    words = main_response.split()
    partial_response = ""

    for word in words:
        partial_response += word + " "
        time.sleep(0.05)  # Add a small delay for a natural typing effect
        yield partial_response


def clear_text():
    """Clear the input and output text"""
    return "", "Ask a question to see the analysis..."


# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # CrediTrust Financial Insights
    Ask questions about customer complaints and get AI-powered analysis of feedback patterns.
    """)

    chatbot = gr.Chatbot(
        value=[],
        label="Analysis Chat",
        height=400
    )

    msg = gr.Textbox(
        placeholder="Ask about customer complaints...",
        lines=2,
        show_label=False
    )

    with gr.Row():
        submit = gr.Button("Ask", variant="primary")
        clear = gr.Button("Clear")

    def user_request(message, history):
        return "", history + [[message, None]]

    def bot_response(history):
        history[-1][1] = ""
        for chunk in analyze_complaints(history[-1][0], history[:-1]):
            history[-1][1] = chunk
            yield history

    submit.click(
        user_request,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False
    ).then(
        bot_response,
        inputs=[chatbot],
        outputs=[chatbot],
        queue=True
    )

    clear.click(lambda: ([], ""), outputs=[chatbot, msg], queue=False)

# Launch the interface
if __name__ == "__main__":
    demo.queue().launch(share=False)
