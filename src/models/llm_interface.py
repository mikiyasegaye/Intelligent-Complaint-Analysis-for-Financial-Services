"""
LLM Interface module for text generation.

This module provides a unified interface for different LLM backends:
- Local models via transformers
- API-based models (if available)
- Mock LLM for testing
"""

from typing import Optional, Dict, Any
from pathlib import Path
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))


class LLMInterface:
    """Interface for LLM-based text generation."""

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: str = "auto",
        max_length: int = 2048,
        temperature: float = 0.7
    ):
        """Initialize the LLM interface.

        Args:
            model_name: Name/path of the model to use
            device: Device to run on ('cpu', 'cuda', 'mps', or 'auto')
            max_length: Maximum length of generated text
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        # Initialize model and tokenizer
        print(f"\nLoading LLM: {model_name}")
        print(f"Using device: {device}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16
            )

            # Move model to device after loading
            if device != "cpu":
                self.model = self.model.to(device)

            # Create generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map=device if device == "cuda" else None  # Only use device_map for CUDA
            )

            print("LLM loaded successfully")

        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to mock LLM")
            self.pipeline = self._mock_pipeline

    def _mock_pipeline(self, prompt: str, **kwargs) -> list:
        """Mock pipeline for testing without a real LLM.

        Args:
            prompt: Input prompt
            **kwargs: Additional arguments

        Returns:
            List containing generated text
        """
        return [{
            "generated_text": (
                "This is a mock response. In a real implementation, "
                "this would be replaced with actual LLM-generated text "
                "based on the provided context and question."
            )
        }]

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate text based on the prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Override default temperature
            **kwargs: Additional arguments for the pipeline

        Returns:
            Generated text
        """
        # Use defaults if not specified
        if max_new_tokens is None:
            max_new_tokens = self.max_length
        if temperature is None:
            temperature = self.temperature

        # Generate text
        result = self.pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        )[0]

        return result["generated_text"]
