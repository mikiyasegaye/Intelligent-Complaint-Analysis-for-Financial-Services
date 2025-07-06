"""
LLM Interface module for text generation.

This module provides a unified interface for different LLM backends:
- Local models via transformers
- API-based models (if available)
- Mock LLM for testing
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
from openai import OpenAI

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))


class LLMInterface:
    """Interface for interacting with language models."""

    def __init__(self, model_name: str = "gpt-3.5-turbo", device: str = "auto"):
        """Initialize the LLM interface.

        Args:
            model_name: Name/path of the model to use
            device: Device to run model on ("cpu", "cuda", "mps", or "auto")
        """
        self.model_name = model_name
        self.device = self._get_device(device)

        # Set up model based on type
        if "gpt-3.5" in model_name or "gpt-4" in model_name:
            self._setup_openai()
        else:
            self._setup_local_model()

    def _setup_openai(self):
        """Set up OpenAI API client."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)
        self.is_openai = True

    def _setup_local_model(self):
        """Set up local HuggingFace model."""
        print(f"\nLoading LLM: {self.model_name}")
        print(f"Using device: {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # Configure model loading for better stability
        model_kwargs = {
            "torch_dtype": torch.float32,
            "trust_remote_code": True,
        }

        # Use device_map="auto" only for larger models
        if "llama" in self.model_name.lower() or "mistral" in self.model_name.lower():
            model_kwargs["device_map"] = "auto"
            # Use 8-bit quantization for memory efficiency
            model_kwargs["load_in_8bit"] = True
        else:
            # For smaller models, use specific device
            if self.device != "cpu":
                model_kwargs["device_map"] = {"": self.device}

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )

        # Set up generation config
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
            self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id

        self.is_openai = False
        print("LLM loaded successfully\n")

    def _get_device(self, device: str) -> str:
        """Get the appropriate device to use.

        Args:
            device: Requested device

        Returns:
            Actual device to use
        """
        if device != "auto":
            return device

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _format_prompt(self, prompt: str) -> str:
        """Format prompt based on model type.

        Args:
            prompt: Raw prompt

        Returns:
            Formatted prompt
        """
        if "llama" in self.model_name.lower():
            # Llama-2 chat format
            return f"<s>[INST] {prompt} [/INST]"
        elif "mistral" in self.model_name.lower():
            # Mistral instruction format
            return f"<s>[INST] {prompt} [/INST]"
        elif "phi" in self.model_name.lower():
            # Phi-2 instruction format
            return f"Instruct: {prompt}\n\nOutput:"
        else:
            # Default format
            return prompt

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """Generate text from the model.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            **kwargs: Additional model-specific parameters

        Returns:
            Generated text
        """
        if self.is_openai:
            # Use OpenAI API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
            return response.choices[0].message.content

        else:
            # Format prompt for local model
            formatted_prompt = self._format_prompt(prompt)

            # Prepare inputs
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )

            # Move inputs to correct device
            if hasattr(self.model, "device"):
                inputs = {k: v.to(self.model.device)
                          for k, v in inputs.items()}

            # Generate with error handling
            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        **kwargs
                    )

                generated_text = self.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                )

                # Remove the prompt from the generated text
                if generated_text.startswith(formatted_prompt):
                    generated_text = generated_text[len(formatted_prompt):]

                return generated_text.strip()

            except RuntimeError as e:
                print(f"Warning: Generation failed with error: {e}")
                print("Falling back to CPU for this generation...")

                # Fall back to CPU if we encounter device-specific errors
                inputs = {k: v.cpu() for k, v in inputs.items()}
                model_device = next(self.model.parameters()).device
                temp_model = self.model.to("cpu")

                with torch.no_grad():
                    outputs = temp_model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        **kwargs
                    )

                # Move model back to original device
                self.model = temp_model.to(model_device)

                generated_text = self.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                )

                if generated_text.startswith(formatted_prompt):
                    generated_text = generated_text[len(formatted_prompt):]

                return generated_text.strip()

    def get_embedding(self, text: str) -> List[float]:
        """Get embeddings for text (not implemented for GPT models).

        Args:
            text: Text to get embeddings for

        Returns:
            Text embeddings
        """
        raise NotImplementedError(
            "Embedding generation not implemented for this model"
        )
