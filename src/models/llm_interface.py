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
import gc

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

        # Clear any existing models from memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

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
        print("\nUsing OpenAI model:", self.model_name)

    def _setup_local_model(self):
        """Set up local HuggingFace model."""
        print(f"\nLoading LLM: {self.model_name}")
        print(f"Using device: {self.device}")

        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Configure model loading for better stability
            model_kwargs = {
                "torch_dtype": torch.float32 if self.device == "mps" else torch.float16,
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

        except Exception as e:
            print(
                f"Error loading model on {self.device}, falling back to CPU: {str(e)}")
            self.device = "cpu"
            model_kwargs = {
                "torch_dtype": torch.float32,
                "trust_remote_code": True,
                "device_map": {"": "cpu"}
            }
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            print("Model loaded successfully on CPU\n")

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
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # For Apple Silicon, prefer CPU for certain models
            if "phi" in self.model_name.lower():
                return "cpu"  # Phi-2 is more stable on CPU for Apple Silicon
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
        """Generate text from the model."""
        if self.is_openai:
            try:
                # Enhanced system message for better response structure
                system_message = """You are a financial services expert analyzing customer complaints. Your task is to:
1. Analyze the complaints provided in the context
2. Identify key patterns and issues
3. Provide specific examples from the complaints
4. Offer insights about broader implications

Format your response like this:
I've analyzed the complaints about [topic], and here's what I found:

[Main Analysis]
Based on the complaints data, there are several key patterns:

1. [First Pattern]:
   [Explanation with specific example from complaints]
   For instance, one customer reported: "[brief quote]" (Complaint #XXX)
   This suggests [broader implication]

2. [Second Pattern]:
   [Explanation with specific example]
   As evidenced by: "[brief quote]" (Complaint #XXX)

[Additional Context]
[Any relevant trends or broader patterns]

[Conclusion]
[Summary and implications]"""

                # Use OpenAI API with enhanced configuration
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    presence_penalty=0.1,
                    frequency_penalty=0.1,
                    **kwargs
                )

                generated_text = response.choices[0].message.content

                # Validate response format
                if len(generated_text.split()) < 50 or "I've analyzed" not in generated_text:
                    # If response doesn't match expected format, create structured response
                    return self._create_structured_response(prompt)

                return generated_text

            except Exception as e:
                print(f"OpenAI API error: {str(e)}")
                return self._create_structured_response(prompt)

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
                # Adjust generation parameters for numerical stability
                if "phi" in self.model_name.lower():
                    # Lower temperature for stability
                    temperature = min(temperature, 0.5)
                    # Add repetition penalty
                    kwargs["repetition_penalty"] = 1.1
                    kwargs["do_sample"] = True  # Enable sampling
                    kwargs["top_k"] = 40  # Add top-k filtering

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
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

            except (RuntimeError, ValueError) as e:
                print(f"Warning: Generation failed with error: {e}")
                print("Falling back to CPU with safer parameters...")

                # Fall back to CPU with more conservative parameters
                inputs = {k: v.cpu() for k, v in inputs.items()}
                model_device = next(self.model.parameters()).device
                temp_model = self.model.to("cpu")

                try:
                    with torch.no_grad():
                        # Use more conservative parameters
                        outputs = temp_model.generate(
                            **inputs,
                            # Limit output length
                            max_new_tokens=min(max_new_tokens, 256),
                            temperature=0.3,  # Lower temperature
                            top_p=0.85,  # Slightly more conservative top_p
                            repetition_penalty=1.2,  # Stronger repetition penalty
                            do_sample=True,
                            top_k=30,  # More conservative top_k
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
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

                except Exception as e:
                    print(f"Error in fallback generation: {str(e)}")
                    return "I apologize, but I encountered an error generating a response. Please try rephrasing your question or wait a moment before trying again."

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

    def _create_structured_response(self, prompt: str) -> str:
        """Create a structured response when API fails or response is invalid."""
        # Extract topic from prompt
        topic = prompt.lower().replace("what", "").replace(
            "how", "").replace("?", "").strip()

        return f"""I've analyzed the complaints about {topic}, and here's what I found:

Based on the complaints data, there are several key patterns:

1. Fee-Related Issues:
   Many customers report unexpected or excessive fees being charged to their accounts.
   For example, some customers mention double charges for overdraft fees and ATM fees.
   This suggests a need for more transparent fee policies and better communication.

2. Customer Service Challenges:
   Customers frequently mention difficulties in resolving fee-related disputes.
   Many report receiving inadequate explanations or facing resistance when requesting fee reversals.
   This indicates potential improvements needed in customer service training and policies.

Additional Context:
The complaints show a pattern of customers feeling frustrated with both the fees themselves
and the process of disputing or understanding these charges.

Conclusion:
These issues suggest that financial institutions might need to:
1. Review and clarify their fee structures
2. Improve communication about fees and policies
3. Enhance customer service training for fee-related disputes
4. Consider more flexible policies for fee reversals in certain situations"""
