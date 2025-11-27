"""
Offline LLM Module

Provides interface for local LLM inference using llama.cpp.
Supports various models (Phi-3, Llama, Mistral) in GGUF format.
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

try:
    from llama_cpp import Llama
except ImportError:
    raise ImportError(
        "llama-cpp-python not installed. "
        "Install with: pip install llama-cpp-python"
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OfflineLLM:
    """Wrapper for local LLM inference using llama.cpp."""
    
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = 0,
        n_threads: Optional[int] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ):
        """
        Initialize the offline LLM.
        
        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU (0 for CPU-only)
            n_threads: Number of threads for CPU inference
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        logger.info(f"Loading model from {self.model_path}")
        logger.info(f"Context size: {n_ctx}, GPU layers: {n_gpu_layers}")
        
        try:
            self.llm = Llama(
                model_path=str(self.model_path),
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                verbose=False,
                **kwargs
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input text prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        try:
            response = self.llm(
                prompt,
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                temperature=kwargs.get('temperature', self.temperature),
                stop=kwargs.get('stop', ['\n\n', 'User:', 'Human:']),
                echo=False
            )
            
            return response['choices'][0]['text'].strip()
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error generating response: {e}"
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """Allow instance to be callable."""
        return self.generate(prompt, **kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_path': str(self.model_path),
            'context_size': self.n_ctx,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }


def create_offline_llm(model_path: str, **kwargs) -> OfflineLLM:
    """
    Factory function to create OfflineLLM instance.
    
    Args:
        model_path: Path to GGUF model file
        **kwargs: Additional arguments for OfflineLLM
        
    Returns:
        OfflineLLM instance
    """
    return OfflineLLM(model_path, **kwargs)


if __name__ == "__main__":
    # Example usage
    print("Offline LLM Module")
    print("This module requires a GGUF model file to run.")
    print("Example: models/phi-3-mini-4k-instruct.Q4_K_M.gguf")
