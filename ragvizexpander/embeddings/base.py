from abc import ABC, abstractmethod
from typing import List, Union, Optional
from chromadb.api.types import Documents, Embeddings


class BaseEmbeddings(ABC):
    """Abstract base class for all embedding models."""

    def __init__(self, batch_size: int = 16):
        self._batch_size = batch_size

    @abstractmethod
    def _get_embeddings_batch(self, batch: List[str]) -> List[float]:
        """Abstract method to get embeddings for a single batch."""
        pass

    def __call__(self, input: Documents) -> Embeddings:
        """Template method that defines the algorithm for batch processing.
        
        Args:
            input (Documents): A list of texts to get embeddings for.
            
        Returns:
            Embeddings: A list of embeddings corresponding to the input texts.
        """
        if not isinstance(input, list):
            input = [input]

        num_batch = max(len(input) // self._batch_size, 1)
        embeddings = []

        for i in range(num_batch):
            if i == num_batch - 1:
                mini_batch = input[self._batch_size * i:]
            else:
                mini_batch = input[self._batch_size * i:self._batch_size * (i + 1)]

            if not isinstance(mini_batch, list):
                mini_batch = [mini_batch]

            batch_embeddings = self._get_embeddings_batch(mini_batch)
            embeddings.extend(batch_embeddings)

        assert len(embeddings) == len(input)
        return embeddings


class EmbeddingsFactory:
    """Factory class for creating embedding instances."""

    @staticmethod
    def create(provider: str, **kwargs) -> BaseEmbeddings:
        """Create an embedding instance based on the provider.
        
        Args:
            provider (str): The embedding provider ('openai', 'huggingface', 'sentence-transformer', 'tei', 'ollama')
            **kwargs: Additional arguments for the specific embedding provider
            
        Returns:
            BaseEmbeddings: An instance of the appropriate embedding class
        """
        if provider == "openai":
            from .openai import OpenAIEmbeddings
            return OpenAIEmbeddings(**kwargs)
        elif provider == "huggingface":
            from .hf_based import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(**kwargs)
        elif provider == "sentence-transformer":
            from .st_based import SentenceTransformerEmbeddings
            return SentenceTransformerEmbeddings(**kwargs)
        elif provider == "tei":
            from .tei_based import TEIEmbeddings
            return TEIEmbeddings(**kwargs)
        elif provider == "ollama":
            from .ollama_based import OllamaEmbeddings
            return OllamaEmbeddings(**kwargs)
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")
