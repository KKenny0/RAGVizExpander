from chromadb.utils.embedding_functions import (
    HuggingFaceEmbeddingFunction,
)
from .base import BaseEmbeddings
from chromadb.api.types import Documents, Embeddings


class HuggingFaceEmbeddings(BaseEmbeddings):
    def __init__(self,
                 model_name: str = None,
                 api_key: str = None,
                 batch_size: int = 16):
        """Initialize HuggingFaceEmbeddings

        Args:
            model_name (str): Name of the HuggingFace model
            api_key (str): API key for accessing the model
            batch_size (int): Number of texts to process in each batch
        """
        super().__init__(batch_size=batch_size)
        self.embed_func = HuggingFaceEmbeddingFunction(
            model_name=model_name,
            api_key=api_key
        )

    def _get_embeddings_batch(self, batch: Documents) -> Embeddings:
        """Get embeddings for a batch of texts.

        Args:
            batch (Documents): A list of texts to get embeddings for.

        Returns:
            Embeddings: A list of embeddings corresponding to the input texts.
        """
        return self.embed_func(batch)
