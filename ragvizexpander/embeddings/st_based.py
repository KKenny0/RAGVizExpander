from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)
from .base import BaseEmbeddings
from chromadb.api.types import Documents, Embeddings

class SentenceTransformerEmbeddings(BaseEmbeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 16):
        """Initialize SentenceTransformerEmbeddings
        
        Args:
            model_name (str): Name of the sentence transformer model
            batch_size (int): Number of texts to process in each batch
        """
        super().__init__(batch_size=batch_size)
        self.embed_func = SentenceTransformerEmbeddingFunction(model_name=model_name)
        
    def _get_embeddings_batch(self, batch: Documents) -> Embeddings:
        """Get embeddings for a batch of texts.
        
        Args:
            batch (Documents): A list of texts to get embeddings for.
            
        Returns:
            Embeddings: A list of embeddings corresponding to the input texts.
        """
        return self.embed_func(batch)
