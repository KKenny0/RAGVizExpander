from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from .base import BaseEmbeddings
from chromadb.api.types import Documents, Embeddings


class OpenAIEmbeddings(BaseEmbeddings):
    def __init__(self,
                 api_base: str = None,
                 api_key: str = None,
                 model_name: str = "text-embedding-ada-002",
                 batch_size: int = 16):
        """Initialize OpenAIEmbeddings
        
        Args:
            api_base (str): OpenAI base URL
            api_key (str): API key
            model_name (str): ID of the model to use.
            batch_size (int): Number of texts to process in each batch
        """
        super().__init__(batch_size=batch_size)
        self.embed_func = OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=model_name,
            api_base=api_base
        )

    def _get_embeddings_batch(self, batch: Documents) -> Embeddings:
        """Get embeddings for a batch of texts.
        
        Args:
            batch (Documents): A list of texts to get embeddings for.
            
        Returns:
            Embeddings: A list of embeddings corresponding to the input texts.
        """
        return self.embed_func(batch)
