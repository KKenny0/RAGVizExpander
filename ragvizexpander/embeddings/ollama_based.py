from chromadb.api.types import Documents, Embeddings
from .base import BaseEmbeddings


class OllamaEmbeddings(BaseEmbeddings):
    """This class is used to get embeddings using the Ollama service"""
    def __init__(self,
                 model_name=None,
                 host=None,
                 batch_size=16
                 ):
        super().__init__(batch_size=batch_size)
        try:
            import ollama
        except ImportError:
            raise ValueError(
                "The ollama python package is not installed. "
                "Please install it with `pip install ollama`."
            )

        if not host:
            self.host = "http://localhost:11434"
        else:
            self.host = host

        self.model_name = model_name
        self._client = ollama.Client(host=self.host)

    def _get_embeddings_batch(self, batch: Documents) -> Embeddings:
        """Get embeddings for a batch of texts.

        Args:
            batch (Documents): A list of texts to get embeddings for.

        Returns:
            Embeddings: A list of embeddings corresponding to the input texts.
        """
        response = self._client.embed(self.model_name, input=batch)
        return response["embeddings"]
