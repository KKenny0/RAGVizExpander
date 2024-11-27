import requests
from chromadb.api.types import Documents, Embeddings
from .base import BaseEmbeddings


class TEIEmbeddings(BaseEmbeddings):
    """This class is used to get embeddings using the TEI service's API."""
    def __init__(self, api_url: str, batch_size: int = 16):
        super().__init__(batch_size=batch_size)
        self._api_url = api_url
        self._session = requests.Session()

    def _get_embeddings_batch(self, batch: Documents) -> Embeddings:
        """Get embeddings for a batch of texts.

        Args:
            batch (Documents): A list of texts to get embeddings for.

        Returns:
            Embeddings: A list of embeddings corresponding to the input texts.
        """
        embed = self._session.post(
            self._api_url,
            json={
                "inputs": batch,
                "normalize": True,
                "truncate": True
            }
        ).json()
        return embed
