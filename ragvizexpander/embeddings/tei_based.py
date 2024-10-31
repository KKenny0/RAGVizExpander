import requests
from chromadb.api.types import Documents, Embeddings


class TEIEmbeddings:
    """This class is used to get embeddings using the TEI service's API."""
    def __init__(self, api_url: str, batch_size: int = 16):
        self._api_url = api_url
        self._session = requests.Session()
        self._batch_size = batch_size

    def __call__(self, input: Documents) -> Embeddings:
        """Get embeddings for a list of texts.

        Args:
            inputs (Documents): A list of texts to get embeddings for.

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

            embed = self._session.post(
                self._api_url,
                json={
                    "inputs": mini_batch,
                    "normalize": True,
                    "truncate": True
                }
            ).json()

            embeddings.extend(embed)
        assert len(embeddings) == len(input)
        return embeddings
