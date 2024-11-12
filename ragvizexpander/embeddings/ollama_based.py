from chromadb.api.types import Documents, Embeddings


class OllamaEmbeddings:
    """This class is used to get embeddings using the Ollama service"""
    def __init__(self,
                 model_name=None,
                 host=None,
                 batch_size=16
                 ):
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

        self._batch_size = batch_size
        self._client = ollama.Client(host=self.host)

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

            response = self._client.embed(self.model_name, input=mini_batch)
            embeds = response["embeddings"]
            embeddings.extend(embeds)

        assert len(embeddings) == len(input)
        return embeddings
