import os
import requests

from chromadb.api.types import (
    Documents,
    Embeddings,
)
from dotenv import load_dotenv

from ragvizexpander import RAGVizChain
from ragvizexpander.llms import ChatOllama

load_dotenv()


class ApiEmbeddingFunction:
    """
    This class is used to get embeddings for a list of texts using the TEI service's API.
    It requires an API url.
    """
    def __init__(self, api_url: str):
        """
        Initialize the ApiEmbeddingFunction.

        Args:
             api_url (str): The URL of the TEI service's API.
        """
        self._api_url = api_url
        self._session = requests.Session()

    def __call__(self, input: Documents) -> Embeddings:
        """
        Get the embeddings for a list of texts.

        Args:
             input (Documents): A list of texts to get embeddings for.

        Returns:
             Embeddings: A list of embeddings corresponding to the input texts.
        """
        # Call TEI Embedding API for each document
        if not isinstance(input, list):
            input = [input]
        batch_size = 16
        num_batch = max(len(input)//batch_size, 1)
        embeddings = []
        for i in range(num_batch):
            if i == num_batch - 1:
                mini_batch = input[batch_size * i:]
            else:
                mini_batch = input[batch_size * i:batch_size * (i + 1)]

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


embedding_func = ApiEmbeddingFunction(os.getenv("EMBEDDING_MODEL_API_BASE"))

client = RAGVizChain(embedding_model=embedding_func,
                     llm=ChatOllama(model_name=os.getenv("OLLAMA_MODEL_NAME")))
client.load_pdf("进出口银行业务介绍培训手册.pdf", verbose=True)
client.visualize_query("贸易合同中应注明在出口买方信贷项下支付的比例，一般为多少？")
