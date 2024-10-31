from chromadb.utils.embedding_functions import (
    HuggingFaceEmbeddingFunction,
)


class HuggingFaceEmbeddings:
    def __init__(self,
                 model_name: str = None,
                 api_key: str = None):
        """Initialize HuggingFaceEmbeddings

        Args:
            model_name (str): Name of the HuggingFace model
            api_key (str): API key for accessing the model
        """
        self.embed_func = HuggingFaceEmbeddingFunction(
            model_name=model_name,
            api_key=api_key
        )

    def __call__(self, *args, **kwargs):
        return self.embed_func(*args, **kwargs)
