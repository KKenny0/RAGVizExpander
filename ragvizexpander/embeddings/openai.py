from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


class OpenAIEmbeddings:
    def __init__(self,
                 api_base: str = None,
                 api_key: str = None,
                 model_name: str = "text-embedding-ada-002"):
        """Initialize OpenAIEmbeddings

        Args:
            api_base (str): OpenAI base URL
            api_key (str): API key
            model_name (str): ID of the model to use.
        """
        self.embed_func = OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=model_name,
            api_base=api_base
        )

    def __call__(self, *args, **kwargs):
        return self.embed_func(*args, **kwargs)
