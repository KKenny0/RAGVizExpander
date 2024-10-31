from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)


class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embed_func = SentenceTransformerEmbeddingFunction(model_name=model_name)

    def __call__(self, *args, **kwargs):
        return self.embed_func(*args, **kwargs)
