import os
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from ragvizexpander import RAGVizChain


embedding_func = SentenceTransformerEmbeddingFunction(model_name=os.getenv("EMBEDDING_MODEL"))

client = RAGVizChain(embedding_model=embedding_func)
client.load_data("presentation.pdf", verbose=True)
client.visualize_query("What are the top revenue drivers for Microsoft?")
