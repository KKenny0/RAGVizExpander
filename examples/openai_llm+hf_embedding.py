import os
from chromadb.utils.embedding_functions import HuggingFaceEmbeddingFunction

from ragvizexpander import RAGVizChain


embedding_func = HuggingFaceEmbeddingFunction(
    model_name=os.getenv("EMBEDDING_MODEL"),
    api_key=os.getenv("EMBEDDING_MODEL_API_KEY")
)


client = RAGVizChain(embedding_model=embedding_func)
client.load_pdf("presentation.pdf", verbose=True)
client.visualize_query("What are the top revenue drivers for Microsoft?")
