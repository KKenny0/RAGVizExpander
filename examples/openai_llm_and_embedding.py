import os
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from ragvizexpander import RAGVizChain


embedding_func = OpenAIEmbeddingFunction(
    model_name=os.getenv("OPENAI_EMBEDDINGS_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_API_BASE"),
)

client = RAGVizChain(embedding_model=embedding_func)
client.load_data("presentation.pdf", verbose=True)
client.visualize_query("What are the top revenue drivers for Microsoft?")
