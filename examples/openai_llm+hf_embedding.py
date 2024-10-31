import os
from chromadb.utils.embedding_functions import HuggingFaceEmbeddingFunction

from ragvizexpander import RAGVizChain
from ragvizexpander.splitters import RecursiveChar2TokenSplitter


embedding_func = HuggingFaceEmbeddingFunction(
    model_name=os.getenv("EMBEDDING_MODEL"),
    api_key=os.getenv("EMBEDDING_MODEL_API_KEY")
)
split_func = RecursiveChar2TokenSplitter()

client = RAGVizChain(embedding_model=embedding_func,
                     split_func=split_func)
client.load_data("presentation.pdf", verbose=True)
client.visualize_query("What are the top revenue drivers for Microsoft?")
