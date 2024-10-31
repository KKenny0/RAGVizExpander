import os
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from ragvizexpander import RAGVizChain
from ragvizexpander.splitters import RecursiveChar2TokenSplitter


embedding_func = OpenAIEmbeddingFunction(
    model_name=os.getenv("OPENAI_EMBEDDINGS_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_API_BASE"),
)
split_func = RecursiveChar2TokenSplitter()

client = RAGVizChain(embedding_model=embedding_func,
                     split_func=split_func)
client.load_data("presentation.pdf", verbose=True)
client.visualize_query("What are the top revenue drivers for Microsoft?")
