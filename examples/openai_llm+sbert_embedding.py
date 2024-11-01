import os

from ragvizexpander import RAGVizChain
from ragvizexpander.splitters import RecursiveChar2TokenSplitter
from ragvizexpander.embeddings import SentenceTransformerEmbeddings


embedding_func = SentenceTransformerEmbeddings(
    model_name=os.getenv("EMBEDDING_MODEL")
)
split_func = RecursiveChar2TokenSplitter()

client = RAGVizChain(embedding_model=embedding_func,
                     split_func=split_func)
client.load_data("Electronic Banking and Accessibility-5265.pdf", verbose=True)
client.visualize_query("What does electronic banking involve according to Asare and Sakoe [12], such as internet banking, mobile banking, agency banking, and automatic teller machines?")
