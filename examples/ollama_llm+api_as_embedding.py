import os
from dotenv import load_dotenv

from ragvizexpander import RAGVizChain
from ragvizexpander.llms import ChatOllama
from ragvizexpander.embeddings import TEIEmbeddings
from ragvizexpander.splitters import RecursiveChar2TokenSplitter

load_dotenv()

embedding_func = TEIEmbeddings(api_url=os.getenv("EMBEDDING_MODEL_API_BASE"))
llm_model = ChatOllama(model_name=os.getenv("OLLAMA_MODEL_NAME"))
split_func = RecursiveChar2TokenSplitter()

client = RAGVizChain(embedding_model=embedding_func,
                     llm=llm_model,
                     split_func=split_func)
client.load_data("Electronic Banking and Accessibility-5265.pdf", verbose=True)
client.visualize_query(
    "What does electronic banking involve according to Asare and Sakoe [12], such as internet banking, mobile banking, agency banking, and automatic teller machines?")
