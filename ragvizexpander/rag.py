"""
rag.py

This module provides functionalities for building and querying a vector database using ChromaDB.
It handles operations like loading PDFs, chunking text, embedding, and retrieving documents based on queries.
"""
from pathlib import Path

import uuid
from typing import List, Any
import chromadb
import numpy as np


def build_vector_database(reader: Any, file: Any, split_func: Any, embedding_model: Any) -> chromadb.Collection:
    """
    Builds a vector database from a PDF file by splitting the text into chunks and embedding them.
    
    Args:
        loader:
        file: The file to process.
        split_func:
        embedding_model:
    
    Returns:
        A Chroma collection object containing the embedded chunks.
    """
    texts = _load_data(reader, file)
    split_texts = _split_text_into_chunks(texts, split_func)
    chroma_collection = _create_and_populate_chroma_collection(split_texts, embedding_model)
    return chroma_collection


def _split_text_into_chunks(texts: List[str], split_func: Any) -> List[str]:
    """
    Splits the text from a file into chunks based on given splitter.
    
    Args:
        texts: List of text extracted from file.

    Returns:
        A list of text chunks.
    """
    return split_func(texts)


def _create_and_populate_chroma_collection(token_split_texts: List[str], embedding_model) -> chromadb.Collection:
    """
    Creates a Chroma collection and populates it with the given text chunks.
    
    Args:
        token_split_texts: List of text chunks split by token count.
    
    Returns:
        A Chroma collection object populated with the text chunks.
    """
    chroma_client = chromadb.Client()
    document_name = uuid.uuid4().hex
    chroma_collection = chroma_client.create_collection(document_name, embedding_function=embedding_model)
    ids = [str(i) for i in range(len(token_split_texts))]
    chroma_collection.add(ids=ids, documents=token_split_texts)
    return chroma_collection


def query_chroma(chroma_collection: chromadb.Collection, query: str | List[str], top_k: int) -> List[str]:
    """
    Queries the Chroma collection for the top_k most relevant chunks to the input query.
    
    Args:
        chroma_collection: The Chroma collection to query.
        query: The input query string.
        top_k: The number of top results to retrieve.
    
    Returns:
        A list of retrieved chunk IDs.
    """
    if isinstance(query, str):
        query = [query]
    results = chroma_collection.query(query_texts=query, n_results=top_k, include=['documents', 'embeddings'])
    retrieved_id = results['ids'][0]
    return retrieved_id


def get_doc_embeddings(chroma_collection: chromadb.Collection) -> np.ndarray:
    """
    Retrieves the document embeddings from the Chroma collection.
    
    Args:
        chroma_collection: The Chroma collection to retrieve embeddings from.
    
    Returns:
        An array of embeddings.
    """
    embeddings = chroma_collection.get(include=['embeddings'])['embeddings']
    return embeddings


def get_docs(chroma_collection: chromadb.Collection) -> List[str]:
    """
    Retrieves the documents from the Chroma collection.
    
    Args:
        chroma_collection: The Chroma collection to retrieve documents from.
    
    Returns:
        A list of documents.
    """
    documents = chroma_collection.get(include=['documents'])['documents']
    return documents


def _load_data(reader, file: str | Path) -> List[str]:
    """
    Loads and extracts text from a file.
    
    Args:
        loader:
        file: The file to load.
    
    Returns:
        A list of strings, each representing the text of a page.
    """
    docs = reader.load_data(file)
    texts = [p.strip() for p in docs if p]
    return texts
