"""
Ragxplorer.py
"""
import os
from os import PathLike
from pathlib import Path
from typing import (
    Optional,
    Any,
    Callable, Union
)
from dotenv import load_dotenv

from pydantic import BaseModel
import pandas as pd
import umap

from chromadb import Collection
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

import plotly.graph_objs as go

from .rag import (
    build_vector_database,
    get_doc_embeddings,
    get_docs,
    query_chroma
)

from .projections import (
    set_up_umap,
    get_projections,
    prepare_projections_df,
    plot_embeddings
)

from .query_expansion import (
    generate_hypothetical_ans,
    generate_sub_qn
)

load_dotenv()


class _Documents(BaseModel):
    text: Optional[Any] = None
    ids: Optional[Any] = None
    embeddings: Optional[Any] = None
    projections: Optional[Any] = None


class _Query(BaseModel):
    original_query: Optional[Any] = None
    original_query_projection: Optional[Any] = None
    actual_search_queries: Optional[Any] = None
    extend_queries: Optional[Any] = None
    retrieved_docs: Optional[Any] = None


class _VizData(BaseModel):
    base_df: Optional[Any] = None
    query_df: Optional[Any] = None
    visualisation_df: Optional[Any] = None


class RAGVizChain(BaseModel):
    """
    RAGVizChain class for managing the RAG exploration process.
    """
    embedding_model: Optional[Callable] = None
    llm: Optional[Callable] = None
    reader: Optional[Any] = None
    split_func: Optional[Callable] = None
    _chosen_embedding_model: Optional[Any] = None
    _chosen_llm: Optional[Any] = None
    _chosen_split_func: Optional[Any] = None
    _vectordb: Optional[Any] = None
    _documents: _Documents = _Documents()
    _projector: Optional[Any] = None
    _query: _Query = _Query()
    _VizData: _VizData = _VizData()

    def __init__(self, **data):
        super().__init__(**data)
        self._set_embedding_model()
        self._set_llm()
        self._set_splitter()

    def _set_embedding_model(self):
        """ Sets the embedding model """
        if self.embedding_model is None:
            if "OPENAI_API_KEY" not in os.environ:
                raise OSError("OPENAI_API_KEY is not set")
            self._chosen_embedding_model = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"),
                                                                   model_name=os.getenv("model_name"))
        else:
            self._chosen_embedding_model = self.embedding_model

    def _set_llm(self):
        """ Sets the LLM model """
        self._chosen_llm = self.llm

    def _set_splitter(self):
        if self.split_func is None:
            from .splitters import RecursiveChar2TokenSplitter
            self._chosen_split_func = RecursiveChar2TokenSplitter()

        else:
            self._chosen_split_func = self.split_func

    def _set_reader(self, file_path):
        from .loaders import extractors

        try:
            ext = Path(file_path).suffix.lower()
        except TypeError:
            ext = Path(file_path.name).suffix.lower()

        extractor = extractors.get(ext)
        self.reader = extractor

    def config_llm(self, config):
        self._chosen_llm.config.update(config)

    def load_data(self,
                  document_path: Union[str, "PathLike[str]"],
                  reader=None,
                  verbose: bool = False,
                  umap_params: dict = None):
        """
        Load data from a file and prepare it for exploration.
        
        Args:
            document_path: Path to the document to load.
            reader:
            verbose:
            umap_params:
        """
        if reader is None:
            self._set_reader(document_path)
        if verbose:
            print(" ~ Building the vector database...")
        self._vectordb = build_vector_database(self.reader,
                                               document_path,
                                               self._chosen_split_func,
                                               self._chosen_embedding_model)
        if verbose:
            print("Completed Building Vector Database ✓")
        self._documents.embeddings = get_doc_embeddings(self._vectordb)
        self._documents.text = get_docs(self._vectordb)
        self._documents.ids = self._vectordb.get()['ids']
        if verbose:
            print(" ~ Reducing the dimensionality of embeddings...")
        self._projector = set_up_umap(embeddings=self._documents.embeddings, umap_params=umap_params)
        self._documents.projections = get_projections(embedding=self._documents.embeddings,
                                                      umap_transform=self._projector)
        self._VizData.base_df = prepare_projections_df(document_ids=self._documents.ids,
                                                       document_projections=self._documents.projections,
                                                       document_text=self._documents.text)
        if verbose:
            print("Completed reducing dimensionality of embeddings ✓")

    def visualize_query(self,
                        query: str,
                        retrieval_method: str = "Naive",
                        top_k: int = 5,
                        query_shape_size: int = 5,
                        import_projection_data: pd.DataFrame = None) -> go.Figure:
        if import_projection_data is not None:
            self._VizData.base_df = import_projection_data
        else:
            if self._vectordb is None or self._VizData.base_df is None:
                raise RuntimeError("Please load the pdf first.")

        if retrieval_method not in ["Naive", "HyAE", "Multi-Sub-Questions"]:
            raise ValueError("Invalid retrieval method. Please use Naive, HyAE, or Multi-Sub-Questions.")

        self._query.original_query = query

        self._query.original_query_projection = get_projections(
            embedding=[self._chosen_embedding_model(self._query.original_query)],
            umap_transform=self._projector
        )

        self._VizData.query_df = pd.DataFrame({"x": [self._query.original_query_projection[0][0]],
                                               "y": [self._query.original_query_projection[1][0]],
                                               "document_cleaned": query,
                                               "category": "Original Query",
                                               "size": query_shape_size})

        if retrieval_method == "Naive":
            self._query.actual_search_queries = self._query.original_query

        elif retrieval_method == "HyAE":
            self._query.extend_queries = [generate_hypothetical_ans(query=self._query.original_query,
                                                                    client=self._chosen_llm)]
            self._query.actual_search_queries = [self._query.extend_queries[0], self._query.original_query]

            hyp_ans_projection = get_projections(
                embedding=[self._chosen_embedding_model(self._query.extend_queries)],
                umap_transform=self._projector
            )
            hyp_ans_df = pd.DataFrame({"x": [hyp_ans_projection[0][0]],
                                       "y": [hyp_ans_projection[1][0]],
                                       "document_cleaned": self._query.extend_queries[0],
                                       "category": "Hypothetical Ans",
                                       "size": query_shape_size})
            self._VizData.query_df = pd.concat([hyp_ans_df, self._VizData.query_df], axis=0)

        elif retrieval_method == "Multi-Sub-Questions":
            self._query.extend_queries = generate_sub_qn(query=self._query.original_query,
                                                         client=self._chosen_llm)
            self._query.actual_search_queries = self._query.extend_queries + [self._query.original_query]

            sub_qn_projection = get_projections(
                embedding=self._chosen_embedding_model(self._query.extend_queries),
                umap_transform=self._projector
            )
            sub_qn_df = pd.DataFrame({"x": sub_qn_projection[0],
                                      "y": sub_qn_projection[1],
                                      "document_cleaned": self._query.extend_queries})
            sub_qn_df['category'] = "Sub-Questions"
            sub_qn_df['size'] = query_shape_size
            self._VizData.query_df = pd.concat([sub_qn_df, self._VizData.query_df], axis=0)

        self._query.retrieved_docs = query_chroma(chroma_collection=self._vectordb,
                                                  query=self._query.actual_search_queries,
                                                  top_k=top_k)

        self._VizData.base_df.loc[
            self._VizData.base_df['id'].isin(self._query.retrieved_docs), "category"] = "Retrieved"

        self._VizData.visualisation_df = pd.concat([self._VizData.base_df, self._VizData.query_df], axis=0)

        fig = plot_embeddings(self._VizData.visualisation_df)
        fig.show()
        return fig

    def visualise_query(self,
                        query: str,
                        retrieval_method: str = "naive",
                        top_k: int = 5,
                        query_shape_size: int = 5,
                        import_projection_data: pd.DataFrame = None) -> go.Figure:
        """
        Visualize the query results in a 2D projection using Plotly.

        Args:
            query (str): The query string to visualize.
            retrieval_method (str): The method used for document retrieval. Defaults to 'naive'.
            top_k (int): The number of top documents to retrieve.
            query_shape_size (int): The size of the shape to represent the query in the plot.

        Returns:
            go.Figure: A Plotly figure object representing the visualization.

        Raises:
            RuntimeError: If the document has not been loaded before visualization.
        """
        return self.visualize_query(query=query,
                                    retrieval_method=retrieval_method,
                                    top_k=top_k,
                                    query_shape_size=query_shape_size,
                                    import_projection_data=None)

    def export_chroma(self) -> Collection:
        """
        Export the ChromaDB collection.
        """
        return self._vectordb

    def load_chroma(self, chroma_collection: Collection, initialize_projector: bool = False,
                    recompute_projections: bool = False, umap_params: dict = None, verbose: bool = True):
        """
        Load ChromaDB collection.
        """
        self._vectordb = chroma_collection
        self._documents.embeddings = get_doc_embeddings(self._vectordb)
        self._documents.text = get_docs(self._vectordb)
        self._documents.ids = self._vectordb.get()['ids']
        if initialize_projector:
            if verbose:
                print("Setting up umap projector")
            self._projector = set_up_umap(embeddings=self._documents.embeddings, umap_params=umap_params)
        if recompute_projections:
            if verbose:
                print("Recomputing projections")
            self._projector = set_up_umap(embeddings=self._documents.embeddings, umap_params=umap_params)
            self._documents.projections = get_projections(embedding=self._documents.embeddings,
                                                          umap_transform=self._projector)
            self._VizData.base_df = prepare_projections_df(document_ids=self._documents.ids,
                                                           document_projections=self._documents.projections,
                                                           document_text=self._documents.text)

    def export_projector(self) -> umap.UMAP:
        """
        Export the UMAP projector.
        """
        return self._projector

    def export_query_extension(self):
        return self._query.extend_queries

    def visualize_chunking(self):
        all_docs = self._documents.text
        retrieved_ids = self._query.retrieved_docs
        return retrieved_ids, all_docs

    def load_projector(self, umap_transform: umap.UMAP, recompute_projections: bool = False):
        """
        Load UMAP projector.
        """
        self._projector = umap_transform
        if recompute_projections:
            self._documents.projections = get_projections(embedding=self._documents.embeddings,
                                                          umap_transform=self._projector)
            self._VizData.base_df = prepare_projections_df(document_ids=self._documents.ids,
                                                           document_projections=self._documents.projections,
                                                           document_text=self._documents.text)

    def run_projector(self):
        """
        Run UMAP projector.
        """
        self._documents.projections = get_projections(embedding=self._documents.embeddings,
                                                      umap_transform=self._projector)
        self._VizData.base_df = prepare_projections_df(document_ids=self._documents.ids,
                                                       document_projections=self._documents.projections,
                                                       document_text=self._documents.text)
