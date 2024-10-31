"""
Streamlit app
"""
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except:
    pass

import os
import streamlit as st
from ragvizexpander import RAGVizChain
from ragvizexpander.llms import *
from ragvizexpander.embeddings import *
from ragvizexpander.splitters import RecursiveChar2TokenSplitter

st.set_page_config(
    page_title="RAGVizExpander Demo",
    page_icon="🔬",
    layout="wide"
)

os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
os.environ['HF_API_KEY'] = st.secrets["HF_API_KEY"]


if "chart" not in st.session_state:
    st.session_state['chart'] = None

if "loaded" not in st.session_state:
    st.session_state['loaded'] = False

st.title("RAGVizExpander Demo🔬")
st.markdown("📦 More details can be found at the GitHub repo [here](https://github.com/KKenny0/RAGVizExpander)")

if not st.session_state['loaded']:
    main_page = st.empty()
    main_button = st.empty()
    with main_page.container():
        uploaded_file = st.file_uploader("Upload your file",
                                         label_visibility="collapsed",
                                         type=['pdf', 'docx', 'txt', 'pptx'])

        # --- setting llm model
        st.markdown("### Settings for *LLM* model")

        st.session_state["llm_model_type"] = st.radio("Select type of llm model",
                                                      ["OpenAI", "Ollama"],
                                                      horizontal=True)

        if st.session_state["llm_model_type"] == "OpenAI":
            st.session_state["openai_llm_base_url"] = st.text_input("Enter OpenAI LLM API Base")
            st.session_state["openai_llm_api_key"] = st.text_input("Enter OpenAI LLM API Key")
            st.session_state["openai_llm_model"] = st.text_input("Enter OpenAI LLM model name")
            st.session_state["chosen_llm_model"] = ChatOpenAI(
                base_url=st.session_state["openai_llm_base_url"],
                api_key=st.session_state["openai_llm_api_key"],
                model_name=st.session_state["openai_llm_model"],
            )
        else:
            st.session_state["ollama_llm_model"] = st.text_input("Enter Ollama model name")
            st.session_state["chosen_llm_model"] = ChatOllama(model_name=st.session_state["ollama_llm_model"])

        st.markdown("""---""")

        # --- setting embedding model
        st.markdown("### Settings for *EMBEDDING* model")

        st.session_state["embedding_model_type"] = st.radio("Select type of embedding model",
                                                            ["OpenAI", "SentenceTransformer", "HuggingFace", "TEI"],
                                                            horizontal=True)

        if st.session_state["embedding_model_type"] == "OpenAI":
            st.session_state["openai_embed_model"] = st.selectbox("Select embedding model",
                                                                      ["text-embedding-3-small",
                                                                       "text-embedding-3-large",
                                                                       "text-embedding-ada-002"])
            st.session_state["openai_embed_api_key"] = st.text_input("Enter OpenAI Embedding API Key")
            st.session_state["openai_embed_api_base"] = st.text_input("Enter OpenAI Embedding API Base")
            st.session_state["chosen_embedding_model"] = OpenAIEmbeddings(
                api_base=st.session_state["openai_embed_api_base"],
                api_key=st.session_state["openai_embed_api_key"],
                model_name=st.session_state["openai_embed_model"],
            )

        elif st.session_state["embedding_model_type"] == "HuggingFace":
            st.session_state["hf_embed_model"] = st.text_input("Enter HF repository name")
            st.session_state["hf_api_key"] = st.text_input("Enter HF API key")
            st.session_state["chosen_embedding_model"] = HuggingFaceEmbeddings(
                model_name=st.session_state["hf_embed_model"],
                api_key=st.session_state["hf_api_key"]
            )

        else:
            st.session_state["tei_api_url"] = st.text_input("Enter TEI(Text-Embedding-Inference) api url")
            st.session_state["chosen_embedding_model"] = TEIEmbeddings(
                api_url=st.session_state["tei_api_url"]
            )

        st.markdown("""---""")

        # --- setting chunking parameters
        st.markdown("### Settings for *CHUNKING* model")
        st.session_state["chunk_size"] = st.number_input("Chunk size", value=500, min_value=100, max_value=1000, step=100)
        st.session_state["chunk_overlap"] = st.number_input("Chunk overlap", value=0, min_value=0, max_value=100, step=10)
        st.session_state["split_func"] = RecursiveChar2TokenSplitter(
            chunk_size=st.session_state["chunk_size"],
            chunk_overlap=st.session_state["chunk_overlap"],
        )

    if st.button("Build Vector DB"):
        st.session_state["client"] = RAGVizChain(embedding_model=st.session_state["chosen_embedding_model"],
                                                 llm=st.session_state["chosen_llm_model"],
                                                 split_func=st.session_state["split_func"])
        main_page.empty()
        main_button.empty()
        with st.spinner("Building Vector DB"):
            st.session_state["client"].load_data(uploaded_file,)
            st.session_state['loaded'] = True
            st.rerun()
else:
    col1, col2 = st.columns(2)
    st.session_state['query'] = col1.text_area("Enter your query here")
    st.session_state['technique'] = col1.radio("Select retrival technique", ["naive", "HyDE", "multi_qns"], horizontal=True)
    st.session_state['top_k'] = col1.number_input("Top k", value=5, min_value=1, max_value=10, step=1)
    if col1.button("Execute Query"):
            st.session_state['chart'] = st.session_state["client"].visualize_query(st.session_state['query'], retrieval_method=st.session_state['technique'], top_k=st.session_state['top_k'])
    if st.session_state['chart'] is not None:
        col2.plotly_chart(st.session_state['chart'])

    if col1.button("Reset Application"):
        st.session_state['loaded'] = False
        st.rerun()
