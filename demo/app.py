"""
Streamlit app
"""
import json

from ragvizexpander.constants import HYDE_SYS_MSG, MULTIPLE_QNS_SYS_MSG

try:
    __import__('pysqlite3')
    import sys

    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except:
    pass

import os
import streamlit as st
import yaml

from ragvizexpander import RAGVizChain
from ragvizexpander.llms import *
from ragvizexpander.embeddings import *
from ragvizexpander.splitters import *
from ragvizexpander.loaders import app_extractors
from utils import YAMLLoader

st.set_page_config(
    page_title="RAGVizExpander Demo",
    page_icon="🔬",
    layout="wide"
)

os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
os.environ['HF_API_KEY'] = st.secrets["HF_API_KEY"]

if "chart" not in st.session_state:
    st.session_state['chart'] = None

if "query_extension" not in st.session_state:
    st.session_state['query_extension'] = None

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
        if uploaded_file is not None:
            for key, loader_map in app_extractors.items():
                if key in uploaded_file.name:
                    st.session_state['file_reader_type'] = st.radio("**Select type of file reader**",
                                                                    list(loader_map.keys()),
                                                                    horizontal=True)

                    st.session_state['file_reader'] = loader_map[st.session_state['file_reader_type']]

        st.divider()

        # --- setting llm model
        llm_on = st.toggle("### Settings for *LLM* model")
        st.caption("For generating hypothetical answers and sub-questions")
        if llm_on:
            st.session_state['llm_model_type'] = st.radio("**Select type of llm model**",
                                                          ["OpenAI", "Ollama", "Llama-Cpp"],
                                                          horizontal=True)

            if st.session_state['llm_model_type'] == "OpenAI":
                st.session_state['openai_base_url'] = st.text_input("Enter OpenAI LLM API Base")
                st.session_state['openai_api_key'] = st.text_input("Enter OpenAI LLM API Key")
                st.session_state['openai_llm_model'] = st.text_input("Enter OpenAI LLM Model")
                st.session_state['chosen_llm_model'] = ChatOpenAI(
                    base_url=st.session_state['openai_base_url'],
                    api_key=st.session_state['openai_api_key'],
                    model=st.session_state['openai_llm_model'])

            elif st.session_state['llm_model_type'] == "Ollama":
                st.session_state['ollama_llm_host'] = st.text_input(
                    "Enter Ollama LLM Host",
                    placeholder="default to http://localhost:11434")
                st.session_state['ollama_llm_model'] = st.text_input("Enter Ollama LLM model")
                st.session_state['chosen_llm_model'] = ChatOllama(host=st.session_state['ollama_llm_host'],
                                                                  model=st.session_state['ollama_llm_model'])

            else:
                st.session_state['llama_cpp_model_path'] = st.text_input("Enter llama.cpp LLM model path ")
                st.session_state['chosen_llm_model'] = ChatLlamaCpp(model_path=st.session_state['llama_cpp_model_path'])

        st.divider()

        # --- setting embedding model
        embedding_on = st.toggle("### Settings for *EMBEDDING* model")
        if embedding_on:
            st.session_state['embedding_model_type'] = st.radio("**Select type of embedding model**",
                                                                ["OpenAI", "Ollama", "SentenceTransformer", "HuggingFace", "TEI"],
                                                                horizontal=True)

            if st.session_state['embedding_model_type'] == "OpenAI":
                st.session_state['openai_embed_model'] = st.selectbox("Select embedding model",
                                                                      ["text-embedding-3-small",
                                                                       "text-embedding-3-large",
                                                                       "text-embedding-ada-002"])
                st.session_state['openai_embed_api_key'] = st.text_input("Enter OpenAI Embedding API Key")
                st.session_state['openai_embed_api_base'] = st.text_input("Enter OpenAI Embedding API Base")
                st.session_state['chosen_embedding_model'] = OpenAIEmbeddings(
                    api_base=st.session_state['openai_embed_api_base'],
                    api_key=st.session_state['openai_embed_api_key'],
                    model_name=st.session_state['openai_embed_model'],
                )

            elif st.session_state['embedding_model_type'] == "Ollama":
                st.session_state['ollama_emb_model'] = st.text_input("Enter Ollama Embedding Model Name, Ref: https://ollama.com/library")
                st.session_state['ollama_host_emb'] = st.text_input("Enter Ollama Embedding service url",
                                                                    placeholder="default to http://localhost:11434")
                st.session_state['chosen_embedding_model'] = OllamaEmbeddings(
                    model_name=st.session_state['ollama_emb_model'],
                    host=st.session_state['ollama_host_emb']
                )

            elif st.session_state['embedding_model_type'] == "SentenceTransformer":
                st.session_state['st_emb_model'] = st.text_input("Enter SentenceTransformer Embedding Model Name, Ref: https://www.sbert.net/docs/sentence_transformer/pretrained_models.html#original-models")
                st.session_state['chosen_embedding_model'] = SentenceTransformerEmbeddings(
                    model_name=st.session_state['st_emb_model']
                )

            elif st.session_state['embedding_model_type'] == "HuggingFace":
                st.session_state['hf_embed_model'] = st.text_input("Enter HF repository name")
                st.session_state['hf_api_key'] = st.text_input("Enter HF API key")
                st.session_state['chosen_embedding_model'] = HuggingFaceEmbeddings(
                    model_name=st.session_state['hf_embed_model'],
                    api_key=st.session_state['hf_api_key']
                )

            else:
                st.session_state['tei_api_url'] = st.text_input("Enter TEI(Text-Embedding-Inference) api url")
                st.session_state['chosen_embedding_model'] = TEIEmbeddings(
                    api_url=st.session_state['tei_api_url']
                )

        st.divider()

        # --- setting chunking parameters
        chunking_on = st.toggle("### Settings for *CHUNKING* model")
        if chunking_on:
            st.session_state['chunk_size'] = st.number_input("Chunk size",
                                                             value=500,
                                                             min_value=100,
                                                             max_value=1000,
                                                             step=100)
            st.session_state['chunk_overlap'] = st.number_input("Chunk overlap",
                                                                value=0,
                                                                min_value=0,
                                                                max_value=100,
                                                                step=10)
            st.session_state['chunking_method'] = st.radio("**Select chunking method**",
                                                           ["Default", "TokenSpitter", "CharSplitter"],
                                                           horizontal=True)

            if st.session_state['chunking_method'] == "CharSplitter":
                st.session_state['chunk_separator'] = st.text_input(
                    "Enter the separator of large input",
                    placeholder="default to \\n\\n"
                )
                st.session_state['split_func'] = CharSplitter(
                    separator=st.session_state['chunk_separator'].encode('utf-8').decode('unicode_escape') if st.session_state["chunk_separator"] else None,
                    chunk_size=st.session_state['chunk_size'],
                    chunk_overlap=st.session_state['chunk_overlap'],
                )
            elif st.session_state['chunking_method'] == "Default":
                st.session_state['chunk_separators'] = st.text_input(
                    'Enter the separators of large input in format `["xxx", "xxx"]`',
                    placeholder='default to ["\\n\\n", "\\n", ". ", " ", ""]',
                )
                st.session_state['token_size'] = st.number_input(
                    "Enter token size of chunk",
                    value=int(st.session_state['chunk_size'] // 2),
                    min_value=50,
                    max_value=st.session_state['chunk_size'],
                    step=10,
                )
                st.session_state['token_overlap'] = st.number_input(
                    "Enter token overlap of chunk",
                    value=0,
                    min_value=0,
                    max_value=100,
                    step=10,
                )
                st.session_state['split_func'] = RecursiveChar2TokenSplitter(
                    chunk_size=st.session_state['chunk_size'],
                    chunk_overlap=st.session_state['chunk_overlap'],
                    separators=json.loads(st.session_state['chunk_separators'], strict=False) if st.session_state['chunk_separators'] else None,
                )
            else:
                st.session_state['split_func'] = TokenSplitter(
                    chunk_size=st.session_state['chunk_size'],
                    chunk_overlap=st.session_state['chunk_overlap'],
                )

    if st.button("Build Vector DB"):
        st.session_state['client'] = RAGVizChain(embedding_model=st.session_state['chosen_embedding_model'],
                                                 llm=st.session_state['chosen_llm_model'],
                                                 reader=st.session_state['file_reader'],
                                                 split_func=st.session_state['split_func'])
        main_page.empty()
        main_button.empty()
        with st.spinner("Building Vector DB"):
            st.session_state['client'].load_data(uploaded_file, )
            st.session_state['loaded'] = True
            st.rerun()
else:
    with st.sidebar:
        st.subheader("Parameters")
        st.caption("Used by `HyAE` and `Multi-Sub-Questions` retrival techniques.")
        st.session_state['llm_temperature'] = st.slider(
            "**Temperature**",
            value=0.8,
            min_value=0.0,
            max_value=1.0,
            step=0.1,
        )
        st.session_state['llm_max_tokens'] = st.slider(
            "**Max Tokens**",
            value=1024,
            min_value=0,
            max_value=8192,
            step=1,
        )
        st.session_state['llm_json_mode'] = st.checkbox("**JSON Mode**")

        st.divider()

        advanced_option_on = st.toggle("### Advanced")
        if advanced_option_on:
            st.session_state['llm_top_p'] = st.slider(
                "**Top P**",
                value=1.0,
                min_value=0.0,
                max_value=1.0,
                step=0.1
            )
            st.session_state['llm_seed'] = st.text_input("**Seed**")

        st.session_state['llm_config'] = {
            "temperature": st.session_state['llm_temperature'],
            "max_tokens": st.session_state['llm_max_tokens'],
            "response_format": "json_object" if st.session_state['llm_json_mode'] else None,
            "top_p": st.session_state['llm_top_p'] if st.session_state['llm_top_p'] else 1.0,
            "seed": None if not st.session_state['llm_seed'] else st.session_state['llm_seed'],
        }

        st.session_state['client'].config_llm(st.session_state['llm_config'])

    with st.container():

        def erase_llm_gen():
            llm_gen_placeholder.empty()

        col1, col2 = st.columns(2)
        st.session_state['query'] = col1.text_area("**Enter your query here**")
        st.session_state['technique'] = col1.radio("**Select retrival technique**",
                                                   ["Naive", "HyAE", "Multi-Sub-Questions"],
                                                   horizontal=True)

        if st.session_state['technique'] == "HyAE":
            st.session_state['hyae_prompt'] = col1.text_area("**Prompt used by LLM to generate a hypothetical answer**",
                                                             value=HYDE_SYS_MSG)
            HYDE_SYS_MSG = st.session_state['hyae_prompt']

        elif st.session_state['technique'] == "Multi-Sub-Questions":
            st.session_state['sub_qns_prompt'] = col1.text_area("**Prompt used by LLM to generate sub-questions**",
                                                                value=MULTIPLE_QNS_SYS_MSG)
            MULTIPLE_QNS_SYS_MSG = st.session_state['sub_qns_prompt']

        llm_gen_placeholder = col1.empty()

        st.session_state["top_k"] = col1.number_input("**Top k**", value=5, min_value=1, max_value=10, step=1)
        if col1.button("Execute Query"):
            st.session_state['chart'] = st.session_state['client'].visualize_query(st.session_state['query'],
                                                                                   retrieval_method=st.session_state[
                                                                                       'technique'],
                                                                                   top_k=st.session_state['top_k'])
            st.session_state['query_extension'] = st.session_state['client'].export_query_extension()

        if st.session_state['chart'] is not None:
            col2.plotly_chart(st.session_state['chart'])

        if st.session_state['query_extension'] is not None:
            llm_gen = []
            for q in st.session_state['query_extension']:
                llm_gen.append(f"- :violet[{q}]")
            llm_gen_placeholder.info("\n".join(llm_gen))

        if col1.button("Reset Application"):
            st.session_state['loaded'] = False
            st.rerun()

    st.divider()

    if st.session_state['chart'] is not None:
        st.session_state['retrieved_ids'], st.session_state['all_docs'] = st.session_state['client'].visualize_chunking()

        id_map = {}
        anchor_text = []
        for i, _id in enumerate(st.session_state['retrieved_ids']):
            anchor_text.append(f":material/arrow_forward:[Retrieved Chunk {i + 1}](#Retrieved-{i + 1})")
            id_map.update({int(_id): i + 1})
        st.markdown("\t".join(anchor_text))

        with st.container(border=True, height=500):
            for i, doc in enumerate(st.session_state['all_docs']):
                if i in id_map:
                    st.subheader(f"Retrieved-{id_map[i]}", anchor=f"Retrieved-{id_map[i]}")
                    st.markdown(f":green[{doc}]")
                else:
                    st.markdown(doc)
