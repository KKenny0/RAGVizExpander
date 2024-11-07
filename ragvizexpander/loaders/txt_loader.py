from typing import List
from unstructured.partition.text import partition_text
from llama_index.core import SimpleDirectoryReader


class TxtLoader:
    def load_data(self, file: str) -> List[str]:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()

        return [text]


class UnstructuredTxtLoader:
    """Load data using unstructured library
    Ref: https://docs.unstructured.io/open-source/core-functionality/partitioning#partition-text
    """

    def load_data(self, file_path: str):
        elements = partition_text(file_path)
        all_text = "\n".join([ele.text.strip() for ele in elements])
        return [all_text]


class LlamaIndexTextLoader:
    """Load data using llama-index library
    Ref: https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader/
    """

    def load_data(self, file_path: str) -> List[str]:
        reader = SimpleDirectoryReader(input_files=[file_path])
        document = reader.load_data()[0]
        return [document.text.strip()]
