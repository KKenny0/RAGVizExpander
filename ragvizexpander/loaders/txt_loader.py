from typing import List
from unstructured.partition.text import partition_text
from llama_index.core import SimpleDirectoryReader

from .base import DocumentLoader, LoaderStrategy, LoaderFactory


class TxtNativeStrategy(LoaderStrategy):
    """Native text loading strategy"""

    def load(self, file_path: str) -> List[str]:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return [text.strip()]


class TxtUnstructuredStrategy(LoaderStrategy):
    """Unstructured library loading strategy for text files"""

    def load(self, file_path: str) -> List[str]:
        elements = partition_text(file_path)
        return ["\n".join([ele.text.strip() for ele in elements])]


class TxtLlamaIndexStrategy(LoaderStrategy):
    """LlamaIndex loading strategy for text files"""

    def load(self, file_path: str) -> List[str]:
        reader = SimpleDirectoryReader(input_files=[file_path])
        document = reader.load_data()[0]
        return [document.text.strip()]


class TxtLoader(DocumentLoader):
    """Text file loader with multiple loading strategies"""

    def __init__(self, strategy: LoaderStrategy = None):
        strategy = strategy or TxtNativeStrategy()
        super().__init__(strategy)

    def load_data(self, file_path: str) -> List[str]:
        if not self.validate_file(file_path):
            raise ValueError(f"Invalid or non-existent file: {file_path}")
        return self._strategy.load(file_path)

    def supported_extensions(self) -> List[str]:
        return ['.txt']


# Register valid strategies for TxtLoader
TxtLoader.register_strategies([
    TxtNativeStrategy,
    TxtUnstructuredStrategy,
    TxtLlamaIndexStrategy
])
