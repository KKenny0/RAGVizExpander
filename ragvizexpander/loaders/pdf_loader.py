import io

import tempfile

from PyPDF2 import PdfReader
from typing import List
from unstructured.partition.pdf import partition_pdf
from llama_index.core import SimpleDirectoryReader
from docling.document_converter import DocumentConverter

from .base import DocumentLoader, LoaderStrategy



class PdfNativeStrategy(LoaderStrategy):
    """Native PDF loading strategy using PyPDF2"""

    def load(self, file_path: str) -> List[str]:
        pdf = PdfReader(file_path)
        return [p.extract_text().strip() for p in pdf.pages if p.extract_text()]


class PdfUnstructuredStrategy(LoaderStrategy):
    """Unstructured library loading strategy for PDF"""

    def load(self, file_path: str) -> List[str]:
        elements = partition_pdf(file_path)
        return ["\n".join([ele.text.strip() for ele in elements])]


class PdfLlamaIndexStrategy(LoaderStrategy):
    """LlamaIndex loading strategy for PDF"""

    def load(self, file_path: str) -> List[str]:
        reader = SimpleDirectoryReader(input_files=[file_path])
        document = reader.load_data()[0]
        return [document.text.strip()]


class PdfDoclingStrategy(LoaderStrategy):
    """docling loading strategy for PDF"""

    def load(self, file_path) -> List[str]:
        converter = DocumentConverter()
        if isinstance(file_path, io.BytesIO):
            with tempfile.NamedTemporaryFile(suffix=".docx") as temp:
                temp.write(file_path.getbuffer())
                temp.seek(0)
                result = converter.convert(temp.name)
        else:
            result = converter.convert(file_path)
        return [result.document.export_to_markdown()]


class PdfLoader(DocumentLoader):
    """PDF document loader with multiple loading strategies"""

    def __init__(self, strategy: LoaderStrategy = None):
        strategy = strategy or PdfNativeStrategy()
        super().__init__(strategy)

    def load_data(self, file_path: str) -> List[str]:
        if not self.validate_file(file_path):
            raise ValueError(f"Invalid or non-existent file: {file_path}")
        return self._strategy.load(file_path)

    def supported_extensions(self) -> List[str]:
        return ['.pdf']


# Register valid strategies for PdfLoader
PdfLoader.register_strategies([
    PdfNativeStrategy,
    PdfUnstructuredStrategy,
    PdfLlamaIndexStrategy
])
