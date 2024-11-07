from PyPDF2 import PdfReader
from typing import List
from unstructured.partition.pdf import partition_pdf
from llama_index.core import SimpleDirectoryReader


class PdfLoader:

    def load_data(
        self,
        file: str,
    ) -> List[str]:
        pdf = PdfReader(file)
        pdf_texts = [p.extract_text().strip()
                     for p in pdf.pages if p.extract_text()]
        return pdf_texts


class UnstructuredPdfLoader:
    """Load data using unstructured library
    Ref: https://docs.unstructured.io/open-source/core-functionality/partitioning#partition-text
    """

    def load_data(self, file_path: str):
        elements = partition_pdf(file_path)
        all_text = "\n".join([ele.text.strip() for ele in elements])
        return [all_text]


class LlamaIndexPdfLoader:
    """Load data using llama-index library
    Ref: https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader/
    """

    def load_data(self, file_path: str) -> List[str]:
        reader = SimpleDirectoryReader(input_files=[file_path])
        document = reader.load_data()[0]
        return [document.text.strip()]
