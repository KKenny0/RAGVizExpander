from typing import List
import re
import docx
from docx.document import Document as doctwo
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph

from unstructured.partition.docx import partition_docx
from llama_index.core import SimpleDirectoryReader

from .base import DocumentLoader, LoaderStrategy, LoaderFactory


class DocxNativeStrategy(LoaderStrategy):
    """Native docx loading strategy using python-docx"""

    toc_pattern = r"|".join("^" + re.escape(name)
                            for name in ["目录", "contents", "table of contents", "致谢", "acknowledge"])

    def _load_single_table(self, table) -> List[List[str]]:
        n_row = len(table.rows)
        n_col = len(table.columns)
        arrays = [["" for _ in range(n_row)] for _ in range(n_col)]
        for i, row in enumerate(table.rows):
            for j, cell in enumerate(row.cells):
                arrays[j][i] = cell.text
        return arrays

    def _iter_block_items(self, parent):
        if isinstance(parent, doctwo):
            parent_elm = parent.element.body
        elif isinstance(parent, _Cell):
            parent_elm = parent._tc
        else:
            raise ValueError("something's not right")

        for child in parent_elm.iterchildren():
            if isinstance(child, CT_P):
                yield Paragraph(child, parent)
            elif isinstance(child, CT_Tbl):
                yield Table(child, parent)

    def load(self, file_path: str) -> List[str]:
        doc = docx.Document(file_path)
        texts = []
        for block in self._iter_block_items(doc):
            if isinstance(block, Paragraph):
                text = block.text.strip()
                if text and not re.search(self.toc_pattern, text.lower()):
                    texts.append(text)
            elif isinstance(block, Table):
                table_data = self._load_single_table(block)
                texts.extend([cell for row in table_data for cell in row if cell.strip()])
        return texts


class DocxUnstructuredStrategy(LoaderStrategy):
    """Unstructured library loading strategy"""

    def load(self, file_path: str) -> List[str]:
        elements = partition_docx(file_path)
        return ["\n".join([ele.text.strip() for ele in elements])]


class DocxLlamaIndexStrategy(LoaderStrategy):
    """LlamaIndex loading strategy"""

    def load(self, file_path: str) -> List[str]:
        reader = SimpleDirectoryReader(input_files=[file_path])
        document = reader.load_data()[0]
        return [document.text.strip()]


class DocxLoader(DocumentLoader):
    """DOCX document loader with multiple loading strategies"""

    def __init__(self, strategy: LoaderStrategy = None):
        strategy = strategy or DocxNativeStrategy()
        super().__init__(strategy)

    def load_data(self, file_path: str) -> List[str]:
        if not self.validate_file(file_path):
            raise ValueError(f"Invalid or non-existent file: {file_path}")
        return self._strategy.load(file_path)

    def supported_extensions(self) -> List[str]:
        return ['.docx']


# Register valid strategies for DocxLoader
DocxLoader.register_strategies([
    DocxNativeStrategy,
    DocxUnstructuredStrategy,
    DocxLlamaIndexStrategy
])
