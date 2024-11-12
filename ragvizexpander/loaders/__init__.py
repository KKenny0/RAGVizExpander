from .pdf_loader import (
    PdfLoader,
    UnstructuredPdfLoader,
    LlamaIndexPdfLoader,
)
from .pptx_loader import (
    PptxLoader,
    UnstructuredPptxReader,
    LlamaIndexPptxReader,
)
from .txt_loader import (
    TxtLoader,
    UnstructuredTxtLoader,
    LlamaIndexTextLoader,
)
from .docx_loader import (
    DocxLoader,
    UnstructuredDocxReader,
    LlamaIndexDocxReader,
)


extractors = {
    ".pdf": PdfLoader(),
    ".pptx": PptxLoader(),
    ".txt": TxtLoader(),
    ".docx": DocxLoader(),
}

app_extractors = {
    ".pdf": {
        "Default": PdfLoader(),
        "Unstructured": UnstructuredPdfLoader(),
        "LlamaIndex": LlamaIndexPdfLoader(),
    },
    ".pptx": {
        "Default": PptxLoader(),
        "Unstructured": UnstructuredPptxReader(),
        "LlamaIndex": LlamaIndexPptxReader(),
    },
    ".txt": {
        "Default": TxtLoader(),
        "Unstructured": UnstructuredTxtLoader(),
        "LlamaIndex": LlamaIndexTextLoader(),
    },
    ".docx": {
        "Default": DocxLoader(),
        "Unstructured": UnstructuredDocxReader(),
        "LlamaIndex": LlamaIndexDocxReader(),
    }
}
