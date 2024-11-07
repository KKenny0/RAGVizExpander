
from .pdf_loader import PdfLoader
from .pptx_loader import PptxLoader
from .txt_loader import TxtLoader
from .docx_loader import DocxLoader


extractors = {
    ".pdf": PdfLoader(),
    ".pptx": PptxLoader(),
    ".txt": TxtLoader(),
    ".docx": DocxLoader(),
}
