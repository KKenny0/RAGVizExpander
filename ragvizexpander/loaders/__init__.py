
from .pdf_loader import PdfLoader
from .ppt_loader import PptLoader
from .txt_loader import TxtLoader
from .docx_loader import DocxLoader


extractors = {
    ".pdf": PdfLoader(),
    ".pptx": PptLoader(),
    ".txt": TxtLoader(),
    ".docx": DocxLoader(),
}
