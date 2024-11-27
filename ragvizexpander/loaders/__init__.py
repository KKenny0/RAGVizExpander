from .pdf_loader import *
from .pptx_loader import *
from .txt_loader import *
from .docx_loader import *
from .base import LoaderFactory


# Register the loader with the factory
loader_factory = LoaderFactory()
loader_factory.register_loader(['.docx'], DocxLoader)
loader_factory.register_loader(['.txt'], TxtLoader)
loader_factory.register_loader(['.pdf'], PdfLoader)
loader_factory.register_loader(['.pptx'], PptxLoader)
