from PyPDF2 import PdfReader
from typing import List


class PdfLoader:

    def load_data(
        self,
        file: str,
    ) -> List[str]:
        pdf = PdfReader(file)
        pdf_texts = [p.extract_text().strip()
                     for p in pdf.pages if p.extract_text()]
        return pdf_texts
