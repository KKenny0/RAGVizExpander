import unicodedata

import pandas as pd
from typing import Tuple, List

import re
import docx
from docx.document import Document as doctwo
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph

import docx2txt
from unstructured.partition.docx import partition_docx
from llama_index.core import SimpleDirectoryReader


class DocxLoader:
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

    def _extract_content(self,) -> Tuple[List[str], List[str]]:
        """Extract table of content and text body from document

        Args:
            doc (docx.Document):

        Returns:
            list[str]: a list contains text body
            list[str]: a list contains table of content
        """
        combined_df = []
        for block in self._iter_block_items(self.doc):
            is_append = False
            if "text" in str(block):
                is_append = False
                run_bold_text = ""
                for run in block.runs:
                    if run.bold:
                        run_bold_text += run.text
                try:
                    style = str(block.style.name)
                except Exception as e:
                    print(f"~ something's wrong: {e}")

                append_txt = str(block.text)
                append_txt = re.sub("\u3000|\xa0|\u00A0|\u2002|\u2003|\u0020", " ", append_txt)
                append_txt = re.sub("\n{2,}", "\n", append_txt)
                append_txt = append_txt.strip()

                if not append_txt:
                    continue

                tab_id = "Novalue"
                is_append = True

            if is_append:
                dftemp = pd.DataFrame({"para_text": [append_txt],
                                       "table_id": [tab_id],
                                       "style": [style]})
                combined_df.append(dftemp)
        combined_df = pd.concat(combined_df, sort=False).reset_index(drop=True)

        toc_list = []
        all_text_temp = []
        for row_idx, doc_row in enumerate(combined_df.to_dict(orient="records")):
            para_text = doc_row["para_text"]

            if "toc" in doc_row["style"] or re.search(self.toc_pattern,
                                                      re.sub("\s", "", para_text),
                                                      re.IGNORECASE):
                toc_list.append(para_text)
            else:
                all_text_temp.append(para_text)

        all_text = []
        cand_text = []
        i = 0
        while i < len(all_text_temp):
            if re.search("(\w)$", all_text_temp[i]) or (
                    all_text_temp[i].endswith((",", "，"))):
                cand_text.append(all_text_temp[i])
            else:
                cand_text.append(all_text_temp[i])
                text_seg = " ".join(cand_text)
                if text_seg:
                    all_text.append(text_seg)
                cand_text = []
            i += 1
        if cand_text:
            all_text.append(" ".join(cand_text))

        return all_text, toc_list

    def load_data(
        self, file_path: str
    ) -> List[str]:
        """Load data using Docx reader, considering only text and tables.

        Args:
            - file_path (str): Path to .docx file

        Returns:
            List[str]: list of documents extracted from file
        """
        self.doc = docx.Document(file_path)
        all_text, all_toc = self._extract_content()
        all_toc = "\n".join(all_toc)
        all_text = "\n".join(
            [unicodedata.normalize("NFKC", text) for text in all_text]
        )
        pages = [all_text]  # 1 page only

        tables = []
        for t in self.doc.tables:
            # return list of columns: list of string
            arrays = self._load_single_table(t)

            tables.append(pd.DataFrame({a[0]: a[1:] for a in arrays}))

        # create output Document with metadata from table
        documents = [table.to_csv(index=False).strip()
                     for table in tables]

        # create Document from non-table text
        documents.extend([non_table_text.strip()
                          for _, non_table_text in enumerate(pages)])

        # create Document from toc
        if all_toc:
            documents.extend([all_toc.strip()])

        return documents


class Docx2txtReader:
    """Load data using docx2txt library
    Ref: https://github.com/ankushshah89/python-docx2txt
    """

    def load_data(self, file_path: str) -> List[str]:
        text = docx2txt.process(file_path)
        return [text]


class UnstructuredDocxReader:
    """Load data using unstructured library
    Ref: https://docs.unstructured.io/open-source/core-functionality/partitioning#partition-docx
    """

    def load_data(self, file_path: str):
        elements = partition_docx(filename=file_path)
        all_text = "\n".join([ele.text.strip() for ele in elements])
        return [all_text]


class LlamaIndexDocxReader:
    """Load data using llama-index library
    Ref: https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader/
    """

    def load_data(self, file_path: str) -> List[str]:
        reader = SimpleDirectoryReader(input_files=[file_path])
        document = reader.load_data()[0]
        return [document.text.strip()]
