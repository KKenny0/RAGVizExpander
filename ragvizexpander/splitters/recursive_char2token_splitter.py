from typing import List
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)


class RecursiveChar2TokenSplitter:
    def __init__(
            self,
            chunk_size: int = 1024,
            chunk_overlap: int = 20,
            separators: List[str] = None,
            token_size: int = 256,
            token_overlap: int = 0,
    ):
        if not separators:
            separators = ["\n\n", "\n", ". ", " ", ""]
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators
        self.token_size = token_size
        self.token_overlap = token_overlap

        self._init_splitter()

    def _init_splitter(self):
        self.char_splitter = RecursiveCharacterTextSplitter(
            separators=self.separators,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        self.token_splitter = TokenTextSplitter(
            chunk_size=self.token_size,
            chunk_overlap=self.token_overlap,
        )

    def __call__(self, texts: List[str]) -> List[str]:
        """Split text into chunks."""
        character_split_texts = self.char_splitter.split_text("\n\n".join(texts))
        return [text
                for chunk in character_split_texts
                for text in self.token_splitter.split_text(chunk)]
