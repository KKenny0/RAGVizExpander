from langchain.text_splitter import CharacterTextSplitter

__all__ = ["CharSplitter"]


class CharSplitter:
    def __init__(
            self,
            separator: str = "\n\n",
            chunk_size: int = 1024,
            chunk_overlap: int = 20
    ):
        self.separator = separator
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.char_splitter = CharacterTextSplitter(
            separator=self.separator,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def __call__(self, texts: list[str]) -> list[str]:
        """Split text into chunks."""
        return [t.strip()
                for t in self.char_splitter.split_text("\n\n".join(texts))]
