from langchain.text_splitter import TokenTextSplitter

__all__ = ["TokenSplitter"]


class TokenSplitter:
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 20):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.token_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def __call__(self, texts: list[str]) -> list[str]:
        """Split text into chunks."""
        return [t.strip()
                for t in self.token_splitter.split_text("\n".join(texts))]
