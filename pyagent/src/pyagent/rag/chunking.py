"""Document chunking strategies."""

import uuid
from typing import Optional

from .base import BaseChunker
from .document import Chunk, Document


class FixedSizeChunker(BaseChunker):
    """Chunk documents into fixed-size pieces with optional overlap."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        if overlap >= chunk_size:
            raise ValueError("Overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content
        chunks = []
        start, chunk_index = 0, 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            chunks.append(Chunk(
                id=f"{document.id}_chunk_{chunk_index}", document_id=document.id, content=chunk_text,
                metadata={**document.metadata, "chunk_index": chunk_index, "chunker": "fixed_size"},
                start_index=start, end_index=end
            ))
            start = end - self.overlap if end < len(text) else end
            chunk_index += 1
        return chunks


class RecursiveChunker(BaseChunker):
    """Recursively chunk documents using multiple separators."""

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, chunk_size: int = 500, overlap: int = 50, separators: Optional[list[str]] = None):
        if overlap >= chunk_size:
            raise ValueError("Overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def chunk(self, document: Document) -> list[Chunk]:
        return self._split_text(document.content, document, self.separators)

    def _split_text(self, text: str, document: Document, separators: list[str]) -> list[Chunk]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [Chunk(id=f"{document.id}_chunk_{uuid.uuid4().hex[:8]}", document_id=document.id, content=text,
                         metadata={**document.metadata, "chunker": "recursive"}, start_index=0, end_index=len(text))]
        separator = next((sep for sep in separators if sep and sep in text), None)
        if separator is None:
            return self._split_fixed(text, document)
        splits = text.split(separator)
        chunks, current_chunk, current_start, chunk_index = [], "", 0, 0
        for i, split in enumerate(splits):
            piece = split + separator if i < len(splits) - 1 else split
            if len(current_chunk) + len(piece) <= self.chunk_size:
                current_chunk += piece
            else:
                if current_chunk.strip():
                    chunks.append(Chunk(id=f"{document.id}_chunk_{chunk_index}", document_id=document.id,
                        content=current_chunk.strip(), metadata={**document.metadata, "chunk_index": chunk_index, "chunker": "recursive"},
                        start_index=current_start, end_index=current_start + len(current_chunk)))
                    chunk_index += 1
                current_chunk = piece
                current_start += len(current_chunk)
        if current_chunk.strip():
            chunks.append(Chunk(id=f"{document.id}_chunk_{chunk_index}", document_id=document.id,
                content=current_chunk.strip(), metadata={**document.metadata, "chunk_index": chunk_index, "chunker": "recursive"},
                start_index=current_start, end_index=current_start + len(current_chunk)))
        return chunks

    def _split_fixed(self, text: str, document: Document) -> list[Chunk]:
        chunks, start, chunk_index = [], 0, 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(Chunk(id=f"{document.id}_chunk_{chunk_index}", document_id=document.id, content=text[start:end],
                metadata={**document.metadata, "chunk_index": chunk_index, "chunker": "fixed_fallback"},
                start_index=start, end_index=end))
            start = end - self.overlap if end < len(text) else end
            chunk_index += 1
        return chunks


class SemanticChunker(BaseChunker):
    """Chunk documents based on semantic boundaries."""

    def __init__(self, min_chunk_size: int = 100, max_chunk_size: int = 1000, separators: Optional[list[str]] = None):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.separators = separators or ["\n\n", "\n", ". ", "! ", "? ", " "]

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content
        paragraphs, chunks, current_chunk, current_start, chunk_index = text.split("\n\n"), [], "", 0, 0
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(para) > self.max_chunk_size:
                if current_chunk and len(current_chunk) >= self.min_chunk_size:
                    chunks.append(self._create_chunk(document, current_chunk, current_start, chunk_index))
                    chunk_index += 1
                    current_chunk = ""
                para_chunks = self._split_large_text(para, document, chunk_index, current_start)
                chunks.extend(para_chunks)
                chunk_index += len(para_chunks)
                continue
            if current_chunk and len(current_chunk) + len(para) + 2 > self.max_chunk_size:
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(self._create_chunk(document, current_chunk, current_start, chunk_index))
                    chunk_index += 1
                current_chunk = para
            else:
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(self._create_chunk(document, current_chunk, current_start, chunk_index))
        return chunks

    def _split_large_text(self, text: str, document: Document, start_index: int, chunk_index: int) -> list[Chunk]:
        recursive = RecursiveChunker(chunk_size=self.max_chunk_size, overlap=0, separators=self.separators)
        temp_doc = Document(id=document.id, content=text, metadata=document.metadata)
        return [Chunk(id=f"{document.id}_chunk_{chunk_index + i}", document_id=document.id, content=c.content,
                     metadata={**c.metadata, "chunk_index": chunk_index + i, "chunker": "semantic"},
                     start_index=start_index + c.start_index, end_index=start_index + c.end_index)
                for i, c in enumerate(recursive.chunk(temp_doc))]

    def _create_chunk(self, document: Document, content: str, start: int, index: int) -> Chunk:
        return Chunk(id=f"{document.id}_chunk_{index}", document_id=document.id, content=content,
                    metadata={**document.metadata, "chunk_index": index, "chunker": "semantic"},
                    start_index=start, end_index=start + len(content))
