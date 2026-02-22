"""Document chunking strategies."""

import uuid
from typing import Optional

from .base import BaseChunker
from .document import Chunk, Document


class FixedSizeChunker(BaseChunker):
    """Chunk documents into fixed-size pieces with optional overlap.

    Simple but effective chunking strategy that splits text into
    chunks of a specified character count.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
    ):
        """Initialize the fixed-size chunker.

        Args:
            chunk_size: Maximum characters per chunk
            overlap: Number of characters to overlap between chunks
        """
        if overlap >= chunk_size:
            raise ValueError("Overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document into fixed-size chunks."""
        text = document.content
        chunks = []

        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]

            chunk = Chunk(
                id=f"{document.id}_chunk_{chunk_index}",
                document_id=document.id,
                content=chunk_text,
                metadata={
                    **document.metadata,
                    "chunk_index": chunk_index,
                    "chunker": "fixed_size",
                },
                start_index=start,
                end_index=end,
            )
            chunks.append(chunk)

            # Move start position, accounting for overlap
            start = end - self.overlap if end < len(text) else end
            chunk_index += 1

        return chunks


class RecursiveChunker(BaseChunker):
    """Recursively chunk documents using multiple separators.

    Tries to split on larger separators first (paragraphs), then
    progressively smaller ones (sentences, words) as needed.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        separators: Optional[list[str]] = None,
    ):
        """Initialize the recursive chunker.

        Args:
            chunk_size: Maximum characters per chunk
            overlap: Number of characters to overlap between chunks
            separators: List of separators to try (in order of preference)
        """
        if overlap >= chunk_size:
            raise ValueError("Overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document recursively."""
        return self._split_text(
            document.content,
            document,
            self.separators,
        )

    def _split_text(
        self,
        text: str,
        document: Document,
        separators: list[str],
    ) -> list[Chunk]:
        """Recursively split text using separators."""
        if not text:
            return []

        # If text is small enough, return as single chunk
        if len(text) <= self.chunk_size:
            return [Chunk(
                id=f"{document.id}_chunk_{uuid.uuid4().hex[:8]}",
                document_id=document.id,
                content=text,
                metadata={
                    **document.metadata,
                    "chunker": "recursive",
                },
                start_index=0,
                end_index=len(text),
            )]

        # Find the best separator
        separator = None
        for sep in separators:
            if sep and sep in text:
                separator = sep
                break

        if separator is None:
            # No separator found, fall back to fixed-size
            return self._split_fixed(text, document)

        # Split by separator
        splits = text.split(separator)
        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_index = 0

        for i, split in enumerate(splits):
            # Add separator back (except for last split)
            piece = split + separator if i < len(splits) - 1 else split

            if len(current_chunk) + len(piece) <= self.chunk_size:
                current_chunk += piece
            else:
                # Save current chunk if not empty
                if current_chunk.strip():
                    chunks.append(Chunk(
                        id=f"{document.id}_chunk_{chunk_index}",
                        document_id=document.id,
                        content=current_chunk.strip(),
                        metadata={
                            **document.metadata,
                            "chunk_index": chunk_index,
                            "chunker": "recursive",
                        },
                        start_index=current_start,
                        end_index=current_start + len(current_chunk),
                    ))
                    chunk_index += 1

                # Start new chunk with overlap
                if self.overlap > 0 and len(current_chunk) > self.overlap:
                    overlap_text = current_chunk[-self.overlap:]
                    current_start = current_start + len(current_chunk) - self.overlap
                    current_chunk = overlap_text + piece
                else:
                    current_start = current_start + len(current_chunk)
                    current_chunk = piece

        # Add final chunk
        if current_chunk.strip():
            chunks.append(Chunk(
                id=f"{document.id}_chunk_{chunk_index}",
                document_id=document.id,
                content=current_chunk.strip(),
                metadata={
                    **document.metadata,
                    "chunk_index": chunk_index,
                    "chunker": "recursive",
                },
                start_index=current_start,
                end_index=current_start + len(current_chunk),
            ))

        return chunks

    def _split_fixed(self, text: str, document: Document) -> list[Chunk]:
        """Fall back to fixed-size splitting."""
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]

            chunks.append(Chunk(
                id=f"{document.id}_chunk_{chunk_index}",
                document_id=document.id,
                content=chunk_text,
                metadata={
                    **document.metadata,
                    "chunk_index": chunk_index,
                    "chunker": "fixed_fallback",
                },
                start_index=start,
                end_index=end,
            ))

            start = end - self.overlap if end < len(text) else end
            chunk_index += 1

        return chunks


class SemanticChunker(BaseChunker):
    """Chunk documents based on semantic boundaries.

    Attempts to split on paragraph and sentence boundaries while
    respecting size constraints.
    """

    def __init__(
        self,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
        separators: Optional[list[str]] = None,
    ):
        """Initialize the semantic chunker.

        Args:
            min_chunk_size: Minimum characters per chunk
            max_chunk_size: Maximum characters per chunk
            separators: Custom separators (defaults to paragraph/sentence)
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.separators = separators or ["\n\n", "\n", ". ", "! ", "? ", " "]

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document semantically."""
        text = document.content
        chunks = []

        # Split into paragraphs first
        paragraphs = text.split("\n\n")
        current_chunk = ""
        current_start = 0
        chunk_index = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If paragraph alone exceeds max size, split it
            if len(para) > self.max_chunk_size:
                # Save current chunk if not empty
                if current_chunk and len(current_chunk) >= self.min_chunk_size:
                    chunks.append(self._create_chunk(
                        document, current_chunk, current_start, chunk_index
                    ))
                    chunk_index += 1
                    current_chunk = ""
                    current_start += len(current_chunk)

                # Split large paragraph
                para_chunks = self._split_large_text(para, document, chunk_index, current_start)
                chunks.extend(para_chunks)
                chunk_index += len(para_chunks)
                if para_chunks:
                    current_start = para_chunks[-1].end_index
                continue

            # Check if adding paragraph exceeds max size
            if current_chunk and len(current_chunk) + len(para) + 2 > self.max_chunk_size:
                # Save current chunk
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(self._create_chunk(
                        document, current_chunk, current_start, chunk_index
                    ))
                    chunk_index += 1
                current_chunk = para
                current_start = text.find(para, current_start)
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                    current_start = text.find(para)

        # Add final chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(self._create_chunk(
                document, current_chunk, current_start, chunk_index
            ))

        return chunks

    def _split_large_text(
        self,
        text: str,
        document: Document,
        start_index: int,
        chunk_index: int,
    ) -> list[Chunk]:
        """Split large text using recursive chunker."""
        recursive = RecursiveChunker(
            chunk_size=self.max_chunk_size,
            overlap=0,
            separators=self.separators,
        )

        temp_doc = Document(
            id=document.id,
            content=text,
            metadata=document.metadata,
        )

        chunks = recursive.chunk(temp_doc)

        # Adjust indices and IDs
        result = []
        for i, chunk in enumerate(chunks):
            result.append(Chunk(
                id=f"{document.id}_chunk_{chunk_index + i}",
                document_id=document.id,
                content=chunk.content,
                metadata={
                    **chunk.metadata,
                    "chunk_index": chunk_index + i,
                    "chunker": "semantic",
                },
                start_index=start_index + chunk.start_index,
                end_index=start_index + chunk.end_index,
            ))

        return result

    def _create_chunk(
        self,
        document: Document,
        content: str,
        start: int,
        index: int,
    ) -> Chunk:
        """Create a chunk with proper metadata."""
        return Chunk(
            id=f"{document.id}_chunk_{index}",
            document_id=document.id,
            content=content,
            metadata={
                **document.metadata,
                "chunk_index": index,
                "chunker": "semantic",
            },
            start_index=start,
            end_index=start + len(content),
        )
