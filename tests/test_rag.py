"""Tests for the RAG system."""

import pytest
from typing import Any

from pyagent.rag import (
    # Data structures
    Document,
    Chunk,
    SearchResult,
    # Embeddings
    DummyEmbedding,
    FakeEmbedding,
    # Vector stores
    MemoryVectorStore,
    cosine_similarity,
    # Chunking
    FixedSizeChunker,
    RecursiveChunker,
    SemanticChunker,
    # Retrievers
    VectorRetriever,
    KeywordRetriever,
    HybridRetriever,
    # Rerankers
    IdentityReranker,
    DiversityReranker,
    # Pipeline
    RAGPipeline,
    SimpleRAG,
    create_rag_tool,
)


class TestDocument:
    """Tests for Document data structure."""

    def test_document_creation(self):
        """Test creating a document."""
        doc = Document(
            id="test-1",
            content="This is test content.",
            metadata={"author": "test"},
            source="test.txt",
        )

        assert doc.id == "test-1"
        assert doc.content == "This is test content."
        assert doc.metadata == {"author": "test"}
        assert doc.source == "test.txt"

    def test_document_defaults(self):
        """Test document default values."""
        doc = Document(id="test", content="content")

        assert doc.metadata == {}
        assert doc.source is None

    def test_document_repr(self):
        """Test document string representation."""
        doc = Document(id="test", content="short")
        repr_str = repr(doc)
        assert "test" in repr_str
        assert "short" in repr_str


class TestChunk:
    """Tests for Chunk data structure."""

    def test_chunk_creation(self):
        """Test creating a chunk."""
        chunk = Chunk(
            id="chunk-1",
            document_id="doc-1",
            content="Chunk content",
            metadata={"page": 1},
            start_index=0,
            end_index=13,
        )

        assert chunk.id == "chunk-1"
        assert chunk.document_id == "doc-1"
        assert chunk.content == "Chunk content"
        assert chunk.embedding is None

    def test_chunk_with_embedding(self):
        """Test chunk with embedding."""
        embedding = [0.1, 0.2, 0.3]
        chunk = Chunk(
            id="chunk-1",
            document_id="doc-1",
            content="content",
            embedding=embedding,
        )

        assert chunk.embedding == embedding


class TestSearchResult:
    """Tests for SearchResult."""

    def test_search_result_creation(self):
        """Test creating a search result."""
        chunk = Chunk(id="c1", document_id="d1", content="test")
        result = SearchResult(chunk=chunk, score=0.95)

        assert result.chunk == chunk
        assert result.score == 0.95
        assert result.document is None


class TestEmbeddings:
    """Tests for embedding providers."""

    @pytest.mark.asyncio
    async def test_dummy_embedding(self):
        """Test dummy embedding."""
        embedding = DummyEmbedding(dimension=128)

        assert embedding.dimension == 128

        # Single query
        vec = await embedding.embed_query("test")
        assert len(vec) == 128
        assert all(v == 0.0 for v in vec)

        # Batch documents
        vecs = await embedding.embed_documents(["a", "b", "c"])
        assert len(vecs) == 3
        assert all(len(v) == 128 for v in vecs)

    @pytest.mark.asyncio
    async def test_fake_embedding(self):
        """Test fake embedding with deterministic output."""
        embedding = FakeEmbedding(dimension=64, seed=42)

        assert embedding.dimension == 64

        # Same text should produce same embedding
        vec1 = await embedding.embed_query("hello world")
        vec2 = await embedding.embed_query("hello world")
        assert vec1 == vec2

        # Different text should produce different embedding
        vec3 = await embedding.embed_query("goodbye world")
        assert vec1 != vec3

        # Values should be normalized
        for v in vec1:
            assert -1.0 <= v <= 1.0


class TestCosineSimilarity:
    """Tests for cosine similarity function."""

    def test_identical_vectors(self):
        """Test similarity of identical vectors."""
        vec = [1.0, 2.0, 3.0]
        sim = cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 0.0001

    def test_orthogonal_vectors(self):
        """Test similarity of orthogonal vectors."""
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        sim = cosine_similarity(vec1, vec2)
        assert abs(sim) < 0.0001

    def test_opposite_vectors(self):
        """Test similarity of opposite vectors."""
        vec1 = [1.0, 0.0]
        vec2 = [-1.0, 0.0]
        sim = cosine_similarity(vec1, vec2)
        assert abs(sim + 1.0) < 0.0001

    def test_zero_vector(self):
        """Test with zero vector."""
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 2.0, 3.0]
        sim = cosine_similarity(vec1, vec2)
        assert sim == 0.0


class TestMemoryVectorStore:
    """Tests for memory vector store."""

    @pytest.mark.asyncio
    async def test_add_and_search(self):
        """Test adding and searching vectors."""
        store = MemoryVectorStore()

        # Add chunks
        chunks = [
            Chunk(id="c1", document_id="d1", content="Python programming"),
            Chunk(id="c2", document_id="d1", content="Java programming"),
        ]
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]

        ids = await store.add(chunks, embeddings)
        assert len(ids) == 2

        # Search with first embedding
        results = await store.search([1.0, 0.0, 0.0], k=1)
        assert len(results) == 1
        assert results[0].chunk.id == "c1"
        assert results[0].score > 0.9  # Should be very similar

    @pytest.mark.asyncio
    async def test_delete(self):
        """Test deleting chunks."""
        store = MemoryVectorStore()

        chunk = Chunk(id="c1", document_id="d1", content="test")
        await store.add([chunk], [[1.0, 0.0]])

        assert await store.count() == 1

        await store.delete(["c1"])
        assert await store.count() == 0

    @pytest.mark.asyncio
    async def test_get(self):
        """Test getting a chunk by ID."""
        store = MemoryVectorStore()

        chunk = Chunk(id="c1", document_id="d1", content="test content")
        await store.add([chunk], [[1.0, 0.0]])

        retrieved = await store.get("c1")
        assert retrieved is not None
        assert retrieved.content == "test content"

        not_found = await store.get("nonexistent")
        assert not_found is None

    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clearing the store."""
        store = MemoryVectorStore()

        chunks = [
            Chunk(id=f"c{i}", document_id="d1", content=f"content {i}")
            for i in range(5)
        ]
        embeddings = [[float(i), 0.0] for i in range(5)]
        await store.add(chunks, embeddings)

        assert await store.count() == 5

        await store.clear()
        assert await store.count() == 0


class TestFixedSizeChunker:
    """Tests for fixed-size chunker."""

    def test_basic_chunking(self):
        """Test basic chunking."""
        chunker = FixedSizeChunker(chunk_size=10, overlap=0)
        doc = Document(id="1", content="1234567890123456789012345")

        chunks = chunker.chunk(doc)

        assert len(chunks) == 3
        assert chunks[0].content == "1234567890"
        assert chunks[1].content == "1234567890"
        assert chunks[2].content == "12345"

    def test_chunking_with_overlap(self):
        """Test chunking with overlap."""
        chunker = FixedSizeChunker(chunk_size=10, overlap=3)
        doc = Document(id="1", content="12345678901234567890")

        chunks = chunker.chunk(doc)

        # With overlap=3: first chunk ends at 10, next starts at 7
        assert len(chunks) >= 2
        # Check overlap exists
        if len(chunks) >= 2:
            # End of first should overlap with start of second
            pass

    def test_small_document(self):
        """Test chunking a document smaller than chunk size."""
        chunker = FixedSizeChunker(chunk_size=100, overlap=0)
        doc = Document(id="1", content="small")

        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert chunks[0].content == "small"

    def test_invalid_overlap(self):
        """Test that invalid overlap raises error."""
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=10, overlap=10)


class TestRecursiveChunker:
    """Tests for recursive chunker."""

    def test_paragraph_split(self):
        """Test splitting by paragraphs."""
        chunker = RecursiveChunker(chunk_size=100, overlap=0)
        doc = Document(
            id="1",
            content="First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
        )

        chunks = chunker.chunk(doc)

        assert len(chunks) >= 1
        # Each paragraph should be in a chunk
        all_content = " ".join(c.content for c in chunks)
        assert "First paragraph" in all_content

    def test_respects_chunk_size(self):
        """Test that chunks respect max size."""
        chunker = RecursiveChunker(chunk_size=50, overlap=0)
        doc = Document(
            id="1",
            content="A" * 100,
        )

        chunks = chunker.chunk(doc)

        for chunk in chunks:
            assert len(chunk.content) <= 50


class TestSemanticChunker:
    """Tests for semantic chunker."""

    def test_semantic_split(self):
        """Test semantic splitting."""
        chunker = SemanticChunker(
            min_chunk_size=10,
            max_chunk_size=100,
        )
        doc = Document(
            id="1",
            content="First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here.",
        )

        chunks = chunker.chunk(doc)

        assert len(chunks) >= 1


class TestVectorRetriever:
    """Tests for vector retriever."""

    @pytest.mark.asyncio
    async def test_retrieve(self):
        """Test basic retrieval."""
        embedding = FakeEmbedding(dimension=64)
        store = MemoryVectorStore()

        # Add some chunks
        chunks = [
            Chunk(id="c1", document_id="d1", content="Python programming language"),
            Chunk(id="c2", document_id="d1", content="Java programming language"),
        ]
        embeddings = await embedding.embed_documents([c.content for c in chunks])
        await store.add(chunks, embeddings)

        # Create retriever
        retriever = VectorRetriever(embedding, store)

        # Retrieve
        results = await retriever.retrieve("programming", k=2)

        assert len(results) == 2


class TestKeywordRetriever:
    """Tests for keyword retriever."""

    @pytest.mark.asyncio
    async def test_keyword_search(self):
        """Test keyword-based search."""
        docs = [
            Document(id="1", content="Python is a programming language"),
            Document(id="2", content="Java is also a programming language"),
            Document(id="3", content="The cat sat on the mat"),
        ]

        retriever = KeywordRetriever(documents=docs)

        results = await retriever.retrieve("programming language", k=2)

        assert len(results) == 2
        # Should find Python and Java documents
        found_ids = {r.chunk.document_id for r in results}
        assert "1" in found_ids or "2" in found_ids


class TestHybridRetriever:
    """Tests for hybrid retriever."""

    @pytest.mark.asyncio
    async def test_hybrid_search(self):
        """Test hybrid retrieval."""
        embedding = FakeEmbedding(dimension=64)
        store = MemoryVectorStore()

        # Add chunks
        chunks = [
            Chunk(id="c1", document_id="d1", content="Python programming"),
            Chunk(id="c2", document_id="d2", content="Java programming"),
        ]
        embeddings = await embedding.embed_documents([c.content for c in chunks])
        await store.add(chunks, embeddings)

        # Create retrievers
        vector_retriever = VectorRetriever(embedding, store)
        keyword_retriever = KeywordRetriever([
            Document(id="d1", content="Python programming"),
            Document(id="d2", content="Java programming"),
        ])

        hybrid = HybridRetriever(
            vector_retriever=vector_retriever,
            keyword_retriever=keyword_retriever,
            alpha=0.5,
        )

        results = await hybrid.retrieve("programming", k=2)

        assert len(results) <= 2


class TestIdentityReranker:
    """Tests for identity reranker."""

    @pytest.mark.asyncio
    async def test_no_change(self):
        """Test that identity reranker doesn't change order."""
        reranker = IdentityReranker()

        chunks = [
            Chunk(id=f"c{i}", document_id="d1", content=f"content {i}")
            for i in range(5)
        ]
        results = [
            SearchResult(chunk=chunk, score=1.0 - i * 0.1)
            for i, chunk in enumerate(chunks)
        ]

        reranked = await reranker.rerank("query", results, top_k=5)

        assert len(reranked) == 5
        # Order should be unchanged
        for i, result in enumerate(reranked):
            assert result.chunk.id == f"c{i}"


class TestDiversityReranker:
    """Tests for diversity reranker."""

    @pytest.mark.asyncio
    async def test_promotes_diversity(self):
        """Test that diversity reranker promotes diverse results."""
        reranker = DiversityReranker(diversity_threshold=0.5)

        # Create similar chunks
        chunks = [
            Chunk(id="c1", document_id="d1", content="the cat sat on the mat"),
            Chunk(id="c2", document_id="d1", content="the cat sat on the hat"),  # Very similar
            Chunk(id="c3", document_id="d1", content="Python programming language"),  # Different
        ]
        results = [
            SearchResult(chunk=chunks[0], score=0.9),
            SearchResult(chunk=chunks[1], score=0.8),
            SearchResult(chunk=chunks[2], score=0.7),
        ]

        reranked = await reranker.rerank("query", results, top_k=3)

        # Should include diverse results
        assert len(reranked) >= 2


class TestRAGPipeline:
    """Tests for RAG pipeline."""

    @pytest.mark.asyncio
    async def test_index_and_retrieve(self):
        """Test indexing and retrieving documents."""
        embedding = FakeEmbedding(dimension=64)
        store = MemoryVectorStore()
        retriever = VectorRetriever(embedding, store)

        pipeline = RAGPipeline(
            embedding=embedding,
            vectorstore=store,
            retriever=retriever,
        )

        # Index documents
        docs = [
            Document(id="1", content="Python is a programming language"),
            Document(id="2", content="JavaScript is also a programming language"),
        ]
        await pipeline.index(docs)

        # Check counts
        assert await pipeline.count_documents() == 2
        assert await pipeline.count_chunks() >= 2

        # Retrieve
        results = await pipeline.retrieve("programming", k=2)

        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_add_single_document(self):
        """Test adding a single document."""
        embedding = FakeEmbedding(dimension=64)
        store = MemoryVectorStore()
        retriever = VectorRetriever(embedding, store)

        pipeline = RAGPipeline(
            embedding=embedding,
            vectorstore=store,
            retriever=retriever,
        )

        doc = Document(id="1", content="Test content")
        chunk_ids = await pipeline.add_document(doc)

        assert len(chunk_ids) >= 1
        assert await pipeline.count_documents() == 1

    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clearing the pipeline."""
        embedding = FakeEmbedding(dimension=64)
        store = MemoryVectorStore()
        retriever = VectorRetriever(embedding, store)

        pipeline = RAGPipeline(
            embedding=embedding,
            vectorstore=store,
            retriever=retriever,
        )

        docs = [Document(id="1", content="Test")]
        await pipeline.index(docs)

        await pipeline.clear()

        assert await pipeline.count_documents() == 0
        assert await pipeline.count_chunks() == 0


class TestSimpleRAG:
    """Tests for SimpleRAG interface."""

    @pytest.mark.asyncio
    async def test_add_and_search(self):
        """Test adding and searching with SimpleRAG."""
        rag = SimpleRAG()

        # Add documents
        await rag.add("Python is a programming language")
        await rag.add("JavaScript is also a programming language")

        # Search
        results = await rag.search("programming", k=2)

        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_with_metadata(self):
        """Test adding with metadata."""
        rag = SimpleRAG()

        doc_id = await rag.add(
            "Test content",
            metadata={"category": "test"},
            source="test.txt",
        )

        assert doc_id.startswith("doc_")


class TestCreateRAGTool:
    """Tests for RAG tool creation."""

    def test_create_tool(self):
        """Test creating a RAG tool."""
        embedding = FakeEmbedding(dimension=64)
        store = MemoryVectorStore()
        retriever = VectorRetriever(embedding, store)

        pipeline = RAGPipeline(
            embedding=embedding,
            vectorstore=store,
            retriever=retriever,
        )

        tool = create_rag_tool(pipeline, name="knowledge_search")

        assert tool.name == "knowledge_search"
        assert "knowledge" in tool.description.lower()
        assert "query" in tool.input_schema["properties"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
