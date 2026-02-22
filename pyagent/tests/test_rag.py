"""Tests for the RAG system."""

import pytest

from pyagent.rag import (
    Document, Chunk, SearchResult,
    DummyEmbedding, FakeEmbedding,
    MemoryVectorStore, cosine_similarity,
    FixedSizeChunker, RecursiveChunker, SemanticChunker,
    VectorRetriever, KeywordRetriever, HybridRetriever,
    IdentityReranker, DiversityReranker,
    RAGPipeline, SimpleRAG, create_rag_tool,
)


class TestDocument:
    def test_document_creation(self):
        doc = Document(id="test-1", content="This is test content.", metadata={"author": "test"}, source="test.txt")
        assert doc.id == "test-1"
        assert doc.content == "This is test content."
        assert doc.metadata == {"author": "test"}

    def test_document_defaults(self):
        doc = Document(id="test", content="content")
        assert doc.metadata == {}
        assert doc.source is None


class TestChunk:
    def test_chunk_creation(self):
        chunk = Chunk(id="chunk-1", document_id="doc-1", content="Chunk content", metadata={"page": 1}, start_index=0, end_index=13)
        assert chunk.id == "chunk-1"
        assert chunk.document_id == "doc-1"
        assert chunk.embedding is None

    def test_chunk_with_embedding(self):
        embedding = [0.1, 0.2, 0.3]
        chunk = Chunk(id="chunk-1", document_id="doc-1", content="content", embedding=embedding)
        assert chunk.embedding == embedding


class TestEmbeddings:
    @pytest.mark.asyncio
    async def test_dummy_embedding(self):
        embedding = DummyEmbedding(dimension=128)
        assert embedding.dimension == 128
        vec = await embedding.embed_query("test")
        assert len(vec) == 128
        vecs = await embedding.embed_documents(["a", "b", "c"])
        assert len(vecs) == 3

    @pytest.mark.asyncio
    async def test_fake_embedding(self):
        embedding = FakeEmbedding(dimension=64, seed=42)
        vec1 = await embedding.embed_query("hello world")
        vec2 = await embedding.embed_query("hello world")
        assert vec1 == vec2
        vec3 = await embedding.embed_query("goodbye world")
        assert vec1 != vec3


class TestCosineSimilarity:
    def test_identical_vectors(self):
        vec = [1.0, 2.0, 3.0]
        sim = cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 0.0001

    def test_orthogonal_vectors(self):
        sim = cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert abs(sim) < 0.0001

    def test_opposite_vectors(self):
        sim = cosine_similarity([1.0, 0.0], [-1.0, 0.0])
        assert abs(sim + 1.0) < 0.0001


class TestMemoryVectorStore:
    @pytest.mark.asyncio
    async def test_add_and_search(self):
        store = MemoryVectorStore()
        chunks = [Chunk(id="c1", document_id="d1", content="Python"), Chunk(id="c2", document_id="d1", content="Java")]
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ids = await store.add(chunks, embeddings)
        assert len(ids) == 2
        results = await store.search([1.0, 0.0, 0.0], k=1)
        assert len(results) == 1
        assert results[0].chunk.id == "c1"

    @pytest.mark.asyncio
    async def test_delete(self):
        store = MemoryVectorStore()
        chunk = Chunk(id="c1", document_id="d1", content="test")
        await store.add([chunk], [[1.0, 0.0]])
        assert await store.count() == 1
        await store.delete(["c1"])
        assert await store.count() == 0


class TestFixedSizeChunker:
    def test_basic_chunking(self):
        chunker = FixedSizeChunker(chunk_size=10, overlap=0)
        doc = Document(id="1", content="1234567890123456789012345")
        chunks = chunker.chunk(doc)
        assert len(chunks) == 3
        assert chunks[0].content == "1234567890"

    def test_invalid_overlap(self):
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=10, overlap=10)


class TestRecursiveChunker:
    def test_paragraph_split(self):
        chunker = RecursiveChunker(chunk_size=100, overlap=0)
        doc = Document(id="1", content="First paragraph.\n\nSecond paragraph.")
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 1


class TestVectorRetriever:
    @pytest.mark.asyncio
    async def test_retrieve(self):
        embedding = FakeEmbedding(dimension=64)
        store = MemoryVectorStore()
        chunks = [Chunk(id="c1", document_id="d1", content="Python"), Chunk(id="c2", document_id="d1", content="Java")]
        embeddings = await embedding.embed_documents([c.content for c in chunks])
        await store.add(chunks, embeddings)
        retriever = VectorRetriever(embedding, store)
        results = await retriever.retrieve("programming", k=2)
        assert len(results) == 2


class TestKeywordRetriever:
    @pytest.mark.asyncio
    async def test_keyword_search(self):
        docs = [Document(id="1", content="Python is a programming language"), Document(id="2", content="The cat sat")]
        retriever = KeywordRetriever(docs)
        results = await retriever.retrieve("programming", k=2)
        assert len(results) >= 1


class TestIdentityReranker:
    @pytest.mark.asyncio
    async def test_no_change(self):
        reranker = IdentityReranker()
        chunks = [Chunk(id=f"c{i}", document_id="d1", content=f"content {i}") for i in range(3)]
        results = [SearchResult(chunk=chunk, score=1.0 - i * 0.1) for i, chunk in enumerate(chunks)]
        reranked = await reranker.rerank("query", results, top_k=3)
        assert len(reranked) == 3


class TestRAGPipeline:
    @pytest.mark.asyncio
    async def test_index_and_retrieve(self):
        embedding = FakeEmbedding(dimension=64)
        store = MemoryVectorStore()
        retriever = VectorRetriever(embedding, store)
        pipeline = RAGPipeline(embedding=embedding, vectorstore=store, retriever=retriever)
        docs = [Document(id="1", content="Python is a programming language")]
        await pipeline.index(docs)
        assert await pipeline.count_documents() == 1
        results = await pipeline.retrieve("programming", k=2)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_clear(self):
        embedding = FakeEmbedding(dimension=64)
        store = MemoryVectorStore()
        retriever = VectorRetriever(embedding, store)
        pipeline = RAGPipeline(embedding=embedding, vectorstore=store, retriever=retriever)
        docs = [Document(id="1", content="Test")]
        await pipeline.index(docs)
        await pipeline.clear()
        assert await pipeline.count_documents() == 0


class TestSimpleRAG:
    @pytest.mark.asyncio
    async def test_add_and_search(self):
        rag = SimpleRAG()
        await rag.add("Python is a programming language")
        await rag.add("JavaScript is also a programming language")
        results = await rag.search("programming", k=2)
        assert len(results) >= 1


class TestCreateRAGTool:
    def test_create_tool(self):
        embedding = FakeEmbedding(dimension=64)
        store = MemoryVectorStore()
        retriever = VectorRetriever(embedding, store)
        pipeline = RAGPipeline(embedding=embedding, vectorstore=store, retriever=retriever)
        tool = create_rag_tool(pipeline, name="knowledge_search")
        assert tool.name == "knowledge_search"
        assert "query" in tool.input_schema.properties


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
