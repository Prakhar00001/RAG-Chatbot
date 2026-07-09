import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_chatbot import RAGPipeline


def test_chunking_and_indexing_work():
    pipeline = RAGPipeline(embedding_model="local")
    docs = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a popular programming language for AI systems.",
    ]

    pipeline.index_documents(docs, sources=["doc1.txt", "doc2.txt"])

    assert len(pipeline.chunks) >= 2
    assert pipeline.vector_store is not None
    assert pipeline.bm25 is not None


def test_query_returns_cited_answer():
    pipeline = RAGPipeline(embedding_model="local")
    docs = [
        "LangChain helps build LLM applications with retrieval pipelines.",
        "FAISS is a vector similarity search library used for fast nearest-neighbor lookup.",
    ]
    pipeline.index_documents(docs, sources=["doc1.txt", "doc2.txt"])

    answer = pipeline.answer_question("What does LangChain help with?")

    assert "LangChain" in answer
    assert "doc1.txt" in answer
