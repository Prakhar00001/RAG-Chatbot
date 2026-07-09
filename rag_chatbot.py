import os
import re
from typing import List, Optional

import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

load_dotenv()


class RAGPipeline:
    def __init__(self, embedding_model: str = "local"):
        self.embedding_model = embedding_model
        self.client = None
        self.chunks: List[dict] = []
        self.vector_store = None
        self.bm25 = None
        self.sources: List[str] = []
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self._splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)

        if embedding_model != "local" and os.getenv("OPENAI_API_KEY"):
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _chunk_text(self, text: str) -> List[str]:
        if not text.strip():
            return []
        return self._splitter.split_text(text)

    def index_documents(self, documents: List[str], sources: Optional[List[str]] = None):
        if sources is None:
            sources = [f"document_{i}.txt" for i in range(len(documents))]

        self.sources = sources
        self.chunks = []

        for doc, source in zip(documents, sources):
            chunks = self._chunk_text(doc)
            for chunk in chunks:
                self.chunks.append({"text": chunk, "source": source})

        if not self.chunks:
            self.vector_store = None
            self.bm25 = None
            return

        texts = [c["text"] for c in self.chunks]
        if self.embedding_model != "local" and self.client is not None:
            embeddings = []
            for text in texts:
                response = self.client.embeddings.create(
                    model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
                    input=text,
                )
                embeddings.append(response.data[0].embedding)
            embeddings = np.array(embeddings, dtype="float32")
        else:
            embeddings_matrix = self.vectorizer.fit_transform(texts).toarray().astype("float32")
            embeddings = embeddings_matrix

        self.vector_store = faiss.IndexFlatL2(embeddings.shape[1])
        self.vector_store.add(embeddings)

        tokenized = [re.findall(r"\w+", c["text"].lower()) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

    def _embed_query(self, query: str) -> np.ndarray:
        if self.embedding_model != "local" and self.client is not None:
            response = self.client.embeddings.create(
                model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
                input=query,
            )
            return np.array(response.data[0].embedding, dtype="float32")
        return self.vectorizer.transform([query]).toarray().astype("float32")[0]

    def _retrieve(self, query: str, top_k: int = 4) -> List[dict]:
        if not self.chunks or self.vector_store is None or self.bm25 is None:
            return []

        query_embedding = self._embed_query(query)
        distances, indices = self.vector_store.search(query_embedding.reshape(1, -1), top_k)

        scored_chunks = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)]
            scored_chunks.append({"text": chunk["text"], "source": chunk["source"], "score": float(dist)})

        bm25_scores = self.bm25.get_scores(re.findall(r"\w+", query.lower()))
        for i, score in enumerate(bm25_scores):
            if score > 0 and not any(item["text"] == self.chunks[i]["text"] for item in scored_chunks):
                scored_chunks.append({"text": self.chunks[i]["text"], "source": self.chunks[i]["source"], "score": float(score)})

        scored_chunks = sorted(scored_chunks, key=lambda item: item["score"], reverse=False)[:top_k]
        return scored_chunks

    def answer_question(self, query: str, top_k: int = 4) -> str:
        if not self.chunks:
            return "Please upload a PDF first."

        context_chunks = self._retrieve(query, top_k=top_k)
        if not context_chunks:
            return "No relevant content found in the uploaded documents."

        context = "\n\n".join([f"[{i + 1}] {chunk['text']}" for i, chunk in enumerate(context_chunks)])
        prompt = f"""You are a helpful research assistant. Use the context below to answer the user's question.

Context:
{context}

Question: {query}

Instructions:
- Answer using only the provided context.
- Cite sources inline as [doc_name].
- Keep the response concise and accurate.
"""

        if self.embedding_model == "local" or self.client is None:
            return self._fallback_answer(query, context_chunks)

        response = self.client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "system", "content": "You answer questions from provided context."}, {"role": "user", "content": prompt}],
            temperature=0.2,
        )
        answer = response.choices[0].message.content or ""
        return self._format_answer(answer, context_chunks)

    def _fallback_answer(self, query: str, context_chunks: List[dict]) -> str:
        joined = " ".join([chunk["text"] for chunk in context_chunks])
        answer = joined[:500]
        if not answer:
            return "No relevant content found."
        citations = ", ".join(sorted({chunk["source"] for chunk in context_chunks}))
        return f"{answer} Sources: {citations}"

    def _format_answer(self, answer: str, context_chunks: List[dict]) -> str:
        citations = ", ".join(sorted({chunk["source"] for chunk in context_chunks}))
        if citations:
            return f"{answer}\n\nSources: {citations}"
        return answer
