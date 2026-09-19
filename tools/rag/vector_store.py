"""ChromaDB-backed vector store using local sentence-transformer embeddings.

Uses LlamaIndex for index management and ChromaDB for persistence.
"""
from pathlib import Path
from functools import lru_cache
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import get_settings


class ResumeVectorStore:
    """Stores resume chunks, past applications, and job snippets for RAG queries."""

    COLLECTION_RESUME = "resume_chunks"
    COLLECTION_APPLICATIONS = "past_applications"
    COLLECTION_JOBS = "job_descriptions"

    def __init__(self) -> None:
        settings = get_settings()
        self._client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._ef = self._build_embedding_function(settings.embedding_model)
        self._resume_col = self._client.get_or_create_collection(
            self.COLLECTION_RESUME, embedding_function=self._ef
        )
        self._app_col = self._client.get_or_create_collection(
            self.COLLECTION_APPLICATIONS, embedding_function=self._ef
        )
        self._job_col = self._client.get_or_create_collection(
            self.COLLECTION_JOBS, embedding_function=self._ef
        )

    @staticmethod
    def _build_embedding_function(model_name: str):
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        return SentenceTransformerEmbeddingFunction(model_name=model_name)

    # ── Indexing ──────────────────────────────────────────────────────────────

    def index_resume(self, chunks: list[str], resume_id: str = "default") -> None:
        """Index resume text chunks, replacing any existing ones for this id."""
        # Delete previous entries for this resume_id
        existing = self._resume_col.get(where={"resume_id": resume_id})
        if existing["ids"]:
            self._resume_col.delete(ids=existing["ids"])

        ids = [f"{resume_id}_{i}" for i in range(len(chunks))]
        self._resume_col.add(
            documents=chunks,
            ids=ids,
            metadatas=[{"resume_id": resume_id, "chunk_index": i} for i in range(len(chunks))],
        )

    def index_application(
        self, chunks: list[str], job_id: str, job_title: str, outcome: str = "unknown"
    ) -> None:
        existing = self._app_col.get(where={"job_id": job_id})
        if existing["ids"]:
            self._app_col.delete(ids=existing["ids"])

        ids = [f"app_{job_id}_{i}" for i in range(len(chunks))]
        self._app_col.add(
            documents=chunks,
            ids=ids,
            metadatas=[
                {"job_id": job_id, "job_title": job_title, "outcome": outcome, "chunk_index": i}
                for i in range(len(chunks))
            ],
        )

    def index_job(self, chunks: list[str], job_id: str, job_title: str) -> None:
        existing = self._job_col.get(where={"job_id": job_id})
        if existing["ids"]:
            self._job_col.delete(ids=existing["ids"])

        ids = [f"job_{job_id}_{i}" for i in range(len(chunks))]
        self._job_col.add(
            documents=chunks,
            ids=ids,
            metadatas=[{"job_id": job_id, "job_title": job_title, "chunk_index": i} for i in range(len(chunks))],
        )

    # ── Querying ──────────────────────────────────────────────────────────────

    def query_resume(self, query: str, n_results: int = 5) -> list[str]:
        """Retrieve most relevant resume chunks for a query (e.g., job description)."""
        results = self._resume_col.query(query_texts=[query], n_results=min(n_results, max(self._resume_col.count(), 1)))
        return results["documents"][0] if results["documents"] else []

    def score_resume_relevance(self, query: str, n_results: int = 3) -> float:
        """Return 0-1 semantic similarity score between query and indexed resume chunks.

        Uses actual ChromaDB L2 distances on normalized vectors rather than chunk count,
        so jobs semantically close to the resume score higher than unrelated ones.
        """
        count = self._resume_col.count()
        if count == 0:
            return 0.0
        n = min(n_results, count)
        results = self._resume_col.query(
            query_texts=[query],
            n_results=n,
            include=["distances"],
        )
        distances = results.get("distances", [[]])[0]
        if not distances:
            return 0.0
        avg_dist = sum(distances) / len(distances)
        # L2 distance on normalized vectors: 0.0=identical, ~2.0=opposite
        return round(max(0.0, 1.0 - avg_dist / 2.0), 4)

    def query_applications(self, query: str, n_results: int = 3) -> list[dict]:
        """Retrieve similar past applications with their outcomes."""
        count = self._app_col.count()
        if count == 0:
            return []
        results = self._app_col.query(
            query_texts=[query],
            n_results=min(n_results, count),
        )
        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        return [{"text": d, "metadata": m} for d, m in zip(docs, metas)]

    def query_jobs(self, query: str, n_results: int = 5) -> list[str]:
        count = self._job_col.count()
        if count == 0:
            return []
        results = self._job_col.query(query_texts=[query], n_results=min(n_results, count))
        return results["documents"][0] if results["documents"] else []

    def resume_chunk_count(self) -> int:
        return self._resume_col.count()

    def application_count(self) -> int:
        return self._app_col.count()


@lru_cache(maxsize=1)
def get_vector_store() -> ResumeVectorStore:
    return ResumeVectorStore()
