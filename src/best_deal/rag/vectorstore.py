"""
Chroma vector-store adapter for Best Deal RAG.

Description:
    This module manages the persistent dense-retrieval collection used by the
    RAG answer service. Index creation is explicit and never triggered on app startup.

Responsibilities:
    - Open or create the persistent Chroma collection.
    - Replace the collection during an explicit full rebuild.
    - Add embedded product chunks in batches.
    - Query the persistent vector store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chromadb import PersistentClient

from best_deal.rag.config import COLLECTION_NAME, RETRIEVAL_K
from best_deal.rag.schemas import Result


class ChromaVectorStore:
    """
    Manage the persistent dense vector index for Best Deal.

    Returns:
        A configured vector-store adapter.
    """

    def __init__(self, path: str | Path, collection_name: str = COLLECTION_NAME) -> None:
        """
        Initialize a persistent Chroma collection reference.

        Args:
            path: Persistent Chroma directory.
            collection_name: Collection name.

        Returns:
            None.
        """
        self.path = Path(path)
        self.collection_name = collection_name
        self.client = PersistentClient(path=str(self.path))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def reset(self) -> None:
        """
        Delete and recreate the configured Chroma collection.

        Returns:
            None.
        """
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def add(self, chunks: list[Result], embeddings: list[list[float]], batch_size: int = 100) -> int:
        """
        Add embedded chunks to Chroma in bounded batches.

        Args:
            chunks: Retrieval chunks to store.
            embeddings: Embeddings aligned with the chunks.
            batch_size: Number of records per Chroma write.

        Returns:
            Number of stored chunks.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        stored = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = embeddings[start : start + batch_size]
            ids = [str(chunk.metadata.get("product_id", start + index)) for index, chunk in enumerate(batch)]
            self.collection.add(
                ids=ids,
                embeddings=vectors,
                documents=[chunk.page_content for chunk in batch],
                metadatas=[chunk.metadata for chunk in batch],
            )
            stored += len(batch)
        return stored

    def query(self, embedding: list[float], k: int = RETRIEVAL_K) -> list[Result]:
        """
        Retrieve the nearest dense documents from Chroma.

        Args:
            embedding: Query embedding.
            k: Maximum number of documents to retrieve.

        Returns:
            Retrieved documents as Result objects.
        """
        if self.collection.count() == 0:
            return []
        results: dict[str, Any] = self.collection.query(query_embeddings=[embedding], n_results=min(k, self.collection.count()))
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        return [Result(page_content=document, metadata=metadata or {}) for document, metadata in zip(documents, metadatas)]
