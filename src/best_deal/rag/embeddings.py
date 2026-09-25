"""
OpenAI embedding adapter for Best Deal RAG.

Description:
    This module isolates the embedding provider from vector-store code so the
    provider can be replaced later without rewriting retrieval logic.

Responsibilities:
    - Create one query embedding.
    - Create embeddings for document batches.
"""

from __future__ import annotations

from openai import OpenAI

from best_deal.rag.config import EMBEDDING_MODEL


class OpenAIEmbedder:
    """
    Generate embeddings with the configured OpenAI model.

    Returns:
        A configured embedding service.
    """

    def __init__(self, client: OpenAI | None = None, model: str = EMBEDDING_MODEL) -> None:
        """
        Initialize the embedding service.

        Args:
            client: Optional OpenAI client for dependency injection.
            model: OpenAI embedding model name.

        Returns:
            None.
        """
        self.client = client or OpenAI()
        self.model = model

    def embed_query(self, text: str) -> list[float]:
        """
        Create an embedding for one search query.

        Args:
            text: Query text.

        Returns:
            Query embedding vector.
        """
        return self.client.embeddings.create(model=self.model, input=[text]).data[0].embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Create embeddings for a batch of product documents.

        Args:
            texts: Product chunk texts.

        Returns:
            Embedding vectors in the same order as the input texts.
        """
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]
