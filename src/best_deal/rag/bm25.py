"""
BM25 sparse retrieval for Best Deal RAG.

Description:
    This module builds and loads the persisted sparse index used together with
    Chroma for hybrid retrieval.

Responsibilities:
    - Tokenize product text consistently.
    - Build a persisted BM25 payload.
    - Search the sparse index and return document ids with lookups.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from best_deal.rag.config import RAG_BM25_PATH, RETRIEVAL_K
from best_deal.rag.schemas import Result


def tokenize(text: str) -> list[str]:
    """
    Tokenize product text for sparse BM25 search.

    Args:
        text: Text to tokenize.

    Returns:
        Lowercase alphanumeric tokens.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Store:
    """
    Manage the persisted Best Deal BM25 index.

    Returns:
        A configured BM25 store.
    """

    def __init__(self, path: str | Path = RAG_BM25_PATH) -> None:
        """
        Initialize the sparse index path.

        Args:
            path: Pickle path for the sparse retrieval payload.

        Returns:
            None.
        """
        self.path = Path(path)
        self.index: BM25Okapi | None = None
        self.payload: dict[str, Any] | None = None
        if self.path.exists():
            self.load()

    def build(self, chunks: list[Result]) -> BM25Okapi:
        """
        Build and persist a BM25 index from retrieval chunks.

        Args:
            chunks: Product chunks used by dense and sparse retrieval.

        Returns:
            The built BM25 index.
        """
        corpus_tokens = [tokenize(chunk.page_content + " " + str(chunk.metadata.get("sku_codes", ""))) for chunk in chunks]
        self.index = BM25Okapi(corpus_tokens)
        self.payload = {
            "corpus_tokens": corpus_tokens,
            "ids": [str(chunk.metadata.get("product_id", index)) for index, chunk in enumerate(chunks)],
            "metadatas": [chunk.metadata for chunk in chunks],
            "documents": [chunk.page_content for chunk in chunks],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("wb") as file:
            pickle.dump(self.payload, file)
        return self.index

    def load(self) -> BM25Okapi:
        """
        Load the persisted BM25 index from disk.

        Returns:
            The loaded BM25 index.
        """
        with self.path.open("rb") as file:
            self.payload = pickle.load(file)
        self.index = BM25Okapi(self.payload["corpus_tokens"])
        return self.index

    def search(self, question: str, k: int = RETRIEVAL_K) -> tuple[list[str], dict[str, tuple[str, dict[str, Any]]]]:
        """
        Search the sparse index for a product question.

        Args:
            question: Product search question.
            k: Maximum number of sparse results.

        Returns:
            Ranked document ids and an id-to-document lookup.
        """
        if self.index is None or self.payload is None:
            return [], {}
        scores = self.index.get_scores(tokenize(question))
        top_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:k]
        ids = [self.payload["ids"][index] for index in top_indices]
        lookup = {
            self.payload["ids"][index]: (
                self.payload["documents"][index],
                self.payload["metadatas"][index],
            )
            for index in top_indices
        }
        return ids, lookup
