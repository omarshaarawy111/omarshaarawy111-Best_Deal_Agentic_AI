"""
Hybrid dense and sparse retriever for Best Deal RAG.

Description:
    This module combines OpenAI dense embeddings and BM25 sparse search with
    Reciprocal Rank Fusion, preserving the retrieval strategy validated in the
    research notebook.

Responsibilities:
    - Retrieve dense candidates.
    - Retrieve sparse candidates.
    - Fuse ranked results with RRF.
    - Return normalized Result objects.
"""

from __future__ import annotations

from best_deal.rag.bm25 import BM25Store
from best_deal.rag.embeddings import OpenAIEmbedder
from best_deal.rag.schemas import Result
from best_deal.rag.vectorstore import ChromaVectorStore


def reciprocal_rank_fusion(rank_lists: list[list[str]], k_rrf: int = 60) -> list[str]:
    """
    Fuse multiple ranked document-id lists with Reciprocal Rank Fusion.

    Args:
        rank_lists: Ranked id lists from retrieval methods.
        k_rrf: RRF constant.

    Returns:
        Fused ids ordered by descending RRF score.
    """
    scores: dict[str, float] = {}
    for rank_list in rank_lists:
        for rank, document_id in enumerate(rank_list, start=1):
            scores[document_id] = scores.get(document_id, 0.0) + 1.0 / (k_rrf + rank)
    return [document_id for document_id, _ in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)]


class HybridRetriever:
    """
    Perform dense plus sparse product retrieval.

    Returns:
        A configured hybrid retriever.
    """

    def __init__(self, embedder: OpenAIEmbedder, vectorstore: ChromaVectorStore, bm25: BM25Store) -> None:
        """
        Initialize the hybrid retrieval dependencies.

        Args:
            embedder: Dense embedding service.
            vectorstore: Chroma vector store.
            bm25: BM25 sparse store.

        Returns:
            None.
        """
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.bm25 = bm25

    def search(self, question: str, k: int = 10) -> list[Result]:
        """
        Retrieve and fuse dense and sparse product candidates.

        Args:
            question: Product search query.
            k: Maximum final candidate count.

        Returns:
            Hybrid-ranked retrieval results.
        """
        query_embedding = self.embedder.embed_query(question)
        dense_chunks = self.vectorstore.query(query_embedding, k)
        dense_ids = [str(chunk.metadata.get("product_id")) for chunk in dense_chunks]
        dense_lookup = {document_id: chunk for document_id, chunk in zip(dense_ids, dense_chunks)}

        sparse_ids, sparse_lookup = self.bm25.search(question, k)
        fused_ids = reciprocal_rank_fusion([dense_ids, sparse_ids])[:k]

        results: list[Result] = []
        for document_id in fused_ids:
            chunk = dense_lookup.get(document_id)
            if chunk is not None:
                results.append(chunk)
                continue
            sparse_document = sparse_lookup.get(document_id)
            if sparse_document is not None:
                document, metadata = sparse_document
                results.append(Result(page_content=document, metadata=metadata))
        return results
