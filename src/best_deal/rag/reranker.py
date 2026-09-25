"""
LLM reranker adapter for Best Deal RAG.

Description:
    This module exposes reranking as a dedicated stage so retrieval and model
    orchestration remain separate responsibilities.

Responsibilities:
    - Accept retrieved product chunks.
    - Delegate relevance ordering to the RAG language-model adapter.
    - Return the requested top-k chunks after reranking.
"""

from __future__ import annotations

from best_deal.rag.llm import RAGLLM
from best_deal.rag.schemas import Result


class RAGReranker:
    """
    Rerank retrieved RAG chunks with an LLM.

    Returns:
        A configured reranker.
    """

    def __init__(self, llm: RAGLLM) -> None:
        """
        Initialize the reranker dependency.

        Args:
            llm: RAG language-model adapter.

        Returns:
            None.
        """
        self.llm = llm

    def rerank(self, question: str, chunks: list[Result], top_k: int) -> list[Result]:
        """
        Reorder and trim retrieved chunks.

        Args:
            question: Original product question.
            chunks: Candidate retrieval results.
            top_k: Maximum returned chunk count.

        Returns:
            Reranked top-k chunks.
        """
        return self.llm.rerank(question, chunks)[:top_k]
