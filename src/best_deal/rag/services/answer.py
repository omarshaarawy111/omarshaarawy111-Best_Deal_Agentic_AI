"""
End-to-end RAG answer service for Best Deal.

Description:
    This module orchestrates query extraction, query rewriting, hybrid
    retrieval, retrieval expansion, reranking, and structured price estimation.
    It is the reusable service behind both the Frontier Agent and future APIs.

Responsibilities:
    - Normalize product questions.
    - Retrieve from both original and rewritten queries.
    - Merge and rerank candidate chunks.
    - Generate a structured price estimate without UI concerns.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from best_deal.config import RAG_VECTORSTORE_PATH
from best_deal.data.models import Item
from best_deal.rag.bm25 import BM25Store
from best_deal.rag.config import COLLECTION_NAME, RETRIEVAL_K, RETRIEVAL_S
from best_deal.rag.embeddings import OpenAIEmbedder
from best_deal.rag.llm import RAGLLM
from best_deal.rag.reranker import RAGReranker
from best_deal.rag.retriever import HybridRetriever
from best_deal.rag.schemas import RAGAnswer, Result
from best_deal.rag.vectorstore import ChromaVectorStore


class RAGAnswerService:
    """
    Provide the reusable Best Deal RAG answer pipeline.

    Returns:
        A configured end-to-end RAG answer service.
    """

    def __init__(
        self,
        vectorstore: ChromaVectorStore | None = None,
        embedder: OpenAIEmbedder | None = None,
        bm25: BM25Store | None = None,
        llm: RAGLLM | None = None,
        retrieval_k: int = RETRIEVAL_K,
        final_k: int = RETRIEVAL_S,
    ) -> None:
        """
        Initialize all RAG pipeline dependencies.

        Args:
            vectorstore: Optional injected Chroma vector store.
            embedder: Optional injected embedding service.
            bm25: Optional injected BM25 store.
            llm: Optional injected RAG LLM service.
            retrieval_k: Candidate count retrieved per query.
            final_k: Number of chunks kept after reranking.

        Returns:
            None.
        """
        self.vectorstore = vectorstore or ChromaVectorStore(RAG_VECTORSTORE_PATH, COLLECTION_NAME)
        self.embedder = embedder or OpenAIEmbedder()
        self.bm25 = bm25 or BM25Store()
        self.llm = llm or RAGLLM()
        self.retriever = HybridRetriever(self.embedder, self.vectorstore, self.bm25)
        self.reranker = RAGReranker(self.llm)
        self.retrieval_k = retrieval_k
        self.final_k = final_k

    @staticmethod
    def get_query(question: str | Item | dict[str, Any] | Any) -> str:
        """
        Normalize supported product-question inputs into search text.

        Args:
            question: Product title, Item, or mapping containing a title/summary.

        Returns:
            Normalized question text.
        """
        if not question:
            return ""
        if isinstance(question, Item):
            return question.title.strip()
        if isinstance(question, dict):
            return str(question.get("title") or question.get("summary") or "").strip()
        return str(question).strip()

    @staticmethod
    def merge_chunks(chunks1: list[Result], chunks2: list[Result]) -> list[Result]:
        """
        Merge two retrieval result lists without duplicate document content.

        Args:
            chunks1: Results from the original query.
            chunks2: Results from the rewritten query.

        Returns:
            Unique merged results in first-seen order.
        """
        merged = list(chunks1)
        existing = {chunk.page_content for chunk in chunks1}
        for chunk in chunks2:
            if chunk.page_content not in existing:
                merged.append(chunk)
                existing.add(chunk.page_content)
        return merged

    def fetch_context(self, original_question: str, rewritten_question: str) -> list[Result]:
        """
        Perform query expansion, hybrid retrieval, merging, and reranking.

        Args:
            original_question: Original normalized product question.
            rewritten_question: LLM-refined retrieval query.

        Returns:
            Final reranked source chunks.
        """
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(self.retriever.search, original_question, self.retrieval_k)
            second = executor.submit(self.retriever.search, rewritten_question, self.retrieval_k)
            chunks1 = first.result()
            chunks2 = second.result()
        merged = self.merge_chunks(chunks1, chunks2)
        return self.reranker.rerank(original_question, merged, self.final_k)

    def answer(self, question: str | Item | dict[str, Any], history: list[dict[str, str]] | None = None) -> RAGAnswer:
        """
        Run the complete RAG price-estimation pipeline.

        Args:
            question: Product question input.
            history: Optional conversation history used by query rewriting.

        Returns:
            Structured RAG answer containing estimate and source chunks.
        """
        question_text = self.get_query(question)
        if not question_text:
            raise ValueError("No question found")
        rewritten = self.llm.rewrite_query(question_text, history)
        chunks = self.fetch_context(question_text, rewritten)
        estimate = self.llm.estimate_price(question_text, chunks)
        return RAGAnswer(estimate=estimate, chunks=chunks)

    def price(self, question: str | Item | dict[str, Any], history: list[dict[str, str]] | None = None) -> tuple[float, list[Result]]:
        """
        Return only the numeric estimate and retrieval context for agent use.

        Args:
            question: Product question input.
            history: Optional conversation history.

        Returns:
            Estimated price and retrieved chunks.
        """
        result = self.answer(question, history)
        return float(result.estimate.estimated_price or 0.0), result.chunks
