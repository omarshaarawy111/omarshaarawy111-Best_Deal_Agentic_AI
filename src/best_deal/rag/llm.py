"""
LLM provider wrappers for Best Deal RAG.

Description:
    This module isolates LiteLLM and structured-output handling from the rest of
    the RAG pipeline.

Responsibilities:
    - Rewrite product queries.
    - Rerank retrieved chunks with structured output.
    - Generate structured price estimates.
"""

from __future__ import annotations

from litellm import completion
from tenacity import retry, stop_after_attempt, wait_exponential

from best_deal.rag.config import GENERATOR_MODEL
from best_deal.rag.prompts import PRICE_ESTIMATE_PROMPT, QUERY_REWRITE_PROMPT, RERANK_SYSTEM_PROMPT
from best_deal.rag.schemas import PriceEstimate, RankOrder, Result

WAIT_CONFIG = wait_exponential(multiplier=1, min=2, max=30)


class RAGLLM:
    """
    Provide the language-model operations required by the RAG answer flow.

    Returns:
        A configured RAG language-model client.
    """

    def __init__(self, model: str = GENERATOR_MODEL) -> None:
        """
        Initialize the RAG language-model configuration.

        Args:
            model: LiteLLM model identifier.

        Returns:
            None.
        """
        self.model = model

    @retry(wait=WAIT_CONFIG, stop=stop_after_attempt(3), reraise=True)
    def rewrite_query(self, question: str, history: list[dict[str, str]] | None = None) -> str:
        """
        Rewrite a product question into a focused retrieval query.

        Args:
            question: Original user question.
            history: Optional conversation history.

        Returns:
            Focused retrieval query.
        """
        prompt = f"{QUERY_REWRITE_PROMPT}\n\nConversation history:\n{history or []}\n\nQuestion:\n{question}"
        response = completion(model=self.model, messages=[{"role": "system", "content": prompt}])
        return str(response.choices[0].message.content or "").strip()

    @retry(wait=WAIT_CONFIG, stop=stop_after_attempt(3), reraise=True)
    def rerank(self, question: str, chunks: list[Result]) -> list[Result]:
        """
        Rerank retrieved chunks using structured LLM output.

        Args:
            question: Original product question.
            chunks: Candidate retrieved chunks.

        Returns:
            Chunks ordered from most relevant to least relevant.
        """
        if not chunks:
            return []
        prompt = f"{RERANK_SYSTEM_PROMPT}\n\nQuestion:\n{question}\n\nChunks:\n"
        for index, chunk in enumerate(chunks, start=1):
            prompt += f"\n# CHUNK ID: {index}\n{chunk.page_content}\n"
        response = completion(
            model=self.model,
            messages=[{"role": "system", "content": prompt}],
            response_format=RankOrder,
        )
        order = RankOrder.model_validate_json(response.choices[0].message.content).order
        valid_ids = set(range(1, len(chunks) + 1))
        cleaned: list[int] = []
        seen: set[int] = set()
        for chunk_id in order:
            if chunk_id in valid_ids and chunk_id not in seen:
                cleaned.append(chunk_id)
                seen.add(chunk_id)
        cleaned.extend(chunk_id for chunk_id in range(1, len(chunks) + 1) if chunk_id not in seen)
        return [chunks[chunk_id - 1] for chunk_id in cleaned]

    @retry(wait=WAIT_CONFIG, stop=stop_after_attempt(3), reraise=True)
    def estimate_price(self, question: str, chunks: list[Result]) -> PriceEstimate:
        """
        Produce a structured price estimate from retrieved context.

        Args:
            question: Product question.
            chunks: Reranked comparison products.

        Returns:
            Structured price estimate with confidence metadata.
        """
        context = "\n\n".join(
            f"Product:\nTitle: {chunk.metadata.get('title')}\n"
            f"Category: {chunk.metadata.get('type')}\n"
            f"Price: ${chunk.metadata.get('price')}\n"
            f"Description: {chunk.page_content}\n"
            f"Source: {chunk.metadata.get('source')}"
            for chunk in chunks
        )
        response = completion(
            model=self.model,
            messages=[
                {"role": "system", "content": PRICE_ESTIMATE_PROMPT.format(context=context)},
                {"role": "user", "content": question},
            ],
            response_format=PriceEstimate,
        )
        return PriceEstimate.model_validate_json(response.choices[0].message.content)
