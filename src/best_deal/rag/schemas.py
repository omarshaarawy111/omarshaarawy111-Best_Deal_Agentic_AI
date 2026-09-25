"""
Pydantic schemas for the Best Deal RAG pipeline.

Description:
    This module defines the structured objects shared by RAG ingestion,
    retrieval, reranking, answer generation, and evaluation.

Responsibilities:
    - Represent retrieved documents.
    - Validate model-produced reranking instructions.
    - Represent structured price estimates.
    - Validate chunks before they are embedded.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from best_deal.rag.config import KNOWN_CATEGORIES


class Result(BaseModel):
    """
    Represent one retrieved RAG document and its metadata.

    Returns:
        A validated retrieval result.
    """

    page_content: str
    metadata: dict[str, Any]


class RankOrder(BaseModel):
    """
    Represent the structured list of chunk ranks returned by the reranker.

    Returns:
        A validated rank order.
    """

    order: list[int] = Field(description="Chunk ids ordered by relevance.")


class PriceEstimate(BaseModel):
    """
    Represent a structured price estimate and confidence metadata.

    Returns:
        A validated price estimate.
    """

    estimated_price: float | None = Field(default=None, description="Estimated price in USD.")
    confidence: Literal["high", "medium", "low", "none"]
    matched_category: bool
    comparable_products_used: int
    reasoning: str


class RAGAnswer(BaseModel):
    """
    Represent the complete output of a RAG price-estimation run.

    Returns:
        A structured estimate plus the retrieved source chunks.
    """

    estimate: PriceEstimate
    chunks: list[Result]


class Chunk(BaseModel):
    """
    Represent a product chunk prepared for the vector database.

    Returns:
        A validated chunk object.
    """

    headline: str
    summary: str
    original_text: str

    def as_result(self, document: dict[str, Any]) -> Result:
        """
        Convert a chunk and source document into a retrieval result.

        Args:
            document: Source document metadata.

        Returns:
            A Result object ready for retrieval.
        """
        metadata = {
            "product_id": document["product_id"],
            "source": document["source"],
            "type": document["type"],
            "title": document["title"],
            "price": document["price"],
        }
        return Result(
            page_content=f"{self.headline}\n\n{self.summary}\n\n{self.original_text}",
            metadata=metadata,
        )


class Chunks(BaseModel):
    """
    Represent a collection of product chunks.

    Returns:
        A validated chunk collection.
    """

    chunks: list[Chunk]


class ChunkValidation(BaseModel):
    """
    Validate a single chunk before embedding.

    Returns:
        A validated chunk metadata record.
    """

    page_content: str
    product_id: int
    type: str
    title: str
    price: float
    source: str

    @field_validator("page_content")
    @classmethod
    def content_not_trivial(cls, value: str) -> str:
        """
        Reject implausibly short or long chunk content.

        Args:
            value: Chunk text.

        Returns:
            Validated chunk text.
        """
        value = value.strip()
        if len(value) < 20:
            raise ValueError("page_content is too short")
        if len(value) > 8000:
            raise ValueError("page_content is too long")
        return value

    @field_validator("type")
    @classmethod
    def category_known(cls, value: str) -> str:
        """
        Validate that the product category is known to the RAG dataset.

        Args:
            value: Product category.

        Returns:
            Validated category.
        """
        if value not in KNOWN_CATEGORIES:
            raise ValueError(f"Unknown category: {value!r}")
        return value

    @field_validator("price")
    @classmethod
    def price_sane(cls, value: float) -> float:
        """
        Validate a positive and bounded product price.

        Args:
            value: Product price.

        Returns:
            Validated price.
        """
        if value <= 0 or value > 100_000:
            raise ValueError(f"Suspicious price: {value}")
        return value

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, value: str) -> str:
        """
        Validate that a product title is not empty.

        Args:
            value: Product title.

        Returns:
            Validated title.
        """
        if not value.strip():
            raise ValueError("Empty title")
        return value
