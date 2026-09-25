"""
RAG evaluation metrics for Best Deal.

Description:
    This module contains pure, UI-independent metrics for retrieval relevance
    and price estimation accuracy.

Responsibilities:
    - Measure Recall@k, MRR, and nDCG@k.
    - Define product relevance from category and price tolerance.
    - Measure price error and tolerance accuracy.
"""

from __future__ import annotations

import math
from typing import Any

from best_deal.rag.config import PRICE_TOLERANCE
from best_deal.rag.schemas import PriceEstimate, Result
from best_deal.utils.parsing import extract_price


def retrieved_product_ids(retrieved_docs: list[Result]) -> list[int]:
    """
    Extract product identifiers from retrieved documents.

    Args:
        retrieved_docs: Retrieved RAG documents.

    Returns:
        Product ids that are present in document metadata.
    """
    return [int(doc.metadata["product_id"]) for doc in retrieved_docs if doc.metadata.get("product_id") is not None]


def is_relevant(expected_category: str, expected_price: float, metadata: dict[str, Any], tolerance: float = PRICE_TOLERANCE) -> bool:
    """
    Determine whether a retrieved product is a useful price comparable.

    Args:
        expected_category: Golden-set product category.
        expected_price: Golden-set product price.
        metadata: Retrieved product metadata.
        tolerance: Relative price tolerance.

    Returns:
        True when category matches and price is within tolerance.
    """
    if str(metadata.get("type", "")).lower() != str(expected_category).lower():
        return False
    doc_price = metadata.get("price")
    if doc_price is None or expected_price == 0:
        return False
    return abs(float(doc_price) - expected_price) / abs(expected_price) <= tolerance


def relevant_flags(expected_category: str, expected_price: float, retrieved_docs: list[Result], tolerance: float = PRICE_TOLERANCE) -> list[bool]:
    """
    Convert retrieved documents into binary relevance flags.

    Args:
        expected_category: Golden-set product category.
        expected_price: Golden-set product price.
        retrieved_docs: Retrieved RAG documents.
        tolerance: Relative price tolerance.

    Returns:
        Relevance flags aligned with retrieved document order.
    """
    return [is_relevant(expected_category, expected_price, doc.metadata, tolerance) for doc in retrieved_docs]


def recall_at_k(expected_category: str, expected_price: float, retrieved_docs: list[Result], k: int, tolerance: float = PRICE_TOLERANCE) -> float:
    """
    Compute binary retrieval recall at k using comparable-product relevance.

    Args:
        expected_category: Golden-set product category.
        expected_price: Golden-set product price.
        retrieved_docs: Retrieved RAG documents.
        k: Retrieval cutoff.
        tolerance: Relative price tolerance.

    Returns:
        Recall@k value between zero and one.
    """
    return float(any(relevant_flags(expected_category, expected_price, retrieved_docs[:k], tolerance)))


def reciprocal_rank(expected_category: str, expected_price: float, retrieved_docs: list[Result], tolerance: float = PRICE_TOLERANCE) -> float:
    """
    Compute reciprocal rank of the first relevant comparable.

    Args:
        expected_category: Golden-set product category.
        expected_price: Golden-set product price.
        retrieved_docs: Retrieved RAG documents.
        tolerance: Relative price tolerance.

    Returns:
        Reciprocal rank value between zero and one.
    """
    for rank, relevant in enumerate(relevant_flags(expected_category, expected_price, retrieved_docs, tolerance), start=1):
        if relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(expected_category: str, expected_price: float, retrieved_docs: list[Result], k: int, tolerance: float = PRICE_TOLERANCE) -> float:
    """
    Compute binary nDCG@k for comparable-product relevance.

    Args:
        expected_category: Golden-set product category.
        expected_price: Golden-set product price.
        retrieved_docs: Retrieved RAG documents.
        k: Retrieval cutoff.
        tolerance: Relative price tolerance.

    Returns:
        nDCG@k value between zero and one.
    """
    for rank, relevant in enumerate(relevant_flags(expected_category, expected_price, retrieved_docs[:k], tolerance), start=1):
        if relevant:
            return 1.0 / math.log2(rank + 1)
    return 0.0



def calculate_price_metrics(estimate: PriceEstimate | str | float | None, reference_answer: float) -> dict[str, float | None | str]:
    """
    Calculate absolute, percentage, and tolerance-based price metrics.

    Args:
        estimate: Structured estimate or raw model output.
        reference_answer: Ground-truth price.

    Returns:
        A mapping of price metrics.
    """
    confidence = None
    if isinstance(estimate, PriceEstimate):
        predicted = estimate.estimated_price
        confidence = estimate.confidence
        if confidence == "none":
            predicted = None
    else:
        predicted = extract_price(estimate)
    actual = float(reference_answer)
    if predicted is None:
        return {
            "predicted_price": None,
            "confidence": confidence,
            "absolute_error": None,
            "percentage_error": None,
            "within_10": 0.0,
            "within_20": 0.0,
            "within_30": 0.0,
        }
    absolute_error = abs(float(predicted) - actual)
    percentage_error = absolute_error / abs(actual) * 100 if actual else None
    return {
        "predicted_price": float(predicted),
        "confidence": confidence,
        "absolute_error": absolute_error,
        "percentage_error": percentage_error,
        "within_10": float(percentage_error is not None and percentage_error <= 10),
        "within_20": float(percentage_error is not None and percentage_error <= 20),
        "within_30": float(percentage_error is not None and percentage_error <= 30),
    }
