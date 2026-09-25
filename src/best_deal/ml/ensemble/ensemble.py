"""
Price-estimation ensemble for Best Deal.

Description:
    This module contains the current inference-time combination of the RAG,
    fine-tuned, and DNN price specialists.

Responsibilities:
    - Validate ensemble weights.
    - Combine three specialist predictions.
    - Keep ensemble behavior independent of agent orchestration.
"""

from __future__ import annotations


class WeightedPriceEnsemble:
    """
    Combine specialist price estimates using configured weights.

    Returns:
        A configured weighted ensemble.
    """

    def __init__(self, frontier_weight: float = 0.8, specialist_weight: float = 0.1, neural_weight: float = 0.1) -> None:
        """
        Initialize ensemble weights.

        Args:
            frontier_weight: Weight assigned to the RAG/frontier estimate.
            specialist_weight: Weight assigned to the fine-tuned model estimate.
            neural_weight: Weight assigned to the DNN estimate.

        Returns:
            None.
        """
        weights = [frontier_weight, specialist_weight, neural_weight]
        if any(weight < 0 for weight in weights) or not abs(sum(weights) - 1.0) < 1e-9:
            raise ValueError("Ensemble weights must be non-negative and sum to 1")
        self.frontier_weight = frontier_weight
        self.specialist_weight = specialist_weight
        self.neural_weight = neural_weight

    def combine(self, frontier: float, specialist: float, neural: float) -> float:
        """
        Combine the three model estimates into one price estimate.

        Args:
            frontier: RAG/frontier model estimate.
            specialist: Fine-tuned model estimate.
            neural: DNN estimate.

        Returns:
            Weighted ensemble estimate.
        """
        return (
            frontier * self.frontier_weight
            + specialist * self.specialist_weight
            + neural * self.neural_weight
        )
