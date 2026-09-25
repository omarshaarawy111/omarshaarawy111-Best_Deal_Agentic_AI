"""
Business deal metrics for Best Deal.

Description:
    This module contains deterministic pricing-gap calculations that are useful
    to both the current agent flow and future database-backed deal engines.

Responsibilities:
    - Calculate price gaps and discount percentages.
    - Determine whether a price gap crosses configured business thresholds.
    - Aggregate deal-level metrics.
"""

from __future__ import annotations

from typing import Any


class DealEvaluator:
    """
    Evaluate a current price against an estimated fair value.

    Returns:
        A configured deal evaluator.
    """

    def __init__(self, min_discount_percent: float = 0.15, min_absolute_gap: float = 10.0) -> None:
        """
        Initialize deal thresholds.

        Args:
            min_discount_percent: Minimum discount fraction.
            min_absolute_gap: Minimum absolute savings in dollars.

        Returns:
            None.
        """
        self.min_discount_percent = min_discount_percent
        self.min_absolute_gap = min_absolute_gap

    def calculate_deal_score(self, estimated_price: float, current_price: float) -> float:
        """
        Calculate the estimated-value discount fraction.

        Args:
            estimated_price: Estimated fair value.
            current_price: Current deal price.

        Returns:
            Non-negative discount fraction.
        """
        if estimated_price <= 0:
            return 0.0
        return max(0.0, (estimated_price - current_price) / estimated_price)

    def is_hot_deal(self, estimated_price: float, current_price: float) -> bool:
        """
        Determine whether a price gap satisfies both deal thresholds.

        Args:
            estimated_price: Estimated fair value.
            current_price: Current deal price.

        Returns:
            True when both percentage and dollar-gap rules are satisfied.
        """
        gap = estimated_price - current_price
        return self.calculate_deal_score(estimated_price, current_price) >= self.min_discount_percent and gap >= self.min_absolute_gap

    def get_deal_details(self, estimated_price: float, current_price: float) -> dict[str, Any]:
        """
        Return a complete deal metric mapping.

        Args:
            estimated_price: Estimated fair value.
            current_price: Current deal price.

        Returns:
            Deal gap, percentage, and hot-deal decision.
        """
        gap = estimated_price - current_price
        score = self.calculate_deal_score(estimated_price, current_price)
        return {
            "estimated_price": round(estimated_price, 2),
            "current_price": round(current_price, 2),
            "deal_gap": round(gap, 2),
            "deal_percent": round(score, 4),
            "is_hot_deal": self.is_hot_deal(estimated_price, current_price),
            "discount_dollars": round(gap, 2),
            "discount_percentage": round(score * 100, 1),
        }


def calculate_business_metrics(deals: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Calculate aggregate savings statistics for a set of deals.

    Args:
        deals: Deal mappings containing estimated_price, current_price, and is_hot_deal/is_hot.

    Returns:
        Aggregate business metrics.
    """
    if not deals:
        return {}
    hot = [deal for deal in deals if deal.get("is_hot_deal", deal.get("is_hot", False))]
    savings = [float(deal.get("estimated_price", 0)) - float(deal.get("current_price", 0)) for deal in deals]
    hot_savings = [float(deal.get("estimated_price", 0)) - float(deal.get("current_price", 0)) for deal in hot]
    return {
        "total_deals_evaluated": len(deals),
        "hot_deals_identified": len(hot),
        "hot_deal_percentage": round(len(hot) / len(deals) * 100, 1),
        "total_savings": round(sum(savings), 2),
        "average_savings_per_deal": round(sum(savings) / len(savings), 2),
        "average_savings_per_hot_deal": round(sum(hot_savings) / len(hot_savings), 2) if hot_savings else 0.0,
    }
