"""
Data validation helpers for Best Deal.

Description:
    This module validates processed tabular or item-level data before it moves
    into downstream training or RAG preparation.

Responsibilities:
    - Validate required columns and basic value ranges.
    - Validate product objects before embedding or model training.
"""

from __future__ import annotations

from typing import Iterable

from best_deal.data.models import Item


class DatasetValidator:
    """
    Validate processed Best Deal records.

    Returns:
        A reusable validator instance.
    """

    def __init__(self, min_price: float = 0.5, max_price: float = 999.49):
        """
        Initialize validation thresholds.

        Args:
            min_price: Minimum accepted price.
            max_price: Maximum accepted price.

        Returns:
            None.
        """
        self.min_price = min_price
        self.max_price = max_price

    def validate_item(self, item: Item) -> bool:
        """
        Validate one processed product item.

        Args:
            item: Product item to validate.

        Returns:
            True when the item satisfies the validation rules.
        """
        if not item.title.strip() or not item.category.strip():
            return False
        if not self.min_price <= item.price <= self.max_price:
            return False
        return bool(item.full and len(item.full.strip()) > 0)

    def validate_items(self, items: Iterable[Item]) -> list[Item]:
        """
        Keep only valid processed items.

        Args:
            items: Product items to validate.

        Returns:
            Valid item records.
        """
        return [item for item in items if self.validate_item(item)]
