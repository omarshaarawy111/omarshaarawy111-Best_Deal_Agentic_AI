"""
Product cleaning and feature engineering for Best Deal.

Description:
    This module extracts the deterministic transformations from the data
    preprocessing notebook. Exploratory plots and ad-hoc analysis remain in
    notebooks and are not imported here.

Responsibilities:
    - Parse product prices and reject invalid/outlier records.
    - Normalize weights into pounds.
    - Build the consolidated product text feature.
    - Deduplicate processed products.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

from best_deal.data.models import Item

MIN_CHARS = 600
MIN_PRICE = 0.5
MAX_PRICE = 999.49
MAX_TEXT_EACH = 3000
MAX_TEXT_TOTAL = 4000
REMOVALS = [
    "Part Number",
    "Best Sellers Rank",
    "Batteries Included?",
    "Batteries Required?",
    "Item model number",
]


class ProductPreprocessor:
    """
    Apply deterministic product preprocessing rules.

    Returns:
        A configured product preprocessor.
    """

    def __init__(self, min_chars: int = MIN_CHARS, min_price: float = MIN_PRICE, max_price: float = MAX_PRICE):
        """
        Initialize preprocessing thresholds.

        Args:
            min_chars: Minimum consolidated text length.
            min_price: Minimum allowed product price.
            max_price: Maximum allowed product price.

        Returns:
            None.
        """
        self.min_chars = min_chars
        self.min_price = min_price
        self.max_price = max_price

    @staticmethod
    def get_weight(details: dict[str, Any]) -> float:
        """
        Convert an Item Weight value into pounds.

        Args:
            details: Parsed product details mapping.

        Returns:
            Product weight in pounds, or zero when unavailable.
        """
        weight_str = details.get("Item Weight")
        if not weight_str:
            return 0.0
        try:
            parts = str(weight_str).split()
            amount = float(parts[0])
            unit = parts[1].lower()
        except (IndexError, ValueError):
            return 0.0
        conversions = {
            "pounds": 1.0,
            "ounces": 1 / 16,
            "grams": 1 / 453.592,
            "milligrams": 1 / 453592,
            "kilograms": 1 / 0.453592,
        }
        if unit in conversions:
            return amount * conversions[unit]
        if unit == "hundredths" and len(parts) > 2 and parts[2].lower() == "pounds":
            return amount / 100
        return 0.0

    @staticmethod
    def simplify(text: Any, max_chars: int = MAX_TEXT_EACH) -> str:
        """
        Normalize whitespace and truncate one text field.

        Args:
            text: Source text value.
            max_chars: Maximum resulting character count.

        Returns:
            Normalized and truncated text.
        """
        return (
            str(text)
            .replace("\n", " ")
            .replace("\r", "")
            .replace("\t", "")
            .replace("  ", " ")
            .strip()[:max_chars]
        )

    @staticmethod
    def scrub(title: str, description: Any, features: Any, details: dict[str, Any], max_total: int = MAX_TEXT_TOTAL) -> str:
        """
        Build the consolidated product text feature.

        Args:
            title: Product title.
            description: Product description.
            features: Product features.
            details: Product detail mapping.
            max_total: Maximum consolidated character count.

        Returns:
            Cleaned product text.
        """
        details = dict(details)
        for field in REMOVALS:
            details.pop(field, None)
        result = title + "\n"
        if description:
            result += ProductPreprocessor.simplify(description) + "\n"
        if features:
            result += ProductPreprocessor.simplify(features) + "\n"
        if details:
            result += json.dumps(details) + "\n"
        pattern = r"\b(?=[A-Z0-9]{7,}\b)(?=.*[A-Z])(?=.*\d)[A-Z0-9]+\b"
        return re.sub(pattern, "", result).strip()[:max_total]

    def parse(self, datapoint: dict[str, Any]) -> Item | None:
        """
        Convert one raw product record into an Item.

        Args:
            datapoint: Raw product mapping from the source dataset.

        Returns:
            A validated Item, or None for a rejected record.
        """
        try:
            price = float(datapoint["price"])
            details = json.loads(datapoint.get("details") or "{}")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if not self.min_price <= price <= self.max_price:
            return None
        title = str(datapoint.get("title") or "").strip()
        if not title:
            return None
        full = self.scrub(
            title,
            datapoint.get("description"),
            datapoint.get("features"),
            details,
        )
        if len(full) < self.min_chars:
            return None
        return Item(
            title=title,
            category=str(datapoint.get("category") or ""),
            price=price,
            full=full,
            weight=self.get_weight(details),
        )

    def transform(self, datapoints: Iterable[dict[str, Any]], deduplicate: bool = True) -> list[Item]:
        """
        Process a collection of raw product records.

        Args:
            datapoints: Iterable of raw product mappings.
            deduplicate: Remove duplicate titles and full-text records.

        Returns:
            List of cleaned product items.
        """
        items = [item for datapoint in datapoints if (item := self.parse(datapoint)) is not None]
        if not deduplicate:
            return items
        seen_titles: set[str] = set()
        items = [item for item in items if not (item.title in seen_titles or seen_titles.add(item.title))]
        seen_full: set[str] = set()
        items = [item for item in items if not (item.full in seen_full or seen_full.add(item.full))]
        return items
