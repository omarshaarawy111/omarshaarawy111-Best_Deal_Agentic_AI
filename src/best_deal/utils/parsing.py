"""
Parsing helpers for the Best Deal application.

Description:
    This module contains small, dependency-light parsing functions that can be
    shared by model adapters, evaluation code, and future API integrations.

Responsibilities:
    - Extract numeric prices from model outputs.
    - Normalize common currency formatting before conversion.
"""

from __future__ import annotations

import re
from typing import Any


def extract_price(text: Any) -> float | None:
    """
    Extract the first numeric price from model output.

    Args:
        text: Model output containing an optional price value.

    Returns:
        Parsed price as a float, or None when no numeric value is present.
    """
    if text is None:
        return None

    value = str(text).replace(",", "")
    dollar_match = re.search(r"\$\s*(\d+(?:\.\d+)?)", value)
    if dollar_match:
        return float(dollar_match.group(1))

    number_match = re.search(r"(?<![A-Za-z])(\d+(?:\.\d+)?)(?![A-Za-z])", value)
    return float(number_match.group(1)) if number_match else None
