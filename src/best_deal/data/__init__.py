"""
Data pipeline modules for Best Deal.

Description:
    This package contains reusable ingestion, cleaning, preprocessing, batch
    preparation, validation, and data-model components extracted from the
    research notebooks.

Responsibilities:
    - Keep each data pipeline responsibility isolated in its own module.
    - Provide importable building blocks for offline data preparation and future
      scheduled data jobs.
"""

from best_deal.data.models import Item
from best_deal.data.collection import HuggingFaceCollector
from best_deal.data.preprocessing import ProductPreprocessor
from best_deal.data.validation import DatasetValidator

__all__ = ["Item", "HuggingFaceCollector", "ProductPreprocessor", "DatasetValidator"]
