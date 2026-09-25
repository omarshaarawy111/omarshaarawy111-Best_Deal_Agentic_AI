"""
Hugging Face data collection for Best Deal research datasets.

Description:
    This module contains the reusable collection logic from the original data
    collection notebook. It is intentionally explicit and is not part of the
    normal user-facing application startup path.

Responsibilities:
    - Load selected Amazon metadata categories.
    - Limit the number of records collected per category.
    - Add normalized category labels.
    - Optionally publish a collected dataset to Hugging Face.
"""

from __future__ import annotations

from typing import Iterable

from datasets import Dataset, concatenate_datasets, load_dataset

DEFAULT_CATEGORIES = [
    "All_Beauty",
    "Electronics",
    "Appliances",
    "Home_and_Kitchen",
    "Gift_Cards",
]


class HuggingFaceCollector:
    """
    Collect category-specific product metadata from Hugging Face.

    Returns:
        A reusable collector configured for the selected source categories.
    """

    def __init__(self, dataset_name: str = "McAuley-Lab/Amazon-Reviews-2023", max_items_per_category: int = 15000):
        """
        Initialize the dataset collector.

        Args:
            dataset_name: Hugging Face source dataset name.
            max_items_per_category: Maximum records loaded from each category.

        Returns:
            None.
        """
        self.dataset_name = dataset_name
        self.max_items_per_category = max_items_per_category

    def collect(self, categories: Iterable[str] = DEFAULT_CATEGORIES) -> Dataset:
        """
        Collect and combine the requested categories.

        Args:
            categories: Iterable of Hugging Face dataset configurations.

        Returns:
            A combined Hugging Face Dataset.
        """
        datasets = []
        for category in categories:
            dataset = load_dataset(
                self.dataset_name,
                f"raw_meta_{category}",
                split="full",
                trust_remote_code=True,
            )
            dataset = dataset.select(range(min(self.max_items_per_category, len(dataset))))
            label = category.replace("_", " ").lower()
            dataset = dataset.map(
                lambda batch, label=label: {"category": [label] * len(batch["title"])},
                batched=True,
            )
            datasets.append(dataset)
        if not datasets:
            raise ValueError("At least one category is required for collection")
        return concatenate_datasets(datasets)

    def collect_and_push(self, categories: Iterable[str], target_dataset: str) -> Dataset:
        """
        Collect source data and publish the result to Hugging Face.

        Args:
            categories: Categories to collect.
            target_dataset: Target Hugging Face dataset repository.

        Returns:
            The collected dataset that was published.
        """
        dataset = self.collect(categories)
        dataset.push_to_hub(target_dataset)
        return dataset
