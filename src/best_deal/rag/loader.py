"""
RAG artifact and dataset loading helpers for Best Deal.

Description:
    This module loads persisted chunk files and research datasets without
    embedding or reprocessing them.

Responsibilities:
    - Load saved RAG chunks from JSON.
    - Load the selected Hugging Face training/validation products for ingestion.
    - Load frozen golden evaluation records.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from best_deal.data.models import Item
from best_deal.rag.config import CHUNKS_PATH, GOLDEN_PATH
from best_deal.rag.schemas import Result


class RAGLoader:
    """
    Load persisted RAG resources and research datasets.

    Returns:
        A configured RAG loader.
    """

    def __init__(self, chunks_path: str | Path = CHUNKS_PATH, golden_path: str | Path = GOLDEN_PATH) -> None:
        """
        Initialize RAG artifact paths.

        Args:
            chunks_path: Persisted knowledge-base chunk JSON path.
            golden_path: Frozen evaluation dataset path.

        Returns:
            None.
        """
        self.chunks_path = Path(chunks_path)
        self.golden_path = Path(golden_path)

    def load_chunks(self) -> list[Result]:
        """
        Load persisted knowledge-base chunks.

        Returns:
            List of RAG Result objects.
        """
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"RAG chunks were not found at {self.chunks_path}")
        data = json.loads(self.chunks_path.read_text(encoding="utf-8"))
        return [Result.model_validate(item) for item in data]

    def load_golden_dataset(self) -> list[dict[str, Any]]:
        """
        Load the frozen RAG golden dataset.

        Returns:
            Golden evaluation records.
        """
        if not self.golden_path.exists():
            raise FileNotFoundError(f"Golden dataset was not found at {self.golden_path}")
        return json.loads(self.golden_path.read_text(encoding="utf-8"))

    def load_training_documents(self, dataset_name: str) -> pd.DataFrame:
        """
        Load train and validation products from a Hugging Face dataset.

        Args:
            dataset_name: Best Deal dataset repository name.

        Returns:
            Product dataframe used for RAG ingestion.
        """
        train, validation, _ = Item.from_hub(dataset_name)
        items = train + validation
        return pd.DataFrame(
            [
                {
                    "product_id": index,
                    "title": item.title,
                    "category": item.category,
                    "price": item.price,
                    "summary": item.summary,
                }
                for index, item in enumerate(items)
            ]
        )
