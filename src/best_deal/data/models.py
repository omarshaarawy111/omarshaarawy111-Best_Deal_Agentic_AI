"""
Data models used by the Best Deal data and RAG pipelines.

Description:
    This module defines validated product records and the small serialization
    helpers used by the Hugging Face research datasets.

Responsibilities:
    - Represent processed products with a stable Pydantic schema.
    - Prepare prompt/completion records for fine-tuning datasets.
    - Load and publish versioned datasets when explicitly requested.
"""

from __future__ import annotations

from typing import Any, Optional, Self

from datasets import Dataset, DatasetDict, load_dataset
from pydantic import BaseModel

PREFIX = "Price is $"
QUESTION = "What does this cost to the nearest dollar?"


class Item(BaseModel):
    """
    Represent a cleaned and curated Best Deal product.

    Returns:
        A validated product record.
    """

    title: str
    category: str
    price: float
    full: Optional[str] = None
    weight: Optional[float] = None
    summary: Optional[str] = None
    prompt: Optional[str] = None
    completion: Optional[str] = None
    id: Optional[int] = None

    def make_prompt(self, text: str) -> None:
        """
        Create the original pricing prompt for the item.

        Args:
            text: Product text supplied to the model.

        Returns:
            None.
        """
        self.prompt = f"{QUESTION}\n\n{text}\n\n{PREFIX}{round(self.price)}.00"

    def test_prompt(self) -> str:
        """
        Return the prompt prefix used to test a pricing model.

        Returns:
            The prompt text ending at the expected price prefix.
        """
        return (self.prompt or "").split(PREFIX)[0] + PREFIX

    def count_tokens(self, tokenizer: Any) -> int:
        """
        Count summary tokens using the supplied tokenizer.

        Args:
            tokenizer: Tokenizer exposing an encode method.

        Returns:
            Number of summary tokens.
        """
        return len(tokenizer.encode(self.summary or "", add_special_tokens=False))

    def make_prompts(self, tokenizer: Any, max_tokens: int, do_round: bool) -> None:
        """
        Create fine-tuning prompt and completion fields.

        Args:
            tokenizer: Tokenizer used to enforce the context limit.
            max_tokens: Maximum summary token count.
            do_round: Whether the target price should be rounded.

        Returns:
            None.
        """
        tokens = tokenizer.encode(self.summary or "", add_special_tokens=False)
        summary = tokenizer.decode(tokens[:max_tokens]).rstrip() if len(tokens) > max_tokens else (self.summary or "")
        self.prompt = f"{QUESTION}\n\n{summary}\n\n{PREFIX}"
        self.completion = f"{round(self.price)}.00" if do_round else str(self.price)

    def count_prompt_tokens(self, tokenizer: Any) -> int:
        """
        Count tokens across prompt and completion.

        Args:
            tokenizer: Tokenizer used for counting.

        Returns:
            Number of combined prompt and completion tokens.
        """
        return len(tokenizer.encode((self.prompt or "") + (self.completion or ""), add_special_tokens=False))

    def to_datapoint(self) -> dict[str, str | None]:
        """
        Convert a fine-tuning record to a simple mapping.

        Returns:
            A prompt/completion dictionary.
        """
        return {"prompt": self.prompt, "completion": self.completion}

    @classmethod
    def from_hub(cls, dataset_name: str) -> tuple[list[Self], list[Self], list[Self]]:
        """
        Load train, validation, and test items from Hugging Face.

        Args:
            dataset_name: Hugging Face dataset repository name.

        Returns:
            Three validated item lists for train, validation, and test data.
        """
        dataset = load_dataset(dataset_name)
        return (
            [cls.model_validate(row) for row in dataset["train"]],
            [cls.model_validate(row) for row in dataset["validation"]],
            [cls.model_validate(row) for row in dataset["test"]],
        )

    @staticmethod
    def push_to_hub(dataset_name: str, train: list["Item"], val: list["Item"], test: list["Item"]) -> None:
        """
        Publish item splits to a Hugging Face dataset repository.

        Args:
            dataset_name: Target Hugging Face dataset repository.
            train: Training items.
            val: Validation items.
            test: Test items.

        Returns:
            None.
        """
        DatasetDict(
            {
                "train": Dataset.from_list([item.model_dump() for item in train]),
                "validation": Dataset.from_list([item.model_dump() for item in val]),
                "test": Dataset.from_list([item.model_dump() for item in test]),
            }
        ).push_to_hub(dataset_name)

    @staticmethod
    def push_prompts_to_hub(dataset_name: str, train: list["Item"], val: list["Item"], test: list["Item"]) -> None:
        """
        Publish prompt/completion splits to Hugging Face.

        Args:
            dataset_name: Target Hugging Face dataset repository.
            train: Training items.
            val: Validation items.
            test: Test items.

        Returns:
            None.
        """
        DatasetDict(
            {
                "train": Dataset.from_list([item.to_datapoint() for item in train]),
                "val": Dataset.from_list([item.to_datapoint() for item in val]),
                "test": Dataset.from_list([item.to_datapoint() for item in test]),
            }
        ).push_to_hub(dataset_name)

    @classmethod
    def prompt_from_hub(cls, dataset_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Load prompt/completion splits from Hugging Face.

        Args:
            dataset_name: Hugging Face dataset repository name.

        Returns:
            Raw train, validation, and test prompt mappings.
        """
        dataset = load_dataset(dataset_name)
        return list(dataset["train"]), list(dataset["val"]), list(dataset["test"])

    def __repr__(self) -> str:
        """
        Return a compact product representation.

        Returns:
            A readable product label and price.
        """
        return f"<{self.title} = ${self.price}>"
