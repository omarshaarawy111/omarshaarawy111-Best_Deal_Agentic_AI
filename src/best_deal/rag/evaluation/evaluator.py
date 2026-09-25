"""
Checkpointed RAG golden-set evaluator for Best Deal.

Description:
    This module contains everything in the RAG evaluation notebook that must
    run before the dashboard: golden-set loading, validation, RAG invocation,
    metric calculation, checkpoint/resume, aggregation, and result persistence.

Responsibilities:
    - Evaluate each golden question through the reusable RAGAnswerService.
    - Save each completed case immediately for crash-safe recovery.
    - Aggregate overall and category-level metrics.
    - Export reproducible evaluation results for CI or future monitoring jobs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from best_deal.config import RAG_EVAL_CHECKPOINT_PATH, RAG_EVAL_RESULTS_PATH
from best_deal.rag.config import RETRIEVAL_K
from best_deal.rag.evaluation.metrics import ndcg_at_k, recall_at_k, reciprocal_rank, calculate_price_metrics
from best_deal.rag.loader import RAGLoader
from best_deal.rag.services.answer import RAGAnswerService


class RAGEvaluator:
    """
    Run and aggregate checkpointed Best Deal RAG evaluation.

    Returns:
        A reusable evaluator that does not depend on Gradio.
    """

    def __init__(self, answer_service: RAGAnswerService | None = None, checkpoint_path: str | Path = RAG_EVAL_CHECKPOINT_PATH, results_path: str | Path = RAG_EVAL_RESULTS_PATH) -> None:
        """
        Initialize the evaluator and result paths.

        Args:
            answer_service: Optional injected RAG answer service.
            checkpoint_path: JSONL checkpoint path.
            results_path: CSV result path.

        Returns:
            None.
        """
        self.answer_service = answer_service or RAGAnswerService()
        self.loader = RAGLoader()
        self.checkpoint_path = Path(checkpoint_path)
        self.results_path = Path(results_path)

    def validate_golden_dataset(self, golden_dataset: list[dict[str, Any]]) -> None:
        """
        Validate uniqueness and required fields in the golden dataset.

        Args:
            golden_dataset: Golden evaluation records.

        Returns:
            None.
        """
        product_ids = [item["product_id"] for item in golden_dataset]
        if len(product_ids) != len(set(product_ids)):
            raise ValueError("Duplicate product_id found in golden dataset")
        required = {"product_id", "title", "category", "question", "reference_answer"}
        for item in golden_dataset:
            if not required.issubset(item):
                raise ValueError(f"Golden record is missing required fields: {item}")

    def evaluate_case(self, test: dict[str, Any]) -> dict[str, Any]:
        """
        Evaluate one golden product through retrieval and price estimation.

        Args:
            test: Golden evaluation record.

        Returns:
            One evaluation result mapping.
        """
        question = str(test["question"])
        expected_category = str(test["category"])
        expected_price = float(test["reference_answer"])
        result = self.answer_service.answer(question)
        chunks = result.chunks
        retrieval_metrics = {
            "recall@1": recall_at_k(expected_category, expected_price, chunks, 1),
            "recall@5": recall_at_k(expected_category, expected_price, chunks, 5),
            "recall@10": recall_at_k(expected_category, expected_price, chunks, RETRIEVAL_K),
            "mrr": reciprocal_rank(expected_category, expected_price, chunks),
            "ndcg@10": ndcg_at_k(expected_category, expected_price, chunks, RETRIEVAL_K),
        }
        price_metrics = calculate_price_metrics(result.estimate, expected_price)
        return {
            "product_id": test["product_id"],
            "category": expected_category,
            "question": question,
            "reference_answer": expected_price,
            "generated_answer": result.estimate.reasoning,
            "matched_category": result.estimate.matched_category,
            "comparable_products_used": result.estimate.comparable_products_used,
            "confidence": result.estimate.confidence,
            **retrieval_metrics,
            **price_metrics,
        }

    def _load_completed_ids(self) -> set[Any]:
        """
        Read successful checkpointed product identifiers.

        Returns:
            Product ids that are already evaluated.
        """
        if not self.checkpoint_path.exists():
            return set()
        completed: set[Any] = set()
        with self.checkpoint_path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    row = json.loads(line)
                    if "error" not in row:
                        completed.add(row["product_id"])
        return completed

    def evaluate_all(self, golden_dataset: list[dict[str, Any]], resume: bool = True) -> Iterator[tuple[dict[str, Any], dict[str, Any], float]]:
        """
        Evaluate the golden dataset with checkpoint-and-resume behavior.

        Args:
            golden_dataset: Golden evaluation records.
            resume: Skip records already completed successfully.

        Yields:
            The source record, result mapping, and progress fraction.
        """
        self.validate_golden_dataset(golden_dataset)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        done_ids = self._load_completed_ids() if resume else set()
        total = len(golden_dataset)
        with self.checkpoint_path.open("a", encoding="utf-8") as file:
            for index, test in enumerate(golden_dataset, start=1):
                if test["product_id"] in done_ids:
                    continue
                try:
                    result = self.evaluate_case(test)
                except Exception as exc:
                    result = {"product_id": test["product_id"], "category": test["category"], "error": str(exc)}
                file.write(json.dumps(result, ensure_ascii=False) + "\n")
                file.flush()
                yield test, result, index / total if total else 1.0

    def load_checkpoint_results(self) -> pd.DataFrame:
        """
        Load all successful checkpoint records into a dataframe.

        Returns:
            Evaluation results dataframe.
        """
        if not self.checkpoint_path.exists():
            return pd.DataFrame()
        rows = []
        with self.checkpoint_path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    row = json.loads(line)
                    if "error" not in row:
                        rows.append(row)
        return pd.DataFrame(rows)

    def aggregate(self, results_df: pd.DataFrame) -> dict[str, pd.DataFrame | dict[str, float]]:
        """
        Aggregate retrieval and price metrics for dashboard or CI output.

        Args:
            results_df: Successful case-level evaluation results.

        Returns:
            Retrieval summary, price summary, and category breakdown tables.
        """
        if results_df.empty:
            raise ValueError("No evaluation results are available")
        retrieval_summary = {
            "Recall@1": results_df["recall@1"].mean(),
            "Recall@5": results_df["recall@5"].mean(),
            "Recall@10": results_df["recall@10"].mean(),
            "MRR": results_df["mrr"].mean(),
            "nDCG@10": results_df["ndcg@10"].mean(),
        }
        valid = results_df.dropna(subset=["absolute_error", "percentage_error"])
        price_summary = {
            "MAE": valid["absolute_error"].mean() if not valid.empty else np.nan,
            "RMSE": np.sqrt(np.mean(valid["absolute_error"] ** 2)) if not valid.empty else np.nan,
            "Median AE": valid["absolute_error"].median() if not valid.empty else np.nan,
            "MAPE": valid["percentage_error"].mean() if not valid.empty else np.nan,
            "Within ±10%": valid["within_10"].mean() if not valid.empty else 0.0,
            "Within ±20%": valid["within_20"].mean() if not valid.empty else 0.0,
            "Within ±30%": valid["within_30"].mean() if not valid.empty else 0.0,
            "Answer extraction rate": len(valid) / len(results_df),
        }
        category = (
            results_df.groupby("category")
            .agg(
                recall_at_10=("recall@10", "mean"),
                mrr=("mrr", "mean"),
                ndcg_at_10=("ndcg@10", "mean"),
                mae=("absolute_error", "mean"),
                within_20=("within_20", "mean"),
            )
            .reset_index()
        )
        return {"retrieval": retrieval_summary, "price": price_summary, "category": category}

    def run(self, resume: bool = True) -> dict[str, Any]:
        """
        Execute pending golden evaluations and persist aggregated results.

        Args:
            resume: Reuse successful checkpoint cases.

        Returns:
            Aggregated evaluation payload with case-level results.
        """
        golden_dataset = self.loader.load_golden_dataset()
        for _, _, _ in self.evaluate_all(golden_dataset, resume=resume):
            pass
        results_df = self.load_checkpoint_results()
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(self.results_path, index=False)
        aggregate = self.aggregate(results_df)
        return {"results": results_df, **aggregate}
