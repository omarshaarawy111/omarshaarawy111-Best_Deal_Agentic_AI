"""
Gradio dashboard for Best Deal RAG evaluation.

Description:
    This module contains only the UI layer that sits after the evaluation
    pipeline. All data loading, evaluation, checkpointing, and aggregation
    happen in RAGEvaluator before these components are rendered.

Responsibilities:
    - Trigger the reusable evaluator.
    - Format evaluation metrics as Gradio-friendly HTML and dataframes.
    - Launch the optional RAG evaluation dashboard.
"""

from __future__ import annotations

import gradio as gr
import pandas as pd

from best_deal.rag.config import PRICE_TOLERANCE
from best_deal.rag.evaluation.evaluator import RAGEvaluator


def _format_metric(label: str, value: float, percentage: bool = False, currency: bool = False) -> str:
    """
    Format one evaluation metric for a Gradio HTML card.

    Args:
        label: Metric label.
        value: Metric value.
        percentage: Display value as a percentage.
        currency: Display value as currency.

    Returns:
        HTML metric card.
    """
    if percentage:
        formatted = f"{value:.1%}"
    elif currency:
        formatted = f"${value:,.2f}"
    else:
        formatted = f"{value:.4f}"
    return f"<div style='margin:10px 0;padding:15px;border-radius:8px;border:1px solid #ddd;'><div>{label}</div><strong style='font-size:28px'>{formatted}</strong></div>"


def run_evaluation(progress=gr.Progress()) -> tuple[str, pd.DataFrame, str, pd.DataFrame]:
    """
    Run pending RAG evaluation cases and prepare dashboard outputs.

    Returns:
        Retrieval HTML, retrieval category dataframe, price HTML, and price category dataframe.
    """
    evaluator = RAGEvaluator()
    golden = evaluator.loader.load_golden_dataset()
    for _, _, progress_value in evaluator.evaluate_all(golden, resume=True):
        progress(progress_value, desc="Evaluating RAG golden set...")
    results = evaluator.load_checkpoint_results()
    summary = evaluator.aggregate(results)
    retrieval = summary["retrieval"]
    price = summary["price"]
    retrieval_html = "".join(
        [
            _format_metric("Recall@10", retrieval["Recall@10"]),
            _format_metric("MRR", retrieval["MRR"]),
            _format_metric("nDCG@10", retrieval["nDCG@10"]),
            f"<p>Relevance = same category + price within {int(PRICE_TOLERANCE * 100)}%.</p>",
        ]
    )
    price_html = "".join(
        [
            _format_metric("MAE", price["MAE"], currency=True),
            _format_metric("RMSE", price["RMSE"], currency=True),
            _format_metric("Within ±20%", price["Within ±20%"], percentage=True),
            _format_metric("Answer extraction rate", price["Answer extraction rate"], percentage=True),
        ]
    )
    category = summary["category"]
    retrieval_chart = category[["category", "recall_at_10"]].rename(columns={"recall_at_10": "Average Recall@10"})
    price_chart = category[["category", "mae"]].rename(columns={"mae": "Average MAE ($)"})
    return retrieval_html, retrieval_chart, price_html, price_chart


def create_dashboard() -> gr.Blocks:
    """
    Build the RAG evaluation Gradio dashboard.

    Returns:
        A configured Gradio Blocks application.
    """
    with gr.Blocks(title="Best Deal RAG Evaluation Dashboard") as app:
        gr.Markdown("# Best Deal RAG Evaluation Dashboard")
        gr.Markdown("Evaluation execution happens in RAGEvaluator before dashboard rendering.")
        button = gr.Button("Run Full Evaluation", variant="primary")
        with gr.Row():
            retrieval_metrics = gr.HTML("Run evaluation to calculate retrieval metrics.")
            retrieval_chart = gr.BarPlot(x="category", y="Average Recall@10", title="Average Recall@10 by Category", y_lim=[0, 1])
        with gr.Row():
            price_metrics = gr.HTML("Run evaluation to calculate price metrics.")
            price_chart = gr.BarPlot(x="category", y="Average MAE ($)", title="Average MAE by Category")
        button.click(run_evaluation, outputs=[retrieval_metrics, retrieval_chart, price_metrics, price_chart])
    return app


def launch_dashboard() -> None:
    """
    Launch the standalone RAG evaluation dashboard.

    Returns:
        None.
    """
    create_dashboard().launch(inbrowser=True)
