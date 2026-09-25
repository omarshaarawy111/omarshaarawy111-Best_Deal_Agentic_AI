"""
Frontier RAG pricing agent for Best Deal.

Description:
    This module adapts the reusable RAGAnswerService to the agent interface.
    RAG retrieval, reranking, prompting, and structured generation remain in
    the dedicated rag package instead of being duplicated here.

Responsibilities:
    - Accept a product description from the ensemble.
    - Delegate estimation to the reusable RAG service.
    - Return the numeric estimate and retrieved context expected by the current workflow.
"""

from __future__ import annotations

from best_deal.agents.agent import Agent
from best_deal.rag.services.answer import RAGAnswerService


class FrontierAgent(Agent):
    """
    Estimate product value using the Best Deal RAG pipeline.

    Returns:
        A configured frontier/RAG pricing agent.
    """

    name = "Frontier Agent"
    color = Agent.BLUE

    def __init__(self, rag_service: RAGAnswerService | None = None) -> None:
        """
        Initialize the RAG pricing adapter.

        Args:
            rag_service: Optional injected RAG answer service.

        Returns:
            None.
        """
        self.rag = rag_service or RAGAnswerService()
        self.log("Frontier Agent is ready")

    def price(self, question: str, history: list[dict[str, str]] | None = None) -> tuple[float, list]:
        """
        Estimate product value and return its retrieval context.

        Args:
            question: Product description or question.
            history: Optional conversation history.

        Returns:
            Numeric estimate and retrieved RAG chunks.
        """
        if not str(question or "").strip():
            return 0.0, []
        result = self.rag.price(question, history)
        self.log(f"Frontier Agent completed - predicting ${result[0]:.2f}")
        return result
