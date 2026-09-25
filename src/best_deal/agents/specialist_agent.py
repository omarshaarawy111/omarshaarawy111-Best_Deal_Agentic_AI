"""
Remote fine-tuned pricing specialist for Best Deal.

Description:
    This module adapts the fine-tuned quantized model already trained and
    deployed remotely on Modal into the local agent interface.

Responsibilities:
    - Connect to the existing Modal Pricer service.
    - Send normalized product descriptions for price inference.
    - Avoid re-training or loading large fine-tuning artifacts locally.
"""

from __future__ import annotations

from best_deal.agents.agent import Agent
from best_deal.integrations.modal_service import ModalPricerService


class SpecialistAgent(Agent):
    """
    Estimate product value with the remotely deployed fine-tuned model.

    Returns:
        A configured specialist agent.
    """

    name = "Specialist Agent"
    color = Agent.RED

    def __init__(self, service: ModalPricerService | None = None) -> None:
        """
        Initialize the Modal pricing adapter.

        Args:
            service: Optional injected Modal pricing service.

        Returns:
            None.
        """
        self.pricer = service or ModalPricerService()
        self.log("Specialist Agent is ready")

    def price(self, description: str | dict[str, str]) -> float:
        """
        Estimate the price of a product using the remote fine-tuned model.

        Args:
            description: Product description or mapping containing a summary.

        Returns:
            Estimated price in US dollars.
        """
        if isinstance(description, dict):
            question_text = str(description.get("summary") or "").strip()
        else:
            question_text = str(description or "").strip()
        if not question_text:
            return 0.0
        self.log("Specialist Agent is calling the remote fine-tuned model")
        result = float(self.pricer.price(question_text))
        self.log(f"Specialist Agent completed - predicting ${result:.2f}")
        return result
