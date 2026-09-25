"""
Multi-model pricing ensemble agent for Best Deal.

Description:
    This module runs preprocessing, then executes the RAG, fine-tuned, and DNN
    specialists concurrently before applying the current weighted ensemble.

Responsibilities:
    - Normalize incoming deal text.
    - Run independent price specialists in parallel.
    - Combine their estimates with the configured ensemble weights.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from best_deal.agents.agent import Agent
from best_deal.agents.frontier_agent import FrontierAgent
from best_deal.agents.neural_network_agent import NeuralNetworkAgent
from best_deal.agents.preprocessor import Preprocessor
from best_deal.agents.specialist_agent import SpecialistAgent
from best_deal.ml.ensemble.ensemble import WeightedPriceEnsemble


class EnsembleAgent(Agent):
    """
    Combine the three Best Deal pricing specialists.

    Returns:
        A configured ensemble agent.
    """

    name = "Ensemble Agent"
    color = Agent.YELLOW

    def __init__(self, ensemble: WeightedPriceEnsemble | None = None) -> None:
        """
        Initialize preprocessing and pricing specialists.

        Args:
            ensemble: Optional injected ensemble combiner.

        Returns:
            None.
        """
        self.preprocessor = Preprocessor()
        self.specialist = SpecialistAgent()
        self.frontier = FrontierAgent()
        self.neural_network = NeuralNetworkAgent()
        self.ensemble = ensemble or WeightedPriceEnsemble()
        self.log("Ensemble Agent is ready")

    def price(self, description: str) -> float:
        """
        Estimate a product value from three concurrent specialists.

        Args:
            description: Raw product description.

        Returns:
            Weighted ensemble price estimate in US dollars.
        """
        question = self.preprocessor.preprocess(description)
        with ThreadPoolExecutor(max_workers=3) as executor:
            frontier_future = executor.submit(lambda: self.frontier.price(question)[0])
            specialist_future = executor.submit(self.specialist.price, question)
            neural_future = executor.submit(self.neural_network.price, question)
            frontier = float(frontier_future.result())
            specialist = float(specialist_future.result())
            neural = float(neural_future.result())
        combined = self.ensemble.combine(frontier, specialist, neural)
        self.log(f"Ensemble Agent complete - returning ${combined:.2f}")
        return round(combined, 2)
