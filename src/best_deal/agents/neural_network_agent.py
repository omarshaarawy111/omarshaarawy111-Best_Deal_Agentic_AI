"""
Deep neural network pricing agent for Best Deal.

Description:
    This module adapts the trained DNN architecture to the agent interface.
    Model weights remain external artifacts and are never committed to Git.

Responsibilities:
    - Initialize DNN inference.
    - Load a configured .pth artifact.
    - Estimate a product price during ensemble inference.
"""

from __future__ import annotations

from pathlib import Path

from best_deal.agents.agent import Agent
from best_deal.config import DNN_WEIGHTS_PATH
from best_deal.deep_learning.inference import DeepNeuralNetworkInference


class NeuralNetworkAgent(Agent):
    """
    Estimate product value with the trained deep neural network.

    Returns:
        A configured neural network agent.
    """

    name = "Neural Network Agent"
    color = Agent.MAGENTA

    def __init__(self, weights_path: str | Path = DNN_WEIGHTS_PATH) -> None:
        """
        Initialize DNN inference and load model weights.

        Args:
            weights_path: Path to the external model checkpoint.

        Returns:
            None.
        """
        self.neural_network = DeepNeuralNetworkInference()
        self.neural_network.setup()
        self.neural_network.load(weights_path)
        self.log("Neural Network Agent is ready and weights are loaded")

    def price(self, description: str | dict[str, str]) -> float:
        """
        Estimate a product price with the loaded DNN.

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
        self.log("Neural Network Agent is starting a prediction")
        result = round(float(self.neural_network.inference(question_text)), 2)
        self.log(f"Neural Network Agent completed - predicting ${result:.2f}")
        return result
