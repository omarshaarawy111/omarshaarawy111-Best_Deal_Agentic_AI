"""
Inference wrapper for the Best Deal deep neural network.

Description:
    This module loads the already-trained DNN checkpoint and reproduces the
    notebook inference transformation without re-training the model.

Responsibilities:
    - Build the feature vectorizer and model.
    - Select an available PyTorch device.
    - Load external checkpoint weights.
    - Convert normalized predictions back to US-dollar prices.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import HashingVectorizer

from best_deal.deep_learning.model import DeepNeuralNetwork

Y_STD = 1.0328539609909058
Y_MEAN = 4.434937953948975


class DeepNeuralNetworkInference:
    """
    Provide deterministic price inference from the trained DNN.

    Returns:
        A reusable inference service.
    """

    def __init__(self) -> None:
        """
        Initialize the inference service without loading model weights yet.

        Returns:
            None.
        """
        self.vectorizer = None
        self.model = None
        self.device = None

    def setup(self) -> None:
        """
        Initialize the vectorizer, model, and compute device.

        Returns:
            None.
        """
        self.vectorizer = HashingVectorizer(n_features=5000, stop_words="english", binary=True)
        self.model = DeepNeuralNetwork(5000)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)
        logging.getLogger(__name__).info("Neural Network is using %s", self.device)

    def load(self, filename: str | Path) -> None:
        """
        Load trained DNN weights from an external checkpoint.

        Args:
            filename: Path to the .pth checkpoint.

        Returns:
            None.
        """
        if self.model is None or self.device is None:
            raise RuntimeError("Call setup() before load().")
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(
                f"DNN checkpoint was not found at {path}. Set BEST_DEAL_DNN_WEIGHTS to the model artifact path."
            )
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)

    def inference(self, text: str) -> float:
        """
        Predict a product price from text.

        Args:
            text: Product description.

        Returns:
            Estimated price in US dollars.
        """
        if self.vectorizer is None or self.model is None or self.device is None:
            raise RuntimeError("The inference service is not set up.")
        self.model.eval()
        with torch.no_grad():
            vector = self.vectorizer.transform([text])
            tensor = torch.tensor(vector.toarray(), dtype=torch.float32, device=self.device)
            prediction = self.model(tensor)[0]
            result = torch.exp(prediction * Y_STD + Y_MEAN) - 1
        return max(0.0, float(result.item()))
