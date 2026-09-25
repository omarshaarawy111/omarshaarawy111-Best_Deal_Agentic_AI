"""
Deep neural network architecture used by Best Deal.

Description:
    This module contains only the architecture needed for inference. Training,
    hyperparameter search, and checkpoint creation remain in notebooks or the
    selected training platform.

Responsibilities:
    - Define the residual block.
    - Define the trained Best Deal price-regression network.
"""

from __future__ import annotations

import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Represent one residual block in the Best Deal DNN.

    Returns:
        A reusable PyTorch residual block.
    """

    def __init__(self, hidden_size: int, dropout_prob: float) -> None:
        """
        Initialize the residual block layers.

        Args:
            hidden_size: Number of hidden units.
            dropout_prob: Dropout probability.

        Returns:
            None.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Execute the residual transformation.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after residual addition and activation.
        """
        residual = x
        output = self.block(x)
        output = output + residual
        return self.relu(output)


class DeepNeuralNetwork(nn.Module):
    """
    Define the trained Best Deal residual DNN architecture.

    Returns:
        A PyTorch price-regression model.
    """

    def __init__(self, input_size: int, num_layers: int = 10, hidden_size: int = 4096, dropout_prob: float = 0.2) -> None:
        """
        Initialize the DNN architecture.

        Args:
            input_size: Feature vector size.
            num_layers: Total layer depth used by the trained architecture.
            hidden_size: Width of the hidden representation.
            dropout_prob: Dropout probability.

        Returns:
            None.
        """
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(hidden_size, dropout_prob) for _ in range(num_layers - 2)]
        )
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Predict normalized log-price from feature vectors.

        Args:
            x: Input feature tensor.

        Returns:
            Model prediction tensor.
        """
        x = self.input_layer(x)
        for block in self.residual_blocks:
            x = block(x)
        return self.output_layer(x)
