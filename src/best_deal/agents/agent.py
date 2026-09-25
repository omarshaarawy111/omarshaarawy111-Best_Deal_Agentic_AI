"""
Base agent logging behavior for Best Deal.

Description:
    This module provides the common logger formatting inherited by all agents.

Responsibilities:
    - Give every agent a consistent name and color.
    - Send agent messages through Python logging instead of printing directly.
"""

from __future__ import annotations

import logging

from best_deal.utils.logging import configure_logging


class Agent:
    """
    Provide shared identity and logging behavior for Best Deal agents.

    Returns:
        A base agent class for specialized agents.
    """

    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BG_BLACK = "\033[40m"
    RESET = "\033[0m"

    name = "Agent"
    color = WHITE

    def log(self, message: str) -> None:
        """
        Write a color-coded message for the current agent.

        Args:
            message: Message to send to the application logger.

        Returns:
            None.
        """
        configure_logging()
        color_code = self.BG_BLACK + self.color
        logging.getLogger().info(f"{color_code}[{self.name}] {message}{self.RESET}")
