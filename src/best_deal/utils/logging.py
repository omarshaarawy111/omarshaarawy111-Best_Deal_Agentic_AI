"""
Logging helpers for the Best Deal runtime.

Description:
    This module standardizes application logging while preserving the ANSI
    colors used by the existing Gradio interface.

Responsibilities:
    - Initialize the application logger.
    - Reformat colored agent logs for Gradio HTML output.
"""

from __future__ import annotations

import logging
import sys

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"
BG_BLACK = "\033[40m"
BG_BLUE = "\033[44m"
RESET = "\033[0m"

COLOR_TO_HTML = {
    BG_BLACK + RED: "#dd0000",
    BG_BLACK + GREEN: "#00dd00",
    BG_BLACK + YELLOW: "#dddd00",
    BG_BLACK + BLUE: "#0000ee",
    BG_BLACK + MAGENTA: "#aa00dd",
    BG_BLACK + CYAN: "#00dddd",
    BG_BLACK + WHITE: "#87CEEB",
    BG_BLUE + WHITE: "#ff7800",
}


def configure_logging() -> logging.Logger:
    """
    Configure the root logger used by the Best Deal runtime.

    Returns:
        The configured root logger.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] [Best Deal] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S %z",
            )
        )
        logger.addHandler(handler)
    return logger


def reformat(message: str) -> str:
    """
    Convert ANSI agent colors into HTML spans for Gradio.

    Args:
        message: Log message containing optional ANSI color sequences.

    Returns:
        The HTML-formatted log message.
    """
    for key, value in COLOR_TO_HTML.items():
        message = message.replace(key, f'<span style="color: {value}">')
    return message.replace(RESET, "</span>")
