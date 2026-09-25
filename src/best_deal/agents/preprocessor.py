"""
LLM-based text preprocessing before Best Deal pricing agents.

Description:
    This module preserves the lightweight local preprocessing model used before
    the three pricing specialists execute.

Responsibilities:
    - Normalize incoming deal descriptions.
    - Produce concise product-focused model input through LiteLLM.
"""

from __future__ import annotations

from litellm import completion

from best_deal.config import PREPROCESSOR_BASE_URL, PREPROCESSOR_MODEL
from best_deal.agents.agent import Agent

SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""


class Preprocessor(Agent):
    """
    Prepare normalized product text for downstream pricing models.

    Returns:
        A configured preprocessing agent.
    """

    name = "Preprocessor Agent"
    color = Agent.CYAN

    def __init__(self) -> None:
        """
        Initialize the preprocessing model configuration.

        Returns:
            None.
        """
        self.model_name = PREPROCESSOR_MODEL
        self.base_url = PREPROCESSOR_BASE_URL

    def messages_for(self, text: str) -> list[dict[str, str]]:
        """
        Build the preprocessing prompt messages.

        Args:
            text: Product text to normalize.

        Returns:
            LiteLLM-compatible chat messages.
        """
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]

    def preprocess(self, text: str | dict[str, str]) -> str:
        """
        Normalize a product description using the configured local LLM.

        Args:
            text: Raw description or mapping containing a summary field.

        Returns:
            Normalized product description.
        """
        if isinstance(text, dict):
            question_text = str(text.get("summary") or "").strip()
        else:
            question_text = str(text or "").strip()
        if not question_text:
            return "No question found."
        response = completion(
            messages=self.messages_for(question_text),
            model=self.model_name,
            api_base=self.base_url,
        )
        return str(response.choices[0].message.content or "").strip()
