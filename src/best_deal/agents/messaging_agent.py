"""
Pushover notification agent for Best Deal.

Description:
    This module sends deal alerts through Pushover and can optionally use an
    LLM to craft a short notification message.

Responsibilities:
    - Build and send push notification payloads.
    - Alert on structured opportunities.
    - Craft readable deal messages through LiteLLM.
"""

from __future__ import annotations

import os

import requests
from litellm import completion

from best_deal.agents.agent import Agent
from best_deal.agents.deals import Opportunity
from best_deal.config import MESSAGING_MODEL


class MessagingAgent(Agent):
    """
    Send Best Deal notifications through Pushover.

    Returns:
        A configured messaging agent.
    """

    name = "Messaging Agent"
    color = Agent.WHITE
    PUSHOVER_URL = "https://api.pushover.net/1/messages.json"

    def __init__(self) -> None:
        """
        Initialize Pushover credentials from environment variables.

        Returns:
            None.
        """
        self.pushover_user = os.getenv("PUSHOVER_USER")
        self.pushover_token = os.getenv("PUSHOVER_TOKEN")
        self.log("Messaging Agent is ready")

    def push(self, text: str) -> None:
        """
        Send a push notification through Pushover.

        Args:
            text: Notification body.

        Returns:
            None.
        """
        if not self.pushover_user or not self.pushover_token:
            raise RuntimeError("PUSHOVER_USER and PUSHOVER_TOKEN must be configured")
        response = requests.post(
            self.PUSHOVER_URL,
            data={"user": self.pushover_user, "token": self.pushover_token, "message": text, "sound": "cashregister"},
            timeout=20,
        )
        response.raise_for_status()

    def alert(self, opportunity: Opportunity) -> None:
        """
        Send a concise notification for an Opportunity.

        Args:
            opportunity: Deal opportunity to notify about.

        Returns:
            None.
        """
        text = (
            f"Deal Alert! Price=${opportunity.deal.price:.2f}, "
            f"Estimate=${opportunity.estimate:.2f}, "
            f"Discount=${opportunity.discount:.2f}: "
            f"{opportunity.deal.product_description[:120]}... {opportunity.deal.url}"
        )
        self.push(text[:900])
        self.log("Messaging Agent completed")

    def craft_message(self, description: str, deal_price: float, estimated_true_value: float) -> str:
        """
        Generate a short user-facing deal message.

        Args:
            description: Product description.
            deal_price: Current deal price.
            estimated_true_value: Estimated fair value.

        Returns:
            Generated notification message.
        """
        prompt = (
            "Please summarize this great deal in 2-3 sentences for a push notification.\n"
            f"Item Description: {description}\n"
            f"Offered Price: {deal_price}\n"
            f"Estimated true value: {estimated_true_value}\n"
            "Respond only with the message."
        )
        response = completion(model=MESSAGING_MODEL, messages=[{"role": "user", "content": prompt}])
        return str(response.choices[0].message.content or "").strip()

    def notify(self, description: str, deal_price: float, estimated_true_value: float, url: str) -> None:
        """
        Craft and send a notification for an LLM-selected deal.

        Args:
            description: Product description.
            deal_price: Current deal price.
            estimated_true_value: Estimated fair value.
            url: Deal URL.

        Returns:
            None.
        """
        text = self.craft_message(description, deal_price, estimated_true_value)
        self.push(f"{text[:800]}... {url}")
        self.log("Messaging Agent completed")
