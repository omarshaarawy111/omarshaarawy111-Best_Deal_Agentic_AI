"""
Autonomous tool-calling planner for Best Deal.

Description:
    This module implements the existing agentic control loop. The language model
    chooses when to call scanning, price-estimation, and notification tools.

Responsibilities:
    - Expose the three Best Deal agents as tool definitions.
    - Execute tool calls safely and concurrently when independent.
    - Maintain per-run memory and opportunity state.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable

from openai import OpenAI

from best_deal.agents.agent import Agent
from best_deal.agents.deals import Deal, Opportunity
from best_deal.agents.ensemble_agent import EnsembleAgent
from best_deal.agents.messaging_agent import MessagingAgent
from best_deal.agents.scanner_agent import ScannerAgent
from best_deal.config import AUTONOMOUS_AGENT_MODEL


class AutonomousPlanningAgent(Agent):
    """
    Run the autonomous Best Deal tool-calling loop.

    Returns:
        A configured autonomous planner.
    """

    name = "Autonomous Planning Agent"
    color = Agent.GREEN

    SYSTEM_MESSAGE = "You find great deals on bargain products using your tools, and notify the user of the best bargain."
    USER_MESSAGE = """First, use your tool to scan the internet for bargain deals. Then for each deal, use your tool to estimate its true value.
Then pick the single most compelling deal where the estimated true value exceeds the deal price by more than $50, and use your tool to notify the user.
If no deal meets this threshold, do not call the notification tool, and just reply OK.
Then just reply OK to indicate success."""

    def __init__(self, client: OpenAI | None = None) -> None:
        """
        Initialize the autonomous planner and its tools.

        Args:
            client: Optional OpenAI client for dependency injection.

        Returns:
            None.
        """
        self.scanner = ScannerAgent()
        self.ensemble = EnsembleAgent()
        self.messenger = MessagingAgent()
        self.openai = client or OpenAI()
        self.memory: Iterable[Opportunity] = ()
        self.opportunity: Opportunity | None = None
        self.log("Autonomous Planning Agent is ready")

    def scan_the_internet_for_bargains(self) -> str:
        """
        Run the scanner tool and serialize its structured result.

        Returns:
            JSON string containing selected deals, or a no-deals message.
        """
        self.log("Autonomous Planning Agent is calling scanner")
        results = self.scanner.scan(memory=self.memory)
        return results.model_dump_json() if results else "No deals found"

    def estimate_true_value(self, description: str) -> str:
        """
        Estimate a product's true value using the model ensemble.

        Args:
            description: Product description.

        Returns:
            Text result containing the estimated value.
        """
        estimate = self.ensemble.price(description)
        return f"The estimated true value of {description} is {estimate:.2f}"

    def notify_user_of_deal(self, description: str, deal_price: float, estimated_true_value: float, url: str) -> str:
        """
        Send the chosen deal notification once per planning run.

        Args:
            description: Product description.
            deal_price: Current deal price.
            estimated_true_value: Estimated fair value.
            url: Deal URL.

        Returns:
            Notification status message.
        """
        if self.opportunity is not None:
            self.log("Autonomous Planning Agent ignored a duplicate notification request")
            return "Notification already sent"
        self.messenger.notify(description, deal_price, estimated_true_value, url)
        deal = Deal(product_description=description, price=deal_price, url=url)
        self.opportunity = Opportunity(
            deal=deal,
            estimate=estimated_true_value,
            discount=estimated_true_value - deal_price,
        )
        return "Notification sent ok"

    def get_tools(self) -> list[dict[str, Any]]:
        """
        Build OpenAI-compatible tool schemas for the planner.

        Returns:
            Tool definitions for scanning, estimating, and notifying.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "scan_the_internet_for_bargains",
                    "description": "Returns top bargains scraped from the internet along with their current prices.",
                    "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "estimate_true_value",
                    "description": "Estimate how much an item is actually worth.",
                    "parameters": {
                        "type": "object",
                        "properties": {"description": {"type": "string", "description": "Description of the item."}},
                        "required": ["description"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "notify_user_of_deal",
                    "description": "Send the user a push notification about the single most compelling deal; call at most once.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "deal_price": {"type": "number"},
                            "estimated_true_value": {"type": "number"},
                            "url": {"type": "string"},
                        },
                        "required": ["description", "deal_price", "estimated_true_value", "url"],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    def handle_tool_call(self, message: Any) -> list[dict[str, Any]]:
        """
        Execute tool calls returned by the planner.

        Args:
            message: Assistant message containing tool calls.

        Returns:
            Tool response messages to append to the model conversation.
        """
        mapping = {
            "scan_the_internet_for_bargains": self.scan_the_internet_for_bargains,
            "estimate_true_value": self.estimate_true_value,
            "notify_user_of_deal": self.notify_user_of_deal,
        }

        def run_one(tool_call: Any) -> dict[str, Any]:
            """
            Execute one OpenAI tool call.

            Args:
                tool_call: OpenAI tool-call object.

            Returns:
                Tool response message.
            """
            name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments or "{}")
            tool = mapping.get(name)
            result = tool(**arguments) if tool else f"Unknown tool: {name}"
            return {"role": "tool", "content": str(result), "tool_call_id": tool_call.id}

        tool_calls = list(message.tool_calls or [])
        if len(tool_calls) <= 1:
            return [run_one(tool_calls[0])] if tool_calls else []
        with ThreadPoolExecutor(max_workers=len(tool_calls)) as executor:
            return list(executor.map(run_one, tool_calls))

    def plan(self, memory: Iterable[Opportunity] = ()) -> Opportunity | None:
        """
        Run the model-directed tool loop for one autonomous planning cycle.

        Args:
            memory: Previously processed opportunities.

        Returns:
            The notified Opportunity, or None when no notification was selected.
        """
        self.memory = memory
        self.opportunity = None
        messages: list[Any] = [
            {"role": "system", "content": self.SYSTEM_MESSAGE},
            {"role": "user", "content": self.USER_MESSAGE},
        ]
        self.log("Autonomous Planning Agent is kicking off a run")
        while True:
            response = self.openai.chat.completions.create(
                model=AUTONOMOUS_AGENT_MODEL,
                messages=messages,
                tools=self.get_tools(),
            )
            message = response.choices[0].message
            if response.choices[0].finish_reason != "tool_calls":
                break
            messages.append(message.model_dump(exclude_none=True) if hasattr(message, "model_dump") else message)
            messages.extend(self.handle_tool_call(message))
        self.log(f"Autonomous Planning Agent completed with: {message.content}")
        return self.opportunity
