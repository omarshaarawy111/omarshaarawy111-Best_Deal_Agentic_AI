"""
Deterministic planning agent for Best Deal.

Description:
    This module preserves the non-agentic planning flow used as a deterministic
    baseline around the scanner, ensemble, and messaging agents.

Responsibilities:
    - Price the scanner's selected deals.
    - Calculate deal gaps.
    - Apply the explicit threshold and send the chosen alert.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

from best_deal.agents.agent import Agent
from best_deal.agents.deals import Deal, Opportunity
from best_deal.agents.ensemble_agent import EnsembleAgent
from best_deal.agents.messaging_agent import MessagingAgent
from best_deal.agents.scanner_agent import ScannerAgent
from best_deal.config import DEAL_THRESHOLD


class PlanningAgent(Agent):
    """
    Execute the deterministic Best Deal planning baseline.

    Returns:
        A configured planning agent.
    """

    name = "Planning Agent"
    color = Agent.GREEN

    def __init__(self) -> None:
        """
        Initialize the scanner, ensemble, and messaging agents.

        Returns:
            None.
        """
        self.scanner = ScannerAgent()
        self.ensemble = EnsembleAgent()
        self.messenger = MessagingAgent()

    def run(self, deal: Deal) -> Opportunity:
        """
        Estimate one deal and convert it to an Opportunity.

        Args:
            deal: Structured candidate deal.

        Returns:
            Price-estimated opportunity.
        """
        estimate = self.ensemble.price(deal.product_description)
        return Opportunity(deal=deal, estimate=estimate, discount=estimate - deal.price)

    def plan(self, memory: Iterable[Opportunity] = ()) -> Opportunity | None:
        """
        Scan deals, estimate candidates, and alert when the threshold is met.

        Args:
            memory: Previously processed opportunities.

        Returns:
            Best qualifying opportunity, or None.
        """
        selection = self.scanner.scan(memory=memory)
        if not selection or not selection.deals:
            return None
        with ThreadPoolExecutor(max_workers=min(5, len(selection.deals))) as executor:
            opportunities = list(executor.map(self.run, selection.deals[:5]))
        best = max(opportunities, key=lambda opportunity: opportunity.discount)
        if best.discount > DEAL_THRESHOLD:
            self.messenger.alert(best)
            return best
        return None
