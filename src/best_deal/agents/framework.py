"""
Runtime orchestration framework for Best Deal agents.

Description:
    This module is the backend application framework used by the Gradio entry
    point. It manages memory and delegates business execution to the autonomous
    planning agent.

Responsibilities:
    - Initialize logging and memory.
    - Lazily initialize the autonomous planner.
    - Persist newly notified opportunities for deduplication.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from best_deal.agents.autonomous_planning_agent import AutonomousPlanningAgent
from best_deal.agents.deals import Opportunity
from best_deal.config import AGENT_MEMORY_PATH
from best_deal.utils.logging import configure_logging


class DealAgentFramework:
    """
    Coordinate Best Deal agent execution and lightweight local memory.

    Returns:
        A configured application framework.
    """

    def __init__(self, memory_path: str | Path = AGENT_MEMORY_PATH) -> None:
        """
        Initialize framework logging and runtime memory.

        Args:
            memory_path: JSON path for lightweight local runtime memory.

        Returns:
            None.
        """
        configure_logging()
        self.memory_path = Path(memory_path)
        self.memory = self.read_memory()
        self.planner: AutonomousPlanningAgent | None = None

    def read_memory(self) -> list[Opportunity]:
        """
        Load persisted opportunities from JSON memory.

        Returns:
            Previously processed opportunities.
        """
        if not self.memory_path.exists():
            return []
        data = json.loads(self.memory_path.read_text(encoding="utf-8"))
        return [Opportunity.model_validate(item) for item in data]

    def write_memory(self) -> None:
        """
        Persist the current opportunity memory as JSON.

        Returns:
            None.
        """
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory_path.write_text(
            json.dumps([opportunity.model_dump() for opportunity in self.memory], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def reset_memory(self, keep_first: int = 0) -> None:
        """
        Reset local opportunity memory while optionally preserving leading records.

        Args:
            keep_first: Number of initial records to preserve.

        Returns:
            None.
        """
        self.memory = self.memory[:keep_first]
        self.write_memory()

    def init_agents_as_needed(self) -> None:
        """
        Lazily initialize the autonomous planning agent.

        Returns:
            None.
        """
        if self.planner is None:
            self.planner = AutonomousPlanningAgent()

    def run(self) -> list[Opportunity]:
        """
        Execute one complete autonomous Best Deal run.

        Returns:
            Updated opportunity memory.
        """
        self.init_agents_as_needed()
        result = self.planner.plan(memory=self.memory)
        if result:
            self.memory.append(result)
            self.write_memory()
        return self.memory
