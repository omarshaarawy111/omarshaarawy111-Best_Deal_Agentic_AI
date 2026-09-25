"""
Main module for the Best Deal Gradio App.

Description:
    This module is the current application entry point. It launches the Gradio
    interface, runs the DealAgentFramework in a worker thread, and streams agent
    logs and deal results into the UI.

Responsibilities:
    - Launch the Best Deal Gradio application.
    - Keep UI concerns separate from agent and RAG business logic.
    - Stream logs from background agent execution without blocking Gradio.
    - Refresh the autonomous deal scan on the existing five-minute timer.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Generator

import gradio as gr

from best_deal.agents.framework import DealAgentFramework
from best_deal.utils.logging import reformat


class QueueHandler(logging.Handler):
    """
    Forward logging records from background agents into a thread-safe queue.

    Returns:
        A queue-backed logging handler.
    """

    def __init__(self, log_queue: queue.Queue[str]) -> None:
        """
        Initialize the queue-backed log handler.

        Args:
            log_queue: Queue that receives formatted log strings.

        Returns:
            None.
        """
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        """
        Add one formatted log record to the queue.

        Args:
            record: Python logging record.

        Returns:
            None.
        """
        self.log_queue.put(self.format(record))


def html_for(log_data: list[str]) -> str:
    """
    Convert recent application logs into the Gradio HTML output.

    Args:
        log_data: Stored formatted log messages.

    Returns:
        HTML container containing the latest log messages.
    """
    output = "<br>".join(log_data[-18:])
    return (
        "<div id='scrollContent' style='height:400px;overflow-y:auto;"
        "border:1px solid #ccc;background-color:#222229;padding:10px;'>"
        f"{output}</div>"
    )


def setup_logging(log_queue: queue.Queue[str]) -> QueueHandler:
    """
    Attach a temporary queue handler to the root application logger.

    Args:
        log_queue: Queue receiving formatted records for the UI.

    Returns:
        The handler so it can be removed after the run completes.
    """
    handler = QueueHandler(log_queue)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S %z"))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return handler


class App:
    """
    Build and run the Best Deal Gradio application.

    Returns:
        A reusable application object.
    """

    def __init__(self, agent_framework: DealAgentFramework | None = None) -> None:
        """
        Initialize the UI and its agent framework reference.

        Args:
            agent_framework: Optional injected framework for testing.

        Returns:
            None.
        """
        self.agent_framework = agent_framework or DealAgentFramework()

    def get_agent_framework(self) -> DealAgentFramework:
        """
        Return the application agent framework.

        Returns:
            The configured DealAgentFramework.
        """
        return self.agent_framework

    @staticmethod
    def table_for(opportunities: list) -> list[list[Any]]:
        """
        Convert Opportunity objects into Gradio table rows.

        Args:
            opportunities: Opportunity objects to display.

        Returns:
            Table rows containing product, price, estimate, discount, and URL.
        """
        return [
            [
                opportunity.deal.product_description,
                f"${opportunity.deal.price:.2f}",
                f"${opportunity.estimate:.2f}",
                f"${opportunity.discount:.2f}",
                opportunity.deal.url,
            ]
            for opportunity in opportunities
        ]

    def update_output(
        self,
        log_data: list[str],
        log_queue: queue.Queue[str],
        result_queue: queue.Queue[list[list[Any]]],
    ) -> Generator[tuple[list[str], str, list[list[Any]]], None, None]:
        """
        Stream logs and results from one background framework execution.

        Args:
            log_data: Existing UI log state.
            log_queue: Queue containing agent logs.
            result_queue: Queue containing final table rows.

        Yields:
            Updated log state, log HTML, and current results table.
        """
        initial_result = self.table_for(self.get_agent_framework().memory)
        final_result: list[list[Any]] | None = None
        while True:
            emitted = False
            try:
                message = log_queue.get_nowait()
                log_data.append(reformat(message))
                emitted = True
            except queue.Empty:
                pass
            try:
                final_result = result_queue.get_nowait()
                emitted = True
            except queue.Empty:
                pass
            if emitted:
                yield log_data, html_for(log_data), final_result or initial_result
            elif final_result is not None:
                break
            else:
                time.sleep(0.1)

    def do_run(self) -> list[list[Any]]:
        """
        Run the agent framework and convert memory to UI table rows.

        Returns:
            Table rows for the Gradio dataframe.
        """
        return self.table_for(self.get_agent_framework().run())

    def run_with_logging(
        self, initial_log_data: list[str]
    ) -> Generator[tuple[list[str], str, list[list[Any]]], None, None]:
        """
        Execute the deal framework in a background thread while streaming logs.

        Args:
            initial_log_data: Existing log state from the Gradio session.

        Yields:
            Updated Gradio outputs while the background run executes.
        """
        log_queue: queue.Queue[str] = queue.Queue()
        result_queue: queue.Queue[list[list[Any]]] = queue.Queue()
        handler = setup_logging(log_queue)

        def worker() -> None:
            """
            Execute one deal-framework run and push its result into the queue.

            Returns:
                None.
            """
            try:
                result_queue.put(self.do_run())
            except Exception as exc:
                logging.getLogger().exception("Best Deal run failed")
                result_queue.put(self.table_for(self.get_agent_framework().memory) + [[f"ERROR: {exc}", "", "", "", ""]])

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        try:
            yield from self.update_output(initial_log_data, log_queue, result_queue)
        finally:
            logging.getLogger().removeHandler(handler)

    def do_select(self, selected_index: gr.SelectData) -> None:
        """
        Send a selected opportunity through the existing notification flow.

        Args:
            selected_index: Gradio row-selection event.

        Returns:
            None.
        """
        opportunities = self.get_agent_framework().memory
        if not opportunities or not selected_index.index:
            return
        row = selected_index.index[0]
        if row >= len(opportunities):
            return
        planner = self.get_agent_framework().planner
        if planner is None:
            self.get_agent_framework().init_agents_as_needed()
            planner = self.get_agent_framework().planner
        opportunity = opportunities[row]
        if planner is not None:
            planner.messenger.notify(
                opportunity.deal.product_description,
                opportunity.deal.price,
                opportunity.estimate,
                opportunity.deal.url,
            )

    def build_ui(self) -> gr.Blocks:
        """
        Build the Best Deal Gradio interface.

        Returns:
            Configured Gradio Blocks application.
        """
        with gr.Blocks(title="Best Deal", fill_width=True) as ui:
            log_data = gr.State([])
            gr.Markdown(
                "<div style='text-align:center;font-size:24px'><strong>Best Deal</strong> - Autonomous Agent Framework that hunts for deals</div>"
            )
            gr.Markdown(
                "<div style='text-align:center;font-size:14px'>A fine-tuned pricing model deployed remotely and a RAG pipeline with a frontier model collaborate to identify online deals.</div>"
            )
            with gr.Row():
                opportunities_dataframe = gr.Dataframe(
                    headers=["Description", "Price", "Estimate", "Discount", "URL"],
                    wrap=True,
                    row_count=(10, "dynamic"),
                    column_count=(5, "fixed"),
                    max_height=400,
                )
            with gr.Row():
                logs = gr.HTML()

            ui.load(
                self.run_with_logging,
                inputs=[log_data],
                outputs=[log_data, logs, opportunities_dataframe],
            )
            timer = gr.Timer(value=300, active=True)
            timer.tick(
                self.run_with_logging,
                inputs=[log_data],
                outputs=[log_data, logs, opportunities_dataframe],
            )
            opportunities_dataframe.select(self.do_select)
        return ui

    def run(self) -> None:
        """
        Launch the Best Deal Gradio application in the browser.

        Returns:
            None.
        """
        self.build_ui().launch(share=False, inbrowser=True)


if __name__ == "__main__":
    App().run()
