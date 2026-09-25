"""
OpenAI batch preparation and execution helpers for Best Deal.

Description:
    This module extracts the expensive LLM batch preparation workflow from the
    research notebook. It is intended for controlled dataset preparation and
    recovery, not normal application startup.

Responsibilities:
    - Build JSONL files for batch requests.
    - Submit, poll, and download batch jobs.
    - Apply batch outputs back to item records.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI


class OpenAIBatchPreparer:
    """
    Prepare and run OpenAI batch jobs for product curation.

    Returns:
        A configured batch helper.
    """

    def __init__(self, client: OpenAI | None = None, model: str = "gpt-4.1-mini"):
        """
        Initialize the OpenAI batch helper.

        Args:
            client: Optional OpenAI client instance.
            model: Model used in generated batch requests.

        Returns:
            None.
        """
        self.client = client or OpenAI()
        self.model = model

    @staticmethod
    def make_request(self, custom_id: str, prompt: str) -> dict[str, Any]:
        """
        Create one JSONL batch request.

        Args:
            custom_id: Stable request identifier.
            prompt: User prompt sent to the model.

        Returns:
            One OpenAI batch request mapping.
        """
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
            },
        }

    def make_file(self, requests: Iterable[dict[str, Any]], filename: str | Path) -> Path:
        """
        Write batch requests to a JSONL file.

        Args:
            requests: Request mappings.
            filename: Destination JSONL path.

        Returns:
            The created file path.
        """
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for request in requests:
                file.write(json.dumps(request, ensure_ascii=False) + "\n")
        return path

    def upload_file(self, filename: str | Path) -> Any:
        """
        Upload a JSONL batch file to OpenAI.

        Args:
            filename: JSONL file path.

        Returns:
            The OpenAI file object.
        """
        with Path(filename).open("rb") as file:
            return self.client.files.create(file=file, purpose="batch")

    def submit_batch(self, file_id: str) -> Any:
        """
        Submit an uploaded batch file for processing.

        Args:
            file_id: OpenAI file identifier.

        Returns:
            The OpenAI batch object.
        """
        return self.client.batches.create(input_file_id=file_id, endpoint="/v1/chat/completions", completion_window="24h")

    def wait_for_completion(self, batch_id: str, poll_seconds: int = 30, timeout_seconds: int = 86400) -> Any:
        """
        Poll a batch until it completes or times out.

        Args:
            batch_id: OpenAI batch identifier.
            poll_seconds: Delay between status checks.
            timeout_seconds: Maximum polling duration.

        Returns:
            The completed batch object.
        """
        started = time.monotonic()
        while time.monotonic() - started < timeout_seconds:
            batch = self.client.batches.retrieve(batch_id)
            if batch.status in {"completed", "failed", "cancelled", "expired"}:
                if batch.status != "completed":
                    raise RuntimeError(f"OpenAI batch ended with status: {batch.status}")
                return batch
            time.sleep(poll_seconds)
        raise TimeoutError(f"Timed out waiting for batch {batch_id}")

    def download_output(self, output_file_id: str, filename: str | Path) -> Path:
        """
        Download the completed batch output.

        Args:
            output_file_id: Output file identifier.
            filename: Destination path.

        Returns:
            The created output file path.
        """
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = self.client.files.content(output_file_id).read()
        path.write_bytes(content)
        return path
