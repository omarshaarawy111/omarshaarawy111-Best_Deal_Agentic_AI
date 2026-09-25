"""
Modal SSL trust helper for the Best Deal project.

Description:
    This module provides the small runtime helper used when local Modal CLI
    execution requires the system trust store. It is intentionally isolated
    from the application business logic and is not executed during normal
    application startup.

Responsibilities:
    - Inject the system certificate store before starting Modal.
    - Preserve command-line arguments when forwarding execution to Modal.
    - Provide a small deployment/support utility without machine-specific paths.
"""

from __future__ import annotations

import runpy
import sys

import truststore


def run_modal() -> None:
    """
    Execute the installed Modal CLI after injecting the system trust store.

    Returns:
        None.
    """
    truststore.inject_into_ssl()
    sys.argv = ["modal", *sys.argv[1:]]
    runpy.run_module("modal", run_name="__main__")


if __name__ == "__main__":
    run_modal()
