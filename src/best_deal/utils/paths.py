"""
Filesystem path helpers for Best Deal.

Description:
    This module provides repository-relative path operations and deliberately
    avoids machine-specific absolute paths.

Responsibilities:
    - Resolve project paths consistently.
    - Convert paths into strings for libraries that expect string values.
"""

from __future__ import annotations

from pathlib import Path

from best_deal.config import PROJECT_ROOT


def project_path(*parts: str) -> Path:
    """
    Resolve a path relative to the Best Deal repository root.

    Args:
        *parts: Path segments below the repository root.

    Returns:
        The resolved project-relative path.
    """
    return PROJECT_ROOT.joinpath(*parts)
