"""
Setup configuration for the Best Deal package.

Description:
    This file enables editable installation so every runtime module can use
    absolute imports from the best_deal package.

Responsibilities:
    - Discover the src-layout package.
    - Install the runtime dependencies required by the current source stage.
"""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent

setup(
    name="best-deal",
    version="0.1.0",
    description="AI-powered multi-agent deal detection and price estimation platform",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.11",
    install_requires=[
        "beautifulsoup4",
        "chromadb",
        "datasets",
        "feedparser",
        "gradio",
        "huggingface-hub",
        "litellm",
        "modal",
        "numpy",
        "openai",
        "pandas",
        "pydantic>=2",
        "python-dotenv",
        "rank-bm25",
        "requests",
        "scikit-learn",
        "tenacity",
        "torch",
        "tqdm",
    ],
    extras_require={
        "dev": ["pytest", "black", "ruff", "mypy"],
        "notebooks": ["jupyter", "jupyterlab", "matplotlib", "plotly"],
    },
)
