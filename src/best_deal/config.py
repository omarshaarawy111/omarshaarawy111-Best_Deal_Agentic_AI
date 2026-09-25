"""
Application configuration for the Best Deal project.

Description:
    This module centralizes environment-backed configuration and project paths.
    Defaults are derived from the repository layout instead of a developer's
    local filesystem.

Responsibilities:
    - Load environment variables from .env when available.
    - Resolve repository-relative artifact paths.
    - Provide configuration values shared by runtime modules.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RUNTIME_DIR = PROJECT_ROOT / "runtime"

DNN_WEIGHTS_PATH = Path(
    os.getenv(
        "BEST_DEAL_DNN_WEIGHTS",
        str(ARTIFACTS_DIR / "models" / "deep_neural_network.pth"),
    )
)

RAG_DIR = ARTIFACTS_DIR / "rag"
RAG_CHUNKS_PATH = Path(
    os.getenv("BEST_DEAL_RAG_CHUNKS", str(RAG_DIR / "chunks" / "kb_chunks.json"))
)
RAG_VECTORSTORE_PATH = Path(
    os.getenv("BEST_DEAL_RAG_VECTORSTORE", str(RAG_DIR / "preprocessed_db"))
)
RAG_BM25_PATH = Path(
    os.getenv("BEST_DEAL_RAG_BM25", str(RAG_DIR / "bm25_index.pkl"))
)
RAG_GOLDEN_PATH = Path(
    os.getenv("BEST_DEAL_RAG_GOLDEN", str(RAG_DIR / "evaluation" / "golden_dataset_200.json"))
)
RAG_EVAL_CHECKPOINT_PATH = Path(
    os.getenv(
        "BEST_DEAL_RAG_EVAL_CHECKPOINT",
        str(RAG_DIR / "evaluation" / "eval_checkpoint.jsonl"),
    )
)
RAG_EVAL_RESULTS_PATH = Path(
    os.getenv(
        "BEST_DEAL_RAG_EVAL_RESULTS",
        str(RAG_DIR / "evaluation" / "rag_evaluation_results.csv"),
    )
)

AGENT_MEMORY_PATH = Path(
    os.getenv("BEST_DEAL_AGENT_MEMORY", str(RUNTIME_DIR / "memory.json"))
)

RAG_COLLECTION_NAME = os.getenv("BEST_DEAL_RAG_COLLECTION", "docs")
RAG_EMBEDDING_MODEL = os.getenv("BEST_DEAL_RAG_EMBEDDING_MODEL", "text-embedding-3-large")
RAG_GENERATOR_MODEL = os.getenv("BEST_DEAL_RAG_GENERATOR_MODEL", "gpt-4.1-mini")
AUTONOMOUS_AGENT_MODEL = os.getenv("BEST_DEAL_AUTONOMOUS_AGENT_MODEL", "gpt-5.1")
SCANNER_MODEL = os.getenv("BEST_DEAL_SCANNER_MODEL", "gpt-5-mini")
MESSAGING_MODEL = os.getenv("BEST_DEAL_MESSAGING_MODEL", "gpt-5-mini")
PREPROCESSOR_MODEL = os.getenv("BEST_DEAL_PREPROCESSOR_MODEL", "ollama/llama3.2")
PREPROCESSOR_BASE_URL = os.getenv("BEST_DEAL_PREPROCESSOR_BASE_URL", "http://localhost:11434")

DEAL_THRESHOLD = float(os.getenv("BEST_DEAL_THRESHOLD", "50"))
RETRIEVAL_K = int(os.getenv("BEST_DEAL_RETRIEVAL_K", "10"))
RETRIEVAL_S = int(os.getenv("BEST_DEAL_RETRIEVAL_S", "10"))
RAG_WORKERS = int(os.getenv("BEST_DEAL_RAG_WORKERS", "3"))
RAG_RETRIEVAL_K = int(os.getenv("BEST_DEAL_RAG_RETRIEVAL_K", "10"))
RAG_RETRIEVAL_S = int(os.getenv("BEST_DEAL_RAG_RETRIEVAL_S", "10"))


def ensure_runtime_directories() -> None:
    """
    Create directories needed for local runtime artifacts.

    Returns:
        None.
    """
    AGENT_MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAG_CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAG_VECTORSTORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAG_BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAG_GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAG_EVAL_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
