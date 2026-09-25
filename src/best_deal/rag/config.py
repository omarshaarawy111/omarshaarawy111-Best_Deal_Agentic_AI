"""
RAG-specific configuration for Best Deal.

Description:
    This module collects constants used by RAG retrieval, ingestion, and
    evaluation while allowing environment-based overrides.

Responsibilities:
    - Define retrieval and embedding defaults.
    - Define document validation rules.
    - Keep RAG paths independent of a developer's machine.
"""

from best_deal.config import (
    RAG_BM25_PATH,
    RAG_CHUNKS_PATH,
    RAG_COLLECTION_NAME,
    RAG_EMBEDDING_MODEL,
    RAG_GENERATOR_MODEL,
    RAG_GOLDEN_PATH,
    RAG_RETRIEVAL_K,
    RAG_RETRIEVAL_S,
    RAG_WORKERS,
)

PRICE_TOLERANCE = 0.30
GOLDEN_SIZE = 200
BATCH_SIZE = 100
RETRIEVAL_K = RAG_RETRIEVAL_K
RETRIEVAL_S = RAG_RETRIEVAL_S
WORKERS = RAG_WORKERS

KNOWN_CATEGORIES = {"appliances", "electronics", "home and kitchen", "all beauty", "gift cards"}
COLLECTION_NAME = RAG_COLLECTION_NAME
EMBEDDING_MODEL = RAG_EMBEDDING_MODEL
GENERATOR_MODEL = RAG_GENERATOR_MODEL
CHUNKS_PATH = RAG_CHUNKS_PATH
GOLDEN_PATH = RAG_GOLDEN_PATH
