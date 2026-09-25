"""
Retrieval-Augmented Generation modules for Best Deal.

Description:
    This package contains the reusable RAG retrieval, reranking, ingestion,
    answer, and evaluation components extracted from the notebooks.

Responsibilities:
    - Keep retrieval stages modular and independently testable.
    - Expose the full answer pipeline through a single service.
    - Keep expensive ingestion and evaluation callable explicitly rather than
      running them when the application starts.
"""
