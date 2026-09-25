"""
Explicit RAG ingestion service for Best Deal.

Description:
    This module converts the validated research dataset into product-level RAG
    chunks, persists the golden evaluation set, creates dense OpenAI embeddings,
    stores them in Chroma, and builds the sparse BM25 index.

Responsibilities:
    - Prepare the held-out golden dataset.
    - Validate and persist knowledge-base chunks.
    - Build the dense and sparse retrieval artifacts.
    - Keep expensive ingestion explicit rather than automatic on startup.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from best_deal.rag.bm25 import BM25Store
from best_deal.config import RAG_VECTORSTORE_PATH
from best_deal.rag.config import BATCH_SIZE, CHUNKS_PATH, GOLDEN_PATH
from best_deal.rag.embeddings import OpenAIEmbedder
from best_deal.rag.loader import RAGLoader
from best_deal.rag.schemas import ChunkValidation, Result
from best_deal.rag.splitter import ProductSplitter
from best_deal.rag.vectorstore import ChromaVectorStore


class RAGIngestionService:
    """
    Build the persistent Best Deal RAG knowledge base on demand.

    Returns:
        A configured RAG ingestion service.
    """

    def __init__(self, embedder: OpenAIEmbedder | None = None) -> None:
        """
        Initialize the RAG ingestion dependencies.

        Args:
            embedder: Optional injected embedding service.

        Returns:
            None.
        """
        self.loader = RAGLoader()
        self.splitter = ProductSplitter()
        self.embedder = embedder or OpenAIEmbedder()

    def create_golden_dataset(self, documents: pd.DataFrame, golden_size: int = 200, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split documents into a knowledge base and a stratified golden set.

        Args:
            documents: Full product dataframe.
            golden_size: Number of held-out evaluation products.
            random_state: Reproducible split seed.

        Returns:
            Tuple containing knowledge-base rows and golden rows.
        """
        if golden_size >= len(documents):
            raise ValueError("golden_size must be smaller than the document count")
        kb_df, golden_df = train_test_split(
            documents,
            test_size=golden_size,
            stratify=documents["category"],
            random_state=random_state,
        )
        return kb_df.reset_index(drop=True), golden_df.reset_index(drop=True)

    def build_valid_chunks(self, rows: pd.DataFrame) -> list[Result]:
        """
        Build and validate product-level chunks from knowledge-base rows.

        Args:
            rows: Knowledge-base product dataframe.

        Returns:
            Valid chunks ready for embedding.
        """
        documents = self.splitter.build_documents(rows.to_dict(orient="records"))
        chunks = self.splitter.create_chunks(documents)
        valid_chunks: list[Result] = []
        for chunk in chunks:
            ChunkValidation(
                page_content=chunk.page_content,
                product_id=int(chunk.metadata["product_id"]),
                type=str(chunk.metadata["type"]),
                title=str(chunk.metadata["title"]),
                price=float(chunk.metadata["price"]),
                source=str(chunk.metadata["source"]),
            )
            valid_chunks.append(chunk)
        return valid_chunks

    def persist_chunks(self, chunks: list[Result], path: str | Path = CHUNKS_PATH) -> Path:
        """
        Persist validated RAG chunks as JSON.

        Args:
            chunks: Valid retrieval chunks.
            path: Destination JSON path.

        Returns:
            Path to the persisted chunks file.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps([chunk.model_dump() for chunk in chunks], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return target

    def persist_golden(self, golden_df: pd.DataFrame, path: str | Path = GOLDEN_PATH) -> Path:
        """
        Persist a frozen golden dataset for reproducible RAG evaluation.

        Args:
            golden_df: Held-out product dataframe.
            path: Destination JSON path.

        Returns:
            Path to the persisted golden dataset.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        records = []
        for row in golden_df.to_dict(orient="records"):
            records.append(
                {
                    "product_id": int(row["product_id"]),
                    "title": row["title"],
                    "category": row["category"],
                    "question": row["title"],
                    "reference_answer": float(row["price"]),
                }
            )
        target.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        return target

    def build_vector_index(self, chunks: list[Result], vectorstore: ChromaVectorStore) -> int:
        """
        Create embeddings and persist them into the Chroma collection.

        Args:
            chunks: Valid chunks to embed.
            vectorstore: Target Chroma vector store.

        Returns:
            Number of embedded and stored chunks.
        """
        vectorstore.reset()
        total = 0
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start : start + BATCH_SIZE]
            embeddings = self.embedder.embed_documents([chunk.page_content for chunk in batch])
            vectorstore.add(batch, embeddings, batch_size=BATCH_SIZE)
            total += len(batch)
        return total

    def run(self, dataset_name: str, golden_size: int = 200, vectorstore_path: str | Path | None = None) -> dict[str, Any]:
        """
        Run the complete explicit RAG ingestion workflow.

        Args:
            dataset_name: Hugging Face dataset containing train/validation/test splits.
            golden_size: Number of held-out golden evaluation products.
            vectorstore_path: Optional Chroma artifact path override.

        Returns:
            Run statistics including product, chunk, vector, and golden counts.
        """
        documents = self.loader.load_training_documents(dataset_name)
        kb_df, golden_df = self.create_golden_dataset(documents, golden_size)
        chunks = self.build_valid_chunks(kb_df)
        self.persist_chunks(chunks)
        self.persist_golden(golden_df)
        vectorstore = ChromaVectorStore(vectorstore_path or RAG_VECTORSTORE_PATH)
        vector_count = self.build_vector_index(chunks, vectorstore)
        bm25 = BM25Store()
        bm25.build(chunks)
        return {
            "products": len(documents),
            "knowledge_base_products": len(kb_df),
            "golden_products": len(golden_df),
            "chunks": len(chunks),
            "vectors": vector_count,
            "bm25_documents": len(chunks),
        }
