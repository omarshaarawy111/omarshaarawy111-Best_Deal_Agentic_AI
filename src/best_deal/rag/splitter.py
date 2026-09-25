"""
Product-level chunking for Best Deal RAG ingestion.

Description:
    The knowledge base uses product-level chunks rather than a generic
    character splitter because each product record is already a meaningful
    retrieval unit.

Responsibilities:
    - Convert product rows into normalized RAG documents.
    - Build one chunk per product.
    - Extract SKU/model-like tokens for sparse retrieval.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from best_deal.rag.schemas import Result

SKU_PATTERN = re.compile(r"\b[A-Z0-9][A-Z0-9\-]{4,}\b")


class ProductSplitter:
    """
    Build product-level RAG chunks from structured product records.

    Returns:
        A configured product splitter.
    """

    def build_documents(self, rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Convert product rows into ingestion documents.

        Args:
            rows: Product mappings containing title, category, summary, price, and id.

        Returns:
            Normalized RAG document mappings.
        """
        documents = []
        for row in rows:
            documents.append(
                {
                    "product_id": int(row["product_id"]),
                    "type": str(row["category"]),
                    "source": f"product_{row['product_id']}",
                    "title": str(row["title"]),
                    "text": f"Title: {row['title']}\n\nCategory: {row['category']}\n\nDescription: {row.get('summary') or ''}".strip(),
                    "price": float(row["price"]),
                }
            )
        return documents

    def create_chunks(self, documents: Iterable[dict[str, Any]]) -> list[Result]:
        """
        Create one product-level Result per source document.

        Args:
            documents: Normalized RAG document mappings.

        Returns:
            Product-level retrieval chunks.
        """
        chunks = []
        for document in documents:
            result = Result(
                page_content=document["title"] + "\n\n" + document["text"],
                metadata={
                    "product_id": document["product_id"],
                    "source": document["source"],
                    "type": document["type"],
                    "title": document["title"],
                    "price": document["price"],
                    "sku_codes": " ".join(self.extract_sku_codes(document["title"])),
                },
            )
            chunks.append(result)
        return chunks

    @staticmethod
    def extract_sku_codes(text: str) -> list[str]:
        """
        Extract likely model or SKU codes from product text.

        Args:
            text: Product title.

        Returns:
            Sorted unique model-like tokens.
        """
        return sorted(set(SKU_PATTERN.findall(text.upper())))
