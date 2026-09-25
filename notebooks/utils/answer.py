import re
import pickle
from pathlib import Path
from typing import Literal
from openai import OpenAI
from chromadb import PersistentClient
from pydantic import BaseModel, Field
from litellm import completion
from tenacity import retry, wait_exponential, stop_after_attempt
from rank_bm25 import BM25Okapi
from pricer.items import Item


_NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent  
_DATA_PIPELINE_DIR = _NOTEBOOKS_DIR / "01_data_pipeline"

DB_NAME = str(_DATA_PIPELINE_DIR / "preprocessed_db")
COLLECTION_NAME = "docs"
EMBEDDING_MODEL = "text-embedding-3-large"
GENERATOR_MODEL = "gpt-4.1-mini"
RETRIEVAL_K = 10
RETRIEVAL_S = 10

assert Path(DB_NAME, "chroma.sqlite3").exists(), (
    f"No vector DB found at {DB_NAME} - run 05_RAG_ingestion.ipynb first."
)

wait = wait_exponential(multiplier=1, min=2, max=30)

openai = OpenAI()
chroma_client = PersistentClient(path=DB_NAME)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

BM25_PATH = Path(DB_NAME) / "bm25_index.pkl"
if BM25_PATH.exists():
    with open(BM25_PATH, "rb") as f:
        bm25_payload = pickle.load(f)
    bm25_index = BM25Okapi(bm25_payload["corpus_tokens"])
else:
    bm25_index = None
    bm25_payload = None
    print(f"WARNING: {BM25_PATH} not found - run 05_RAG_ingestion.ipynb first. "
          f"Falling back to dense-only retrieval.")



class Result(BaseModel):
    page_content: str
    metadata: dict


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


class PriceEstimate(BaseModel):
    """Structured result of a price estimation - safe for an agent to branch on."""

    estimated_price: float | None = Field(
        default=None,
        description="Estimated price in USD. Must be null (not 0.0) if no reasonable estimate is possible.",
    )
    confidence: Literal["high", "medium", "low", "none"] = Field(
        description="'none' means the retrieved context could not support any estimate "
                    "(e.g. wrong category / no comparable products) - treat this as "
                    "'need another source', not as a $0 product."
    )
    matched_category: bool = Field(
        description="Whether the retrieved context was from the same product category as the target"
    )
    comparable_products_used: int = Field(
        description="Number of retrieved products actually used as comparables for the estimate"
    )
    reasoning: str = Field(description="1-2 sentence explanation of how the estimate was derived")


STRUCTURED_SYSTEM_PROMPT = """
You are a pricing assistant for an e-commerce platform.

Estimate the price of the target product using the retrieved products and their
prices as reference examples. Use products that are similar even if not an exact
match (same category, similar brand tier/features/size are enough) and reason
about how the target differs to adjust your estimate.

Set confidence to:
- "high": very similar comparables found, price estimate should be reliable
- "medium": same-category comparables found but with notable differences
- "low": only loosely related comparables, estimate is a rough guess
- "none": retrieved context is a different category or otherwise unusable -
  in this case estimated_price MUST be null, not 0.0

Knowledge base context:
{context}
"""


def get_query(question):
    if not question:
        return "No question found."
    if isinstance(question, Item):
        question_text = question.title
    elif isinstance(question, dict):
        question_text = question.get("title", "")
    else:
        question_text = str(question).strip()
    if not question_text:
        return "No question found."
    return question_text


@retry(wait=wait, stop=stop_after_attempt(3), reraise=True)
def rewrite_query(question):
    message = f"""
You are a pricing assistant. You will search a knowledge base of product descriptions and prices.
You have a user's question about a product, and you need to formulate a short, focused query to find the most relevant product descriptions.

And this is the user's current question:
{question}

Respond only with a single, refined query that you will use to search the knowledge base.
It should be a VERY short, specific description of the product, including key features, brand, and category.
Do not mention pricing or dollar amounts - just the product details.
IMPORTANT: Respond ONLY with the knowledgebase query, nothing else.
"""
    response = completion(model=GENERATOR_MODEL, messages=[{"role": "system", "content": message}])
    return response.choices[0].message.content



def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@retry(wait=wait, stop=stop_after_attempt(3), reraise=True)
def fetch_context_unranked(question):
    query_vec = openai.embeddings.create(model=EMBEDDING_MODEL, input=[question]).data[0].embedding
    results = collection.query(query_embeddings=[query_vec], n_results=RETRIEVAL_K)
    chunks = []
    for result in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(Result(page_content=result[0], metadata=result[1]))
    return chunks


def reciprocal_rank_fusion(rank_lists: list[list[str]], k_rrf: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for rank_list in rank_lists:
        for rank, doc_id in enumerate(rank_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def bm25_search(question: str, k: int = RETRIEVAL_K):
    if bm25_index is None:
        return [], {}
    tokens = tokenize(question)
    scores = bm25_index.get_scores(tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    ids = [bm25_payload["ids"][i] for i in top_idx]
    lookup = {
        bm25_payload["ids"][i]: (bm25_payload["documents"][i], bm25_payload["metadatas"][i])
        for i in top_idx
    }
    return ids, lookup


@retry(wait=wait, stop=stop_after_attempt(3), reraise=True)
def fetch_context_hybrid(question: str, k: int = RETRIEVAL_K) -> list[Result]:
    query_vec = openai.embeddings.create(model=EMBEDDING_MODEL, input=[question]).data[0].embedding
    dense_results = collection.query(query_embeddings=[query_vec], n_results=k)
    dense_ids = dense_results["ids"][0]
    dense_lookup = {
        doc_id: (doc, meta)
        for doc_id, doc, meta in zip(dense_ids, dense_results["documents"][0], dense_results["metadatas"][0])
    }
    bm25_ids, bm25_lookup = bm25_search(question, k)
    fused_ids = reciprocal_rank_fusion([dense_ids, bm25_ids])[:k]
    chunks = []
    for doc_id in fused_ids:
        doc, meta = dense_lookup.get(doc_id) or bm25_lookup.get(doc_id)
        chunks.append(Result(page_content=doc, metadata=meta))
    return chunks


def merge_chunks(chunks1, chunks2):
    merged = chunks1[:]
    existing = [chunk.page_content for chunk in chunks1]
    for chunk in chunks2:
        if chunk.page_content not in existing:
            merged.append(chunk)
    return merged


@retry(wait=wait, stop=stop_after_attempt(3), reraise=True)
def rerank(question, chunks):
    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""
    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the chunks:\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = completion(model=GENERATOR_MODEL, messages=messages, response_format=RankOrder)
    reply = response.choices[0].message.content
    order = RankOrder.model_validate_json(reply).order
    valid_ids = set(range(1, len(chunks) + 1))
    seen = set()
    clean_order = []
    for i in order:
        if i in valid_ids and i not in seen:
            clean_order.append(i)
            seen.add(i)
    for i in valid_ids - seen:
        clean_order.append(i)
    return [chunks[i - 1] for i in clean_order]


def fetch_context(original_question, rewritten_question):
    chunks1 = fetch_context_hybrid(original_question)
    chunks2 = fetch_context_hybrid(rewritten_question)
    chunks = merge_chunks(chunks1, chunks2)
    reranked = rerank(original_question, chunks)
    return reranked[:RETRIEVAL_S]



def get_structured_answer(question: str, chunks: list[Result]) -> PriceEstimate:
    context = "\n\n".join(
        f"""Product:
    Title: {chunk.metadata['title']}
    Category: {chunk.metadata['type']}
    Price: ${chunk.metadata['price']}
    Description:{chunk.page_content}
    Source: {chunk.metadata['source']}"""
        for chunk in chunks
    )
    messages = [
        {"role": "system", "content": STRUCTURED_SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": question},
    ]
    response = completion(model=GENERATOR_MODEL, messages=messages, response_format=PriceEstimate)
    return PriceEstimate.model_validate_json(response.choices[0].message.content)


def answer_question(question: str) -> tuple[PriceEstimate, list[Result]]:
    question_text = get_query(question)
    query = rewrite_query(question_text)
    chunks = fetch_context(question_text, query)
    estimate = get_structured_answer(question_text, chunks)
    return estimate.estimated_price, chunks



def make_rag_messages(question, chunks):
    context = "\n\n".join(
        f"""Product:
    Title: {chunk.metadata['title']}
    Category: {chunk.metadata['type']}
    Price: ${chunk.metadata['price']}
    Description:{chunk.page_content}
    Source: {chunk.metadata['source']}"""
        for chunk in chunks
    )
    system_prompt = STRUCTURED_SYSTEM_PROMPT.format(context=context)
    return [{"role": "system", "content": system_prompt}] + [{"role": "user", "content": question}]


def get_answer(messages):
    response = completion(model=GENERATOR_MODEL, messages=messages)
    return response.choices[0].message.content


print(f"RAG pipeline loaded.")
