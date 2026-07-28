"""
RAG module for the Retail Sales.

Builds a knowledge base from the cleaned DataFrame after ETL,
stores embeddings in a persistent ChromaDB, and answers questions
using retrieval + an OpenRouter LLM.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple
import tempfile

import pandas as pd
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from prompts import RAG_PROMPT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

CHROMA_DIR = Path(tempfile.gettempdir()) / "retail_sales_chroma"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
COLLECTION_NAME = "retail_sales"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 6

# Simple product → category mapping (extend as needed)
PRODUCT_CATEGORY = {
    "iphone 15": "Smartphones",
    "macbook pro": "Laptops",
    "airpods pro": "Audio",
    "airpods": "Audio",
    "ipad air": "Tablets",
    "ipad": "Tablets",
}


def _derive_category(product: str) -> str:
    key = str(product).strip().lower()
    for k, cat in PRODUCT_CATEGORY.items():
        if k in key:
            return cat
    return "Other"


def _row_to_document(row: pd.Series, idx: int) -> Document:
    """
    Convert one cleaned sales row into a LangChain Document.

    Uses the actual columns produced by the existing Transform node:
        date, product, quantity, price, customer_id, store_id, total_sales
    """
    order_id = f"ORD-{idx:04d}"
    product = str(row.get("product", "UNKNOWN"))
    category = _derive_category(product)
    quantity = row.get("quantity", "")
    unit_price = row.get("price", "")
    total_sales = row.get("total_sales", "")
    date_val = row.get("date", "")
    if hasattr(date_val, "strftime"):
        date_str = date_val.strftime("%Y-%m-%d")
    else:
        date_str = str(date_val)[:10]
    region = str(row.get("store_id", "UNKNOWN"))
    customer = str(row.get("customer_id", "UNKNOWN"))

    page_content = (
        f"Order ID: {order_id}. "
        f"Date: {date_str}. "
        f"Product: {product}. "
        f"Category: {category}. "
        f"Quantity: {quantity}. "
        f"Unit Price: {unit_price}. "
        f"Total Sales: {total_sales}. "
        f"Region (Store): {region}. "
        f"Customer: {customer}."
    )

    metadata = {
        "order_id": order_id,
        "product": product,
        "category": category,
        "quantity": int(quantity) if pd.notna(quantity) else 0,
        "unit_price": float(unit_price) if pd.notna(unit_price) else 0.0,
        "total_sales": float(total_sales) if pd.notna(total_sales) else 0.0,
        "date": date_str,
        "region": region,
        "customer": customer,
        "row_index": idx,
    }
    return Document(page_content=page_content, metadata=metadata)


def build_documents(cleaned_df: pd.DataFrame) -> List[Document]:
    """Turn every row of the cleaned DataFrame into a Document."""
    if cleaned_df is None or cleaned_df.empty:
        return []
    docs: List[Document] = []
    for idx, row in cleaned_df.reset_index(drop=True).iterrows():
        docs.append(_row_to_document(row, int(idx)))
    logger.info("Created %d documents from cleaned sales data", len(docs))
    return docs


def get_embeddings() -> HuggingFaceEmbeddings:
    """Local sentence-transformer embeddings (no API key required)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_or_update_vectorstore(
    cleaned_df: pd.DataFrame,
    persist_directory: Optional[Path] = None,
) -> Chroma:
    """
    Build (or rebuild) a persistent Chroma vector store from the cleaned data.
    """
    persist_directory = persist_directory or CHROMA_DIR
    persist_directory.mkdir(parents=True, exist_ok=True)

    documents = build_documents(cleaned_df)
    if not documents:
        raise ValueError("No documents to index – cleaned DataFrame is empty.")

    embeddings = get_embeddings()

    # Always rebuild for a fresh upload (simple & deterministic)
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_directory),
    )
    logger.info(
        "Chroma vector store ready at %s (%d docs)",
        persist_directory,
        len(documents),
    )
    return vectorstore


def load_vectorstore(persist_directory: Optional[Path] = None) -> Optional[Chroma]:
    """Load an existing persistent Chroma store (returns None if missing)."""
    persist_directory = persist_directory or CHROMA_DIR
    if not persist_directory.exists():
        return None
    embeddings = get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )


def get_retriever(vectorstore: Chroma, k: int = TOP_K):
    return vectorstore.as_retriever(search_kwargs={"k": k})


def get_llm():
    """
    OpenRouter chat model (OpenAI-compatible API).
    Requires OPENROUTER_API_KEY in the environment (never hard-coded).

    Optional env vars:
      OPENROUTER_MODEL    – default: meta-llama/llama-3.1-8b-instruct
      OPENROUTER_BASE_URL – default: https://openrouter.ai/api/v1
    """
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file or Streamlit secrets."
        )

    model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    return ChatOpenAI(
        model=model,
        temperature=0.1,
        api_key=api_key,
        base_url=base_url,
    )


def format_docs(docs: List[Document]) -> str:
    if not docs:
        return "(no relevant records retrieved)"
    return "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))


def answer_question(
    question: str,
    vectorstore: Chroma,
    return_sources: bool = True,
) -> Tuple[str, List[Document]]:
    """
    Retrieve relevant documents and generate an answer with the OpenRouter LLM.
    The LLM is instructed to answer ONLY from the retrieved context.
    """
    retriever = get_retriever(vectorstore)
    llm = get_llm()

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(question)
    sources: List[Document] = []
    if return_sources:
        sources = retriever.invoke(question)
    return answer.strip(), sources


def clear_vectorstore(persist_directory: Optional[Path] = None) -> None:
    """Optional helper to wipe the local Chroma DB (e.g. between uploads)."""
    import shutil

    persist_directory = persist_directory or CHROMA_DIR
    if persist_directory.exists():
        shutil.rmtree(persist_directory)
        logger.info("Cleared Chroma directory %s", persist_directory)
