"""
RAG module for Retail Sales.

Builds a knowledge base from the cleaned DataFrame after ETL,
creates local Hugging Face embeddings, stores them temporarily in
an in-memory ChromaDB collection, and answers questions using
retrieval with an OpenRouter LLM.

No ChromaDB files are written to disk. This avoids read-only database
errors on Streamlit Community Cloud.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple
from uuid import uuid4

import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings

from prompts import RAG_PROMPT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLLECTION_NAME_PREFIX = "retail_sales"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 6

# Simple product-to-category mapping.
PRODUCT_CATEGORY = {
    "iphone 15": "Smartphones",
    "iphone": "Smartphones",
    "macbook pro": "Laptops",
    "macbook": "Laptops",
    "laptop": "Laptops",
    "airpods pro": "Audio",
    "airpods": "Audio",
    "headphones": "Audio",
    "ipad air": "Tablets",
    "ipad": "Tablets",
    "tablet": "Tablets",
}


# ---------------------------------------------------------------------------
# Data conversion helpers
# ---------------------------------------------------------------------------

def _derive_category(product: str) -> str:
    """
    Derive a product category from the product name.

    Args:
        product: Product name from the cleaned dataset.

    Returns:
        A matching category or "Other".
    """
    normalized_product = str(product).strip().lower()

    for keyword, category in PRODUCT_CATEGORY.items():
        if keyword in normalized_product:
            return category

    return "Other"


def _safe_int(value: object, default: int = 0) -> int:
    """Convert a value safely to an integer."""
    if pd.isna(value):
        return default

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value: object, default: float = 0.0) -> float:
    """Convert a value safely to a floating-point number."""
    if pd.isna(value):
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _row_to_document(row: pd.Series, idx: int) -> Document:
    """
    Convert one cleaned sales row into a LangChain Document.

    Expected cleaned columns:
    - date
    - product
    - quantity
    - price
    - customer_id
    - store_id
    - total_sales
    """
    order_id = f"ORD-{idx + 1:04d}"

    product = str(row.get("product", "UNKNOWN")).strip()
    category = _derive_category(product)

    quantity = _safe_int(row.get("quantity"))
    unit_price = _safe_float(row.get("price"))
    total_sales = _safe_float(row.get("total_sales"))

    date_value = row.get("date", "")

    if hasattr(date_value, "strftime"):
        date_string = date_value.strftime("%Y-%m-%d")
    else:
        date_string = str(date_value).strip()[:10]

    if not date_string:
        date_string = "UNKNOWN"

    region = str(row.get("store_id", "UNKNOWN")).strip()
    customer = str(row.get("customer_id", "UNKNOWN")).strip()

    page_content = (
        f"Order ID: {order_id}. "
        f"Date: {date_string}. "
        f"Product: {product}. "
        f"Category: {category}. "
        f"Quantity: {quantity}. "
        f"Unit Price: {unit_price:.2f}. "
        f"Total Sales: {total_sales:.2f}. "
        f"Region or Store: {region}. "
        f"Customer: {customer}."
    )

    metadata = {
        "order_id": order_id,
        "product": product,
        "category": category,
        "quantity": quantity,
        "unit_price": unit_price,
        "total_sales": total_sales,
        "date": date_string,
        "region": region,
        "customer": customer,
        "row_index": idx,
    }

    return Document(
        page_content=page_content,
        metadata=metadata,
    )


def build_documents(cleaned_df: pd.DataFrame) -> List[Document]:
    """
    Convert every row in the cleaned DataFrame into a Document.

    Args:
        cleaned_df: Cleaned retail-sales DataFrame.

    Returns:
        List of LangChain Documents.

    Raises:
        ValueError: If the DataFrame is missing or empty.
    """
    if cleaned_df is None:
        raise ValueError("The cleaned DataFrame is missing.")

    if cleaned_df.empty:
        raise ValueError("The cleaned DataFrame is empty.")

    documents: List[Document] = []

    reset_df = cleaned_df.reset_index(drop=True)

    for idx, row in reset_df.iterrows():
        documents.append(
            _row_to_document(
                row=row,
                idx=int(idx),
            )
        )

    if not documents:
        raise ValueError(
            "No documents could be created from the cleaned dataset."
        )

    logger.info(
        "Created %d documents from cleaned retail-sales data.",
        len(documents),
    )

    return documents


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return local Hugging Face sentence-transformer embeddings.

    No external API key is required for embeddings.
    The model is downloaded when it is used for the first time.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )


# ---------------------------------------------------------------------------
# In-memory ChromaDB
# ---------------------------------------------------------------------------

def build_or_update_vectorstore(
    cleaned_df: pd.DataFrame,
    persist_directory: Optional[object] = None,
) -> Chroma:
    """
    Build a fresh in-memory Chroma vector store.

    The persist_directory parameter is accepted only for compatibility with
    older app code. It is intentionally ignored because this implementation
    does not write ChromaDB data to disk.

    Args:
        cleaned_df: Cleaned retail-sales DataFrame.
        persist_directory: Ignored compatibility parameter.

    Returns:
        An in-memory Chroma vector store.

    Raises:
        ValueError: If no documents can be indexed.
        RuntimeError: If ChromaDB creation fails.
    """
    del persist_directory

    documents = build_documents(cleaned_df)

    if not documents:
        raise ValueError(
            "No documents are available for RAG indexing."
        )

    embeddings = get_embeddings()

    # A unique collection name prevents stale data from an earlier upload
    # being reused during the same Streamlit server process.
    collection_name = (
        f"{COLLECTION_NAME_PREFIX}_{uuid4().hex[:12]}"
    )

    try:
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name,
        )
    except Exception as exc:
        logger.exception(
            "Failed to create the in-memory ChromaDB collection."
        )

        raise RuntimeError(
            "Unable to build the RAG knowledge base in memory. "
            f"Original error: {exc}"
        ) from exc

    try:
        indexed_count = int(vectorstore._collection.count())
    except Exception:
        indexed_count = len(documents)

    if indexed_count <= 0:
        raise RuntimeError(
            "The ChromaDB collection was created, but no documents "
            "were indexed."
        )

    logger.info(
        "In-memory ChromaDB collection '%s' created with %d documents.",
        collection_name,
        indexed_count,
    )

    return vectorstore


def load_vectorstore(
    persist_directory: Optional[object] = None,
) -> Optional[Chroma]:
    """
    Return None because this project uses an in-memory vector store.

    The knowledge base must be rebuilt from the cleaned uploaded dataset
    whenever the application session restarts.
    """
    del persist_directory

    logger.info(
        "Persistent loading is disabled because ChromaDB runs in memory."
    )

    return None


def clear_vectorstore(
    persist_directory: Optional[object] = None,
) -> None:
    """
    Compatibility helper for the previous persistent implementation.

    No folder needs to be removed because the vector store is held only
    in memory. Removing the vector-store object from Streamlit session
    state is enough to release it.
    """
    del persist_directory

    logger.info(
        "No persistent ChromaDB directory exists. "
        "The in-memory collection will be released with its session."
    )


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def get_retriever(
    vectorstore: Chroma,
    k: int = TOP_K,
):
    """
    Create a similarity-search retriever.

    Args:
        vectorstore: Initialized Chroma vector store.
        k: Number of relevant documents to retrieve.

    Returns:
        A LangChain retriever.
    """
    if vectorstore is None:
        raise ValueError(
            "The vector store is missing. Run the ETL pipeline and "
            "build the RAG knowledge base first."
        )

    if k <= 0:
        raise ValueError(
            "The number of retrieved documents must be greater than zero."
        )

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
        },
    )


def format_docs(docs: List[Document]) -> str:
    """
    Format retrieved documents as context for the LLM.
    """
    if not docs:
        return "(no relevant sales records retrieved)"

    formatted_documents: List[str] = []

    for index, document in enumerate(docs, start=1):
        formatted_documents.append(
            f"[Record {index}]\n{document.page_content}"
        )

    return "\n\n".join(formatted_documents)


# ---------------------------------------------------------------------------
# OpenRouter LLM
# ---------------------------------------------------------------------------

def get_llm():
    """
    Create the OpenRouter chat model.

    Required environment variable:
        OPENROUTER_API_KEY

    Optional environment variables:
        OPENROUTER_MODEL
        OPENROUTER_BASE_URL
    """
    from langchain_openai import ChatOpenAI

    api_key = os.getenv(
        "OPENROUTER_API_KEY",
        "",
    ).strip()

    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not configured. Add it to the "
            "local .env file or Streamlit Community Cloud secrets."
        )

    if api_key.startswith("sk-or-v1-your"):
        raise EnvironmentError(
            "OPENROUTER_API_KEY still contains the placeholder value."
        )

    model = os.getenv(
        "OPENROUTER_MODEL",
        "meta-llama/llama-3.1-8b-instruct",
    ).strip()

    base_url = os.getenv(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1",
    ).strip()

    if not model:
        raise EnvironmentError(
            "OPENROUTER_MODEL cannot be empty."
        )

    return ChatOpenAI(
        model=model,
        temperature=0.1,
        api_key=api_key,
        base_url=base_url,
        max_retries=2,
    )


# ---------------------------------------------------------------------------
# Question answering
# ---------------------------------------------------------------------------

def answer_question(
    question: str,
    vectorstore: Chroma,
    return_sources: bool = True,
) -> Tuple[str, List[Document]]:
    """
    Retrieve relevant sales records and generate an answer.

    The LLM receives only the retrieved context through RAG_PROMPT.

    Args:
        question: User's natural-language question.
        vectorstore: In-memory Chroma vector store.
        return_sources: Whether to return retrieved Documents.

    Returns:
        Tuple containing:
        - Generated answer
        - Retrieved source Documents
    """
    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError(
            "Please enter a question before running the RAG assistant."
        )

    if vectorstore is None:
        raise ValueError(
            "The RAG knowledge base is not ready. Run the ETL pipeline first."
        )

    retriever = get_retriever(vectorstore)
    llm = get_llm()

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    try:
        answer = chain.invoke(cleaned_question)
    except Exception as exc:
        logger.exception(
            "RAG answer generation failed."
        )

        raise RuntimeError(
            "The Retail Sales Assistant could not generate an answer. "
            f"Original error: {exc}"
        ) from exc

    retrieved_sources: List[Document] = []

    if return_sources:
        try:
            retrieved_sources = retriever.invoke(cleaned_question)
        except Exception as exc:
            logger.warning(
                "The answer was generated, but source retrieval failed: %s",
                exc,
            )

    cleaned_answer = str(answer).strip()

    if not cleaned_answer:
        cleaned_answer = (
            "The uploaded sales dataset does not contain enough "
            "information to answer this question."
        )

    return cleaned_answer, retrieved_sources
