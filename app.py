"""
Streamlit frontend for the Retail Sales (LangGraph) + RAG assistant.

Run locally:
    streamlit run app.py
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment & LangSmith (never hard-code keys)
# ---------------------------------------------------------------------------
load_dotenv()  # loads .env if present (local development)

# LangSmith tracing is controlled purely by environment variables.
# Set these in .env (local) or in Streamlit Community Cloud secrets.
#   LANGCHAIN_TRACING_V2=true
#   LANGCHAIN_API_KEY=lsv2_pt_...
#   LANGCHAIN_PROJECT=retail-sales
# Never print or display the actual key.

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Retail Sales",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 Retail Sales")
st.markdown(
    """
    **Linear LangGraph ETL + RAG**  
    Upload a messy retail sales CSV → Extract → Transform → Load → download an analytics-ready dataset,  
    then ask questions with the Retail Sales Assistant.
    """
)

# ---------------------------------------------------------------------------
# Sidebar – instructions & env status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("How to use")
    st.markdown(
        """
        1. Upload a CSV (or use the sample).
        2. Preview the raw data.
        3. Click **Run ETL Pipeline**.
        4. Inspect cleaning statistics & download cleaned CSV.
        5. Ask questions in the **Retail Sales Assistant** (RAG).
        """
    )
    st.divider()
    st.subheader("LangSmith status")
    tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    project = os.getenv("LANGCHAIN_PROJECT", "(not set)")
    st.write(f"Tracing enabled: **{tracing}**")
    st.write(f"Project: `{project}`")
    if not tracing:
        st.info(
            "To enable LangSmith, create a `.env` file from `.env.example` "
            "and set `LANGCHAIN_TRACING_V2=true` + your API key."
        )
    st.divider()
    st.subheader("OpenRouter (RAG) status")
    openrouter_set = bool(os.getenv("OPENROUTER_API_KEY"))
    st.write(f"OPENROUTER_API_KEY set: **{openrouter_set}**")
    model_name = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")
    st.caption(f"Model: `{model_name}`")
    if not openrouter_set:
        st.info(
            "Add `OPENROUTER_API_KEY` to `.env` (see `.env.example`) to enable the "
            "Retail Sales Assistant."
        )

# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload messy sales CSV",
    type=["csv"],
    help="Expected columns (flexible names): date, product, quantity, price",
)

# Optional: load the bundled sample
use_sample = st.checkbox("Use built-in sample data instead", value=False)

raw_df: pd.DataFrame | None = None
temp_path: str | None = None

if use_sample:
    sample_path = Path(__file__).parent / "sample_data" / "messy_sales.csv"
    if sample_path.exists():
        raw_df = pd.read_csv(sample_path)
        # Write a temporary copy so the Extract node can read from disk
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="sample_")
        raw_df.to_csv(tmp.name, index=False)
        temp_path = tmp.name
        tmp.close()
        st.success(f"Loaded sample data ({len(raw_df)} rows)")
    else:
        st.error("Sample file not found. Please upload a CSV.")
elif uploaded_file is not None:
    # Persist upload to a temp file for the Extract node
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="upload_")
    tmp.write(uploaded_file.getvalue())
    tmp.close()
    temp_path = tmp.name
    raw_df = pd.read_csv(temp_path)
    st.success(f"Uploaded file loaded ({len(raw_df)} rows)")

# ---------------------------------------------------------------------------
# Preview raw data
# ---------------------------------------------------------------------------
if raw_df is not None:
    st.subheader("Raw data preview")
    st.dataframe(raw_df.head(20), use_container_width=True)
    st.caption(f"Showing first 20 of {len(raw_df)} rows · {list(raw_df.columns)}")

# ---------------------------------------------------------------------------
# Session state for RAG (survives Streamlit reruns)
# ---------------------------------------------------------------------------
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False
if "etl_stats" not in st.session_state:
    st.session_state.etl_stats = {}

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
run_button = st.button(
    "🚀 Run ETL Pipeline",
    type="primary",
    disabled=raw_df is None,
)

if run_button and temp_path is not None:
    with st.spinner("Running Extract → Transform → Load …"):
        try:
            from src.graph import etl_graph
            from src.state import ETLState

            initial_state: ETLState = {
                "uploaded_file_path": temp_path,
                "raw_df": None,
                "cleaned_df": None,
                "stats": {},
                "errors": [],
                "success": False,
            }

            # Invoke the compiled graph (LangSmith traces automatically when env vars are set)
            final_state = etl_graph.invoke(initial_state)

            if final_state.get("errors"):
                for err in final_state["errors"]:
                    st.error(err)

            if final_state.get("success") and final_state.get("cleaned_df") is not None:
                cleaned = final_state["cleaned_df"]
                stats = final_state.get("stats", {})

                # Persist for RAG section across reruns
                st.session_state.cleaned_df = cleaned
                st.session_state.etl_stats = stats

                st.subheader("✅ Cleaning statistics")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Original rows", stats.get("original_rows", "–"))
                col2.metric("Duplicates removed", stats.get("duplicates_removed", "–"))
                col3.metric("Rows dropped (validation)", stats.get("rows_dropped_validation", "–"))
                col4.metric("Final rows", stats.get("final_rows", "–"))

                col5, col6, col7 = st.columns(3)
                col5.metric("Total revenue", f"${stats.get('total_revenue', 0):,.2f}")
                col6.metric("Unique products", stats.get("unique_products", "–"))
                date_range = stats.get("date_range", {})
                col7.metric(
                    "Date range",
                    f"{date_range.get('min', '–')} → {date_range.get('max', '–')}",
                )

                st.subheader("Cleaned data preview")
                st.dataframe(cleaned.head(30), use_container_width=True)

                # Download button
                output_path = stats.get("output_path")
                if output_path and Path(output_path).exists():
                    with open(output_path, "rb") as f:
                        st.download_button(
                            label="📥 Download cleaned CSV",
                            data=f,
                            file_name="cleaned_sales.csv",
                            mime="text/csv",
                        )
                else:
                    csv_bytes = cleaned.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download cleaned CSV",
                        data=csv_bytes,
                        file_name="cleaned_sales.csv",
                        mime="text/csv",
                    )

                # ----------------------------------------------------------
                # Build RAG knowledge base from cleaned data
                # ----------------------------------------------------------
                with st.spinner("Building RAG knowledge base (Chroma + embeddings) …"):
                    try:
                        from rag import build_or_update_vectorstore

                        # Build a fresh in-memory vector store and keep the
                        # Python object in Streamlit session state.
                        st.session_state.vectorstore = None
                        st.session_state.vectorstore_ready = False

                        vectorstore = build_or_update_vectorstore(cleaned)
                        st.session_state.vectorstore = vectorstore
                        st.session_state.vectorstore_ready = True

                        st.success(
                            f"Knowledge base ready – {len(cleaned)} sales records indexed."
                        )
                    except Exception as rag_exc:  # noqa: BLE001
                        st.session_state.vectorstore = None
                        st.session_state.vectorstore_ready = False
                        logger.exception("RAG indexing failed")
                        st.warning(
                            f"ETL succeeded but RAG indexing failed: {rag_exc}. "
                            "You can still download the cleaned CSV."
                        )
            else:
                st.warning(
                    "Pipeline finished but no cleaned data was produced. Check errors above."
                )

        except Exception as exc:  # noqa: BLE001
            logger.exception("Pipeline failed")
            st.error(f"Unexpected error while running the pipeline: {exc}")

# ---------------------------------------------------------------------------
# Retail Sales Assistant (RAG) – shown when knowledge base is ready
# ---------------------------------------------------------------------------
st.divider()
st.header("Retail Sales Assistant")
st.markdown(
    "Ask questions about the **cleaned** sales data. "
    "Answers are generated only from retrieved records (RAG)."
)

if not st.session_state.vectorstore_ready:
    st.info(
        "Run the ETL pipeline first. After cleaning finishes, the knowledge base "
        "is built automatically and this assistant becomes available."
    )
else:
    question = st.text_input(
        "Your question",
        placeholder="e.g. Which product generated the highest sales?",
        key="rag_question",
    )
    col_ask, col_show = st.columns([1, 3])
    with col_ask:
        ask_clicked = st.button("Ask", type="primary")
    with col_show:
        show_sources = st.checkbox("Show retrieved documents", value=False)

    if ask_clicked and question.strip():
        with st.spinner("Retrieving relevant records and generating answer …"):
            try:
                from rag import answer_question

                # The in-memory Chroma object is stored in session state.
                vs = st.session_state.get("vectorstore")

                if vs is None:
                    st.session_state.vectorstore_ready = False
                    st.error("Vector store not found. Please re-run the ETL pipeline.")
                else:
                    answer, sources = answer_question(
                        question=question.strip(),
                        vectorstore=vs,
                        return_sources=show_sources,
                    )
                    st.subheader("Answer")
                    st.write(answer)

                    if show_sources and sources:
                        with st.expander("Retrieved documents", expanded=False):
                            for i, doc in enumerate(sources, 1):
                                st.markdown(f"**[{i}]** {doc.page_content}")
                                st.caption(f"Metadata: {doc.metadata}")
            except EnvironmentError as env_err:
                st.error(str(env_err))
            except Exception as qa_exc:  # noqa: BLE001
                logger.exception("RAG Q&A failed")
                st.error(f"Failed to answer: {qa_exc}")

    # Example questions for discoverability
    with st.expander("Example questions"):
        st.markdown(
            """
            - Which product generated the highest sales?
            - Show all laptop-related sales.
            - Which region had the highest revenue?
            - Summarize the cleaned sales dataset.
            - How many iPhone 15 units were sold?
            - List sales for customer C003.
            """
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Built · LangGraph ETL + RAG · Streamlit · Chroma · OpenRouter · LangSmith-ready · "
    "Made for learning – rotate any key that appears in chat before deployment."
)
