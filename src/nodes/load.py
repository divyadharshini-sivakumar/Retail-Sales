"""
Load node – persist the cleaned DataFrame to a temporary CSV for download.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from src.state import ETLState

logger = logging.getLogger(__name__)


def load_node(state: ETLState) -> dict[str, Any]:
    """
    Write cleaned_df to a temporary CSV and store the path in stats.
    """
    errors: list[str] = list(state.get("errors") or [])
    cleaned_df = state.get("cleaned_df")
    stats: dict[str, Any] = dict(state.get("stats") or {})

    if cleaned_df is None or cleaned_df.empty:
        errors.append("No cleaned data available for Load node.")
        return {
            "stats": stats,
            "errors": errors,
            "success": False,
        }

    try:
        # Create a named temporary file that survives until the Streamlit session ends
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            prefix="cleaned_sales_",
            delete=False,
            encoding="utf-8",
        )
        tmp_path = Path(tmp.name)
        cleaned_df.to_csv(tmp_path, index=False)
        tmp.close()

        stats["output_path"] = str(tmp_path)
        stats["output_filename"] = tmp_path.name

        logger.info("Cleaned data written to %s (%d rows)", tmp_path, len(cleaned_df))
        return {
            "stats": stats,
            "errors": errors,
            "success": True,
        }

    except Exception as exc:  # noqa: BLE001
        msg = f"Failed to write cleaned CSV: {exc}"
        logger.exception(msg)
        errors.append(msg)
        return {
            "stats": stats,
            "errors": errors,
            "success": False,
        }
