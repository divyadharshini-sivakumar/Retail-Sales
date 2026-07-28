"""
Extract node – load a CSV file into a pandas DataFrame.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.state import ETLState

logger = logging.getLogger(__name__)


def extract_node(state: ETLState) -> dict[str, Any]:
    """
    Load the uploaded CSV into state["raw_df"].

    Expected columns (flexible):
        date, product, quantity, price, (optional) customer_id, store_id, ...

    Returns partial state update.
    """
    errors: list[str] = list(state.get("errors") or [])
    file_path = state.get("uploaded_file_path")

    if not file_path:
        errors.append("No file path provided to Extract node.")
        return {
            "raw_df": None,
            "errors": errors,
            "success": False,
        }

    path = Path(file_path)
    if not path.exists():
        errors.append(f"File not found: {file_path}")
        return {
            "raw_df": None,
            "errors": errors,
            "success": False,
        }

    try:
        # Try common encodings; fall back to latin-1
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")

        if df.empty:
            errors.append("Uploaded CSV is empty.")
            return {
                "raw_df": df,
                "errors": errors,
                "success": False,
            }

        logger.info("Extracted %d rows × %d columns from %s", *df.shape, path.name)
        return {
            "raw_df": df,
            "errors": errors,
            "success": True,
        }

    except Exception as exc:  # noqa: BLE001 – surface any read error
        msg = f"Failed to read CSV: {exc}"
        logger.exception(msg)
        errors.append(msg)
        return {
            "raw_df": None,
            "errors": errors,
            "success": False,
        }
