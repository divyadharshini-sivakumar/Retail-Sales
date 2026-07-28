"""
Transform node – clean, validate and enrich the raw sales data.
"""
from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

from src.state import ETLState

logger = logging.getLogger(__name__)

# Canonical product name mapping (example business rules)
PRODUCT_CANONICAL = {
    "iphone 15": "iPhone 15",
    "iphone15": "iPhone 15",
    "iphone-15": "iPhone 15",
    "macbook pro": "MacBook Pro",
    "macbookpro": "MacBook Pro",
    "airpods pro": "AirPods Pro",
    "airpods": "AirPods",
    "ipad air": "iPad Air",
    "ipad": "iPad",
}


def _standardize_product(name: Any) -> str:
    """Lower-case, strip, map to canonical name when possible."""
    if pd.isna(name):
        return "UNKNOWN"
    cleaned = re.sub(r"\s+", " ", str(name).strip().lower())
    return PRODUCT_CANONICAL.get(cleaned, cleaned.title())


def _parse_date(val: Any) -> pd.Timestamp | pd.NaT:
    """Try several common date formats."""
    if pd.isna(val):
        return pd.NaT
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            return pd.to_datetime(val, format=fmt)
        except (ValueError, TypeError):
            continue
    # last resort
    try:
        return pd.to_datetime(val, infer_datetime_format=True)
    except Exception:  # noqa: BLE001
        return pd.NaT


def transform_node(state: ETLState) -> dict[str, Any]:
    """
    Clean the raw DataFrame:
      - drop exact duplicates
      - standardize column names (lower + snake)
      - handle missing values
      - standardize dates & product names
      - validate quantity (>0) and price (>=0)
      - compute total_sales = quantity * price
    """
    errors: list[str] = list(state.get("errors") or [])
    raw_df = state.get("raw_df")

    if raw_df is None or raw_df.empty:
        errors.append("No data available for Transform node.")
        return {
            "cleaned_df": None,
            "stats": {},
            "errors": errors,
            "success": False,
        }

    df = raw_df.copy()
    original_rows = len(df)

    # ------------------------------------------------------------------
    # 1. Normalize column names
    # ------------------------------------------------------------------
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.replace(r"\s+", "_", regex=True)
    )

    # Map common aliases
    column_aliases = {
        "qty": "quantity",
        "qty_sold": "quantity",
        "units": "quantity",
        "unit_price": "price",
        "sale_price": "price",
        "amount": "price",
        "product_name": "product",
        "item": "product",
        "item_name": "product",
        "transaction_date": "date",
        "sale_date": "date",
        "order_date": "date",
    }
    df = df.rename(columns={k: v for k, v in column_aliases.items() if k in df.columns})

    required = {"date", "product", "quantity", "price"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns after normalization: {sorted(missing_cols)}")
        return {
            "cleaned_df": None,
            "stats": {"original_rows": original_rows},
            "errors": errors,
            "success": False,
        }

    # ------------------------------------------------------------------
    # 2. Drop exact duplicates
    # ------------------------------------------------------------------
    before_dedup = len(df)
    df = df.drop_duplicates()
    duplicates_removed = before_dedup - len(df)

    # ------------------------------------------------------------------
    # 3. Standardize product names
    # ------------------------------------------------------------------
    df["product"] = df["product"].apply(_standardize_product)

    # ------------------------------------------------------------------
    # 4. Parse dates
    # ------------------------------------------------------------------
    df["date"] = df["date"].apply(_parse_date)
    invalid_dates = df["date"].isna().sum()

    # ------------------------------------------------------------------
    # 5. Coerce quantity & price to numeric
    # ------------------------------------------------------------------
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # ------------------------------------------------------------------
    # 6. Validation rules
    # ------------------------------------------------------------------
    # quantity must be positive integer-like
    invalid_qty = (df["quantity"].isna()) | (df["quantity"] <= 0)
    # price must be non-negative
    invalid_price = (df["price"].isna()) | (df["price"] < 0)

    rows_dropped_validation = int((invalid_qty | invalid_price | df["date"].isna()).sum())

    # Keep only valid rows
    mask_valid = ~(invalid_qty | invalid_price | df["date"].isna())
    df = df.loc[mask_valid].copy()

    # ------------------------------------------------------------------
    # 7. Fill remaining missing values (if any optional columns exist)
    # ------------------------------------------------------------------
    if "customer_id" in df.columns:
        df["customer_id"] = df["customer_id"].fillna("UNKNOWN")
    if "store_id" in df.columns:
        df["store_id"] = df["store_id"].fillna("UNKNOWN")

    # ------------------------------------------------------------------
    # 8. Calculate total_sales
    # ------------------------------------------------------------------
    df["total_sales"] = (df["quantity"] * df["price"]).round(2)

    # Ensure sensible dtypes
    df["quantity"] = df["quantity"].astype(int)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    # Sort for analytics friendliness
    df = df.sort_values(["date", "product"]).reset_index(drop=True)

    stats = {
        "original_rows": original_rows,
        "duplicates_removed": duplicates_removed,
        "invalid_dates": int(invalid_dates),
        "rows_dropped_validation": rows_dropped_validation,
        "final_rows": len(df),
        "total_revenue": float(df["total_sales"].sum()) if not df.empty else 0.0,
        "unique_products": int(df["product"].nunique()) if not df.empty else 0,
        "date_range": {
            "min": str(df["date"].min().date()) if not df.empty else None,
            "max": str(df["date"].max().date()) if not df.empty else None,
        },
    }

    logger.info(
        "Transform complete: %d → %d rows (removed %d duplicates, %d invalid)",
        original_rows,
        len(df),
        duplicates_removed,
        rows_dropped_validation,
    )

    return {
        "cleaned_df": df,
        "stats": stats,
        "errors": errors,
        "success": True,
    }
