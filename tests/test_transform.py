"""Unit tests for the Transform node."""
from __future__ import annotations

import pandas as pd
import pytest

from src.nodes.transform import transform_node
from src.state import ETLState


@pytest.fixture
def messy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": ["2024-01-15", "15/01/2024", "2024-01-16", "not-a-date", "2024-01-17"],
            "Product": ["iPhone 15", "iphone15", "MacBook Pro", "iPhone 15", "AirPods Pro"],
            "Qty": [2, 1, 1, 1, 3],
            "Price": [999.99, 999.99, 2499.00, 999.99, 249.50],
            "Customer_ID": ["C001", "C002", "C003", "C004", "C005"],
        }
    )


def test_transform_basic(messy_df: pd.DataFrame) -> None:
    state: ETLState = {
        "uploaded_file_path": None,
        "raw_df": messy_df,
        "cleaned_df": None,
        "stats": {},
        "errors": [],
        "success": True,
    }
    result = transform_node(state)

    assert result["success"] is True
    cleaned = result["cleaned_df"]
    assert cleaned is not None
    assert "total_sales" in cleaned.columns
    assert "date" in cleaned.columns
    assert "product" in cleaned.columns
    assert "quantity" in cleaned.columns
    assert "price" in cleaned.columns

    # Invalid date row should be dropped
    assert len(cleaned) == 4  # one bad date removed

    # Product names standardized
    products = set(cleaned["product"].tolist())
    assert "iPhone 15" in products
    assert "MacBook Pro" in products

    # total_sales calculated
    assert (cleaned["total_sales"] == cleaned["quantity"] * cleaned["price"]).all()

    stats = result["stats"]
    assert stats["original_rows"] == 5
    assert stats["final_rows"] == 4
    assert stats["invalid_dates"] >= 1


def test_transform_empty() -> None:
    state: ETLState = {
        "uploaded_file_path": None,
        "raw_df": pd.DataFrame(),
        "cleaned_df": None,
        "stats": {},
        "errors": [],
        "success": True,
    }
    result = transform_node(state)
    assert result["success"] is False
    assert any("no data" in e.lower() for e in result["errors"])


def test_transform_negative_qty_and_price() -> None:
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "product": ["A", "B", "C"],
            "quantity": [1, -5, 2],
            "price": [10.0, 20.0, -3.0],
        }
    )
    state: ETLState = {
        "uploaded_file_path": None,
        "raw_df": df,
        "cleaned_df": None,
        "stats": {},
        "errors": [],
        "success": True,
    }
    result = transform_node(state)
    cleaned = result["cleaned_df"]
    assert cleaned is not None
    # Only the first row is valid
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["product"] == "A"
