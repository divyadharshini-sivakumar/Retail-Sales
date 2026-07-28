"""Unit tests for the Extract node."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.nodes.extract import extract_node
from src.state import ETLState


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Create a tiny valid CSV for testing."""
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "product": ["Widget", "Gadget"],
            "quantity": [1, 2],
            "price": [10.0, 20.0],
        }
    )
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    return path


def test_extract_success(sample_csv: Path) -> None:
    state: ETLState = {
        "uploaded_file_path": str(sample_csv),
        "raw_df": None,
        "cleaned_df": None,
        "stats": {},
        "errors": [],
        "success": False,
    }
    result = extract_node(state)
    assert result["success"] is True
    assert result["raw_df"] is not None
    assert len(result["raw_df"]) == 2
    assert result["errors"] == []


def test_extract_missing_file() -> None:
    state: ETLState = {
        "uploaded_file_path": "/nonexistent/path.csv",
        "raw_df": None,
        "cleaned_df": None,
        "stats": {},
        "errors": [],
        "success": False,
    }
    result = extract_node(state)
    assert result["success"] is False
    assert result["raw_df"] is None
    assert any("not found" in e.lower() for e in result["errors"])


def test_extract_no_path() -> None:
    state: ETLState = {
        "uploaded_file_path": None,
        "raw_df": None,
        "cleaned_df": None,
        "stats": {},
        "errors": [],
        "success": False,
    }
    result = extract_node(state)
    assert result["success"] is False
    assert any("no file path" in e.lower() for e in result["errors"])
