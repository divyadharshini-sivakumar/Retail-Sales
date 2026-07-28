"""Integration test for the full LangGraph ETL pipeline."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.graph import build_etl_graph
from src.state import ETLState


@pytest.fixture
def sample_messy_csv(tmp_path: Path) -> Path:
    """Write the same content as sample_data/messy_sales.csv."""
    content = """date,product,quantity,price,customer_id,store_id
2024-01-15,iPhone 15,2,999.99,C001,S01
15/01/2024,iphone15,1,999.99,C002,S01
2024-01-16,MacBook Pro,1,2499.00,C003,S02
01/16/2024,macbook pro,1,2499,C003,S02
2024-01-17,AirPods Pro,3,249.50,C004,S01
2024-01-17,airpods pro,3,249.50,C004,S01
2024-01-18,iPad Air,2,599.00,C005,S03
18-01-2024,ipad air,2,599,C005,S03
2024-01-19,UNKNOWN ITEM,,199.99,C006,S01
2024-01-20,iPhone 15,-1,999.99,C007,S02
2024-01-21,MacBook Pro,1,-50,C008,S01
2024-01-22,AirPods,5,179.00,C009,S03
2024/01/23,iPad,1,499.00,C010,S02
not-a-date,iPhone 15,1,999.99,C011,S01
2024-01-25,iphone-15,2,999.99,C012,S03
2024-01-26,MacBookPro,1,2499.00,C013,S01
2024-01-27,airpods,4,179.00,,S02
2024-01-28,iPad Air,1,599.00,C015,
2024-01-29,iPhone 15,3,999.99,C016,S01
2024-01-30,MacBook Pro,2,2499.00,C017,S02
"""
    path = tmp_path / "messy.csv"
    path.write_text(content, encoding="utf-8")
    return path


def test_full_pipeline(sample_messy_csv: Path) -> None:
    graph = build_etl_graph()

    initial: ETLState = {
        "uploaded_file_path": str(sample_messy_csv),
        "raw_df": None,
        "cleaned_df": None,
        "stats": {},
        "errors": [],
        "success": False,
    }

    final = graph.invoke(initial)

    assert final["success"] is True
    assert final["cleaned_df"] is not None
    assert len(final["cleaned_df"]) > 0
    assert "total_sales" in final["cleaned_df"].columns
    assert final["stats"]["final_rows"] == final["stats"]["original_rows"] - (
        final["stats"]["duplicates_removed"] + final["stats"]["rows_dropped_validation"]
    )
    # Output file should exist
    out_path = final["stats"].get("output_path")
    assert out_path is not None
    assert Path(out_path).exists()

    # Clean up
    Path(out_path).unlink(missing_ok=True)
