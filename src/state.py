"""
Shared state for the LangGraph ETL workflow.
"""
from typing import Any, Dict, List, Optional, TypedDict
import pandas as pd


class ETLState(TypedDict):
    """State passed between Extract → Transform → Load nodes."""

    # Input
    uploaded_file_path: Optional[str]  # temporary path of uploaded CSV

    # DataFrames
    raw_df: Optional[pd.DataFrame]
    cleaned_df: Optional[pd.DataFrame]

    # Metrics & diagnostics
    stats: Dict[str, Any]
    errors: List[str]
    success: bool
