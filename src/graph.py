"""
LangGraph workflow definition – linear Extract → Transform → Load.
"""
from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, StateGraph

from src.nodes import extract_node, load_node, transform_node
from src.state import ETLState

logger = logging.getLogger(__name__)


def _should_continue(state: ETLState) -> Literal["transform", "load", "__end__"]:
    """
    Simple router: stop early on hard failures.
    """
    if not state.get("success", False):
        return "__end__"
    # After extract we always go to transform if success
    # After transform we go to load
    # The graph edges themselves enforce the linear order;
    # this helper is only used for conditional edges if needed later.
    return "transform"


def build_etl_graph():
    """
    Compile a linear three-node LangGraph.

    Returns
    -------
    CompiledGraph
        Ready to invoke with an initial ETLState.
    """
    workflow = StateGraph(ETLState)

    # Register nodes
    workflow.add_node("extract", extract_node)
    workflow.add_node("transform", transform_node)
    workflow.add_node("load", load_node)

    # Linear edges
    workflow.set_entry_point("extract")
    workflow.add_edge("extract", "transform")
    workflow.add_edge("transform", "load")
    workflow.add_edge("load", END)

    graph = workflow.compile()
    logger.info("ETL LangGraph compiled successfully")
    return graph


# Convenience singleton for the Streamlit app
etl_graph = build_etl_graph()
