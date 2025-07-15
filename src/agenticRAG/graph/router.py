from typing import Literal
from src.agenticRAG.models.state import AgentState

def route_query(state: AgentState) -> Literal["rag_path", "web_search", "direct_llm"]:
    """Route to appropriate path based on decision"""
    route_map = {
        "RAG": "rag_path",
        "WEB": "web_search", 
        "DIRECT": "direct_llm"
    }
    return route_map.get(state.route_decision, "direct_llm")