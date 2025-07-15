from langgraph.graph import StateGraph, END
from src.agenticRAG.models.state import AgentState
from src.agenticRAG.nodes.query_upgrader import query_upgrader_node
from src.agenticRAG.nodes.query_router import query_router_node
from src.agenticRAG.nodes.rag_node import rag_node
from src.agenticRAG.nodes.web_search_node import web_search_node
from src.agenticRAG.nodes.direct_llm_node import direct_llm_node
from src.agenticRAG.graph.router import route_query

class GraphBuilder:
    """Builder for the AgenticRAG graph"""
    
    @staticmethod
    def create_graph():
        """Create the LangGraph workflow"""
        
        # Initialize graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("query_upgrader", query_upgrader_node)
        workflow.add_node("query_router", query_router_node)
        workflow.add_node("rag_path", rag_node)
        workflow.add_node("web_search", web_search_node)
        workflow.add_node("direct_llm", direct_llm_node)
        
        # Set entry point
        workflow.set_entry_point("query_upgrader")
        
        # Add edges
        workflow.add_edge("query_upgrader", "query_router")
        
        # Add conditional edges based on routing decision
        workflow.add_conditional_edges(
            "query_router",
            route_query,
            {
                "rag_path": "rag_path",
                "web_search": "web_search",
                "direct_llm": "direct_llm"
            }
        )
        
        # All paths end at END
        workflow.add_edge("rag_path", END)
        workflow.add_edge("web_search", END)
        workflow.add_edge("direct_llm", END)
        
        # Compile the graph
        return workflow.compile()