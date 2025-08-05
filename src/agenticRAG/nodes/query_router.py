from typing import Literal
from src.agenticRAG.models.state import AgentState
from src.agenticRAG.components.llm_factory import LLMFactory
from src.agenticRAG.prompt.prompts import Prompts
from src.config.settings import settings

class QueryRouter:
    """Node for routing queries to appropriate paths"""
    
    def __init__(self):
        self.llm = LLMFactory.get_llm()
        self.prompt = Prompts.query_router()
    
    def route_query(self, state: AgentState) -> AgentState:
        """Route query to appropriate path"""
        
        chain = self.prompt | self.llm
        
        try:
            response = chain.invoke({"query": state.upgraded_query})
            route_decision = response.content.strip().upper()
            
            # Validate route decision
            if route_decision not in ["RAG", "WEB", "DIRECT"]:
                route_decision = settings.DEFAULT_ROUTE
                
            state.route_decision = route_decision
            state.metadata["routing_success"] = True
            
        except Exception as e:
            state.route_decision = settings.DEFAULT_ROUTE
            state.metadata["routing_success"] = False
            state.metadata["routing_error"] = str(e)
        
        return state

# Node function for LangGraph
def query_router_node(state: AgentState) -> AgentState:
    """Node function for query routing"""
    router = QueryRouter()
    return router.route_query(state)