from models.state import AgentState
from components.llm_factory import LLMFactory
from config.prompts import Prompts
from config.settings import settings

class QueryUpgrader:
    """Node for upgrading user queries"""
    
    def __init__(self):
        self.llm = LLMFactory.get_llm()
        self.prompt = Prompts.QUERY_UPGRADER
    
    def upgrade_query(self, state: AgentState) -> AgentState:
        """Upgrade/enhance the user query"""
        
        chain = self.prompt | self.llm
        
        try:
            response = chain.invoke({"query": state.user_query})
            upgraded_query = response.content.strip()
            
            # Fallback to original if upgrade fails
            if not upgraded_query or len(upgraded_query) > settings.MAX_QUERY_LENGTH:
                upgraded_query = state.user_query
                
            state.upgraded_query = upgraded_query
            state.metadata["upgrade_success"] = True
            
        except Exception as e:
            state.upgraded_query = state.user_query
            state.metadata["upgrade_success"] = False
            state.metadata["upgrade_error"] = str(e)
        
        return state

# Node function for LangGraph
def query_upgrader_node(state: AgentState) -> AgentState:
    """Node function for query upgrading"""
    upgrader = QueryUpgrader()
    return upgrader.upgrade_query(state)