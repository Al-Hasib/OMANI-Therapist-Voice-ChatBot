from src.agenticRAG.models.state import AgentState
from src.agenticRAG.components.llm_factory import LLMFactory
from src.agenticRAG.prompt.prompts import Prompts

class DirectLLMNode:
    """Node for direct LLM processing"""
    
    def __init__(self):
        self.llm = LLMFactory.get_llm()
        self.prompt = Prompts.DIRECT_RESPONSE
    
    def process_direct_llm(self, state: AgentState) -> AgentState:
        """Process direct LLM path"""
        
        try:
            chain = self.prompt | self.llm
            
            response = chain.invoke({"query": state.upgraded_query})
            state.final_response = response.content
            state.metadata["direct_llm_success"] = True
            
        except Exception as e:
            state.final_response = "Sorry, I couldn't process your request at the moment."
            state.metadata["direct_llm_success"] = False
            state.metadata["direct_llm_error"] = str(e)
        
        return state

# Node function for LangGraph
def direct_llm_node(state: AgentState) -> AgentState:
    """Node function for direct LLM processing"""
    direct_processor = DirectLLMNode()
    return direct_processor.process_direct_llm(state)