from src.AgenticRAG.models.state import AgentState
from src.AgenticRAG.components.llm_factory import LLMFactory
from src.AgenticRAG.components.search_tools import SearchToolFactory
from src.config.prompts import Prompts

class WebSearchNode:
    """Node for web search processing"""
    
    def __init__(self):
        self.llm = LLMFactory.get_llm()
        self.search_tool = SearchToolFactory.get_search_tool()
        self.prompt = Prompts.WEB_RESPONSE
    
    def process_web_search(self, state: AgentState) -> AgentState:
        """Process web search path"""
        
        try:
            # Perform web search
            search_results = self.search_tool.run(state.upgraded_query)
            state.search_results = [search_results]
            
            # Generate response with search results
            chain = self.prompt | self.llm
            
            response = chain.invoke({
                "query": state.upgraded_query,
                "search_results": search_results
            })
            
            state.final_response = response.content
            state.metadata["web_search_success"] = True
            
        except Exception as e:
            state.final_response = "Sorry, I couldn't perform web search at the moment."
            state.metadata["web_search_success"] = False
            state.metadata["web_search_error"] = str(e)
        
        return state

# Node function for LangGraph
def web_search_node(state: AgentState) -> AgentState:
    """Node function for web search processing"""
    web_processor = WebSearchNode()
    return web_processor.process_web_search(state)