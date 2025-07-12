from src.AgenticRAG.models.state import AgentState
from src.AgenticRAG.components.llm_factory import LLMFactory
from src.AgenticRAG.components.vectorstore import VectorStoreManager
from src.config.prompts import Prompts

class RAGNode:
    """Node for RAG processing"""
    
    def __init__(self):
        self.llm = LLMFactory.get_llm()
        self.vectorstore_manager = VectorStoreManager()
        self.prompt = Prompts.RAG_RESPONSE
        
        # Load vectorstore
        self.vectorstore_manager.load_vectorstore()
    
    def process_rag(self, state: AgentState) -> AgentState:
        """Process RAG path - retrieve from knowledge base"""
        
        try:
            # Retrieve documents
            docs = self.vectorstore_manager.search_documents(state.upgraded_query, k=3)
            state.retrieved_docs = docs
            
            # Generate response with retrieved context
            chain = self.prompt | self.llm
            
            context = "\n".join(docs) if docs else "No relevant documents found."
            response = chain.invoke({
                "query": state.upgraded_query,
                "context": context
            })
            
            state.final_response = response.content
            state.metadata["rag_success"] = True
            
        except Exception as e:
            state.final_response = "Sorry, I couldn't retrieve information from the knowledge base."
            state.metadata["rag_success"] = False
            state.metadata["rag_error"] = str(e)
        
        return state

# Node function for LangGraph
def rag_node(state: AgentState) -> AgentState:
    """Node function for RAG processing"""
    rag_processor = RAGNode()
    return rag_processor.process_rag(state)