import time
from typing import List
from src.config.settings import settings
from src.agenticRAG.models.state import AgentState
from src.agenticRAG.models.schemas import QueryRequest, QueryResponse
from src.agenticRAG.graph.builder import GraphBuilder
from loguru import logger

class AgenticRAGSystem:
    """Main AgenticRAG system"""
    
    def __init__(self):
        # Validate settings
        settings.validate()
        
        # Create graph
        self.app = GraphBuilder.create_graph()
        
        logger.info("AgenticRAG system initialized successfully")
    
    def process_query(self, query: str) -> QueryResponse:
        """Process a single query"""
        
        start_time = time.time()
        
        try:
            # Initialize state
            initial_state = AgentState(user_query=query)
            
            # Run the graph
            final_state = self.app.invoke(initial_state)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create response
            response = QueryResponse(
                query=final_state.user_query,
                upgraded_query=final_state.upgraded_query,
                route_taken=final_state.route_decision,
                response=final_state.final_response,
                metadata=final_state.metadata,
                processing_time=processing_time
            )
            
            logger.info(f"Query processed successfully in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
    
    def process_batch(self, queries: List[str]) -> List[QueryResponse]:
        """Process multiple queries"""
        
        responses = []
        for query in queries:
            try:
                response = self.process_query(query)
                responses.append(response)
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
        
        return responses

def main():
    """Main function"""
    
    # Initialize system
    system = AgenticRAGSystem()
    
    # Test queries
    test_queries = [
        "What is machine learning?",
        "Latest news about AI",
        "Write a poem about spring"
    ]
    
    # Process queries
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        
        try:
            response = system.process_query(query)
            
            print(f"Original Query: {response.query}")
            print(f"Upgraded Query: {response.upgraded_query}")
            print(f"Route Taken: {response.route_taken}")
            print(f"Response: {response.response}")
            print(f"Processing Time: {response.processing_time:.2f}s")
            print(f"Metadata: {response.metadata}")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()