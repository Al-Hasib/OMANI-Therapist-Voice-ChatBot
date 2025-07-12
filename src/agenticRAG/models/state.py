from typing import Dict, List, Any
from pydantic import BaseModel, Field

class AgentState(BaseModel):
    """State schema for the AgenticRAG workflow"""
    
    user_query: str = Field(description="Original user query")
    upgraded_query: str = Field(default="", description="Enhanced query")
    route_decision: str = Field(default="", description="Routing decision")
    retrieved_docs: List[str] = Field(default_factory=list, description="Retrieved documents")
    search_results: List[str] = Field(default_factory=list, description="Web search results")
    final_response: str = Field(default="", description="Final response")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        """Pydantic configuration"""
        arbitrary_types_allowed = True