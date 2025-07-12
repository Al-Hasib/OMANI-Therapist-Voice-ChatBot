from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class QueryRequest(BaseModel):
    """Request schema for query processing"""
    query: str
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    """Response schema for query processing"""
    query: str
    upgraded_query: str
    route_taken: str
    response: str
    metadata: Dict[str, Any]
    processing_time: float

class ProcessingMetadata(BaseModel):
    """Metadata for processing steps"""
    upgrade_success: bool = False
    routing_success: bool = False
    path_success: bool = False
    errors: List[str] = []
    processing_time: float = 0.0