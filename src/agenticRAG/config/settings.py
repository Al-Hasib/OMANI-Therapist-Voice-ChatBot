import os
from typing import Dict, Any

class Settings:
    """Configuration settings for the AgenticRAG system"""
    
    # API Keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GOOGLE_CSE_ID: str = os.getenv("GOOGLE_CSE_ID", "")
    
    # Model Configuration
    GROQ_MODEL: str = "llama3-8b-8192"
    GROQ_TEMPERATURE: float = 0.1
    
    # Embedding Model
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Vector Store
    VECTORSTORE_PATH: str = "data/vectorstore"
    
    # Search Configuration
    SEARCH_RESULTS_COUNT: int = 5
    
    # Query Enhancement
    MAX_QUERY_LENGTH: int = 200
    
    # Routing Configuration
    DEFAULT_ROUTE: str = "DIRECT"
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required settings"""
        required_keys = ["GROQ_API_KEY"]
        for key in required_keys:
            if not getattr(cls, key):
                raise ValueError(f"Missing required setting: {key}")
        return True

settings = Settings()