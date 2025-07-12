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

    OPENAI_MODEL: str = "gpt-4.1-nano-2025-04-14"
    OPENAI_TEMPERATURE: float = 0.3
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Embedding Model
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
    # Vector Store
    VECTORSTORE_PATH: str = "data/vectorstore"
    
    # Search Configuration
    SEARCH_RESULTS_COUNT: int = 5

    SERPER_API_KEY: str = os.getenv("SERPER_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    
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