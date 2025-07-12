from langchain_groq import ChatGroq
from config.settings import settings

class LLMFactory:
    """Factory for creating LLM instances"""
    
    _instance = None
    
    @classmethod
    def get_llm(cls) -> ChatGroq:
        """Get or create LLM instance (singleton pattern)"""
        if cls._instance is None:
            cls._instance = ChatGroq(
                model=settings.GROQ_MODEL,
                temperature=settings.GROQ_TEMPERATURE,
                groq_api_key=settings.GROQ_API_KEY
            )
        return cls._instance
    
    @classmethod
    def create_new_llm(cls, **kwargs) -> ChatGroq:
        """Create a new LLM instance with custom parameters"""
        return ChatGroq(
            model=kwargs.get("model", settings.GROQ_MODEL),
            temperature=kwargs.get("temperature", settings.GROQ_TEMPERATURE),
            groq_api_key=kwargs.get("api_key", settings.GROQ_API_KEY)
        )