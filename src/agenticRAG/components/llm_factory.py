from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from config.settings import settings
from typing import Union, Literal

class LLMFactory:
    """Factory for creating LLM instances"""
    
    _groq_instance = None
    _openai_instance = None
    
    @classmethod
    def get_llm(cls, provider: Literal["groq", "openai"] = "groq") -> Union[ChatGroq, ChatOpenAI]:
        """Get or create LLM instance (singleton pattern)"""
        if provider == "groq":
            if cls._groq_instance is None:
                cls._groq_instance = ChatGroq(
                    model=settings.GROQ_MODEL,
                    temperature=settings.GROQ_TEMPERATURE,
                    groq_api_key=settings.GROQ_API_KEY
                )
            return cls._groq_instance
        elif provider == "openai":
            if cls._openai_instance is None:
                cls._openai_instance = ChatOpenAI(
                    model=settings.OPENAI_MODEL,
                    temperature=settings.OPENAI_TEMPERATURE,
                    openai_api_key=settings.OPENAI_API_KEY
                )
            return cls._openai_instance
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @classmethod
    def create_new_llm(cls, provider: Literal["groq", "openai"] = "groq", **kwargs) -> Union[ChatGroq, ChatOpenAI]:
        """Create a new LLM instance with custom parameters"""
        if provider == "groq":
            return ChatGroq(
                model=kwargs.get("model", settings.GROQ_MODEL),
                temperature=kwargs.get("temperature", settings.GROQ_TEMPERATURE),
                groq_api_key=kwargs.get("api_key", settings.GROQ_API_KEY)
            )
        elif provider == "openai":
            return ChatOpenAI(
                model=kwargs.get("model", settings.OPENAI_MODEL),
                temperature=kwargs.get("temperature", settings.OPENAI_TEMPERATURE),
                openai_api_key=kwargs.get("api_key", settings.OPENAI_API_KEY)
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @classmethod
    def get_groq_llm(cls) -> ChatGroq:
        """Convenience method to get Groq LLM"""
        return cls.get_llm("groq")
    
    @classmethod
    def get_openai_llm(cls) -> ChatOpenAI:
        """Convenience method to get OpenAI LLM"""
        return cls.get_llm("openai")
    
    @classmethod
    def reset_instances(cls):
        """Reset singleton instances (useful for testing)"""
        cls._groq_instance = None
        cls._openai_instance = None