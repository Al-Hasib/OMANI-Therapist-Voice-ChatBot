from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from src.config.settings import settings
from typing import Union, Literal

class EmbeddingFactory:
    """Factory for creating embedding instances"""
    
    _huggingface_instance = None
    _openai_instance = None
    
    @classmethod
    def get_embeddings(cls, provider: Literal["huggingface", "openai"] = "huggingface") -> Union[HuggingFaceEmbeddings, OpenAIEmbeddings]:
        """Get or create embeddings instance (singleton pattern)"""
        if provider == "huggingface":
            if cls._huggingface_instance is None:
                cls._huggingface_instance = HuggingFaceEmbeddings(
                    model_name=settings.EMBEDDING_MODEL
                )
            return cls._huggingface_instance
        elif provider == "openai":
            if cls._openai_instance is None:
                cls._openai_instance = OpenAIEmbeddings(
                    model=settings.OPENAI_EMBEDDING_MODEL,
                    openai_api_key=settings.OPENAI_API_KEY
                )
            return cls._openai_instance
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @classmethod
    def create_new_embeddings(cls, provider: Literal["huggingface", "openai"] = "huggingface", **kwargs) -> Union[HuggingFaceEmbeddings, OpenAIEmbeddings]:
        """Create a new embeddings instance with custom parameters"""
        if provider == "huggingface":
            return HuggingFaceEmbeddings(
                model_name=kwargs.get("model_name", settings.EMBEDDING_MODEL),
                **{k: v for k, v in kwargs.items() if k != "model_name"}
            )
        elif provider == "openai":
            return OpenAIEmbeddings(
                model=kwargs.get("model", settings.OPENAI_EMBEDDING_MODEL),
                openai_api_key=kwargs.get("api_key", settings.OPENAI_API_KEY),
                **{k: v for k, v in kwargs.items() if k not in ["model", "api_key"]}
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @classmethod
    def get_huggingface_embeddings(cls) -> HuggingFaceEmbeddings:
        """Convenience method to get HuggingFace embeddings"""
        return cls.get_embeddings("huggingface")
    
    @classmethod
    def get_openai_embeddings(cls) -> OpenAIEmbeddings:
        """Convenience method to get OpenAI embeddings"""
        return cls.get_embeddings("openai")
    
    @classmethod
    def reset_instances(cls):
        """Reset singleton instances (useful for testing)"""
        cls._huggingface_instance = None
        cls._openai_instance = None