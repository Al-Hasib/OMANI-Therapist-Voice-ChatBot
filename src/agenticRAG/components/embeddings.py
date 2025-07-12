from langchain_community.embeddings import HuggingFaceEmbeddings
from config.settings import settings

class EmbeddingFactory:
    """Factory for creating embedding instances"""
    
    _instance = None
    
    @classmethod
    def get_embeddings(cls) -> HuggingFaceEmbeddings:
        """Get or create embeddings instance (singleton pattern)"""
        if cls._instance is None:
            cls._instance = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL
            )
        return cls._instance