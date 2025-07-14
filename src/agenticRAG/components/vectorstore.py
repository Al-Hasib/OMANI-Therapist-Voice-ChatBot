from langchain_community.vectorstores import FAISS
from langchain_huggingface  import HuggingFaceEmbeddings
from typing import List, Optional
from src.config.settings import settings
from src.agenticRAG.components.embeddings import EmbeddingFactory
import os

class VectorStoreManager:
    """Manager for vector store operations"""
    
    def __init__(self):
        self.embeddings = EmbeddingFactory.get_embeddings()
        self.vectorstore = None
    
    def load_vectorstore(self, path: Optional[str] = None) -> bool:
        """Load vector store from path"""
        try:
            path = path or settings.VECTORSTORE_PATH
            if os.path.exists(path):
                self.vectorstore = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
                return True
            return False
        except Exception as e:
            print(f"Error loading vectorstore: {e}")
            return False
    
    def search_documents(self, query: str, k: int = 3) -> List[str]:
        """Search for similar documents"""
        if not self.vectorstore:
            return []
        
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []
    
    def add_documents(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        """Add documents to vector store"""
        if not self.vectorstore:
            self.vectorstore = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
        else:
            self.vectorstore.add_texts(texts, metadatas=metadatas)
    
    def save_vectorstore(self, path: Optional[str] = None):
        """Save vector store to path"""
        if self.vectorstore:
            path = path or settings.VECTORSTORE_PATH
            self.vectorstore.save_local(path)