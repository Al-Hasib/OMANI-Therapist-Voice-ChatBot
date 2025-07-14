from langchain_community.vectorstores import FAISS
from langchain_huggingface  import HuggingFaceEmbeddings
from typing import List, Optional
from src.config.settings import settings
from src.agenticRAG.components.embeddings import EmbeddingFactory
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
from src.agenticRAG.components.document_chunker import DocumentChunker


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




def store_documents_in_vectorstore(
    file_paths: List[str],
    vectorstore_manager: Optional[VectorStoreManager] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    save_path: Optional[str] = None,
    include_metadata: bool = True
) -> Dict[str, Any]:
    """
    Process documents and store them in vector store
    
    Args:
        file_paths (List[str]): List of file paths to process
        vectorstore_manager (VectorStoreManager, optional): Existing manager instance
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap between chunks
        save_path (str, optional): Path to save the vector store
        include_metadata (bool): Whether to include file metadata
        
    Returns:
        Dict[str, Any]: Processing results with statistics
    """
    # Initialize components
    if vectorstore_manager is None:
        vectorstore_manager = VectorStoreManager()
    
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Load existing vectorstore if available
    vectorstore_manager.load_vectorstore(save_path)
    
    # Track processing statistics
    results = {
        "total_files": len(file_paths),
        "processed_files": 0,
        "failed_files": [],
        "total_chunks": 0,
        "chunks_by_file": {}
    }
    
    try:
        for file_path in file_paths:
            try:
                print(f"Processing file: {file_path}")
                
                # Process file into chunks
                chunks = chunker.process_file(file_path)
                
                if chunks:
                    # Prepare metadata if requested
                    metadatas = None
                    if include_metadata:
                        file_name = Path(file_path).name
                        file_extension = Path(file_path).suffix
                        metadatas = [
                            {
                                "source": file_path,
                                "file_name": file_name,
                                "file_extension": file_extension,
                                "chunk_index": i
                            }
                            for i in range(len(chunks))
                        ]
                    
                    # Add documents to vector store
                    vectorstore_manager.add_documents(chunks, metadatas)
                    
                    # Update statistics
                    results["processed_files"] += 1
                    results["total_chunks"] += len(chunks)
                    results["chunks_by_file"][file_path] = len(chunks)
                    
                    print(f"Successfully processed {file_path}: {len(chunks)} chunks")
                    
                else:
                    print(f"No chunks extracted from {file_path}")
                    results["failed_files"].append(file_path)
                    
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                results["failed_files"].append(file_path)
        
        # Save the vector store
        if results["total_chunks"] > 0:
            vectorstore_manager.save_vectorstore(save_path)
            print(f"Vector store saved with {results['total_chunks']} total chunks")
        
        return results
        
    except Exception as e:
        print(f"Error in store_documents_in_vectorstore: {e}")
        results["error"] = str(e)
        return results


def store_single_document_in_vectorstore(
    file_path: str,
    vectorstore_manager: Optional[VectorStoreManager] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    save_path: Optional[str] = None
) -> bool:
    """
    Process and store a single document in vector store
    
    Args:
        file_path (str): Path to the file to process
        vectorstore_manager (VectorStoreManager, optional): Existing manager instance
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap between chunks
        save_path (str, optional): Path to save the vector store
        
    Returns:
        bool: Success status
    """
    results = store_documents_in_vectorstore(
        file_paths=[file_path],
        vectorstore_manager=vectorstore_manager,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        save_path=save_path
    )
    
    return results["processed_files"] > 0


def batch_store_documents(
    directory_path: str,
    file_extensions: List[str] = [".pdf", ".docx", ".txt", ".md"],
    vectorstore_manager: Optional[VectorStoreManager] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process and store all documents from a directory
    
    Args:
        directory_path (str): Path to directory containing documents
        file_extensions (List[str]): List of file extensions to process
        vectorstore_manager (VectorStoreManager, optional): Existing manager instance
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap between chunks
        save_path (str, optional): Path to save the vector store
        
    Returns:
        Dict[str, Any]: Processing results
    """
    # Find all files with specified extensions
    directory = Path(directory_path)
    file_paths = []
    
    for extension in file_extensions:
        file_paths.extend(directory.glob(f"*{extension}"))
    
    # Convert to string paths
    file_paths = [str(path) for path in file_paths]
    
    if not file_paths:
        print(f"No files found in {directory_path} with extensions {file_extensions}")
        return {"total_files": 0, "processed_files": 0, "failed_files": [], "total_chunks": 0}
    
    print(f"Found {len(file_paths)} files to process")
    
    return store_documents_in_vectorstore(
        file_paths=file_paths,
        vectorstore_manager=vectorstore_manager,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        save_path=save_path
    )


# Example usage
def main():
    """Example usage of the vector store functions"""
    
    # Initialize vector store manager
    vs_manager = VectorStoreManager()
    
    # Example 1: Store a single document
    print("=== Storing Single Document ===")
    file_path = "/home/ubuntu/OMANI-Therapist-Voice-ChatBot/KnowledgebaseFile/SuicideGuard_An_NLP-Based_Chrome_Extension_for_Detecting_Suicidal_Thoughts_in_Bengali.pdf"
    success = store_single_document_in_vectorstore(
        file_path=file_path,
        vectorstore_manager=vs_manager,
        chunk_size=1000,
        chunk_overlap=150
    )
    print(f"Single document processing: {'Success' if success else 'Failed'}")
    
    # # Example 2: Store multiple documents
    # print("\n=== Storing Multiple Documents ===")
    # file_paths = [
    #     "document1.pdf",
    #     "document2.docx",
    #     "document3.txt"
    # ]
    
    # results = store_documents_in_vectorstore(
    #     file_paths=file_paths,
    #     vectorstore_manager=vs_manager,
    #     chunk_size=1000,
    #     chunk_overlap=200
    # )
    
    # print(f"Processing Results:")
    # print(f"  Total files: {results['total_files']}")
    # print(f"  Processed files: {results['processed_files']}")
    # print(f"  Failed files: {results['failed_files']}")
    # print(f"  Total chunks: {results['total_chunks']}")
    
    # # Example 3: Batch process directory
    # print("\n=== Batch Processing Directory ===")
    # directory_path = "/home/ubuntu/OMANI-Therapist-Voice-ChatBot/KnowledgebaseFile/"
    
    # batch_results = batch_store_documents(
    #     directory_path=directory_path,
    #     file_extensions=[".pdf", ".docx", ".txt", ".md"],
    #     vectorstore_manager=vs_manager
    # )
    
    # print(f"Batch Processing Results:")
    # print(f"  Total files: {batch_results['total_files']}")
    # print(f"  Processed files: {batch_results['processed_files']}")
    # print(f"  Total chunks: {batch_results['total_chunks']}")
    
    # Example 4: Search the vector store
    print("\n=== Searching Vector Store ===")
    query = "suicide prevention techniques"
    search_results = vs_manager.search_documents(query, k=3)
    
    print(f"Search results for '{query}':")
    for i, result in enumerate(search_results):
        print(f"  Result {i+1}: {result[:200]}...")


if __name__ == "__main__":
    main()