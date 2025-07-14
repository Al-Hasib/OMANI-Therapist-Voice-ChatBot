import os
from typing import List, Union
from pathlib import Path

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from langchain.schema import Document

class DocumentChunker:
    """
    A class to read various document types and chunk them using LangChain
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the DocumentChunker
        
        Args:
            chunk_size (int): Size of each chunk in characters
            chunk_overlap (int): Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def read_pdf(self, file_path: str) -> List[Document]:
        """Read PDF file and return documents"""
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            return documents
        except Exception as e:
            print(f"Error reading PDF file {file_path}: {e}")
            return []
    
    def read_docx(self, file_path: str) -> List[Document]:
        """Read DOCX file and return documents"""
        try:
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
            return documents
        except Exception as e:
            print(f"Error reading DOCX file {file_path}: {e}")
            return []
    
    def read_txt(self, file_path: str) -> List[Document]:
        """Read TXT file and return documents"""
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
            return documents
        except Exception as e:
            print(f"Error reading TXT file {file_path}: {e}")
            return []
    
    def read_md(self, file_path: str) -> List[Document]:
        """Read Markdown file and return documents"""
        try:
            loader = UnstructuredMarkdownLoader(file_path)
            documents = loader.load()
            return documents
        except Exception as e:
            print(f"Error reading MD file {file_path}: {e}")
            return []
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load document based on file extension
        
        Args:
            file_path (str): Path to the document file
            
        Returns:
            List[Document]: List of loaded documents
        """
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            return self.read_pdf(file_path)
        elif file_extension == '.docx':
            return self.read_docx(file_path)
        elif file_extension == '.txt':
            return self.read_txt(file_path)
        elif file_extension == '.md':
            return self.read_md(file_path)
        else:
            print(f"Unsupported file type: {file_extension}")
            return []
    
    def chunk_documents(self, documents: List[Document]) -> List[str]:
        """
        Chunk documents and return list of strings
        
        Args:
            documents (List[Document]): List of documents to chunk
            
        Returns:
            List[str]: List of chunked text strings
        """
        if not documents:
            return []
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Extract text content from chunks
        chunk_texts = [chunk.page_content for chunk in chunks]
        
        return chunk_texts
    
    def process_file(self, file_path: str) -> List[str]:
        """
        Process a single file: load and chunk it
        
        Args:
            file_path (str): Path to the file to process
            
        Returns:
            List[str]: List of chunked text strings
        """
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return []
        
        # Load document
        documents = self.load_document(file_path)
        
        if not documents:
            print(f"No content loaded from {file_path}")
            return []
        
        # Chunk documents
        chunks = self.chunk_documents(documents)
        
        print(f"Successfully processed {file_path}: {len(chunks)} chunks created")
        return chunks
    
    def process_multiple_files(self, file_paths: List[str]) -> List[str]:
        """
        Process multiple files and return combined chunks
        
        Args:
            file_paths (List[str]): List of file paths to process
            
        Returns:
            List[str]: Combined list of chunked text strings
        """
        all_chunks = []
        
        for file_path in file_paths:
            chunks = self.process_file(file_path)
            all_chunks.extend(chunks)
        
        return all_chunks


# Example usage and utility functions
def main():
    """Example usage of the DocumentChunker class"""
    
    # Initialize chunker with custom parameters
    chunker = DocumentChunker(chunk_size=800, chunk_overlap=100)
    
    # Example: Process a single file
    file_path = "example.pdf"  # Replace with your file path
    chunks = chunker.process_file(file_path)
    
    if chunks:
        print(f"Total chunks: {len(chunks)}")
        print("\nFirst chunk preview:")
        print(chunks[0][:200] + "..." if len(chunks[0]) > 200 else chunks[0])
    
    # Example: Process multiple files
    file_paths = [
        "document1.pdf",
        "document2.docx",
        "document3.txt",
        "document4.md"
    ]
    
    all_chunks = chunker.process_multiple_files(file_paths)
    print(f"\nTotal chunks from all files: {len(all_chunks)}")
    
    return all_chunks


def create_chunker_with_custom_settings(chunk_size: int = 1000, 
                                       chunk_overlap: int = 200) -> DocumentChunker:
    """
    Create a DocumentChunker with custom settings
    
    Args:
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        DocumentChunker: Configured chunker instance
    """
    return DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


if __name__ == "__main__":
    main()