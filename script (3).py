# Create the document processor module
document_processor_code = '''"""
Document processing module for RAG system.
Handles PDF, DOCX, and plain text file ingestion with error handling and metadata extraction.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

import PyPDF2
import pdfplumber
from docx import Document as DocxDocument

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Metadata container for processed documents."""
    title: str
    source_type: str  # 'pdf', 'docx', 'txt'
    file_path: str
    file_size: int
    page_count: Optional[int] = None
    created_at: datetime = None
    processed_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.processed_at is None:
            self.processed_at = datetime.now()


class DocumentProcessor:
    """
    Handles document ingestion, parsing, and chunking for RAG system.
    Supports PDF, DOCX, and plain text files with comprehensive error handling.
    """
    
    def __init__(
        self,
        chunk_size: int = 200,  # ~200 words
        chunk_overlap: int = 50,  # 50 word overlap
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size * 4  # Convert words to approximate characters
        self.chunk_overlap = chunk_overlap * 4
        
        # Default separators for RecursiveCharacterTextSplitter
        self.separators = separators or ["\\n\\n", "\\n", " ", ""]
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
        )
    
    def process_documents(
        self, 
        file_paths: List[str]
    ) -> Tuple[List[Document], List[str]]:
        """
        Process multiple documents and return chunks with metadata.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            Tuple of (successfully processed documents, error messages)
        """
        all_documents = []
        errors = []
        
        for file_path in file_paths:
            try:
                documents = self.process_single_document(file_path)
                all_documents.extend(documents)
                logger.info(f"Successfully processed: {file_path}")
            except Exception as e:
                error_msg = f"Error processing {file_path}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        return all_documents, errors
    
    def process_single_document(self, file_path: str) -> List[Document]:
        """
        Process a single document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects with chunks and metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = Path(file_path).suffix.lower()
        file_size = os.path.getsize(file_path)
        
        # Extract text based on file type
        if file_extension == '.pdf':
            text, page_count = self._extract_pdf_text(file_path)
            source_type = 'pdf'
        elif file_extension == '.docx':
            text, page_count = self._extract_docx_text(file_path)
            source_type = 'docx'
        elif file_extension == '.txt':
            text, page_count = self._extract_txt_text(file_path)
            source_type = 'txt'
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        if not text.strip():
            raise ValueError(f"No extractable text found in: {file_path}")
        
        # Create metadata
        metadata = DocumentMetadata(
            title=Path(file_path).stem,
            source_type=source_type,
            file_path=file_path,
            file_size=file_size,
            page_count=page_count
        )
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create Document objects with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = {
                'title': metadata.title,
                'source_type': metadata.source_type,
                'file_path': metadata.file_path,
                'file_size': metadata.file_size,
                'page_count': metadata.page_count,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'processed_at': metadata.processed_at.isoformat(),
            }
            
            documents.append(Document(
                page_content=chunk,
                metadata=doc_metadata
            ))
        
        return documents
    
    def _extract_pdf_text(self, file_path: str) -> Tuple[str, int]:
        """Extract text from PDF using multiple methods for robustness."""
        text = ""
        page_count = 0
        
        try:
            # First, try pdfplumber (better for complex layouts)
            with pdfplumber.open(file_path) as pdf:
                page_count = len(pdf.pages)
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\\n\\n"
        except Exception as e:
            logger.warning(f"pdfplumber failed for {file_path}: {e}")
            
            # Fallback to PyPDF2
            try:
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    page_count = len(reader.pages)
                    
                    for page_num, page in enumerate(reader.pages):
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\\n\\n"
                        except Exception as page_error:
                            logger.warning(f"Failed to extract page {page_num}: {page_error}")
                            
            except Exception as fallback_error:
                raise Exception(f"Both PDF extraction methods failed: {fallback_error}")
        
        if not text.strip():
            raise ValueError("No text could be extracted from PDF")
            
        return text.strip(), page_count
    
    def _extract_docx_text(self, file_path: str) -> Tuple[str, int]:
        """Extract text from DOCX files."""
        try:
            doc = DocxDocument(file_path)
            text = ""
            
            # Extract text from paragraphs
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\\n\\n"
            
            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text.strip():
                            text += cell.text + " "
                    text += "\\n"
            
            # Approximate page count (assuming ~500 words per page)
            word_count = len(text.split())
            page_count = max(1, word_count // 500)
            
            if not text.strip():
                raise ValueError("No text found in DOCX file")
                
            return text.strip(), page_count
            
        except Exception as e:
            raise Exception(f"Failed to extract DOCX text: {e}")
    
    def _extract_txt_text(self, file_path: str) -> Tuple[str, int]:
        """Extract text from plain text files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            # Approximate page count
            word_count = len(text.split())
            page_count = max(1, word_count // 500)
            
            if not text.strip():
                raise ValueError("Empty text file")
                
            return text.strip(), page_count
            
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    text = file.read()
                
                word_count = len(text.split())
                page_count = max(1, word_count // 500)
                return text.strip(), page_count
                
            except Exception as e:
                raise Exception(f"Failed to read text file with multiple encodings: {e}")
        except Exception as e:
            raise Exception(f"Failed to extract text: {e}")


class HybridChunker:
    """
    Advanced chunking strategy that combines semantic and structural approaches.
    """
    
    def __init__(self, base_chunk_size: int = 200, overlap: int = 50):
        self.base_chunk_size = base_chunk_size * 4  # Convert to characters
        self.overlap = overlap * 4
        
        # Create multiple splitters for different content types
        self.paragraph_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.base_chunk_size,
            chunk_overlap=self.overlap,
            separators=["\\n\\n", "\\n", ". ", "! ", "? ", " ", ""]
        )
        
        self.sentence_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.base_chunk_size // 2,
            chunk_overlap=self.overlap // 2,
            separators=[". ", "! ", "? ", "\\n", " ", ""]
        )
    
    def adaptive_chunk(self, text: str, content_type: str = "general") -> List[str]:
        """
        Adaptively chunk text based on content type and structure.
        
        Args:
            text: Input text to chunk
            content_type: Type of content (general, technical, narrative)
            
        Returns:
            List of text chunks
        """
        if content_type == "technical":
            # Use smaller chunks for technical content
            return self.sentence_splitter.split_text(text)
        elif content_type == "narrative":
            # Use larger chunks for narrative content
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.base_chunk_size * 2,
                chunk_overlap=self.overlap * 2,
                separators=["\\n\\n", "\\n", ". ", " ", ""]
            )
            return splitter.split_text(text)
        else:
            # Default paragraph-based chunking
            return self.paragraph_splitter.split_text(text)
'''

# Save the document processor
with open('src/rag_system/document_processors/processor.py', 'w') as f:
    f.write(document_processor_code)

print("✅ Document processor module created")