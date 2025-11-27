"""
Document Embeddings Module

Handles document processing and embedding generation for offline vector search.
Uses sentence-transformers for local embedding generation.
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Union, Optional
import chardet

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "sentence-transformers not installed. "
        "Install with: pip install sentence-transformers"
    )

from pypdf import PdfReader
from docx import Document
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentEmbedder:
    """Handles document processing and embedding generation."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        cache_folder: Optional[str] = None
    ):
        """
        Initialize the document embedder.
        
        Args:
            model_name: Name of sentence-transformers model
            device: Device to run on ('cpu' or 'cuda')
            cache_folder: Cache folder for model
        """
        self.model_name = model_name
        self.device = device
        
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=cache_folder
        )
        logger.info("Embedding model loaded successfully")
    
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text.
        
        Args:
            text: Single text or list of texts
            
        Returns:
            numpy array of embeddings
        """
        if isinstance(text, str):
            text = [text]
        
        embeddings = self.model.encode(
            text,
            show_progress_bar=len(text) > 10,
            convert_to_numpy=True
        )
        return embeddings
    
    def process_pdf(self, file_path: str) -> Dict[str, Union[str, List[str]]]:
        """
        Extract text from PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary with metadata and text chunks
        """
        try:
            reader = PdfReader(file_path)
            text_chunks = []
            
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    text_chunks.append({
                        'page': i + 1,
                        'text': text.strip()
                    })
            
            return {
                'file_path': file_path,
                'file_type': 'pdf',
                'num_pages': len(reader.pages),
                'chunks': text_chunks
            }
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            return {'error': str(e)}
    
    def process_docx(self, file_path: str) -> Dict[str, Union[str, List[str]]]:
        """
        Extract text from DOCX file.
        
        Args:
            file_path: Path to DOCX file
            
        Returns:
            Dictionary with metadata and text chunks
        """
        try:
            doc = Document(file_path)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            
            return {
                'file_path': file_path,
                'file_type': 'docx',
                'num_paragraphs': len(paragraphs),
                'chunks': [{'index': i, 'text': p} for i, p in enumerate(paragraphs)]
            }
        except Exception as e:
            logger.error(f"Error processing DOCX {file_path}: {e}")
            return {'error': str(e)}
    
    def process_txt(self, file_path: str) -> Dict[str, Union[str, List[str]]]:
        """
        Extract text from TXT file.
        
        Args:
            file_path: Path to TXT file
            
        Returns:
            Dictionary with metadata and text chunks
        """
        try:
            # Detect encoding
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                detected = chardet.detect(raw_data)
                encoding = detected['encoding'] or 'utf-8'
            
            # Read with detected encoding
            with open(file_path, 'r', encoding=encoding) as f:
                text = f.read()
            
            # Split into paragraphs
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            return {
                'file_path': file_path,
                'file_type': 'txt',
                'encoding': encoding,
                'chunks': [{'index': i, 'text': p} for i, p in enumerate(paragraphs)]
            }
        except Exception as e:
            logger.error(f"Error processing TXT {file_path}: {e}")
            return {'error': str(e)}
    
    def process_document(self, file_path: str) -> Dict:
        """
        Process document based on file extension.
        
        Args:
            file_path: Path to document
            
        Returns:
            Processed document data
        """
        path = Path(file_path)
        
        if not path.exists():
            return {'error': f'File not found: {file_path}'}
        
        extension = path.suffix.lower()
        
        if extension == '.pdf':
            return self.process_pdf(str(path))
        elif extension == '.docx':
            return self.process_docx(str(path))
        elif extension == '.txt':
            return self.process_txt(str(path))
        else:
            return {'error': f'Unsupported file type: {extension}'}
    
    def embed_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Generate embeddings for list of document chunks.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            Documents with embeddings added
        """
        all_texts = []
        for doc in documents:
            if 'chunks' in doc:
                for chunk in doc['chunks']:
                    all_texts.append(chunk.get('text', ''))
        
        if not all_texts:
            logger.warning("No text found to embed")
            return documents
        
        logger.info(f"Generating embeddings for {len(all_texts)} text chunks")
        embeddings = self.embed_text(all_texts)
        
        # Add embeddings back to documents
        idx = 0
        for doc in documents:
            if 'chunks' in doc:
                for chunk in doc['chunks']:
                    chunk['embedding'] = embeddings[idx]
                    idx += 1
        
        return documents
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self.model.get_sentence_embedding_dimension()


def create_embedder(model_name: str = "all-MiniLM-L6-v2", **kwargs) -> DocumentEmbedder:
    """
    Factory function to create DocumentEmbedder.
    
    Args:
        model_name: Name of embedding model
        **kwargs: Additional arguments
        
    Returns:
        DocumentEmbedder instance
    """
    return DocumentEmbedder(model_name=model_name, **kwargs)


if __name__ == "__main__":
    print("Document Embeddings Module")
    print("Supports: PDF, DOCX, TXT files")
