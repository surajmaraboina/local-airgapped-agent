"""FAISS Vector Store Module - Handles vector indexing and similarity search."""
import logging
import numpy as np
from typing import List, Dict

try:
    import faiss
except ImportError:
    raise ImportError("faiss not installed. Run: pip install faiss-cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, embedding_dim: int, index_type: str = "Flat"):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.metadata = []
        
        if index_type == "Flat":
            self.index = faiss.IndexFlatL2(embedding_dim)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        logger.info(f"Initialized {index_type} index (dim={embedding_dim})")
    
    def add_documents(self, documents: List[Dict]) -> None:
        embeddings_list = []
        for doc in documents:
            if 'chunks' not in doc:
                continue
            for chunk in doc['chunks']:
                if 'embedding' not in chunk:
                    continue
                embedding = chunk['embedding']
                if isinstance(embedding, list):
                    embedding = np.array(embedding)
                embeddings_list.append(embedding)
                self.metadata.append({
                    'text': chunk.get('text', ''),
                    'source': doc.get('file_path', 'unknown'),
                    'file_type': doc.get('file_type', 'unknown')
                })
        if embeddings_list:
            embeddings_array = np.vstack(embeddings_list).astype('float32')
            self.index.add(embeddings_array)
            logger.info(f"Added {len(embeddings_list)} vectors. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        if self.index.ntotal == 0:
            return []
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype('float32')
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, k)
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['score'] = float(1 / (1 + dist))
                result['rank'] = i + 1
                results.append(result)
        return results
