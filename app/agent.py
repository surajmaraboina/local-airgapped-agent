"""AI Agent Module - Orchestrates LLM, embeddings, and vector store for Q&A."""
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OfflineAgent:
    def __init__(self, llm, embedder, vector_store, max_context_docs: int = 3):
        self.llm = llm
        self.embedder = embedder
        self.vector_store = vector_store
        self.max_context_docs = max_context_docs
        logger.info("Agent initialized")
    
    def query(self, question: str, return_sources: bool = True) -> Dict:
        logger.info(f"Query: {question}")
        query_embedding = self.embedder.embed_text(question)
        results = self.vector_store.search(query_embedding, k=self.max_context_docs)
        if not results:
            return {'answer': "No relevant information found.", 'sources': []}
        context_parts = [f"[Doc {i+1}]: {r['text']}" for i, r in enumerate(results)]
        context = "\n\n".join(context_parts)
        prompt = f"""Use the context to answer concisely.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"""
        answer = self.llm.generate(prompt, max_tokens=300)
        response = {'answer': answer.strip()}
        if return_sources:
            response['sources'] = [{'text': r['text'][:150] + '...', 'source': r['source'], 'score': round(r['score'], 3)} for r in results]
        return response
    
    def add_documents(self, file_paths: List[str]) -> None:
        logger.info(f"Processing {len(file_paths)} documents...")
        all_docs = []
        for file_path in file_paths:
            doc = self.embedder.process_document(file_path)
            if 'error' not in doc:
                all_docs.append(doc)
        if all_docs:
            embedded_docs = self.embedder.embed_documents(all_docs)
            self.vector_store.add_documents(embedded_docs)
            logger.info(f"Added {len(all_docs)} documents to knowledge base")
