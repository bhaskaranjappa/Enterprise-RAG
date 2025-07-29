# Create the main retrieval coordinator
retrieval_code = '''"""
Main retrieval coordinator that handles embedding generation, vector search, and reranking.
Supports hybrid search and multiple retrieval strategies.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

from langchain_core.documents import Document

from ..embeddings.embedding_manager import EmbeddingManager
from ..retrieval.vector_store import FAISSVectorStore, HybridSearchVectorStore


logger = logging.getLogger(__name__)


class RetrievalEngine:
    """
    Main retrieval engine that coordinates embedding generation and vector search.
    Supports multiple retrieval strategies including hybrid search and reranking.
    """
    
    def __init__(
        self,
        embedding_manager: Optional[EmbeddingManager] = None,
        vector_store: Optional[FAISSVectorStore] = None,
        use_reranking: bool = False,
        reranker_model: Optional[str] = None
    ):
        # Initialize components
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.vector_store = vector_store or FAISSVectorStore(
            embedding_dimension=self.embedding_manager.embedding_dim
        )
        
        self.use_reranking = use_reranking
        self.reranker = None
        
        # Initialize reranker if requested
        if use_reranking and reranker_model:
            self._initialize_reranker(reranker_model)
        
        logger.info("Initialized retrieval engine")
    
    def _initialize_reranker(self, model_name: str):
        """Initialize cross-encoder reranking model."""
        try:
            from sentence_transformers import CrossEncoder
            
            self.reranker = CrossEncoder(model_name)
            logger.info(f"Initialized reranker: {model_name}")
            
        except ImportError:
            logger.warning("sentence-transformers not available for reranking")
            self.use_reranking = False
        except Exception as e:
            logger.error(f"Failed to load reranker: {e}")
            self.use_reranking = False
    
    def index_documents(self, documents: List[Document]) -> bool:
        """
        Index a list of documents in the vector store.
        
        Args:
            documents: List of Document objects to index
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embeddings
            embeddings, metadata = self.embedding_manager.embed_documents(documents)
            
            # Add to vector store
            doc_ids = self.vector_store.add_embeddings(
                embeddings=embeddings,
                metadata=metadata,
                documents=documents
            )
            
            logger.info(f"Indexed {len(documents)} documents with IDs: {doc_ids[:3]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            return False
    
    def retrieve_documents(
        self,
        query: str,
        k: int = 5,
        retrieval_method: str = "semantic",
        score_threshold: Optional[float] = None,
        rerank_top_k: Optional[int] = None
    ) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            retrieval_method: Method to use ("semantic", "hybrid")
            score_threshold: Minimum similarity score threshold
            rerank_top_k: Number of documents to rerank (if reranking enabled)
            
        Returns:
            List of relevant Document objects
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_manager.generate_embeddings([query])[0]
            
            # Retrieve based on method
            if retrieval_method == "hybrid" and isinstance(self.vector_store, HybridSearchVectorStore):
                # Use hybrid search if available
                search_results = self.vector_store.hybrid_search(
                    query=query,
                    query_embedding=query_embedding,
                    k=k * 2 if self.use_reranking else k
                )
            else:
                # Default semantic search
                search_results = self.vector_store.similarity_search(
                    query_embedding=query_embedding,
                    k=k * 2 if self.use_reranking else k,
                    score_threshold=score_threshold
                )
            
            # Convert to documents
            documents = []
            for doc_id, score, metadata in search_results:
                # Try to get full document from docstore
                doc = self.vector_store.docstore.search(doc_id)
                if doc:
                    # Add similarity score to metadata
                    enhanced_metadata = dict(doc.metadata)
                    enhanced_metadata['similarity_score'] = score
                    enhanced_metadata['retrieval_method'] = retrieval_method
                    doc.metadata = enhanced_metadata
                    documents.append(doc)
            
            # Apply reranking if enabled
            if self.use_reranking and self.reranker and len(documents) > 1:
                documents = self._rerank_documents(query, documents, rerank_top_k or k)
            
            # Return top-k documents
            return documents[:k]
            
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            return []
    
    def _rerank_documents(
        self,
        query: str,
        documents: List[Document],
        top_k: int
    ) -> List[Document]:
        """
        Rerank documents using cross-encoder model.
        
        Args:
            query: Original query
            documents: Documents to rerank
            top_k: Number of top documents to return
            
        Returns:
            Reranked list of documents
        """
        try:
            # Prepare query-document pairs
            pairs = [(query, doc.page_content) for doc in documents]
            
            # Get reranking scores
            rerank_scores = self.reranker.predict(pairs)
            
            # Combine documents with rerank scores
            scored_docs = []
            for doc, rerank_score in zip(documents, rerank_scores):
                # Update metadata with rerank score
                enhanced_metadata = dict(doc.metadata)
                enhanced_metadata['rerank_score'] = float(rerank_score)
                doc.metadata = enhanced_metadata
                scored_docs.append((doc, rerank_score))
            
            # Sort by rerank score
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Return top documents
            reranked_docs = [doc for doc, _ in scored_docs[:top_k]]
            
            logger.info(f"Reranked {len(documents)} documents, returning top {len(reranked_docs)}")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return documents[:top_k]
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about the retrieval system."""
        stats = {
            'embedding_model': self.embedding_manager.model_name,
            'embedding_dimension': self.embedding_manager.embedding_dim,
            'vector_store_stats': self.vector_store.get_stats(),
            'reranking_enabled': self.use_reranking,
        }
        
        if self.use_reranking and self.reranker:
            stats['reranker_model'] = getattr(self.reranker, 'model_name', 'unknown')
        
        return stats
    
    def save_index(self, path: Optional[str] = None):
        """Save the vector store index."""
        self.vector_store.save(path)
    
    def load_index(self, path: Optional[str] = None) -> bool:
        """Load the vector store index."""
        return self.vector_store.load(path)
    
    def clear_index(self):
        """Clear all indexed data."""
        self.vector_store.clear()


class MultiStrategyRetriever:
    """
    Advanced retriever that combines multiple retrieval strategies.
    """
    
    def __init__(
        self,
        primary_retriever: RetrievalEngine,
        strategies: Optional[List[str]] = None,
        fusion_method: str = "rrf"  # Reciprocal Rank Fusion
    ):
        self.primary_retriever = primary_retriever
        self.strategies = strategies or ["semantic", "hybrid"]
        self.fusion_method = fusion_method
        
        logger.info(f"Initialized multi-strategy retriever with strategies: {self.strategies}")
    
    def retrieve_with_fusion(
        self,
        query: str,
        k: int = 5,
        fusion_k: int = 10
    ) -> List[Document]:
        """
        Retrieve documents using multiple strategies and fuse results.
        
        Args:
            query: Search query
            k: Final number of documents to return
            fusion_k: Number of documents to retrieve per strategy before fusion
            
        Returns:
            Fused list of documents
        """
        all_results = {}  # doc_id -> {doc, scores_by_strategy}
        
        # Retrieve using each strategy
        for strategy in self.strategies:
            try:
                documents = self.primary_retriever.retrieve_documents(
                    query=query,
                    k=fusion_k,
                    retrieval_method=strategy
                )
                
                # Process results
                for rank, doc in enumerate(documents):
                    doc_key = self._get_document_key(doc)
                    
                    if doc_key not in all_results:
                        all_results[doc_key] = {
                            'document': doc,
                            'strategies': {},
                            'ranks': {}
                        }
                    
                    all_results[doc_key]['strategies'][strategy] = True
                    all_results[doc_key]['ranks'][strategy] = rank
                    
            except Exception as e:
                logger.error(f"Strategy {strategy} failed: {e}")
                continue
        
        # Fuse results
        if self.fusion_method == "rrf":
            fused_documents = self._reciprocal_rank_fusion(all_results, k)
        else:
            # Fallback to simple rank-based fusion
            fused_documents = self._simple_fusion(all_results, k)
        
        logger.info(f"Fused results from {len(self.strategies)} strategies, returning {len(fused_documents)} documents")
        return fused_documents
    
    def _get_document_key(self, doc: Document) -> str:
        """Generate unique key for document."""
        # Use content hash or metadata-based key
        title = doc.metadata.get('title', '')
        chunk_idx = doc.metadata.get('chunk_index', 0)
        return f"{title}_{chunk_idx}_{hash(doc.page_content[:100])}"
    
    def _reciprocal_rank_fusion(
        self,
        results: Dict[str, Dict],
        k: int,
        k_param: int = 60
    ) -> List[Document]:
        """Apply Reciprocal Rank Fusion to combine results."""
        scores = {}
        
        for doc_key, result_info in results.items():
            score = 0.0
            
            for strategy, rank in result_info['ranks'].items():
                # RRF formula: 1 / (k + rank)
                score += 1.0 / (k_param + rank + 1)
            
            scores[doc_key] = score
        
        # Sort by fused score
        sorted_results = sorted(
            results.items(),
            key=lambda x: scores[x[0]],
            reverse=True
        )
        
        # Return top-k documents
        return [result_info['document'] for _, result_info in sorted_results[:k]]
    
    def _simple_fusion(self, results: Dict[str, Dict], k: int) -> List[Document]:
        """Simple fusion based on number of strategies that retrieved each document."""
        # Score by number of strategies + average rank
        scores = {}
        
        for doc_key, result_info in results.items():
            num_strategies = len(result_info['strategies'])
            avg_rank = sum(result_info['ranks'].values()) / num_strategies
            
            # Higher score for more strategies, lower average rank
            scores[doc_key] = num_strategies - (avg_rank / 100.0)
        
        # Sort and return top-k
        sorted_results = sorted(
            results.items(),
            key=lambda x: scores[x[0]],
            reverse=True
        )
        
        return [result_info['document'] for _, result_info in sorted_results[:k]]
'''

# Save the retrieval module
with open('src/rag_system/retrieval/retriever.py', 'w') as f:
    f.write(retrieval_code)

print("✅ Retrieval coordinator module created")