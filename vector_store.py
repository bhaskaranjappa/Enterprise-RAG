"""
FAISS-based vector store for efficient similarity search and retrieval.
Supports different index types based on dataset size and includes persistence.
"""

import os
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import numpy as np

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document


logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    FAISS-based vector store with automatic index optimization based on dataset size.
    Supports persistence, metadata storage, and efficient similarity search.
    """

    def __init__(
        self,
        embedding_dimension: int = 384,  # all-MiniLM-L6-v2 default
        index_type: str = "auto",
        distance_metric: str = "cosine",
        nlist: int = 100,  # For IVF indices
        nprobe: int = 10,  # Search parameter for IVF
        store_path: Optional[str] = None
    ):
        self.embedding_dimension = embedding_dimension
        self.index_type = index_type
        self.distance_metric = distance_metric
        self.nlist = nlist
        self.nprobe = nprobe
        self.store_path = store_path or "data/vector_store"

        # Initialize components
        self.index = None
        self.docstore = InMemoryDocstore({})
        self.index_to_docstore_id = {}
        self.metadata_store = []
        self.is_trained = False

        # Create storage directory
        os.makedirs(self.store_path, exist_ok=True)

        logger.info(f"Initialized FAISS vector store with dimension {embedding_dimension}")

    def _create_index(self, n_vectors: int) -> faiss.Index:
        """
        Create appropriate FAISS index based on dataset size and configuration.

        Args:
            n_vectors: Number of vectors to be indexed

        Returns:
            FAISS index instance
        """
        if self.index_type == "auto":
            if n_vectors < 1000:
                index_type = "flat"
            elif n_vectors < 10000:
                index_type = "ivf_flat"
            else:
                index_type = "ivf_pq"
        else:
            index_type = self.index_type.lower()

        logger.info(f"Creating {index_type} index for {n_vectors} vectors")

        if self.distance_metric == "cosine":
            # For cosine similarity, we'll use inner product with normalized vectors
            measure = faiss.METRIC_INNER_PRODUCT
        else:
            measure = faiss.METRIC_L2

        if index_type == "flat":
            # Exact search - good for small datasets
            if measure == faiss.METRIC_INNER_PRODUCT:
                index = faiss.IndexFlatIP(self.embedding_dimension)
            else:
                index = faiss.IndexFlatL2(self.embedding_dimension)

        elif index_type == "ivf_flat":
            # IVF with flat quantizer - balance between speed and accuracy
            quantizer = faiss.IndexFlatL2(self.embedding_dimension)
            index = faiss.IndexIVFFlat(
                quantizer, 
                self.embedding_dimension, 
                min(self.nlist, n_vectors // 4)  # Adjust nlist based on dataset size
            )

        elif index_type == "ivf_pq":
            # IVF with product quantization - for large datasets
            quantizer = faiss.IndexFlatL2(self.embedding_dimension)

            # PQ parameters
            m = min(64, self.embedding_dimension // 4)  # Number of sub-vectors
            if self.embedding_dimension % m != 0:
                m = self.embedding_dimension // (self.embedding_dimension // m)

            index = faiss.IndexIVFPQ(
                quantizer,
                self.embedding_dimension,
                min(self.nlist, n_vectors // 4),
                m,
                8  # bits per sub-vector
            )

        else:
            raise ValueError(f"Unsupported index type: {index_type}")

        return index

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
        documents: Optional[List[Document]] = None
    ) -> List[str]:
        """
        Add embeddings and associated metadata to the vector store.

        Args:
            embeddings: Numpy array of embeddings
            metadata: List of metadata dictionaries
            documents: Optional list of Document objects

        Returns:
            List of document IDs
        """
        if len(embeddings) != len(metadata):
            raise ValueError("Number of embeddings must match number of metadata entries")

        # Normalize embeddings for cosine similarity
        if self.distance_metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-12)

        # Create or reinitialize index if needed
        if self.index is None:
            self.index = self._create_index(len(embeddings))

        # Train index if required
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            logger.info("Training index...")
            self.index.train(embeddings.astype(np.float32))
            self.is_trained = True

        # Generate document IDs
        start_idx = len(self.metadata_store)
        doc_ids = [f"doc_{start_idx + i}" for i in range(len(embeddings))]

        # Add to FAISS index
        self.index.add(embeddings.astype(np.float32))

        # Update metadata store and mappings
        for i, (doc_id, meta) in enumerate(zip(doc_ids, metadata)):
            faiss_idx = start_idx + i
            self.index_to_docstore_id[faiss_idx] = doc_id
            self.metadata_store.append(meta)

            # Add to docstore if documents provided
            if documents and i < len(documents):
                self.docstore.add({doc_id: documents[i]})

        # Set search parameters for IVF indices
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = min(self.nprobe, getattr(self.index, 'nlist', self.nprobe))

        logger.info(f"Added {len(embeddings)} embeddings to vector store")
        return doc_ids

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Perform similarity search against the vector store.

        Args:
            query_embedding: Query embedding vector
            k: Number of top results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            List of tuples (document_id, similarity_score, metadata)
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []

        # Normalize query embedding for cosine similarity
        if self.distance_metric == "cosine":
            query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Perform search
        try:
            scores, indices = self.index.search(
                query_embedding.astype(np.float32), 
                min(k, self.index.ntotal)
            )

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for empty results
                    continue

                # Apply score threshold if specified
                if score_threshold is not None and score < score_threshold:
                    continue

                doc_id = self.index_to_docstore_id.get(idx)
                if doc_id and idx < len(self.metadata_store):
                    metadata = self.metadata_store[idx]
                    results.append((doc_id, float(score), metadata))

            logger.info(f"Found {len(results)} results for similarity search")
            return results

        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []

    def get_relevant_documents(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """
        Get relevant documents based on similarity search.

        Args:
            query_embedding: Query embedding vector
            k: Number of documents to retrieve
            score_threshold: Minimum similarity score threshold

        Returns:
            List of relevant Document objects
        """
        search_results = self.similarity_search(query_embedding, k, score_threshold)

        documents = []
        for doc_id, score, metadata in search_results:
            # Try to get document from docstore
            doc = self.docstore.search(doc_id)
            if doc:
                # Add similarity score to metadata
                enhanced_metadata = dict(doc.metadata)
                enhanced_metadata['similarity_score'] = score
                doc.metadata = enhanced_metadata
                documents.append(doc)
            else:
                # Create document from metadata if not in docstore
                content = metadata.get('content', '')
                if content:
                    enhanced_metadata = dict(metadata)
                    enhanced_metadata['similarity_score'] = score
                    doc = Document(page_content=content, metadata=enhanced_metadata)
                    documents.append(doc)

        return documents

    def save(self, index_path: Optional[str] = None, metadata_path: Optional[str] = None):
        """
        Save the vector store to disk.

        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata
        """
        if self.index is None:
            logger.warning("No index to save")
            return

        index_path = index_path or os.path.join(self.store_path, "faiss_index")
        metadata_path = metadata_path or os.path.join(self.store_path, "metadata.pkl")

        try:
            # Save FAISS index
            faiss.write_index(self.index, index_path)

            # Save metadata and mappings
            metadata_dict = {
                'metadata_store': self.metadata_store,
                'index_to_docstore_id': self.index_to_docstore_id,
                'embedding_dimension': self.embedding_dimension,
                'distance_metric': self.distance_metric,
                'index_type': self.index_type,
                'is_trained': self.is_trained
            }

            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata_dict, f)

            logger.info(f"Saved vector store to {self.store_path}")

        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            raise

    def load(self, index_path: Optional[str] = None, metadata_path: Optional[str] = None):
        """
        Load the vector store from disk.

        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata file
        """
        index_path = index_path or os.path.join(self.store_path, "faiss_index")
        metadata_path = metadata_path or os.path.join(self.store_path, "metadata.pkl")

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            logger.warning("Vector store files not found")
            return False

        try:
            # Load FAISS index
            self.index = faiss.read_index(index_path)

            # Load metadata
            with open(metadata_path, 'rb') as f:
                metadata_dict = pickle.load(f)

            self.metadata_store = metadata_dict['metadata_store']
            self.index_to_docstore_id = metadata_dict['index_to_docstore_id']
            self.embedding_dimension = metadata_dict['embedding_dimension']
            self.distance_metric = metadata_dict['distance_metric']
            self.index_type = metadata_dict.get('index_type', 'auto')
            self.is_trained = metadata_dict.get('is_trained', True)

            # Set search parameters
            if hasattr(self.index, 'nprobe'):
                self.index.nprobe = min(self.nprobe, getattr(self.index, 'nlist', self.nprobe))

            logger.info(f"Loaded vector store with {self.index.ntotal} vectors")
            return True

        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        if self.index is None:
            return {'total_vectors': 0, 'is_trained': False}

        stats = {
            'total_vectors': self.index.ntotal,
            'embedding_dimension': self.embedding_dimension,
            'distance_metric': self.distance_metric,
            'index_type': type(self.index).__name__,
            'is_trained': self.is_trained
        }

        # Add index-specific stats
        if hasattr(self.index, 'nlist'):
            stats['nlist'] = self.index.nlist
        if hasattr(self.index, 'nprobe'):
            stats['nprobe'] = self.index.nprobe

        return stats

    def clear(self):
        """Clear all data from the vector store."""
        self.index = None
        self.docstore = InMemoryDocstore({})
        self.index_to_docstore_id = {}
        self.metadata_store = []
        self.is_trained = False
        logger.info("Cleared vector store")


class HybridSearchVectorStore(FAISSVectorStore):
    """
    Extended vector store that supports hybrid search combining semantic and keyword matching.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bm25_index = None

    def add_documents_with_bm25(
        self,
        documents: List[Document],
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]]
    ):
        """Add documents with both vector and BM25 indexing for hybrid search."""
        # Add to vector store
        doc_ids = self.add_embeddings(embeddings, metadata, documents)

        # Create BM25 index
        try:
            from rank_bm25 import BM25Okapi

            # Tokenize documents for BM25
            tokenized_docs = []
            for doc in documents:
                tokens = doc.page_content.lower().split()
                tokenized_docs.append(tokens)

            self.bm25_index = BM25Okapi(tokenized_docs)
            logger.info("Created BM25 index for hybrid search")

        except ImportError:
            logger.warning("rank_bm25 not available, hybrid search disabled")

        return doc_ids

    def hybrid_search(
        self,
        query: str,
        query_embedding: np.ndarray,
        k: int = 5,
        alpha: float = 0.7  # Weight for semantic vs keyword search
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Perform hybrid search combining semantic and keyword matching.

        Args:
            query: Text query for keyword matching
            query_embedding: Embedding vector for semantic search  
            k: Number of results to return
            alpha: Weight for semantic search (1-alpha for keyword)

        Returns:
            List of hybrid search results
        """
        if self.bm25_index is None:
            logger.warning("BM25 index not available, falling back to semantic search")
            return self.similarity_search(query_embedding, k)

        # Get semantic search results
        semantic_results = self.similarity_search(query_embedding, k * 2)

        # Get BM25 keyword results  
        query_tokens = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(query_tokens)

        # Normalize BM25 scores
        if len(bm25_scores) > 0:
            max_bm25 = max(bm25_scores)
            if max_bm25 > 0:
                bm25_scores = bm25_scores / max_bm25

        # Combine scores
        hybrid_results = {}

        # Add semantic results
        for doc_id, sem_score, metadata in semantic_results:
            hybrid_results[doc_id] = {
                'semantic_score': sem_score,
                'bm25_score': 0.0,
                'metadata': metadata
            }

        # Add BM25 results
        for idx, bm25_score in enumerate(bm25_scores):
            doc_id = self.index_to_docstore_id.get(idx)
            if doc_id:
                if doc_id in hybrid_results:
                    hybrid_results[doc_id]['bm25_score'] = bm25_score
                else:
                    hybrid_results[doc_id] = {
                        'semantic_score': 0.0,
                        'bm25_score': bm25_score,
                        'metadata': self.metadata_store[idx] if idx < len(self.metadata_store) else {}
                    }

        # Calculate hybrid scores and sort
        final_results = []
        for doc_id, scores in hybrid_results.items():
            hybrid_score = (
                alpha * scores['semantic_score'] + 
                (1 - alpha) * scores['bm25_score']
            )
            final_results.append((doc_id, hybrid_score, scores['metadata']))

        # Sort by hybrid score and return top-k
        final_results.sort(key=lambda x: x[1], reverse=True)
        return final_results[:k]
