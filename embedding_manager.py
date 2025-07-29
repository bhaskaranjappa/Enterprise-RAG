"""
Embedding generation and management module for RAG system.
Uses sentence-transformers for efficient embedding generation with batch processing.
"""

import os
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

from sentence_transformers import SentenceTransformer
import torch

from langchain_core.documents import Document


logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages embedding generation with support for batch processing and caching.
    Uses sentence-transformers/all-MiniLM-L6-v2 for efficient embedding generation.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 100,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_dir = cache_dir or "data/embedding_cache"

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Initializing embedding model on device: {self.device}")

        # Initialize model
        self._load_model()

        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)

    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device
            )

            # Get embedding dimension
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def generate_embeddings(
        self, 
        texts: List[str],
        normalize: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts with batch processing.

        Args:
            texts: List of text strings to embed
            normalize: Whether to normalize embeddings for cosine similarity
            show_progress: Whether to show progress bar

        Returns:
            Numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])

        try:
            # Generate embeddings in batches for memory efficiency
            all_embeddings = []

            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]

                # Generate embeddings for batch
                batch_embeddings = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    normalize_embeddings=normalize,
                    show_progress_bar=show_progress and i == 0
                )

                all_embeddings.append(batch_embeddings)

                if show_progress and len(texts) > self.batch_size:
                    logger.info(f"Processed {min(i + self.batch_size, len(texts))}/{len(texts)} texts")

            # Concatenate all batches
            embeddings = np.vstack(all_embeddings)

            logger.info(f"Generated {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}")
            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    def embed_documents(self, documents: List[Document]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Generate embeddings for Document objects.

        Args:
            documents: List of Document objects

        Returns:
            Tuple of (embeddings array, list of metadata dicts)
        """
        texts = [doc.page_content for doc in documents]
        metadata_list = [doc.metadata for doc in documents]

        embeddings = self.generate_embeddings(texts)

        return embeddings, metadata_list

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
        file_path: str
    ):
        """
        Save embeddings and metadata to disk.

        Args:
            embeddings: Numpy array of embeddings
            metadata: List of metadata dictionaries
            file_path: Path to save the data
        """
        try:
            data = {
                'embeddings': embeddings,
                'metadata': metadata,
                'model_name': self.model_name,
                'embedding_dim': self.embedding_dim
            }

            with open(file_path, 'wb') as f:
                pickle.dump(data, f)

            logger.info(f"Saved embeddings to {file_path}")

        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
            raise

    def load_embeddings(self, file_path: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Load embeddings and metadata from disk.

        Args:
            file_path: Path to the saved embeddings

        Returns:
            Tuple of (embeddings array, metadata list)
        """
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            # Verify model compatibility
            if data.get('model_name') != self.model_name:
                logger.warning(f"Model mismatch: saved with {data.get('model_name')}, current: {self.model_name}")

            embeddings = data['embeddings']
            metadata = data['metadata']

            logger.info(f"Loaded {embeddings.shape[0]} embeddings from {file_path}")
            return embeddings, metadata

        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            raise

    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray,
        top_k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute cosine similarity between query and document embeddings.

        Args:
            query_embedding: Query embedding vector
            document_embeddings: Matrix of document embeddings
            top_k: Number of top results to return

        Returns:
            Tuple of (similarity scores, indices of top matches)
        """
        try:
            # Ensure query embedding is 2D
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)

            # Compute cosine similarity
            similarities = np.dot(query_embedding, document_embeddings.T).flatten()

            # Get top-k indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            top_scores = similarities[top_indices]

            return top_scores, top_indices

        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            raise

    def optimize_for_inference(self):
        """Optimize model for inference (disable gradients, use half precision if supported)."""
        try:
            self.model.eval()

            # Use half precision on GPU if available
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.half()
                logger.info("Enabled half precision for GPU inference")

        except Exception as e:
            logger.warning(f"Could not optimize for inference: {e}")


class HybridEmbedding:
    """
    Experimental hybrid embedding approach combining multiple embedding strategies.
    """

    def __init__(
        self,
        semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.7
    ):
        self.semantic_model = EmbeddingManager(semantic_model)
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight

        # Validate weights
        if abs(keyword_weight + semantic_weight - 1.0) > 1e-6:
            raise ValueError("Keyword and semantic weights must sum to 1.0")

    def create_hybrid_embeddings(
        self,
        texts: List[str],
        use_tfidf: bool = True
    ) -> np.ndarray:
        """
        Create hybrid embeddings combining semantic and keyword-based representations.

        Args:
            texts: List of input texts
            use_tfidf: Whether to include TF-IDF based keyword embeddings

        Returns:
            Hybrid embedding matrix
        """
        # Generate semantic embeddings
        semantic_embeddings = self.semantic_model.generate_embeddings(texts)

        if not use_tfidf:
            return semantic_embeddings

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            # Generate keyword-based embeddings using TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=semantic_embeddings.shape[1],  # Match semantic dimension
                stop_words='english',
                ngram_range=(1, 2)
            )

            tfidf_embeddings = vectorizer.fit_transform(texts).toarray()

            # Pad or truncate TF-IDF to match semantic embedding dimension
            if tfidf_embeddings.shape[1] < semantic_embeddings.shape[1]:
                padding = np.zeros((
                    tfidf_embeddings.shape[0],
                    semantic_embeddings.shape[1] - tfidf_embeddings.shape[1]
                ))
                tfidf_embeddings = np.hstack([tfidf_embeddings, padding])
            elif tfidf_embeddings.shape[1] > semantic_embeddings.shape[1]:
                tfidf_embeddings = tfidf_embeddings[:, :semantic_embeddings.shape[1]]

            # Normalize both embedding types
            semantic_embeddings = semantic_embeddings / np.linalg.norm(
                semantic_embeddings, axis=1, keepdims=True
            )
            tfidf_embeddings = tfidf_embeddings / np.linalg.norm(
                tfidf_embeddings, axis=1, keepdims=True
            )

            # Combine embeddings
            hybrid_embeddings = (
                self.semantic_weight * semantic_embeddings +
                self.keyword_weight * tfidf_embeddings
            )

            return hybrid_embeddings

        except ImportError:
            logger.warning("scikit-learn not available, falling back to semantic embeddings only")
            return semantic_embeddings
        except Exception as e:
            logger.error(f"Error creating hybrid embeddings: {e}")
            return semantic_embeddings
