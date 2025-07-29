"""
Main RAG system that coordinates all components: document processing, embedding, 
retrieval, and generation to provide end-to-end RAG functionality.
"""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

from langchain_core.documents import Document

from .document_processors.processor import DocumentProcessor
from .embeddings.embedding_manager import EmbeddingManager
from .retrieval.vector_store import FAISSVectorStore, HybridSearchVectorStore
from .retrieval.retriever import RetrievalEngine, MultiStrategyRetriever
from .generation.llm_manager import RAGGenerator, create_llm


logger = logging.getLogger(__name__)


class RAGSystem:
    """
    Complete RAG system that integrates document processing, embedding generation,
    vector storage, retrieval, and answer generation.
    """

    def __init__(
        self,
        llm_type: str = "ollama",
        llm_model: str = "llama2",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_store_path: str = "data/vector_store",
        use_hybrid_search: bool = False,
        use_reranking: bool = False,
        reranker_model: Optional[str] = None,
        chunk_size: int = 200,
        chunk_overlap: int = 50,
        **llm_kwargs
    ):
        """
        Initialize RAG system with specified configuration.

        Args:
            llm_type: Type of LLM ("ollama", "huggingface")
            llm_model: Name of the LLM model
            embedding_model: Name of the embedding model
            vector_store_path: Path to store vector index
            use_hybrid_search: Whether to enable hybrid search
            use_reranking: Whether to enable reranking
            reranker_model: Cross-encoder model for reranking
            chunk_size: Document chunk size in words
            chunk_overlap: Overlap between chunks in words
            **llm_kwargs: Additional arguments for LLM initialization
        """
        self.config = {
            'llm_type': llm_type,
            'llm_model': llm_model,
            'embedding_model': embedding_model,
            'vector_store_path': vector_store_path,
            'use_hybrid_search': use_hybrid_search,
            'use_reranking': use_reranking,
            'reranker_model': reranker_model,
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'initialized_at': datetime.now()
        }

        # Initialize components
        self._initialize_components(**llm_kwargs)

        # Load existing index if available
        self.load_index()

        logger.info("RAG system initialized successfully")

    def _initialize_components(self, **llm_kwargs):
        """Initialize all RAG components."""
        try:
            # Document processor
            self.document_processor = DocumentProcessor(
                chunk_size=self.config['chunk_size'],
                chunk_overlap=self.config['chunk_overlap']
            )

            # Embedding manager
            self.embedding_manager = EmbeddingManager(
                model_name=self.config['embedding_model']
            )

            # Vector store
            if self.config['use_hybrid_search']:
                self.vector_store = HybridSearchVectorStore(
                    embedding_dimension=self.embedding_manager.embedding_dim,
                    store_path=self.config['vector_store_path']
                )
            else:
                self.vector_store = FAISSVectorStore(
                    embedding_dimension=self.embedding_manager.embedding_dim,
                    store_path=self.config['vector_store_path']
                )

            # Retrieval engine
            self.retrieval_engine = RetrievalEngine(
                embedding_manager=self.embedding_manager,
                vector_store=self.vector_store,
                use_reranking=self.config['use_reranking'],
                reranker_model=self.config['reranker_model']
            )

            # Multi-strategy retriever (optional)
            if self.config['use_hybrid_search']:
                self.multi_retriever = MultiStrategyRetriever(self.retrieval_engine)
            else:
                self.multi_retriever = None

            # LLM and generator
            self.llm = create_llm(
                llm_type=self.config['llm_type'],
                model_name=self.config['llm_model'],
                **llm_kwargs
            )

            self.generator = RAGGenerator(self.llm)

            logger.info("All components initialized")

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    def ingest_documents(
        self,
        file_paths: List[str],
        batch_size: int = 50
    ) -> Dict[str, Any]:
        """
        Ingest and index documents from file paths.

        Args:
            file_paths: List of file paths to process
            batch_size: Number of documents to process in each batch

        Returns:
            Dictionary with ingestion results
        """
        start_time = datetime.now()
        results = {
            'total_files': len(file_paths),
            'successful_files': 0,
            'failed_files': 0,
            'total_chunks': 0,
            'errors': [],
            'processing_time': 0.0
        }

        try:
            # Process documents
            documents, errors = self.document_processor.process_documents(file_paths)

            if errors:
                results['errors'].extend(errors)
                results['failed_files'] = len(errors)

            results['successful_files'] = results['total_files'] - results['failed_files']
            results['total_chunks'] = len(documents)

            if not documents:
                logger.warning("No documents to index")
                return results

            # Index documents in batches
            indexed_count = 0
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]

                if self.retrieval_engine.index_documents(batch):
                    indexed_count += len(batch)
                else:
                    logger.error(f"Failed to index batch {i//batch_size + 1}")

            results['indexed_chunks'] = indexed_count

            # Save index
            self.save_index()

            # Calculate processing time
            end_time = datetime.now()
            results['processing_time'] = (end_time - start_time).total_seconds()

            logger.info(f"Ingested {results['successful_files']} files, "
                       f"{results['total_chunks']} chunks in {results['processing_time']:.2f}s")

            return results

        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            results['errors'].append(f"System error: {str(e)}")
            return results

    def query(
        self,
        question: str,
        k: int = 5,
        retrieval_method: str = "semantic",
        use_multi_strategy: bool = False,
        max_tokens: int = 200,
        temperature: float = 0.7,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Process a question and generate an answer using RAG.

        Args:
            question: User's question
            k: Number of documents to retrieve
            retrieval_method: Retrieval method ("semantic", "hybrid")
            use_multi_strategy: Whether to use multi-strategy retrieval
            max_tokens: Maximum tokens in generated answer
            temperature: Generation temperature
            include_sources: Whether to include source information

        Returns:
            Dictionary with answer and metadata
        """
        start_time = datetime.now()

        try:
            # Retrieve relevant documents
            if use_multi_strategy and self.multi_retriever:
                documents = self.multi_retriever.retrieve_with_fusion(question, k)
                method_used = "multi_strategy"
            else:
                documents = self.retrieval_engine.retrieve_documents(
                    query=question,
                    k=k,
                    retrieval_method=retrieval_method
                )
                method_used = retrieval_method

            # Generate answer
            response = self.generator.generate_answer(
                question=question,
                retrieved_documents=documents,
                max_tokens=max_tokens,
                temperature=temperature,
                include_sources=include_sources
            )

            # Add retrieval information
            response['retrieval_method'] = method_used
            response['processing_time'] = (datetime.now() - start_time).total_seconds()

            # Add document snippets if requested
            if include_sources and documents:
                response['document_snippets'] = [
                    {
                        'content': doc.page_content[:200] + "...",
                        'score': doc.metadata.get('similarity_score', 0.0),
                        'source': doc.metadata.get('title', 'Unknown')
                    }
                    for doc in documents[:3]  # Top 3 snippets
                ]

            logger.info(f"Processed query in {response['processing_time']:.2f}s")
            return response

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                'answer': "I'm sorry, I encountered an error while processing your question.",
                'question': question,
                'error': str(e),
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'retrieval_method': method_used if 'method_used' in locals() else 'unknown',
                'num_sources': 0
            }

    def query_stream(
        self,
        question: str,
        k: int = 5,
        retrieval_method: str = "semantic",
        max_tokens: int = 200,
        temperature: float = 0.7
    ):
        """
        Process a question and generate a streaming answer.

        Args:
            question: User's question
            k: Number of documents to retrieve
            retrieval_method: Retrieval method
            max_tokens: Maximum tokens in generated answer
            temperature: Generation temperature

        Yields:
            Streaming response chunks
        """
        try:
            # Retrieve documents
            documents = self.retrieval_engine.retrieve_documents(
                query=question,
                k=k,
                retrieval_method=retrieval_method
            )

            # Generate streaming answer
            for chunk in self.generator.generate_answer_stream(
                question=question,
                retrieved_documents=documents,
                max_tokens=max_tokens,
                temperature=temperature
            ):
                yield chunk

        except Exception as e:
            logger.error(f"Streaming query failed: {e}")
            yield "I'm sorry, I encountered an error while processing your question."

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        stats = {
            'config': dict(self.config),
            'retrieval_stats': self.retrieval_engine.get_retrieval_stats(),
            'system_status': {
                'components_initialized': True,
                'index_loaded': self.vector_store.index is not None,
                'total_indexed_documents': self.vector_store.index.ntotal if self.vector_store.index else 0
            }
        }

        return stats

    def save_index(self, path: Optional[str] = None):
        """Save the current vector index."""
        try:
            self.retrieval_engine.save_index(path)
            logger.info("Index saved successfully")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise

    def load_index(self, path: Optional[str] = None) -> bool:
        """Load vector index from disk."""
        try:
            success = self.retrieval_engine.load_index(path)
            if success:
                logger.info("Index loaded successfully")
            else:
                logger.info("No existing index found")
            return success
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False

    def clear_index(self):
        """Clear all indexed data."""
        try:
            self.retrieval_engine.clear_index()
            logger.info("Index cleared")
        except Exception as e:
            logger.error(f"Failed to clear index: {e}")
            raise

    def update_llm_config(self, **kwargs):
        """Update LLM configuration."""
        try:
            # Recreate LLM with new config
            self.llm = create_llm(
                llm_type=self.config['llm_type'],
                model_name=self.config['llm_model'],
                **kwargs
            )

            # Update generator
            self.generator = RAGGenerator(self.llm)

            logger.info("LLM configuration updated")

        except Exception as e:
            logger.error(f"Failed to update LLM config: {e}")
            raise

    def health_check(self) -> Dict[str, bool]:
        """Perform system health check."""
        checks = {}

        try:
            # Check document processor
            checks['document_processor'] = self.document_processor is not None

            # Check embedding manager
            checks['embedding_manager'] = self.embedding_manager.model is not None

            # Check vector store
            checks['vector_store'] = self.vector_store is not None

            # Check LLM
            try:
                test_response = self.llm.generate("Test", max_tokens=5, temperature=0.1)
                checks['llm'] = len(test_response.strip()) > 0
            except:
                checks['llm'] = False

            # Overall health
            checks['overall_healthy'] = all(checks.values())

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            checks['overall_healthy'] = False

        return checks
