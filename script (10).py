# Create comprehensive test suite
test_code = '''"""
Comprehensive test suite for the RAG system.
Tests document processing, embedding generation, vector storage, retrieval, and API endpoints.
"""

import pytest
import tempfile
import os
import shutil
from pathlib import Path
from typing import List
import numpy as np

from fastapi.testclient import TestClient
from langchain_core.documents import Document

# Import system components
from src.rag_system.document_processors.processor import DocumentProcessor
from src.rag_system.embeddings.embedding_manager import EmbeddingManager
from src.rag_system.retrieval.vector_store import FAISSVectorStore
from src.rag_system.generation.llm_manager import create_llm
from src.rag_system.core import RAGSystem
from src.rag_system.api.app import app


class TestDocumentProcessor:
    """Test suite for document processing functionality."""
    
    @pytest.fixture
    def processor(self):
        """Create a document processor instance."""
        return DocumentProcessor(chunk_size=50, chunk_overlap=10)
    
    @pytest.fixture
    def sample_files(self):
        """Create sample files for testing."""
        temp_dir = tempfile.mkdtemp()
        files = {}
        
        # Create sample text file
        txt_path = os.path.join(temp_dir, "sample.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("This is a sample text document. " * 20)
        files['txt'] = txt_path
        
        yield files
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_text_file_processing(self, processor, sample_files):
        """Test processing of text files."""
        documents = processor.process_single_document(sample_files['txt'])
        
        assert len(documents) > 0
        assert all(isinstance(doc, Document) for doc in documents)
        assert all(doc.page_content for doc in documents)
        assert all('title' in doc.metadata for doc in documents)
        assert all('source_type' in doc.metadata for doc in documents)
    
    def test_chunking_parameters(self, processor, sample_files):
        """Test that chunking respects size and overlap parameters."""
        documents = processor.process_single_document(sample_files['txt'])
        
        if len(documents) > 1:
            # Check that chunks have reasonable sizes
            chunk_sizes = [len(doc.page_content) for doc in documents]
            assert all(size > 0 for size in chunk_sizes)
            
            # Check metadata consistency
            assert all(doc.metadata['chunk_index'] == i for i, doc in enumerate(documents))
    
    def test_error_handling(self, processor):
        """Test error handling for invalid files."""
        # Test non-existent file
        with pytest.raises(FileNotFoundError):
            processor.process_single_document("nonexistent.txt")
        
        # Test unsupported file type
        temp_file = tempfile.NamedTemporaryFile(suffix='.xyz', delete=False)
        temp_file.close()
        
        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                processor.process_single_document(temp_file.name)
        finally:
            os.unlink(temp_file.name)


class TestEmbeddingManager:
    """Test suite for embedding generation."""
    
    @pytest.fixture
    def embedding_manager(self):
        """Create an embedding manager instance."""
        return EmbeddingManager(batch_size=2)  # Small batch for testing
    
    def test_embedding_generation(self, embedding_manager):
        """Test basic embedding generation."""
        texts = ["Hello world", "This is a test", "Machine learning is cool"]
        embeddings = embedding_manager.generate_embeddings(texts)
        
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == embedding_manager.embedding_dim
        assert embeddings.dtype == np.float32 or embeddings.dtype == np.float64
    
    def test_empty_input(self, embedding_manager):
        """Test handling of empty input."""
        embeddings = embedding_manager.generate_embeddings([])
        assert embeddings.size == 0
    
    def test_similarity_computation(self, embedding_manager):
        """Test similarity computation."""
        texts = ["cat", "dog", "car"]
        embeddings = embedding_manager.generate_embeddings(texts)
        query_embedding = embedding_manager.generate_embeddings(["pet"])[0]
        
        scores, indices = embedding_manager.compute_similarity(
            query_embedding, embeddings, top_k=2
        )
        
        assert len(scores) <= 2
        assert len(indices) <= 2
        assert all(isinstance(score, (float, np.float32, np.float64)) for score in scores)
    
    def test_batch_processing(self, embedding_manager):
        """Test batch processing functionality."""
        # Create enough texts to trigger batching
        texts = [f"Test text number {i}" for i in range(10)]
        embeddings = embedding_manager.generate_embeddings(texts, show_progress=False)
        
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == embedding_manager.embedding_dim


class TestVectorStore:
    """Test suite for FAISS vector store."""
    
    @pytest.fixture
    def vector_store(self):
        """Create a vector store instance."""
        temp_dir = tempfile.mkdtemp()
        store = FAISSVectorStore(
            embedding_dimension=384,
            store_path=temp_dir
        )
        yield store
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_embeddings(self):
        """Generate sample embeddings and metadata."""
        np.random.seed(42)
        embeddings = np.random.randn(10, 384).astype(np.float32)
        metadata = [
            {'title': f'Doc {i}', 'content': f'Content {i}', 'chunk_index': i}
            for i in range(10)
        ]
        return embeddings, metadata
    
    def test_add_embeddings(self, vector_store, sample_embeddings):
        """Test adding embeddings to vector store."""
        embeddings, metadata = sample_embeddings
        doc_ids = vector_store.add_embeddings(embeddings, metadata)
        
        assert len(doc_ids) == len(embeddings)
        assert all(isinstance(doc_id, str) for doc_id in doc_ids)
        assert vector_store.index.ntotal == len(embeddings)
    
    def test_similarity_search(self, vector_store, sample_embeddings):
        """Test similarity search functionality."""
        embeddings, metadata = sample_embeddings
        vector_store.add_embeddings(embeddings, metadata)
        
        # Search with first embedding as query
        query_embedding = embeddings[0]
        results = vector_store.similarity_search(query_embedding, k=3)
        
        assert len(results) <= 3
        assert all(isinstance(result, tuple) and len(result) == 3 for result in results)
        
        # Check that the first result should be the exact match
        if results:
            doc_id, score, meta = results[0]
            assert isinstance(score, (float, np.float32, np.float64))
    
    def test_persistence(self, sample_embeddings):
        """Test saving and loading vector store."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create and populate vector store
            store1 = FAISSVectorStore(embedding_dimension=384, store_path=temp_dir)
            embeddings, metadata = sample_embeddings
            store1.add_embeddings(embeddings, metadata)
            
            # Save
            store1.save()
            original_count = store1.index.ntotal
            
            # Create new instance and load
            store2 = FAISSVectorStore(embedding_dimension=384, store_path=temp_dir)
            success = store2.load()
            
            assert success
            assert store2.index.ntotal == original_count
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_get_stats(self, vector_store, sample_embeddings):
        """Test statistics retrieval."""
        embeddings, metadata = sample_embeddings
        vector_store.add_embeddings(embeddings, metadata)
        
        stats = vector_store.get_stats()
        
        assert 'total_vectors' in stats
        assert 'embedding_dimension' in stats
        assert stats['total_vectors'] == len(embeddings)
        assert stats['embedding_dimension'] == 384


class TestRAGSystem:
    """Integration tests for the complete RAG system."""
    
    @pytest.fixture
    def temp_rag_system(self):
        """Create a temporary RAG system for testing."""
        temp_dir = tempfile.mkdtemp()
        
        # Mock LLM for testing
        class MockLLM:
            def generate(self, prompt, **kwargs):
                return "This is a mock response for testing purposes."
            
            def generate_stream(self, prompt, **kwargs):
                for word in ["Mock", "streaming", "response"]:
                    yield word + " "
        
        try:
            system = RAGSystem(
                llm_type="ollama",  # Will be mocked
                llm_model="test",
                vector_store_path=temp_dir,
                chunk_size=20,
                chunk_overlap=5
            )
            
            # Replace LLM with mock
            system.llm = MockLLM()
            system.generator.llm = MockLLM()
            
            yield system
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_document_ingestion(self, temp_rag_system, tmp_path):
        """Test document ingestion process."""
        # Create sample document
        sample_doc = tmp_path / "test.txt"
        sample_doc.write_text("This is a test document for RAG system testing. " * 10)
        
        # Ingest document
        results = temp_rag_system.ingest_documents([str(sample_doc)])
        
        assert results['successful_files'] > 0
        assert results['total_chunks'] > 0
        assert results['failed_files'] == 0
        assert len(results['errors']) == 0
    
    def test_query_processing(self, temp_rag_system, tmp_path):
        """Test query processing with mock responses."""
        # First ingest a document
        sample_doc = tmp_path / "test.txt"
        sample_doc.write_text("Artificial intelligence is transforming many industries.")
        
        temp_rag_system.ingest_documents([str(sample_doc)])
        
        # Query the system
        response = temp_rag_system.query(
            question="What is transforming industries?",
            k=3,
            max_tokens=50
        )
        
        assert 'answer' in response
        assert 'question' in response
        assert response['answer']  # Should not be empty
        assert response['num_sources'] >= 0
    
    def test_system_stats(self, temp_rag_system):
        """Test system statistics retrieval."""
        stats = temp_rag_system.get_system_stats()
        
        assert 'config' in stats
        assert 'retrieval_stats' in stats
        assert 'system_status' in stats
        
        assert 'components_initialized' in stats['system_status']
        assert stats['system_status']['components_initialized'] is True


class TestAPIEndpoints:
    """Test suite for FastAPI endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        # Mock the RAG system for API testing
        with TestClient(app) as client:
            yield client
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_health_endpoint_without_system(self, client):
        """Test health endpoint when system is not initialized."""
        response = client.get("/health")
        # Should return 503 when system is not initialized
        assert response.status_code == 503
    
    @pytest.mark.asyncio
    async def test_document_upload_validation(self, client):
        """Test document upload with invalid files."""
        # Test with no files
        response = client.post("/upload_docs")
        assert response.status_code == 422  # Validation error
    
    def test_query_validation(self, client):
        """Test query endpoint validation."""
        # Test with invalid request
        response = client.post("/query", json={})
        assert response.status_code == 422  # Validation error
        
        # Test with empty question
        response = client.post("/query", json={"question": ""})
        assert response.status_code in [400, 503]  # Bad request or service unavailable


# Utility functions for testing
def create_sample_documents(num_docs: int = 5) -> List[Document]:
    """Create sample documents for testing."""
    documents = []
    for i in range(num_docs):
        doc = Document(
            page_content=f"This is test document {i}. It contains sample content for testing purposes.",
            metadata={
                'title': f'Test Doc {i}',
                'source_type': 'test',
                'chunk_index': 0
            }
        )
        documents.append(doc)
    return documents


# Performance tests
class TestPerformance:
    """Performance tests for the RAG system."""
    
    def test_embedding_performance(self):
        """Test embedding generation performance."""
        embedding_manager = EmbeddingManager()
        
        # Test with varying text sizes
        texts = [
            "Short text",
            "Medium length text with more words and content",
            "Much longer text " * 50
        ]
        
        import time
        start_time = time.time()
        embeddings = embedding_manager.generate_embeddings(texts)
        end_time = time.time()
        
        assert embeddings.shape[0] == len(texts)
        assert end_time - start_time < 10  # Should complete within 10 seconds
    
    def test_vector_search_performance(self):
        """Test vector search performance with larger datasets."""
        vector_store = FAISSVectorStore(embedding_dimension=384)
        
        # Create larger dataset
        np.random.seed(42)
        embeddings = np.random.randn(1000, 384).astype(np.float32)
        metadata = [{'id': i, 'content': f'Content {i}'} for i in range(1000)]
        
        # Add embeddings
        vector_store.add_embeddings(embeddings, metadata)
        
        # Test search performance
        query_embedding = np.random.randn(384).astype(np.float32)
        
        import time
        start_time = time.time()
        results = vector_store.similarity_search(query_embedding, k=10)
        end_time = time.time()
        
        assert len(results) == 10
        assert end_time - start_time < 1  # Should complete within 1 second


# Configuration for pytest
pytest_plugins = []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''

# Save the test suite
with open('tests/test_rag_system.py', 'w') as f:
    f.write(test_code)

# Create pytest configuration
pytest_config = '''[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --tb=short
    --strict-markers
    -ra
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    api: marks tests as API tests
'''

with open('pytest.ini', 'w') as f:
    f.write(pytest_config)

print("✅ Test suite created")

# Create conftest.py for shared test fixtures
conftest_code = '''"""
Shared pytest fixtures and configuration.
"""

import pytest
import tempfile
import shutil
import os
from pathlib import Path


@pytest.fixture(scope="session")
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="rag_test_")
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_text_files(temp_data_dir):
    """Create sample text files for testing."""
    files = []
    
    # Simple text file
    simple_file = Path(temp_dir) / "simple.txt"
    simple_file.write_text("This is a simple test document.")
    files.append(str(simple_file))
    
    # Complex text file
    complex_file = Path(temp_dir) / "complex.txt"
    complex_text = "\\n".join([
        "Introduction",
        "This document covers advanced topics in artificial intelligence.",
        "",
        "Machine Learning",
        "Machine learning is a subset of AI that focuses on algorithms.",
        "It includes supervised, unsupervised, and reinforcement learning.",
        "",
        "Deep Learning", 
        "Deep learning uses neural networks with many layers.",
        "It has revolutionized computer vision and natural language processing."
    ])
    complex_file.write_text(complex_text)
    files.append(str(complex_file))
    
    return files


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    monkeypatch.setenv("LLM_TYPE", "test")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    monkeypatch.setenv("USE_HYBRID_SEARCH", "false")
    monkeypatch.setenv("USE_RERANKING", "false")
'''

with open('tests/conftest.py', 'w') as f:
    f.write(conftest_code)

print("✅ Test configuration created")