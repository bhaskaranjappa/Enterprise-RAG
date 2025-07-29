"""
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
    complex_text = "\n".join([
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
