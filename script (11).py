# Create Dockerfile
dockerfile_content = '''# Multi-stage build for efficient Docker image
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PIP_NO_CACHE_DIR=1 \\
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY tests/ ./tests/
COPY pytest.ini .

# Create necessary directories
RUN mkdir -p data/uploaded_docs data/vector_store logs

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "src.rag_system.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
'''

with open('Dockerfile', 'w') as f:
    f.write(dockerfile_content)

print("✅ Dockerfile created")

# Create docker-compose.yml
compose_content = '''version: '3.8'

services:
  rag-system:
    build: .
    container_name: rag-document-search
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - LLM_TYPE=ollama
      - LLM_MODEL=llama2
      - OLLAMA_BASE_URL=http://ollama:11434
      - EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
      - USE_HYBRID_SEARCH=false
      - USE_RERANKING=false
    depends_on:
      - ollama
    restart: unless-stopped
    networks:
      - rag-network

  ollama:
    image: ollama/ollama:latest
    container_name: ollama-server
    ports:
      - "11434:11434"
    volumes:
      - ollama-data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0
    restart: unless-stopped
    networks:
      - rag-network
    # Uncomment for GPU support
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]

  # Optional: Redis for caching (future enhancement)
  redis:
    image: redis:7-alpine
    container_name: rag-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped
    networks:
      - rag-network
    command: redis-server --appendonly yes

volumes:
  ollama-data:
  redis-data:

networks:
  rag-network:
    driver: bridge
'''

with open('docker-compose.yml', 'w') as f:
    f.write(compose_content)

print("✅ Docker Compose configuration created")

# Create main application entry point
main_app_content = '''#!/usr/bin/env python3
"""
Main entry point for the RAG Document Search System.
Provides CLI interface for running the system and managing components.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.rag_system.core import RAGSystem
from src.rag_system.api.app import app
import uvicorn


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    os.makedirs("logs", exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/rag_system.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server."""
    print(f"Starting RAG system server on {host}:{port}")
    
    # Ensure required directories exist
    os.makedirs("data/uploaded_docs", exist_ok=True)
    os.makedirs("data/vector_store", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    uvicorn.run(
        "src.rag_system.api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


def ingest_documents(file_paths: list, llm_type: str = "ollama", llm_model: str = "llama2"):
    """Ingest documents into the system via CLI."""
    print(f"Ingesting {len(file_paths)} documents...")
    
    try:
        # Initialize RAG system
        rag_system = RAGSystem(
            llm_type=llm_type,
            llm_model=llm_model
        )
        
        # Process documents
        results = rag_system.ingest_documents(file_paths)
        
        print(f"Ingestion completed:")
        print(f"  - Successful files: {results['successful_files']}")
        print(f"  - Failed files: {results['failed_files']}")
        print(f"  - Total chunks: {results['total_chunks']}")
        print(f"  - Processing time: {results['processing_time']:.2f}s")
        
        if results['errors']:
            print("Errors:")
            for error in results['errors']:
                print(f"  - {error}")
    
    except Exception as e:
        print(f"Error during ingestion: {e}")
        sys.exit(1)


def query_system(question: str, k: int = 5, llm_type: str = "ollama", llm_model: str = "llama2"):
    """Query the system via CLI."""
    print(f"Processing query: {question}")
    
    try:
        # Initialize RAG system
        rag_system = RAGSystem(
            llm_type=llm_type,
            llm_model=llm_model
        )
        
        # Process query
        response = rag_system.query(
            question=question,
            k=k,
            include_sources=True
        )
        
        print(f"\\nAnswer: {response['answer']}")
        print(f"Sources used: {response['num_sources']}")
        print(f"Processing time: {response['processing_time']:.2f}s")
        
        if response.get('sources'):
            print("\\nSources:")
            for i, source in enumerate(response['sources'], 1):
                print(f"  {i}. {source.get('title', 'Unknown')} "
                     f"(Relevance: {source.get('relevance_score', 0):.3f})")
    
    except Exception as e:
        print(f"Error during query: {e}")
        sys.exit(1)


def clear_index():
    """Clear the document index."""
    print("Clearing document index...")
    
    try:
        rag_system = RAGSystem()
        rag_system.clear_index()
        print("Index cleared successfully.")
    
    except Exception as e:
        print(f"Error clearing index: {e}")
        sys.exit(1)


def system_stats(llm_type: str = "ollama", llm_model: str = "llama2"):
    """Display system statistics."""
    try:
        rag_system = RAGSystem(
            llm_type=llm_type,
            llm_model=llm_model
        )
        
        stats = rag_system.get_system_stats()
        
        print("RAG System Statistics:")
        print(f"  - Total indexed documents: {stats['system_status']['total_indexed_documents']}")
        print(f"  - Embedding model: {stats['retrieval_stats']['embedding_model']}")
        print(f"  - Vector store: {stats['retrieval_stats']['vector_store_stats']['index_type']}")
        
        if stats['config']:
            print("\\nConfiguration:")
            for key, value in stats['config'].items():
                if key != 'initialized_at':
                    print(f"  - {key}: {value}")
    
    except Exception as e:
        print(f"Error getting system stats: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAG Document Search System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run the API server
  python main.py server --port 8000
  
  # Ingest documents
  python main.py ingest document1.pdf document2.txt
  
  # Query the system
  python main.py query "What is machine learning?"
  
  # View system statistics
  python main.py stats
  
  # Clear the index
  python main.py clear
        """
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level"
    )
    
    parser.add_argument(
        "--llm-type",
        choices=["ollama", "huggingface"],
        default="ollama",
        help="Type of LLM to use"
    )
    
    parser.add_argument(
        "--llm-model",
        default="llama2",
        help="Name of the LLM model to use"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Run the API server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    server_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    server_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("files", nargs="+", help="Files to ingest")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the system")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("-k", type=int, default=5, help="Number of results to retrieve")
    
    # Stats command
    subparsers.add_parser("stats", help="Show system statistics")
    
    # Clear command
    subparsers.add_parser("clear", help="Clear the document index")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute commands
    if args.command == "server":
        run_server(args.host, args.port, args.reload)
    elif args.command == "ingest":
        ingest_documents(args.files, args.llm_type, args.llm_model)
    elif args.command == "query":
        query_system(args.question, args.k, args.llm_type, args.llm_model)
    elif args.command == "stats":
        system_stats(args.llm_type, args.llm_model)
    elif args.command == "clear":
        clear_index()


if __name__ == "__main__":
    main()
'''

with open('main.py', 'w') as f:
    f.write(main_app_content)

# Make it executable
os.chmod('main.py', 0o755)

print("✅ Main application entry point created")