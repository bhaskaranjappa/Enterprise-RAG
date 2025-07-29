"""
FastAPI application for RAG system with document upload, querying, and management endpoints.
Provides RESTful API interface for all RAG functionality.
"""

import os
import logging
import asyncio
import tempfile
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import json

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from ..core import RAGSystem


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/rag_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Request model for querying the RAG system."""
    question: str = Field(..., description="The question to ask")
    k: int = Field(5, description="Number of documents to retrieve", ge=1, le=20)
    retrieval_method: str = Field("semantic", description="Retrieval method")
    use_multi_strategy: bool = Field(False, description="Use multi-strategy retrieval")
    max_tokens: int = Field(200, description="Maximum tokens in response", ge=10, le=1000)
    temperature: float = Field(0.7, description="Generation temperature", ge=0.0, le=2.0)
    include_sources: bool = Field(True, description="Include source information")


class QueryResponse(BaseModel):
    """Response model for queries."""
    answer: str
    question: str
    retrieval_method: str
    processing_time: float
    num_sources: int
    context_used: bool
    sources: Optional[List[Dict[str, Any]]] = None
    document_snippets: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


class DocumentUploadResponse(BaseModel):
    """Response model for document uploads."""
    total_files: int
    successful_files: int
    failed_files: int
    total_chunks: int
    indexed_chunks: int
    processing_time: float
    errors: List[str]
    message: str


class SystemStatsResponse(BaseModel):
    """Response model for system statistics."""
    config: Dict[str, Any]
    retrieval_stats: Dict[str, Any]
    system_status: Dict[str, Any]


class HealthCheckResponse(BaseModel):
    """Response model for health checks."""
    status: str
    timestamp: datetime
    checks: Dict[str, bool]


# Global RAG system instance
rag_system: Optional[RAGSystem] = None


def get_rag_system() -> RAGSystem:
    """Dependency to get the RAG system instance."""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    return rag_system


# FastAPI app initialization
app = FastAPI(
    title="RAG Document Search System",
    description="A complete RAG system for document ingestion, indexing, and intelligent querying",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup."""
    global rag_system

    try:
        # Initialize with default configuration
        # In production, load from environment variables or config file
        rag_system = RAGSystem(
            llm_type=os.getenv("LLM_TYPE", "ollama"),
            llm_model=os.getenv("LLM_MODEL", "llama2"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            use_hybrid_search=os.getenv("USE_HYBRID_SEARCH", "false").lower() == "true",
            use_reranking=os.getenv("USE_RERANKING", "false").lower() == "true",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")  # For Ollama
        )

        logger.info("RAG system initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        # Don't raise here - let the app start but endpoints will return 503


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global rag_system

    if rag_system:
        try:
            # Save index before shutdown
            rag_system.save_index()
            logger.info("RAG system shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# API Endpoints

@app.get("/", summary="Root endpoint")
async def root():
    """Root endpoint with basic system information."""
    return {
        "message": "RAG Document Search System API",
        "version": "1.0.0",
        "status": "online",
        "timestamp": datetime.now(),
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthCheckResponse, summary="Health check")
async def health_check(rag_system: RAGSystem = Depends(get_rag_system)):
    """Perform comprehensive health check of all system components."""
    try:
        checks = rag_system.health_check()
        status = "healthy" if checks.get("overall_healthy", False) else "unhealthy"

        return HealthCheckResponse(
            status=status,
            timestamp=datetime.now(),
            checks=checks
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status="error",
            timestamp=datetime.now(),
            checks={"error": str(e)}
        )


@app.post("/upload_docs", response_model=DocumentUploadResponse, summary="Upload and index documents")
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    rag_system: RAGSystem = Depends(get_rag_system)
):
    """
    Upload and index multiple documents (PDF, DOCX, or TXT).
    Documents are processed asynchronously and indexed into the vector store.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # Validate file types
    allowed_extensions = {'.pdf', '.docx', '.txt'}
    invalid_files = []

    for file in files:
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in allowed_extensions:
            invalid_files.append(f"{file.filename}: {file_extension}")

    if invalid_files:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file types: {invalid_files}. Allowed: {list(allowed_extensions)}"
        )

    # Save uploaded files temporarily
    temp_files = []
    try:
        upload_dir = Path("data/uploaded_docs")
        upload_dir.mkdir(exist_ok=True)

        for file in files:
            # Create unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = f"{timestamp}_{file.filename}"
            file_path = upload_dir / safe_filename

            # Save file
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)

            temp_files.append(str(file_path))
            logger.info(f"Saved uploaded file: {file_path}")

        # Process documents
        results = rag_system.ingest_documents(temp_files)

        # Add success message
        if results['successful_files'] > 0:
            results['message'] = f"Successfully processed {results['successful_files']} files"
        else:
            results['message'] = "No files were processed successfully"

        logger.info(f"Document upload completed: {results['message']}")

        return DocumentUploadResponse(**results)

    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

    finally:
        # Cleanup temporary files in background
        background_tasks.add_task(cleanup_temp_files, temp_files)


@app.post("/query", response_model=QueryResponse, summary="Query the RAG system")
async def query_documents(
    request: QueryRequest,
    rag_system: RAGSystem = Depends(get_rag_system)
):
    """
    Query the RAG system with a natural language question.
    Returns an AI-generated answer based on the indexed documents.
    """
    try:
        # Validate request
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        # Process query
        response = rag_system.query(
            question=request.question,
            k=request.k,
            retrieval_method=request.retrieval_method,
            use_multi_strategy=request.use_multi_strategy,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            include_sources=request.include_sources
        )

        logger.info(f"Query processed: '{request.question[:50]}...' in {response.get('processing_time', 0):.2f}s")

        return QueryResponse(**response)

    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/query_stream", summary="Stream query response")
async def query_documents_stream(
    request: QueryRequest,
    rag_system: RAGSystem = Depends(get_rag_system)
):
    """
    Query the RAG system with streaming response.
    Returns Server-Sent Events (SSE) for real-time response generation.
    """
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        async def generate_response():
            """Generate streaming response."""
            try:
                for chunk in rag_system.query_stream(
                    question=request.question,
                    k=request.k,
                    retrieval_method=request.retrieval_method,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature
                ):
                    # Format as SSE
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"

                # Send completion signal
                yield f"data: {json.dumps({'done': True})}\n\n"

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            generate_response(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )

    except Exception as e:
        logger.error(f"Streaming query setup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Streaming failed: {str(e)}")


@app.get("/stats", response_model=SystemStatsResponse, summary="Get system statistics")
async def get_system_stats(rag_system: RAGSystem = Depends(get_rag_system)):
    """Get comprehensive statistics about the RAG system."""
    try:
        stats = rag_system.get_system_stats()
        return SystemStatsResponse(**stats)

    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")


@app.post("/clear_index", summary="Clear the document index")
async def clear_index(rag_system: RAGSystem = Depends(get_rag_system)):
    """Clear all indexed documents from the vector store."""
    try:
        rag_system.clear_index()

        return {
            "message": "Document index cleared successfully",
            "timestamp": datetime.now()
        }

    except Exception as e:
        logger.error(f"Failed to clear index: {e}")
        raise HTTPException(status_code=500, detail=f"Index clearing failed: {str(e)}")


@app.post("/save_index", summary="Save the current index")
async def save_index(rag_system: RAGSystem = Depends(get_rag_system)):
    """Manually save the current document index to disk."""
    try:
        rag_system.save_index()

        return {
            "message": "Document index saved successfully",
            "timestamp": datetime.now()
        }

    except Exception as e:
        logger.error(f"Failed to save index: {e}")
        raise HTTPException(status_code=500, detail=f"Index saving failed: {str(e)}")


# Helper functions

async def cleanup_temp_files(file_paths: List[str]):
    """Clean up temporary files in the background."""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up {file_path}: {e}")


# Exception handlers

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled error in {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )


# Development server
if __name__ == "__main__":
    # Ensure required directories exist
    os.makedirs("data/uploaded_docs", exist_ok=True)
    os.makedirs("data/vector_store", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Run server
    uvicorn.run(
        "src.rag_system.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
