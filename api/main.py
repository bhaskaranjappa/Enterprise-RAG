# C:\Users\ABhas\projects\enterprise-rag\api\main.py
import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

# Enforce paths for internal imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'retrieval'))

from hybrid_search import connect_and_load, load_all_chunks
from api.generate import full_pipeline

app = FastAPI(
    title="UPSC Enterprise RAG System",
    description="Production-grade API layer serving OnlyIAS PRAHAAR modules.",
    version="1.0.0"
)

# Global shared resources loaded into memory once at startup
SHARED_RESOURCES = {}

class ChatRequest(BaseModel):
    query: str = Field(..., example="What are the main principles of Ashoka's Dhamma?")

class ContextSource(BaseModel):
    source: str
    page: int
    text: str

class ChatResponse(BaseModel):
    answer: str
    citations: List[ContextSource]

@app.on_event("startup")
def startup_event():
    """Warm up models and open database connections on app spin up."""
    print("[Startup] Initializing shared application context...")
    try:
        SHARED_RESOURCES["collection"] = connect_and_load()
        SHARED_RESOURCES["embed_model"] = SentenceTransformer("BAAI/bge-small-en-v1.5")
        SHARED_RESOURCES["all_chunks"] = load_all_chunks()
        print("[Startup] All vector data and embeddings loaded successfully.")
    except Exception as e:
        print(f"[Fatal Startup Error] Failed to initialize backend systems: {str(e)}")
        raise e

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    """
    Main stateless endpoint processing raw queries through the complete 
    Retrieval-Rerank-Generation infrastructure.
    """
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
        
    try:
        # Pass shared dependencies cleanly to avoid reloading files per request
        answer, reranked_chunks = full_pipeline(
            query=payload.query,
            collection=SHARED_RESOURCES["collection"],
            embed_model=SHARED_RESOURCES["embed_model"],
            all_chunks=SHARED_RESOURCES["all_chunks"]
        )
        
        # Guardrail: If data is missing from database, completely wipe citations out of payload
        if answer == "NOT FOUND":
            return ChatResponse(answer="NOT FOUND", citations=[])
        
        # Structure citations cleanly for front-end parsing
        citations = [
            ContextSource(
                source=chunk.get("source", "Unknown"),
                page=int(chunk.get("page", 0)),
                text=chunk.get("text", "")
            )
            for chunk in reranked_chunks
        ]
        
        return ChatResponse(answer=answer, citations=citations)
        
    except Exception as e:
        # Bubble up explicit error traces for production debugging
        raise HTTPException(status_code=500, detail=f"Internal Pipeline Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # 1. Dynamically read the PORT environment variable injected by Google Cloud Run
    # 2. Fall back to 8000 automatically so your local testing commands never break
    target_port = int(os.getenv("PORT", 8000))
    
    # Disable reload in production to maximize concurrency throughput
    is_development = os.getenv("ENV", "production") == "development"
    
    print(f"[Boot] Launching FastAPI matrix on port {target_port}...")
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=target_port, 
        reload=is_development
    )