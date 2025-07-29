# First, let me create the complete RAG system implementation
# Starting with the requirements.txt file

requirements_content = """
# Core FastAPI and server dependencies
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
pydantic==2.5.0

# Document processing libraries
PyPDF2==3.0.1
pdfplumber==0.10.3
python-docx==1.1.0

# LangChain and related libraries
langchain==0.1.0
langchain-community==0.0.9
langchain-core==0.1.12
langchain-text-splitters==0.0.1

# Embedding and vector store
sentence-transformers==2.2.2
faiss-cpu==1.7.4
numpy==1.24.4

# LLM integration
transformers==4.36.2
torch==2.1.2
accelerate==0.25.0

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1

# Logging and monitoring
loguru==0.7.2

# Optional: For advanced features
rank_bm25==0.2.2
"""

# Save the requirements file
with open('requirements.txt', 'w') as f:
    f.write(requirements_content)

print("✅ Requirements file created")