# Create the main project structure and core modules

import os
os.makedirs('src', exist_ok=True)
os.makedirs('src/rag_system', exist_ok=True)
os.makedirs('src/rag_system/document_processors', exist_ok=True)
os.makedirs('src/rag_system/embeddings', exist_ok=True)
os.makedirs('src/rag_system/retrieval', exist_ok=True)
os.makedirs('src/rag_system/generation', exist_ok=True)
os.makedirs('src/rag_system/api', exist_ok=True)
os.makedirs('tests', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('data/uploaded_docs', exist_ok=True)
os.makedirs('data/vector_store', exist_ok=True)
os.makedirs('logs', exist_ok=True)

print("✅ Project structure created")

# Create __init__.py files
init_dirs = [
    'src',
    'src/rag_system',
    'src/rag_system/document_processors',
    'src/rag_system/embeddings', 
    'src/rag_system/retrieval',
    'src/rag_system/generation',
    'src/rag_system/api',
    'tests'
]

for dir_path in init_dirs:
    with open(f'{dir_path}/__init__.py', 'w') as f:
        f.write('"""RAG System Module"""\n')

print("✅ __init__.py files created")