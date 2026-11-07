# ingestion/migrate_to_cloud.py
import os
import sys
from dotenv import load_dotenv
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
from sentence_transformers import SentenceTransformer

# Setup module pathing
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ingestion'))
from load_chunk import extract_text_from_pdf, chunk_pages

load_dotenv()

COLLECTION_NAME = "knowledge_base"
DATA_FOLDER = "./data"

def migrate():
    endpoint = os.getenv("ZILLIZ_ENDPOINT")
    token = os.getenv("ZILLIZ_TOKEN")
    
    if not endpoint or not token:
        print("[Error] Missing ZILLIZ_ENDPOINT or ZILLIZ_TOKEN in .env configuration.")
        return

    print(f"Connecting to Zilliz Cloud Endpoint: {endpoint}")
    connections.connect(
        alias="default",
        uri=endpoint,
        token=token,
        secure=True
    )

    # If the database is completely fresh, we programmatically provision the Collection Schema
    if not utility.has_collection(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' not found. Initializing enterprise schema...")
        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="page", dtype=DataType.INT64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384) # BGE-small-en vector dimension
        ]
        schema = CollectionSchema(fields, description="Enterprise RAG document knowledge base")
        collection = Collection(name=COLLECTION_NAME, schema=schema)
        
        # Build vector index parameters for cloud search optimization
        index_params = {
            "metric_type": "COSINE",
            "index_type": "AUTOINDEX", # Zilliz automatically handles cloud vector speed indexing
            "params": {}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        print("Schema creation and AUTOINDEX allocation complete.")
    else:
        collection = Collection(COLLECTION_NAME)
        print(f"Target cluster collection '{COLLECTION_NAME}' successfully targeted.")

    print("Initializing embedding extraction layer...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    
    pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} files for cloud migration.")
    
    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file}")
        filepath = os.path.join(DATA_FOLDER, pdf_file)
        pages_text = extract_text_from_pdf(filepath)
        chunks = chunk_pages(pages_text, pdf_file)
        
        if not chunks:
            continue
            
        texts = [c["text"] for c in chunks]
        embeddings = model.encode(texts, normalize_embeddings=True).tolist()
        
        # Structure fields into rows matching schema orientation
        data_payload = [
            [c["source"] for c in chunks],
            [int(c["page"]) for c in chunks],
            texts,
            embeddings
        ]
        
        print(f"Uploading {len(chunks)} vectors up to Zilliz Cloud...")
        collection.insert(data_payload)
        
    collection.flush()
    print("Cloud migration completed successfully.")

if __name__ == "__main__":
    migrate()