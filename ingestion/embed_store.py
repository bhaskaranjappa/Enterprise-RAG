import sys
import os
sys.path.append(os.path.dirname(__file__))

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer
from load_chunk import extract_text_from_pdf, chunk_pages

COLLECTION_NAME = "knowledge_base"
EMBED_DIM = 384  # bge-small output dimension
DATA_FOLDER = "./data"
BATCH_SIZE = 100  # insert 100 chunks at a time

def connect_milvus():
    connections.connect("default", host="localhost", port="19530")
    print("Connected to Milvus")

def create_collection():
    if utility.has_collection(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists — dropping and recreating.")
        utility.drop_collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, 
                    is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, 
                    max_length=2000),
        FieldSchema(name="source", dtype=DataType.VARCHAR, 
                    max_length=500),
        FieldSchema(name="page", dtype=DataType.INT64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, 
                    dim=EMBED_DIM),
    ]
    schema = CollectionSchema(
        fields, 
        description="Enterprise RAG document chunks"
    )
    collection = Collection(COLLECTION_NAME, schema)

    # IVF_FLAT index — good balance of speed vs accuracy
    # nlist=128 means 128 cluster centroids for search
    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index(
        field_name="embedding", 
        index_params=index_params
    )
    print(f"Collection '{COLLECTION_NAME}' created with index.")
    return collection

def load_all_chunks():
    pdf_files = [f for f in os.listdir(DATA_FOLDER) 
                 if f.endswith(".pdf")]
    all_chunks = []
    for pdf_file in pdf_files:
        filepath = os.path.join(DATA_FOLDER, pdf_file)
        pages_text = extract_text_from_pdf(filepath)
        chunks = chunk_pages(pages_text, pdf_file)
        all_chunks.extend(chunks)
        print(f"Loaded: {pdf_file} -> {len(chunks)} chunks")
    print(f"\nTotal chunks to embed: {len(all_chunks)}")
    return all_chunks

def embed_and_insert(collection, chunks, model):
    total_inserted = 0

    # Process in batches to avoid memory issues
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        # Generate embeddings for this batch
        embeddings = model.encode(
            texts, 
            normalize_embeddings=True,  # required for COSINE metric
            show_progress_bar=False
        )

        
        # Prepare data as list of row dicts (PyMilvus 3.0 format)
        data = [
            {
                "text": batch[j]["text"][:2000],
                "source": batch[j]["source"],
                "page": batch[j]["page"],
                "embedding": embeddings[j].tolist()
            }
            for j in range(len(batch))
        ]

        collection.insert(data)

        
        total_inserted += len(batch)

        if total_inserted % 500 == 0:
            print(f"  Inserted {total_inserted}/{len(chunks)} chunks...")

    collection.flush()  # write to disk
    print(f"\nTotal inserted: {total_inserted} chunks")

def verify_search(collection, model):
    """Run one test query to confirm everything works."""
    collection.load()
    query = "What are the causes of the Cold War?"
    query_embedding = model.encode(
        [query], 
        normalize_embeddings=True
    ).tolist()

    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    results = collection.search(
        data=query_embedding,
        anns_field="embedding",
        param=search_params,
        limit=3,
        output_fields=["text", "source", "page"]
    )

    print("\n--- Verification search: 'Causes of Cold War' ---")
    for i, hit in enumerate(results[0]):
        print(f"\nResult {i+1}:")
        print(f"  Score: {hit.score:.4f}")
        print(f"  Source: {hit.entity.get('source')}, "
              f"Page: {hit.entity.get('page')}")
        print(f"  Text: {hit.entity.get('text')[:200]}...")

def main():
    print("=== Phase 2: Embedding + Milvus Storage ===\n")

    # Step 1: Connect
    connect_milvus()

    # Step 2: Create collection
    collection = create_collection()

    # Step 3: Load embedding model (downloads ~130MB first time)
    print("\nLoading embedding model (bge-small-en-v1.5)...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    print("Model loaded.")

    # Step 4: Load all chunks from PDFs
    print("\nLoading chunks from PDFs...")
    chunks = load_all_chunks()

    # Step 5: Embed and insert
    print("\nEmbedding and inserting into Milvus...")
    embed_and_insert(collection, chunks, model)

    # Step 6: Verify with a test search
    verify_search(collection, model)

    print("\n=== Phase 2 Complete ===")

if __name__ == "__main__":
    main()