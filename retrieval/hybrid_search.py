# retrieval/hybrid_search.py
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ingestion'))
from load_chunk import extract_text_from_pdf, chunk_pages

COLLECTION_NAME = "upsc_knowledge_base"
DATA_FOLDER = "./data"

def connect_and_load():
    """Polymorphic connection router: scales smoothly between local and Zilliz Cloud grids."""
    endpoint = os.getenv("ZILLIZ_ENDPOINT")
    token = os.getenv("ZILLIZ_TOKEN")
    
    # If cloud credentials exist, route securely to Zilliz via TLS
    if endpoint and token:
        # Strip any accidental whitespace or wrapping quotes from the strings
        clean_endpoint = endpoint.strip().strip('"').strip("'")
        clean_token = token.strip().strip('"').strip("'")
        
        connections.connect(
            alias="default",
            uri=clean_endpoint,
            token=clean_token,
            secure=True  # Enforces mandatory SSL for cloud verification
        )
    else:
        # Fallback cleanly to local development container mesh
        connections.connect("default", host="localhost", port="19530")
        
    collection = Collection(COLLECTION_NAME)
    collection.load()
    return collection

def vector_search(collection, model, sub_queries, top_k=20):
    """Dense vector search — queries multiple sub-query targets cleanly."""
    aggregated_hits = {}

    for sub_q in sub_queries:
        query_embedding = model.encode([sub_q], normalize_embeddings=True).tolist()
        results = collection.search(
            data=query_embedding,
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["text", "source", "page"]
        )
        
        if results and len(results) > 0:
            for hit in results[0]:
                text = hit.entity.get("text")
                key = text[:100]
                if key not in aggregated_hits or hit.score > aggregated_hits[key]["vector_score"]:
                    aggregated_hits[key] = {
                        "text": text,
                        "source": hit.entity.get("source"),
                        "page": hit.entity.get("page"),
                        "vector_score": hit.score
                    }
                    
    sorted_hits = sorted(aggregated_hits.values(), key=lambda x: x["vector_score"], reverse=True)
    return sorted_hits[:top_k]

def bm25_search(all_chunks, sub_queries, top_k=20):
    """Keyword search — scans multi-query terms across tokenized corpus."""
    if not all_chunks:
        return []

    tokenized_corpus = [c["text"].lower().split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    aggregated_hits = {}

    for sub_q in sub_queries:
        tokenized_query = sub_q.lower().split()
        scores = bm25.get_scores(tokenized_query)
        
        for idx, score in enumerate(scores):
            if score > 0:
                text = all_chunks[idx]["text"]
                key = text[:100]
                if key not in aggregated_hits or score > aggregated_hits[key]["bm25_score"]:
                    aggregated_hits[key] = {
                        "text": text,
                        "source": all_chunks[idx]["source"],
                        "page": all_chunks[idx]["page"],
                        "bm25_score": score
                    }
                    
    sorted_hits = sorted(aggregated_hits.values(), key=lambda x: x["bm25_score"], reverse=True)
    if not sorted_hits:
        return [{**c, "bm25_score": 0} for c in all_chunks[:top_k]]
        
    return sorted_hits[:top_k]

def hybrid_merge(vector_hits, bm25_hits, top_k=20):
    k = 60
    scores = {}
    
    for rank, hit in enumerate(vector_hits):
        key = hit["text"][:100]
        if key not in scores:
            scores[key] = {"hit": hit, "rrf_score": 0}
        scores[key]["rrf_score"] += 1 / (rank + k)
    
    for rank, hit in enumerate(bm25_hits):
        key = hit["text"][:100]
        if key not in scores:
            scores[key] = {"hit": hit, "rrf_score": 0}
        scores[key]["rrf_score"] += 1 / (rank + k)
    
    merged = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)[:top_k]
    return [item["hit"] for item in merged]

def load_all_chunks():
    if not os.path.isdir(DATA_FOLDER):
        print(f"[Retrieval] {DATA_FOLDER} is absent; BM25 corpus will be empty.")
        return []

    pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith(".pdf")]
    all_chunks = []
    for pdf_file in pdf_files:
        filepath = os.path.join(DATA_FOLDER, pdf_file)
        pages_text = extract_text_from_pdf(filepath)
        chunks = chunk_pages(pages_text, pdf_file)
        all_chunks.extend(chunks)
    return all_chunks