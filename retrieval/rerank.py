import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ingestion'))

from hybrid_search import (connect_and_load, vector_search, 
                            bm25_search, hybrid_merge, load_all_chunks)
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def rerank(query, hits, top_k=5):
    """
    CrossEncoder scores each query-chunk pair directly.
    Unlike embeddings (which score query and chunk separately),
    CrossEncoder sees both together — much more accurate relevance scoring.
    Takes top-20 hybrid hits, returns top-5 reranked.
    """
    cross_encoder = CrossEncoder(RERANK_MODEL)
    
    pairs = [[query, hit["text"]] for hit in hits]
    scores = cross_encoder.predict(pairs)
    
    # Attach rerank score to each hit
    for i, hit in enumerate(hits):
        hit["rerank_score"] = float(scores[i])
    
    # Sort by rerank score, return top_k
    reranked = sorted(hits, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]

def main():
    print("=== Phase 4: Reranking ===\n")
    
    collection = connect_and_load()
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    all_chunks = load_all_chunks()
    
    test_queries = [
        "What are the causes of the Cold War?",
        "Article 370 Kashmir special status"
    ]
    
    for query in test_queries:
        print(f"Query: '{query}'")
        print("-" * 60)
        
        # Get top-20 hybrid results
        vector_hits = vector_search(collection, model, query, top_k=20)
        bm25_hits = bm25_search(all_chunks, query, top_k=20)
        hybrid_hits = hybrid_merge(vector_hits, bm25_hits, top_k=20)
        
        # Rerank down to top-5
        reranked = rerank(query, hybrid_hits, top_k=5)
        
        print("TOP 3 AFTER RERANKING:")
        for i, hit in enumerate(reranked[:3]):
            print(f"\n  Result {i+1}:")
            print(f"  Rerank Score: {hit['rerank_score']:.4f}")
            print(f"  Source: {hit['source']}, Page: {hit['page']}")
            print(f"  Text: {hit['text'][:200]}...")
        print("\n")

if __name__ == "__main__":
    main()