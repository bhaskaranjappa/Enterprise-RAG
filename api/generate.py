# api/generate.py
import sys
import os
import time
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ingestion'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'retrieval'))

from hybrid_search import connect_and_load, vector_search, bm25_search, hybrid_merge, load_all_chunks
from rerank import rerank
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
GROQ_REASONING_EFFORT = "none"  # Instruct mode: efficient, non-reasoning RAG responses

def decompose_query(query: str) -> list:
    """Runs exactly once per request lifecycle to protect API limits."""
    trigger_words = ["relate", "compare", "influence", "connect", "relationship", "versus", "vs", "and"]
    if not any(word in query.lower() for word in trigger_words):
        return [query]
        
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return [query]

    client = Groq(api_key=api_key)
    system_prompt = (
        "You are an expert cross-document search optimizer. Break down the user's complex cross-document query "
        "into exactly two separate, simplified keyword search queries targeting separate subjects.\n"
        "CRITICAL: If the query mentions an explicit material like 'geography', 'ethics', 'governance', "
        "or 'sin', make sure that keyword is explicitly present in its respective sub-query.\n"
        "Output ONLY the two queries separated by a semicolon. No conversational fluff or numbers."
    )
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Split: {query}"}],
            temperature=0.7,
            top_p=0.8,
            reasoning_effort=GROQ_REASONING_EFFORT,
            reasoning_format="hidden"
        )
        parts = [p.strip() for p in response.choices[0].message.content.split(";")]
        return parts if len(parts) == 2 else [query]
    except Exception:
        return [query]

def build_prompt(query, reranked_chunks):
    context_blocks = [f"[Source {i+1}: {c['source']}, Page {c['page']}]\n{c['text']}" for i, c in enumerate(reranked_chunks)]
    context = "\n\n".join(context_blocks)
    return f"""You are a strict, closed-loop Enterprise RAG extraction assistant. Your task is to answer the question using ONLY the context provided below.

RULES:
1. Base your answer ONLY on the provided explicit facts in the context block.
2. After every factual claim, cite the source like this: (Source 1) or (Source 2).
3. If the context does not contain explicit, verifiable facts to fully answer the query, you must respond with exactly: NOT FOUND
4. Do not use any outside knowledge, assumptions, historical training data, or logical extrapolations.
5. If it is not explicitly written in the provided context, it is NOT FOUND.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

def generate_answer(query, reranked_chunks):
    # If the database extraction yields nothing, exit instantly without calling the LLM
    if not reranked_chunks or len(reranked_chunks) == 0:
        return "NOT FOUND"
        
    client = Groq()
    prompt = build_prompt(query, reranked_chunks)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        top_p=0.8,
        reasoning_effort=GROQ_REASONING_EFFORT,
        reasoning_format="hidden"
    )
    
    raw_answer = response.choices[0].message.content.strip()
    
    # Post-processing normalization guardrail:
    # If the model emits the string but wraps it in sentences, normalize it down.
    lower_ans = raw_answer.lower()
    if (
        "not found" in lower_ans or 
        "insufficient_context" in lower_ans or 
        "insufficient context" in lower_ans or
        "does not contain" in lower_ans
    ):
        return "NOT FOUND"
        
    return raw_answer

def full_pipeline(query, collection, embed_model, all_chunks):
    """Complete optimized RAG execution flow with up-front single-call decomposition."""
    # 1. Decompose once upstream
    sub_queries = decompose_query(query)
    
    # 2. Retrieve candidates smoothly via non-duplicated calls
    vector_hits = vector_search(collection, embed_model, sub_queries, top_k=20)
    bm25_hits = bm25_search(all_chunks, sub_queries, top_k=20)
    hybrid_hits = hybrid_merge(vector_hits, bm25_hits, top_k=20)
    
    # 3. Scale context pool limits based on complexity
    is_complex = len(sub_queries) > 1
    target_top_k = 10 if is_complex else 5
    reranked = rerank(query, hybrid_hits, top_k=target_top_k)
    
    # 4. Generate answer with a short cooldown sleep to smooth out parallel evaluation pacing
    #if is_complex:
    #    time.sleep(1.5)
        
    answer = generate_answer(query, reranked)
    
    # If the verdict is NOT FOUND, clean the output by stripping citations from the return payload
    if answer == "NOT FOUND":
        return answer, []
        
    return answer, reranked