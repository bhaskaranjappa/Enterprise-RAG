# app.py
import streamlit as st
import os
import sys
from sentence_transformers import SentenceTransformer
from groq import Groq
from pymilvus import connections, Collection

# 1. Page Configuration Setup
st.set_page_config(
    page_title="UPSC Enterprise RAG System",
    page_icon="📖",
    layout="wide"
)

st.title("📖 UPSC Enterprise RAG Platform")
st.caption("Production Closed-Loop Document Extraction (OnlyIAS PRAHAAR Modules)")
st.markdown("---")

# 2. System Environmental Secrets Loading
# These will be bound securely inside your Hugging Face space dashboard settings
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
ZILLIZ_URI = os.environ.get("ZILLIZ_URI")
ZILLIZ_TOKEN = os.environ.get("ZILLIZ_TOKEN")
GROQ_MODEL = "llama-3.3-70b-versatile"

@st.cache_resource
def initialize_shared_resources():
    """Warms up local search indices and establishes Zilliz cloud pools once."""
    # Note: Ensure all your fallback retrieval helper definitions (vector_search, 
    # bm25_search, hybrid_merge, load_all_chunks, rerank) are imported or defined here.
    # For a direct clean cloud run, we pull directly via standard Milvus SDK hooks:
    connections.connect(alias="default", uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
    collection = Collection("YOUR_COLLECTION_NAME")
    collection.load()
    
    embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return collection, embed_model

try:
    if not GROQ_API_KEY or not ZILLIZ_URI or not ZILLIZ_TOKEN:
        st.error("🔑 Missing Infrastructure Credentials. Ensure Secrets are added to the space.")
        st.stop()
        
    collection, embed_model = initialize_shared_resources()
except Exception as e:
    st.error(f"❌ Failed to boot core backend matrix: {str(e)}")
    st.stop()

# 3. Dynamic RAG Query Processing Logic
def build_strict_prompt(query, chunks):
    context = "\n\n".join([f"[Source {i+1}: {c['source']}, Page {c['page']}]\n{c['text']}" for i, c in enumerate(chunks)])
    return f"""You are a strict, closed-loop Enterprise RAG extraction assistant. Your task is to answer the question using ONLY the context provided below.
RULES:
1. Base your answer ONLY on the provided explicit facts in the context block.
2. After every factual claim, cite the source like this: (Source 1) or (Source 2).
3. If the context does not contain explicit, verifiable facts to fully answer the query, you must respond with exactly: NOT FOUND
4. Do not use any outside knowledge, assumptions, historical training data, or logical extrapolations.

CONTEXT:
{context}

QUESTION: {query}
ANSWER:"""

# 4. Streamlit Interactive Interface Chat Loop
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask a question strictly inside module bounds..."):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    with st.chat_message("assistant"):
        with st.spinner("Searching internal Zilliz cloud layers..."):
            # Execute your exact optimized pipeline search steps here
            # (decompose_query -> vector_search -> bm25_search -> hybrid_merge -> rerank)
            
            # Simulated outcome mapping for structure clarity:
            reranked_hits = [] # Replace with your active retrieval return arrays
            
            if not reranked_hits:
                answer = "NOT FOUND"
            else:
                client = Groq(api_key=GROQ_API_KEY)
                prompt = build_strict_prompt(user_input, reranked_hits)
                response = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                answer = response.choices[0].message.content.strip()
                if "not found" in answer.lower():
                    answer = "NOT FOUND"
            
            if answer == "NOT FOUND":
                display = "❌ **NOT FOUND**: Information not present within active document matrices."
            else:
                display = answer
                
            st.markdown(display)
            st.session_state.chat_history.append({"role": "assistant", "content": display})