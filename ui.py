# ui.py
import streamlit as st
import sys
import os

# Enforce correct internal system paths for modular imports within the container
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'retrieval')))

from api.generate import full_pipeline
from retrieval.hybrid_search import connect_and_load, load_all_chunks
from sentence_transformers import SentenceTransformer

# Set up clean layout configurations
st.set_page_config(
    page_title="UPSC Enterprise RAG Platform",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📖 UPSC Enterprise RAG Platform")
st.caption("Production Closed-Loop Document Extraction Engine (OnlyIAS PRAHAAR Modules)")
st.markdown("---")

# 1. Warm up resources once globally inside the cloud space container
@st.cache_resource
def load_rag_resources():
    st.write("⏳ Initializing Zilliz Cloud connection & loading embeddings core...")
    try:
        # Reads ZILLIZ_URI and ZILLIZ_TOKEN securely from Hugging Face Settings -> Secrets
        collection = connect_and_load()
        embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        all_chunks = load_all_chunks()
        return collection, embed_model, all_chunks
    except Exception as e:
        st.error(f"❌ Critical Resource Loading Failure: {str(e)}")
        st.stop()

# Load the resources into the global context
collection, embed_model, all_chunks = load_rag_resources()

# 2. Configuration Controls and Constraints Display in Sidebar
st.sidebar.header("🛠️ Infrastructure Settings")
st.sidebar.success("⚡ In-Memory Cloud Pipeline: Active")

st.sidebar.markdown("""
### Security Constraints Enforced:
* 🔒 **Closed-Loop Retrieval Only**
* 🛑 **Zero Web Contamination**
* 📉 **Zero Hallucination Guardrails**
""")

# Initialize state history arrays for chat logs
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Display Chat Thread History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Capture Live User Input Query
if user_query := st.chat_input("Ask a question based entirely on internal modules..."):
    
    # Render user query instantly in chat view
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    # Trigger processing state spinner
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        with st.spinner("Executing Hybrid Search Matrix & Citations Audit..."):
            try:
                # Direct in-memory execution bypassing network endpoints completely
                answer, reranked_chunks = full_pipeline(
                    query=user_query,
                    collection=collection,
                    embed_model=embed_model,
                    all_chunks=all_chunks
                )
                
                # Formulate display text based on strict RAG constraints
                if answer == "NOT FOUND":
                    display_text = "❌ **NOT FOUND**: The requested information is not present in the internal vector database."
                else:
                    display_text = answer
                
                # Update active response view screen
                response_placeholder.markdown(display_text)
                st.session_state.messages.append({"role": "assistant", "content": display_text})
                
                # 5. Render Dynamic Citation Modules if Valid Data Exists
                if answer != "NOT FOUND" and reranked_chunks:
                    st.markdown("### 📌 Supporting Citations Found:")
                    for idx, cite in enumerate(reranked_chunks):
                        with st.expander(f"Source {idx+1}: {cite.get('source', 'Unknown')} (Page {cite.get('page', 0)})"):
                            st.markdown(f"*{cite.get('text', '')}*")
                            
                # 6. Developer Debugging Window (Collapsible Payload View)
                with st.sidebar.expander("🔍 Developer Payload Inspection"):
                    st.json({
                        "query": user_query,
                        "status": "SUCCESS",
                        "answer_returned": answer,
                        "chunks_retrieved_count": len(reranked_chunks)
                    })
                    
            except Exception as e:
                response_placeholder.error(f"⚠️ Pipeline Error: Execution thread failed. Details: {str(e)}")