<div align="center">
  <h1>Enterprise RAG</h1>
  <p><strong>Document-grounded question answering with hybrid retrieval, neural reranking, and source citations.</strong></p>
  <p>
    <a href="https://huggingface.co/spaces/BhaskarAnjappa/Enterprise_RAG">Live demo</a>
    ·
    <a href="#architecture">Architecture</a>
    ·
    <a href="#quickstart">Quickstart</a>
    ·
    <a href="#evaluation">Evaluation</a>
  </p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python 3.11">
    <img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
    <img src="https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI">
    <img src="https://img.shields.io/badge/Vector%20DB-Milvus-00A1EA" alt="Milvus">
  </p>
</div>

---

## Overview

Enterprise RAG is a retrieval-augmented question-answering application designed to answer questions from a defined collection of PDF documents. It combines semantic vector search with BM25 keyword retrieval, merges the candidate lists using Reciprocal Rank Fusion (RRF), and applies a CrossEncoder reranker before answer generation.

The generation layer uses Groq-hosted Llama 3.3 70B Versatile with a context-constrained prompt. Answers are expected to use the retrieved context, include source references, and return **NOT FOUND** when the available context is insufficient.

The repository includes a Streamlit interface, a FastAPI endpoint, PDF ingestion utilities, local Milvus infrastructure, and an evaluation harness.

> **Demo:** [Open Enterprise RAG on Hugging Face Spaces](https://huggingface.co/spaces/BhaskarAnjappa/Enterprise_RAG)  
> The live Space is deployed separately. Its runtime configuration and document store are not created by this GitHub repository.

## Capabilities

- **PDF ingestion** — extracts text page by page, preserves source-page metadata, and splits content into overlapping chunks.
- **Hybrid retrieval** — combines dense vector search in Milvus with BM25 keyword search.
- **Rank fusion** — uses Reciprocal Rank Fusion to combine candidate rankings.
- **Neural reranking** — scores query-and-passage pairs with a CrossEncoder.
- **Grounded generation** — sends the selected context to the configured Groq model with instructions to avoid unsupported answers.
- **Source traceability** — returns source names and page numbers alongside API answers; the Streamlit UI exposes supporting passages.
- **Two interfaces** — a Streamlit application and a FastAPI chat endpoint.

## Architecture

### System architecture

The system has two connected paths: a document-indexing path that prepares searchable content, and a request path that retrieves evidence and generates an answer.

#### 1. Document indexing

~~~mermaid
flowchart LR
    PDF["PDF documents"] --> EXTRACT["Extract text by page"]
    EXTRACT --> CLEAN["Clean and filter"]
    CLEAN --> CHUNK["Split into overlapping chunks"]

    CHUNK --> EMBED["Generate embeddings<br/>BGE small English"]
    EMBED --> VECTOR[("Milvus vector collection")]

    CHUNK --> CORPUS["Build searchable text corpus<br/>with source and page metadata"]

    classDef source fill:#EEF2FF,stroke:#6366F1,color:#1E1B4B,stroke-width:1.5px
    classDef process fill:#F8FAFC,stroke:#94A3B8,color:#0F172A,stroke-width:1.2px
    classDef storage fill:#ECFEFF,stroke:#0891B2,color:#164E63,stroke-width:1.5px

    class PDF source
    class EXTRACT,CLEAN,CHUNK,EMBED process
    class VECTOR,CORPUS storage
~~~

The vector database stores embeddings and chunk metadata. The text corpus is also loaded by the application for keyword retrieval.

#### 2. Live question answering

~~~mermaid
flowchart TB
    QUESTION["User question"] --> ENTRY["Streamlit interface<br/>or FastAPI endpoint"]
    ENTRY --> ORCHESTRATE["Query orchestration<br/>optional decomposition"]

    ORCHESTRATE --> DENSE["Semantic retrieval"]
    ORCHESTRATE --> KEYWORD["Keyword retrieval"]

    VECTOR[("Milvus vector collection")] --> DENSE
    CORPUS["In-memory text corpus"] --> KEYWORD

    DENSE --> FUSION["Reciprocal Rank Fusion"]
    KEYWORD --> FUSION

    FUSION --> RERANK["CrossEncoder reranking"]
    RERANK --> CONTEXT["Select evidence and build prompt"]
    CONTEXT --> GENERATE["Groq-hosted language model"]
    GENERATE --> ANSWER["Grounded response"]
    ANSWER --> CITATIONS["Answer with source/page references<br/>or NOT FOUND when evidence is insufficient"]

    classDef input fill:#EEF2FF,stroke:#6366F1,color:#1E1B4B,stroke-width:1.5px
    classDef process fill:#F8FAFC,stroke:#94A3B8,color:#0F172A,stroke-width:1.2px
    classDef retrieval fill:#ECFEFF,stroke:#0891B2,color:#164E63,stroke-width:1.5px
    classDef model fill:#F5F3FF,stroke:#8B5CF6,color:#2E1065,stroke-width:1.5px
    classDef result fill:#ECFDF5,stroke:#10B981,color:#064E3B,stroke-width:1.5px

    class QUESTION input
    class ENTRY,ORCHESTRATE,CONTEXT process
    class DENSE,KEYWORD,FUSION,RERANK retrieval
    class GENERATE model
    class ANSWER,CITATIONS result
~~~

The vector database can be local Milvus or a configured Zilliz Cloud cluster. The application loads the text corpus at startup to support BM25 retrieval.

### Retrieval and generation stages

1. **Ingestion:** PDF text is extracted page by page. The first two pages are skipped. Remaining text is split into 600-character chunks with 90-character overlap.
2. **Embedding:** Chunks are encoded using **BAAI/bge-small-en-v1.5** and stored in the Milvus collection **knowledge_base**.
3. **Candidate retrieval:** The query is searched through Milvus dense retrieval and BM25 keyword retrieval. Each path returns up to 20 candidates.
4. **Fusion:** Reciprocal Rank Fusion combines the ranked result lists.
5. **Reranking:** **cross-encoder/ms-marco-MiniLM-L-6-v2** scores query-passage pairs. The pipeline selects up to five passages for a simple query and up to ten for a decomposed query.
6. **Generation:** The selected context is passed to Groq using the configured **llama-3.3-70b-versatile** model ID. The prompt requires context-based answers and a **NOT FOUND** response when evidence is insufficient.

## Technology

| Layer | Components |
|---|---|
| User interface | Streamlit |
| API | FastAPI, Uvicorn, Pydantic |
| Document processing | pypdf, LangChain text splitters |
| Embeddings | Sentence Transformers — BAAI/bge-small-en-v1.5 |
| Dense vector search | Milvus, PyMilvus |
| Keyword search | rank-bm25 |
| Rank fusion | Reciprocal Rank Fusion |
| Reranking | CrossEncoder — ms-marco-MiniLM-L-6-v2 |
| Answer generation | Groq API — configured Llama 3.3 70B Versatile model ID |
| Local infrastructure | Docker Compose, etcd, MinIO, Milvus |
| Evaluation | JSON gold-standard questions and a Python API evaluation script |
| Demo hosting | Hugging Face Spaces |

## Quickstart

### Prerequisites

- Python 3.11
- Docker Desktop or Docker Engine with Docker Compose
- A Groq API key
- A document collection you are authorized to process
- A Zilliz Cloud endpoint and token only when using the cloud vector database

### 1. Clone the repository

~~~powershell
git clone https://github.com/bhaskaranjappa/Enterprise-RAG.git
cd Enterprise-RAG
~~~

### 2. Create an environment and install dependencies

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
~~~

### 3. Configure environment variables

~~~powershell
Copy-Item .env.example .env
~~~

Open **.env** and add your Groq API key. For local Milvus, leave the Zilliz variables empty. For Zilliz Cloud, set both cloud variables.

### 4. Add your PDF documents

~~~powershell
New-Item -ItemType Directory -Force data
~~~

Place your authorized PDF files in the **data/** directory. The source documents are intentionally not included in this repository. The application needs access to the local PDFs for ingestion and for rebuilding the BM25 corpus. If the directory is empty or absent, the app skips BM25 keyword retrieval and continues with semantic retrieval from the configured vector collection. Add authorized PDFs to enable hybrid retrieval.

### 5. Start local Milvus infrastructure

~~~powershell
docker compose up -d
docker compose ps
~~~

The Compose file starts etcd, MinIO, and standalone Milvus. It does not start the Streamlit application or the FastAPI server.

### 6. Ingest and index the PDFs

~~~powershell
python ingestion/embed_store.py
~~~

This creates the **knowledge_base** collection, generates embeddings, inserts chunks, and runs a verification search.

> **Important:** The local ingestion script drops and recreates the collection when it already exists. Running it again replaces the existing local collection data.

### 7. Launch the Streamlit application

~~~powershell
streamlit run ui.py
~~~

Open the local URL printed by Streamlit, normally **http://localhost:8501**.

### 8. Launch the FastAPI endpoint (optional)

In a separate terminal, activate the virtual environment and run:

~~~powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000
~~~

The chat endpoint is:

~~~http
POST http://localhost:8000/chat
Content-Type: application/json

{
  "query": "What are the main principles of Ashoka's Dhamma?"
}
~~~

Interactive API documentation is available at **http://localhost:8000/docs**.

## Configuration

Copy **.env.example** to **.env**. Do not commit the real **.env** file.

| Variable | Required | Description |
|---|---:|---|
| **GROQ_API_KEY** | Yes for answer generation | API key used by the Groq client. |
| **ZILLIZ_ENDPOINT** | Only for Zilliz Cloud | Cloud cluster URI. |
| **ZILLIZ_TOKEN** | Only for Zilliz Cloud | Cloud authentication token. |
| **ENV** | No | API runtime mode; defaults to production in **api/main.py**. |
| **PORT** | No | FastAPI port; defaults to 8000. |
| **HF_TOKEN** | Optional | Hugging Face authentication for workflows that use the Hugging Face Hub client. |

When both **ZILLIZ_ENDPOINT** and **ZILLIZ_TOKEN** are present, the retrieval connector uses Zilliz Cloud over a secure connection. Otherwise, it attempts to connect to local Milvus at **localhost:19530**.

The Streamlit and API processes also need access to the PDF files in **data/**, because the BM25 corpus is rebuilt from those files when resources are initialized.

### Zilliz Cloud ingestion

To populate a Zilliz Cloud collection instead of local Milvus:

1. Set **ZILLIZ_ENDPOINT** and **ZILLIZ_TOKEN** in **.env**.
2. Keep the authorized PDFs in **data/**.
3. Run:

~~~powershell
python ingestion/migrate_to_cloud.py
~~~

The migration script creates the collection if it does not exist. It inserts the processed chunks into an existing collection without clearing it; avoid rerunning it blindly, or duplicate records may be added.

## API

### **POST /chat**

Request:

~~~json
{
  "query": "What are the main principles of Ashoka's Dhamma?"
}
~~~

Response shape:

~~~json
{
  "answer": "Generated answer with inline source references.",
  "citations": [
    {
      "source": "document.pdf",
      "page": 12,
      "text": "Retrieved passage used to support the answer."
    }
  ]
}
~~~

When the retrieved context is insufficient, the pipeline can return:

~~~json
{
  "answer": "NOT FOUND",
  "citations": []
}
~~~

## Evaluation

The evaluation assets are under **eval/**:

- **gold_standard.json** — a curated set of 22 question and expected-answer records, including out-of-scope prompts.
- **evaluate.py** — sends each question to the local FastAPI **/chat** endpoint and records evaluation output.
- **eval_results.json** — a checked-in snapshot of five completed evaluation records.

Run the API first, then execute:

~~~powershell
python eval/evaluate.py
~~~

The script writes its output to **eval/eval_results.json**, replacing the existing snapshot.

**Interpretation note:** the current **hit_success** field primarily measures whether a retrieved source filename matches an expected source subject for in-scope questions. It does not establish answer correctness, faithfulness, citation precision, or end-to-end accuracy. Also, the current out-of-scope check looks for **INSUFFICIENT_CONTEXT**, while the generation pipeline returns **NOT FOUND**; align these labels before reporting out-of-scope metrics.

For that reason, this repository does not claim a validated overall accuracy percentage. The checked-in five-record snapshot is partial and should not be presented as a representative benchmark.

## Repository layout

~~~text
Enterprise-RAG/
├── api/
│   ├── generate.py          # Query decomposition, retrieval orchestration, generation
│   └── main.py              # FastAPI application and /chat endpoint
├── eval/
│   ├── evaluate.py          # API-based evaluation harness
│   ├── eval_results.json    # Partial recorded evaluation snapshot
│   └── gold_standard.json   # Curated evaluation questions
├── ingestion/
│   ├── load_chunk.py        # PDF extraction and chunking
│   ├── embed_store.py       # Local Milvus ingestion
│   └── migrate_to_cloud.py  # Zilliz Cloud ingestion
├── retrieval/
│   ├── hybrid_search.py     # Dense retrieval, BM25, and RRF
│   └── rerank.py            # CrossEncoder reranking
├── data/                    # Local PDF inputs; intentionally not committed
├── volumes/                 # Local Milvus/etcd/MinIO state; not committed
├── app.py                   # Earlier Streamlit prototype
├── ui.py                    # Streamlit entrypoint used by the Dockerfile
├── sync_data.py             # Hugging Face Space data synchronization utility
├── Dockerfile               # Streamlit container entrypoint on port 7860
├── docker-compose.yml       # Local Milvus infrastructure
├── requirements.txt
└── .env.example
~~~

## Deployment

The repository's Dockerfile starts **ui.py** and exposes port **7860**, matching the Hugging Face Spaces Docker runtime convention. The Docker Compose file is for local vector-database infrastructure; it is not the application deployment definition.

The public demo is hosted separately:

**[BhaskarAnjappa / Enterprise_RAG — Hugging Face Space](https://huggingface.co/spaces/BhaskarAnjappa/Enterprise_RAG)**

The deployed Space must be configured with its own secrets and access to its vector collection and document corpus. Pushing changes to this GitHub repository does not automatically update the Space unless a deployment connection is configured.

## Data, security, and licensing

- PDF source documents are not included. Add only documents you have permission to process and redistribute.
- Do not commit **.env**, API keys, access tokens, database volumes, or generated vector-store files.
- **volumes/** and **.codegraph/** are local runtime or analysis artifacts and are excluded from version control.
- The evaluation JSON contains expected-answer text derived from the document collection. Confirm redistribution rights before sharing that evaluation content publicly.
- This repository currently has no explicit open-source license. Public visibility permits people to view the code, but does not grant general permission to reuse, modify, or redistribute it.

## License

No license has been applied. All rights reserved by default unless the repository owner adds a license.

---

<div align="center">
  <p>Built by <a href="https://github.com/bhaskaranjappa">Bhaskar Anjappa</a></p>
  <p><a href="https://huggingface.co/spaces/BhaskarAnjappa/Enterprise_RAG">Try the live demo</a> · <a href="https://github.com/bhaskaranjappa/Enterprise-RAG">Explore the source</a></p>
</div>
