**🤖 AI Book Assistant (RAG System)**

A production-ready Retrieval-Augmented Generation (RAG) API built with FastAPI, FAISS, and Ollama that allows users to ask questions about a book and get AI-generated answers based on semantic search over the book content.

**🌟 Features**
🔍 Core RAG Pipeline
Semantic search using FAISS vector database
High-quality embeddings using Sentence Transformers (BAAI models)
Optional cross-encoder reranking
Query rewriting using LLM (improves search quality)
Context-aware answer generation using Ollama LLMs
⚡ Performance
Fast vector similarity search
Optional caching system (if enabled in config)
Lightweight API with FastAPI
Optimized chunk retrieval pipeline

**🧠 AI Capabilities**
Book-based question answering
Context-aware responses only from provided documents
Query-type aware prompts (general, summary, explanation, comparison)

**🐳 Deployment Ready**
Docker support (single command deployment)
Docker Compose for multi-service setup (API + Ollama)
Volume persistence for models and data

**🏗️ Project Structure**
AI_BookAssistant_Deployment/
│
├── app/
│   ├── main.py            # FastAPI entry point
│   ├── models.py          # Pydantic request/response models
│   ├── routes.py          # API endpoints
│   ├── services.py        # RAG pipeline logic
│
├── data/
│   └── HP_book_all.pdf    # Source document
│
├── resource/
│   ├── chunks.pkl         # Preprocessed text chunks
│   ├── faiss_index.bin    # FAISS vector index
│   ├── models--BAAI-*     # Embedding & reranker models
│
├── logs/
│   └── app.log
│
├── tests/
│   └── test_api.py
│
├── config.py              # Configuration settings
├── Dockerfile             # API container
├── docker-compose.yml     # Multi-container setup
├── requirements.txt
├── README.md
└── .env

**⚙️ Architecture**
User Query
    ↓
FastAPI (/ask)
    ↓
Query Rewrite (LLM - optional)
    ↓
Embedding Model (Sentence Transformer)
    ↓
FAISS Vector Search (top-k chunks)
    ↓
Reranker (Cross Encoder - optional)
    ↓
Context Builder
    ↓
Ollama LLM (Answer Generation)
    ↓
Final Response

**🚀 Installation & Setup**
🔹 1. Clone Project
git clone <your-repo-url>
cd AI_BookAssistant_Deployment
🔹 2. Local Setup (Without Docker)
Install dependencies
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
Run Ollama
ollama serve
Pull models
ollama pull llama3.2:1b
ollama pull mistral:latest
Start API
uvicorn app.main:app --reload
🔹 3. Docker Setup (Recommended)
Build & run everything
docker-compose up -d
Pull LLM models inside container
docker-compose exec ollama ollama pull llama3.2:1b
docker-compose exec ollama ollama pull mistral:latest
Access API
http://localhost:8000/docs
📡 API Endpoints
🔹 Ask a Question (Main Endpoint)
POST /api/v1/ask
Request Body
{
  "query": "What is the main theme of the book?",
  "query_type": "summary",
  "top_k": 10,
  "rerank_top_k": 3,
  "use_reranker": true,
  "include_sources": false
}
Response
{
  "response": "The main theme of the book is friendship and courage...",
  "query_rewritten": "Explain main themes of the book",
  "chunks_used": 3,
  "processing_time_ms": 320.5,
  "model_used": "llama3.2:1b"
}
🔹 Health Check
GET /api/v1/health
🔹 Stats (optional debug endpoint)
GET /api/v1/stats

**🧠 How RAG Works in This Project**
Step 1: Query Input

User asks a question.

Step 2: Query Rewrite (optional)

LLM improves query clarity.

Step 3: Embedding

Query converted into vector using:

BAAI/bge-large-en-v1.5
Step 4: Vector Search

FAISS finds most similar chunks.

Step 5: Reranking (optional)

Cross-encoder improves relevance ordering.

Step 6: Context Building

Top chunks are merged into prompt.

Step 7: LLM Generation

Ollama model generates final answer.

**🐳 Docker Overview**
Services
Service	Purpose
api	FastAPI backend
ollama	LLM inference engine
Run flow
docker-compose up
→ starts ollama
→ starts api
→ api connects to ollama internally

**⚙️ Environment Variables**
OLLAMA_BASE_URL=http://ollama:11434
SMALL_LLM=llama3.2:1b
LARGE_LLM=mistral:latest

RAG_TOP_K=10
RAG_RERANK_K=3
RAG_USE_RERANKER=true

LOG_LEVEL=INFO

**📊 Performance**
Stage	Time
Embedding	~50–100ms
FAISS search	~10ms
Reranking	~100–200ms
LLM response	~300–1000ms

**🧪 Testing**
pytest tests/

**📌 Requirements**
Python 3.9+
FAISS
Sentence Transformers
FastAPI
Ollama (for LLM)

**🚀 Future Improvements**
 Chat memory (multi-turn conversation)
 Multi-document support
 Streaming responses
 Redis caching
 User authentication
 Cloud deployment (AWS ECS/K8s)

**👨‍💻 Author**
ZABEEH ULLAH NOOR

**⭐ If this project helps you**
Give it a ⭐ on GitHub 🙂
