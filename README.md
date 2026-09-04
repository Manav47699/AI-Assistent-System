

## Architecture Diagram

```
+-------------------------------------------------------------+
|                Streamlit Web UI (ui.py)                     |
|                   http://localhost:8501                     |
|   - Model Selection, Temperature and Top-P Controls         |
|   - System Prompt Options, Structured Output and Tool Toggles|
|   - Tabs for Chat, Document RAG, and Ingestion              |
+------------------------------+------------------------------+
                               |
                               | HTTP POST (/chat, /rag-chat, /ingest)
                               v
+-------------------------------------------------------------+
|                 FastAPI Backend (main.py)                   |
|                   http://localhost:8000                     |
|   - OpenAI SDK / Local vLLM Base URL Support                |
|   - Function Calling Tool (`get_weather`)                   |
|   - Pydantic Structured Output (`StudentEvaluation`)        |
|   - Rate Limiting Middleware & Async Request Handling       |
|   - Fallback Mechanism (Primary -> Fallback -> Mock)        |
+---------------+-----------------------------+---------------+
                |                             |
                | Similarity Search           | Chat Completion
                v                             v
+-------------------------------+ +---------------------------+
|    ChromaDB Vector Store      | |    LLM / vLLM Server      |
|           (rag.py)            | |   (OpenAI or Local vLLM)  |
|  - Character Chunking         | |   - Primary: gpt-4o-mini  |
|  - Local Document Ingestion   | |   - Fallback: gpt-3.5     |
|  - Chroma Embeddings          | |   - Local: vLLM (:8000/v1)|
+-------------------------------+ +---------------------------+
```

---

## Project Structure

```
.
├── docs/                   # Knowledge base text documents for RAG
│   └── ai_fellowship.txt   # Sample document
├── main.py                 # FastAPI backend (LLM calls, tools, RAG, fallback, rate limit)
├── rag.py                  # RAG pipeline (chunking, ChromaDB vector store, search)
├── ui.py                   # Streamlit web application
├── Dockerfile              # Docker container definition
├── docker-compose.yml      # Multi-container setup for backend + frontend
├── requirements.txt        # Python package dependencies
├── .gitignore              # Ignored files (venv, chroma_db, caches, env)
└── README.md               # Documentation and execution instructions
```

---

## Running with Docker Compose

To start both the FastAPI backend and Streamlit UI together:

```bash
docker compose up --build
```

Access the services:
- Streamlit Web UI: http://localhost:8501
- FastAPI Docs (Swagger): http://localhost:8000/docs
- Health Check: http://localhost:8000/health

To stop:
```bash
docker compose down
```

---

## Local Setup (Using Virtual Environment)

1. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Set environment variables if using OpenAI or a running vLLM server:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   export VLLM_BASE_URL="http://localhost:8000/v1"
   ```
   *Note: If no API key is set, the backend falls back to offline mock mode so you can test all workflows without needing paid credits.*

4. Ingest documents into ChromaDB:
   ```bash
   python rag.py
   ```

5. Start the backend:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

6. In another terminal, start the Streamlit UI:
   ```bash
   source venv/bin/activate
   streamlit run ui.py
   ```

---

## Assignment Requirements Checklist

### Task 1: Build an AI Assistant
- **LLM Integration**: Uses OpenAI SDK with custom `base_url` support for local vLLM serving.
- **Prompt Engineering**: System prompt dropdown, temperature slider, and top-p slider in UI.
- **Structured Output**: Pydantic model (`StudentEvaluation`) enforcing JSON schema responses.
- **Tool Calling**: Dummy function calling (`get_weather`) with schema and automatic parameter extraction.
- **RAG Pipeline**: Character-level chunking with overlap and ChromaDB persistent storage.
- **Containerization**: Single `Dockerfile` packaging the app.

### Task 2: Productionize the AI Assistant
- **Web UI**: Streamlit application connected directly to FastAPI via HTTP POST.
- **Async Processing**: Asynchronous request handling using FastAPI `async def`.
- **Reliability**:
  - In-memory rate limiting middleware (max 60 requests per minute).
  - Retry mechanism with automatic model fallback (`gpt-4o-mini` -> `gpt-3.5-turbo` -> graceful offline simulation).
  - Clean error handling and HTTP status codes.
- **Docker Compose**: Orchestrates `backend` and `frontend` with environment configuration and persistent volumes.