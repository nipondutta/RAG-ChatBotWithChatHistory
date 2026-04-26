# ChatBotUsingRAG

A FastAPI-based Retrieval-Augmented Generation (RAG) chatbot using:
- HuggingFace embeddings
- Ollama (LLaMA 3) as the LLM

## Project Structure

- `App/app.py` - FastAPI application and HTTP endpoints.
- `Src/Generation/llm.py` - LLM loader using HuggingFaceHub.
- `Src/Generation/prompt.py` - RAG prompt template.
- `Src/Ingestion/loader.py` - PDF loader.
- `Src/Ingestion/splitter.py` - text splitter.
- `Src/Ingestion/embedder.py` - vector store creation with HuggingFace embeddings.
- `Src/Pipeline/ragChain.py` - builds the RAG chain and manages session chat history.
- `Src/Retrieval/retriever.py` - retriever creation.
- `Src/Utils/config.py` - environment and path configuration.
- `VectorStore/chroma` - persisted Chroma vector store.
- `Data/raw` - uploaded PDF files.

## Requirements

Install Ollama:

```bash
ollama pull llama3
ollama serve
```

Make sure your `requirements.txt` looks like:

```bash
fastapi
uvicorn
langchain
langchain-community
langchain-ollama
chromadb
pypdf
python-dotenv
sentence-transformers
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the repo root with:

```bash
OLLAMA_MODEL=model_name
OLLAMA_BASE_URL=Localhost_URL
TEMPERATURE=float
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

If you use a different HF model, update `HUGGINGFACE_MODEL` accordingly.

## Run the App

From the project root:

```bash
python App/app.py
```

The API will start on:

- `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## API Endpoints

- `GET /health` - health check and RAG readiness.
- `POST /upload` - upload a PDF file and build the RAG chain.
- `POST /ask` - ask a question after upload.
- `GET /chat-history/{session_id}` - fetch chat history for a session.
- `DELETE /chat-history/{session_id}` - clear a session's chat history.
- `GET /chat-sessions` - list active chat sessions.

## Usage Example

1. Upload a PDF using `POST /upload`.
2. Ask questions using `POST /ask` with JSON:

```json
{
  "question": "What is the main topic of the PDF?",
  "session_id": "default"
}
```

3. Retrieve the session history:

```http
GET /chat-history/default
```

## Notes

- On startup, the app clears `Data/raw`, the Chroma directory, and session history.
- The app writes logs to `Logs/app.log` and recreates a fresh log file each run.
- The app supports only PDF uploads.
