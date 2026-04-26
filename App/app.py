from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pathlib import Path
import shutil
import logging
import sys
from typing import List

sys.path.append(str(Path(__file__).parent.parent))

from Src.Ingestion.loader import load_documents
from Src.Ingestion.splitter import split_documents
from Src.Ingestion.embedder import create_vectorstore
from Src.Pipeline.ragChain import build_rag_chain, get_chat_history, clear_chat_history, get_all_chat_histories
from Src.Utils.config import DATA_DIR

# Configure logging with both console and file handlers
log_dir = Path(__file__).parent.parent / "Logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "app.log"

# Delete previous log file on startup
if log_file.exists():
    log_file.unlink()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG BOT API with Chat History", version="1.0")

UPLOAD_DIR = Path(DATA_DIR)

# Global RAG chain
rag_chain = None

@app.on_event("startup")
def startup_event():
    """Clear all data and database on startup."""
    global rag_chain
    try:
        from Src.Utils.config import VECTOR_DB_DIR
        
        # Clear uploaded documents
        if UPLOAD_DIR.exists():
            logger.info(f"Clearing uploaded documents at {UPLOAD_DIR}")
            shutil.rmtree(UPLOAD_DIR)
            logger.info("Uploaded documents cleared")
        
        # Recreate upload directory
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        # Clear ChromaDB on startup (VectorStore/chroma)
        vector_db_path = Path(VECTOR_DB_DIR)
        if vector_db_path.exists():
            logger.info(f"Clearing vector database at {VECTOR_DB_DIR}")
            shutil.rmtree(vector_db_path)
            logger.info("Vector database cleared")
        
        # Recreate empty directory
        vector_db_path.mkdir(parents=True, exist_ok=True)
        
        # Reset chat history
        from Src.Pipeline.ragChain import chat_histories
        chat_histories.clear()
        logger.info("Chat history cleared")
        
        # Don't build RAG chain yet - wait for document upload
        rag_chain = None
        logger.info("RAG system ready. Waiting for document upload...")
    except Exception as e:
        logger.error(f"Startup error: {str(e)}", exc_info=True)


class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default"  # Session ID for chat history


class Message(BaseModel):
    role: str  # "human" or "ai"
    content: str


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: List[Message]


@app.get("/health")
def health():
    return {"status": "ok", "rag_ready": rag_chain is not None}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global rag_chain

    try:
        if not file.filename.endswith(".pdf"):
            logger.error(f"Invalid file type: {file.filename}")
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        file_path = UPLOAD_DIR / file.filename
        logger.info(f"Uploading file: {file.filename}")

        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load and process documents
        documents = load_documents(UPLOAD_DIR)
        logger.info(f"Loaded {len(documents)} documents")
        
        chunks = split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks")

        if not chunks:
            raise HTTPException(status_code=400, detail="No text extracted from PDF")

        # Create vector store and rebuild chain
        create_vectorstore(chunks)
        rag_chain = build_rag_chain()
        logger.info(f"Document indexed and RAG chain rebuilt")

        return {
            "message": "Document uploaded and indexed successfully",
            "filename": file.filename,
            "chunks": len(chunks)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
def ask_question(payload: QuestionRequest):
    if not rag_chain:
        logger.error("RAG chain not initialized")
        raise HTTPException(status_code=503, detail="RAG chain not initialized. Please upload a document first.")
    
    logger.info(f"Question received: {payload.question} | Session: {payload.session_id}")

    try:
        # Get chat history for this session
        history = get_chat_history(payload.session_id)
        
        # Format chat history as context
        chat_history_str = ""
        for msg in history.messages:
            role = "User" if msg.type == "human" else "Assistant"
            chat_history_str += f"{role}: {msg.content}\n"
        
        if not chat_history_str:
            chat_history_str = "No previous conversation history"
        
        # Invoke the chain with the question and chat history
        response = rag_chain.invoke({
            "input": payload.question,
            "chat_history": chat_history_str
        })
        answer = response["answer"]
        
        # Add to chat history
        history.add_user_message(payload.question)
        history.add_ai_message(answer)
        
        logger.info(f"Answer generated successfully | Session: {payload.session_id}")

        return {
            "question": payload.question,
            "answer": answer,
            "session_id": payload.session_id
        }

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat-history/{session_id}")
def get_session_history(session_id: str):
    """Get chat history for a specific session."""
    try:
        history = get_chat_history(session_id)
        messages = []
        
        for msg in history.messages:
            messages.append({
                "role": msg.type,
                "content": msg.content
            })
        
        return {
            "session_id": session_id,
            "messages": messages
        }
    except Exception as e:
        logger.error(f"Error fetching chat history: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/chat-history/{session_id}")
def delete_session_history(session_id: str):
    """Clear chat history for a specific session."""
    try:
        clear_chat_history(session_id)
        logger.info(f"Cleared chat history for session: {session_id}")
        return {"message": f"Chat history cleared for session: {session_id}"}
    except Exception as e:
        logger.error(f"Error clearing chat history: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat-sessions")
def get_all_sessions():
    """Get list of all active chat sessions."""
    try:
        histories = get_all_chat_histories()
        sessions = []
        
        for session_id, history in histories.items():
            sessions.append({
                "session_id": session_id,
                "message_count": len(history.messages)
            })
        
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"Error fetching sessions: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

