from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from Src.Generation.llm import get_llm
from Src.Generation.prompt import RAG_PROMPT
from Src.Utils.config import VECTOR_DB_DIR
import logging
from typing import Dict

logger = logging.getLogger(__name__)

TOP_K_RESULTS = 3

# In-memory chat histories by session
chat_histories: Dict[str, BaseChatMessageHistory] = {}

def build_rag_chain():
    """Build and return the complete RAG chain."""
    try:
        # Initialize LLM from config
        llm = get_llm()

        # Initialize embeddings using the same HuggingFace model class as the vector store
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False}
        )

        # Load vector database
        vectordb = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=embeddings
        )

        # Create retriever
        retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K_RESULTS})

        # Create the RAG chain in classic style using the LLM and prompt template
        combine_docs_chain = create_stuff_documents_chain(llm, RAG_PROMPT)
        qa_chain = create_retrieval_chain(retriever, combine_docs_chain)

        logger.info("RAG chain built successfully")
        return qa_chain
        
    except Exception as e:
        logger.error(f"Failed to build RAG chain: {str(e)}")
        raise


def get_chat_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create chat history for a session."""
    if session_id not in chat_histories:
        chat_histories[session_id] = ChatMessageHistory()
        logger.info(f"Created new chat history for session: {session_id}")
    return chat_histories[session_id]


def clear_chat_history(session_id: str) -> None:
    """Clear chat history for a session."""
    if session_id in chat_histories:
        del chat_histories[session_id]
        logger.info(f"Cleared chat history for session: {session_id}")


def get_all_chat_histories() -> Dict[str, BaseChatMessageHistory]:
    """Return all chat histories."""
    return chat_histories
