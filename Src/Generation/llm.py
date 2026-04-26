from langchain_ollama import ChatOllama
import os

def get_llm():
    # You can change model to whichever Ollama model you pulled
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=float(os.getenv("TEMPERATURE", 0.7)),
    )
