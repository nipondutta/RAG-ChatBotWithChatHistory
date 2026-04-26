from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

def load_documents(data_dir):
    data_dir = Path(data_dir)   # ✅ ensure Path object

    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    documents = []
    pdf_files = list(data_dir.glob("*.pdf"))

    if not pdf_files:
        raise ValueError("No PDF files found in data directory")

    for pdf in pdf_files:
        loader = PyPDFLoader(str(pdf))
        documents.extend(loader.load())

    return documents
