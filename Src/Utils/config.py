from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = str(BASE_DIR / "Data" / "raw")
VECTOR_DB_DIR = str(BASE_DIR / "VectorStore" / "chroma")
