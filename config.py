from pathlib import Path
import os

from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

UPWORK_API_DOC_PATH = Path(
    os.getenv(
        "UPWORK_API_DOC_PATH",
        r"C:\Users\hp\Desktop\ProAnalyst_assignment\API Documentation Partial.pdf",
    )
)
FAISS_INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR", BASE_DIR / ".faiss_index"))

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY", "")
DEEPINFRA_MODEL = os.getenv(
    "DEEPINFRA_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
)
DEEPINFRA_CHAT_URL = os.getenv(
    "DEEPINFRA_CHAT_URL", "https://api.deepinfra.com/v1/openai/chat/completions"
)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3
