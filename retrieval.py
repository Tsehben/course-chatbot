from __future__ import annotations

from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Version-compatible imports
try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import OpenAIEmbeddings
    except ImportError:
        from langchain.embeddings import OpenAIEmbeddings

# Prefer the new standalone package; fall back to community stub if not installed.
try:
    from langchain_chroma import Chroma  # pip install -U langchain-chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

load_dotenv()
PERSIST_DIR = Path(__file__).parent / "chroma"
EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_K = 4

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def get_relevant_docs(query: str, k: int = DEFAULT_K) -> List[str]:
    """Return top-k relevant document page contents for a given query."""
    if not query.strip():
        return []

    if not PERSIST_DIR.exists():
        raise RuntimeError("Vector store not found. Run `python ingest.py` first.")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectordb = Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
    )

    docs = vectordb.similarity_search(query=query, k=k)
    return [doc.page_content for doc in docs]
