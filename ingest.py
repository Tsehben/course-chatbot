import sys
from pathlib import Path
from typing import List

import textract
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# -----------------------------------------------------------------------------
# Version-compatible LangChain imports
# -----------------------------------------------------------------------------

try:
    # New split package (preferred)
    from langchain_openai import OpenAIEmbeddings  # type: ignore
except ImportError:  # pragma: no cover
    try:
        # LangChain ≥ 0.2 moved community integrations here
        from langchain_community.embeddings import OpenAIEmbeddings  # type: ignore
    except ImportError:
        # Fallback for very old versions
        from langchain.embeddings import OpenAIEmbeddings  # type: ignore

try:
    from langchain_community.vectorstores import Chroma  # type: ignore
except ImportError:  # pragma: no cover
    from langchain.vectorstores import Chroma  # type: ignore

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

load_dotenv()  # pick up OPENAI_API_KEY from .env

BASE_DIR = Path(__file__).parent
MATERIALS_DIR = BASE_DIR / "materials"
PERSIST_DIR = BASE_DIR / "chroma"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
EMBED_MODEL_NAME = "text-embedding-3-small"

SUPPORTED_EXTS = {".txt", ".md", ".pdf", ".docx", ".pptx"}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _extract_text(path: Path) -> str:
    """Return raw UTF-8 text from *path* or an empty string on failure."""
    try:
        return textract.process(str(path)).decode("utf-8", errors="ignore")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[skip] Could not extract {path.name}: {exc}")
        return ""


def _load_documents() -> List[Document]:
    """Iterate over files in materials/ and yield chunked LangChain Documents."""
    documents: List[Document] = []

    if not MATERIALS_DIR.exists():
        print(f"[warn] Materials directory not found at {MATERIALS_DIR}")
        return documents

    for path in MATERIALS_DIR.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTS:
            # e.g. .DS_Store or other unsupported files
            continue

        text = _extract_text(path)
        if not text.strip():
            continue

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        for chunk in splitter.split_text(text):
            documents.append(Document(page_content=chunk, metadata={"source": path.name}))

    return documents

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    docs = _load_documents()
    if not docs:
        print("[warn] No documents loaded; nothing to ingest.")
        return

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL_NAME)

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR),
    )
    vectordb.persist()
    print("✓ embeddings stored")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
