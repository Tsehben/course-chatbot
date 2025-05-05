from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict, deque
from uuid import uuid4

# import openai  # Removed for faster startup; will be lazy-loaded if LLM enabled
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

# NOTE: Import of `get_relevant_docs` is deferred (see commented RAG block below) to avoid heavy LangChain/Chroma startup cost.
# from retrieval import get_relevant_docs
from auth import router as auth_router, get_current_user

# -----------------------------------------------------------------------------
# Configuration & setup
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
# Front-end assets (Bootstrap template)
FRONT_DIR = BASE_DIR / "front-end"
TEMPLATES_DIR = BASE_DIR / "templates"
# Pick chat.html from front-end if present, else fallback to old template
INDEX_HTML = (FRONT_DIR / "index.html") if (FRONT_DIR / "index.html").exists() else (TEMPLATES_DIR / "index.html")

load_dotenv()
# Prevent uvicorn/watchfiles reloader from scanning huge dirs like node_modules or build artifacts
os.environ.setdefault(
    "WATCHFILES_IGNORE_PATHS",
    "node_modules/*,front-end/*,frontend_dist/*,materials/*,chroma/*,.venv/*,.git/*",
)

def _chat_completion(**kwargs):
    """Lazily import OpenAI client on first call to keep startup fast.

    Supports both `openai>=1.0` (new SDK) and legacy `openai<1.0`.
    The detected client is cached on the function object to avoid re-importing.
    """
    if not hasattr(_chat_completion, "_client"):
        # First invocation: detect and initialise client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment.")

        try:
            # Preferred: new SDK (>=1.0)
            from openai import OpenAI  # type: ignore

            _chat_completion._client = OpenAI(api_key=api_key)  # type: ignore[attr-defined]
            _chat_completion._is_new = True  # type: ignore[attr-defined]
        except ImportError:
            import openai as _openai_legacy  # type: ignore

            _openai_legacy.api_key = api_key  # type: ignore[attr-defined]
            _chat_completion._client = _openai_legacy  # type: ignore[attr-defined]
            _chat_completion._is_new = False  # type: ignore[attr-defined]

    if getattr(_chat_completion, "_is_new", False):  # type: ignore[attr-defined]
        return _chat_completion._client.chat.completions.create(**kwargs)  # type: ignore[attr-defined]
    return _chat_completion._client.ChatCompletion.create(**kwargs)  # type: ignore[attr-defined]

MODEL_NAME = "gpt-4o-mini"

app = FastAPI(title="Course Q&A Chatbot")

# CORS (allow React dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers BEFORE mounting the front-end SPA to ensure paths like /auth/* resolve to FastAPI endpoints.
app.include_router(auth_router)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _build_messages(question: str, context_docs: List[str], history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    system_prompt = (
        "You are a helpful teaching assistant for GENED 1188. When possible, ground your "
        "answers in the COURSE CONTEXT provided below. If the context lacks the exact "
        "information, you may still offer reasonable study guidance or explanations "
        "based on standard educational best-practices, but NEVER invent specific course "
        "facts (e.g., instructor names, grading policies) that are not in the context. "
        "If asked about such missing specifics, say: 'I'm not sure based on the course materials.' "
        "Do NOT fabricate information."
    )
    context = "\n\n".join(context_docs) if context_docs else ""

    messages = history + [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Context:\n{context}"},
        {"role": "user", "content": question},
    ]
    return messages


def _sync_llm_call(messages: List[Dict[str, str]]) -> str:
    """Call OpenAI ChatCompletion synchronously."""
    try:
        response = _chat_completion(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
            max_tokens=256,
        )
        # new client returns object with .choices[0].message.content; old returns dict similar.
        choice = response.choices[0]
        # handle both cases where message might be attribute or dict
        answer = (
            choice.message.content.strip()
            if hasattr(choice.message, "content")
            else choice["message"]["content"].strip()
        )
    except Exception as exc:  # pylint: disable=broad-except
        answer = f"Error contacting language model: {exc}"
    return answer

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.get("/", response_class=FileResponse, include_in_schema=False)
async def index() -> FileResponse:
    """Serve the chat UI."""
    return FileResponse(INDEX_HTML)

# Compatibility: still serve SPA when templates link to chat.html
@app.get("/chat.html", response_class=FileResponse, include_in_schema=False)
async def chat_alias() -> FileResponse:
    return FileResponse(INDEX_HTML)

@app.post("/ask")
async def ask(
    question: str = Form(...),
    session: str | None = Form(None),
    user: str = Depends(get_current_user),
) -> JSONResponse:
    """RAG endpoint that returns answer JSON."""
    if not session:
        session = str(uuid4())

    # quick greeting shortcut
    GREETINGS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}
    if question.lower().strip() in GREETINGS:
        answer = "Hi, I am an AI assistant for GENED 1188. How can I help you today?"
        CONVERSATIONS[session].append({"role": "user", "content": question})
        CONVERSATIONS[session].append({"role": "assistant", "content": answer})
        return JSONResponse({"answer": answer, "session": session})

    # ---------------------------------------------------------------------
    # Retrieve relevant docs & generate answer with OpenAI ChatCompletion
    # ---------------------------------------------------------------------

    # 1. Lazy-import retrieval to keep startup snappy.
    try:
        from retrieval import get_relevant_docs  # heavy import (langchain/chroma)

        context_docs = await run_in_threadpool(get_relevant_docs, question, 8)
    except Exception as exc:  # pylint: disable=broad-except
        # If vector store missing or retrieval fails, fall back to empty context
        context_docs = []
        # Optionally log: print(f"Retrieval error: {exc}")

    # 2. Build messages (system + dialogue history + context) & call LLM
    history = list(CONVERSATIONS[session])
    messages = _build_messages(question, context_docs, history)
    answer: str = await run_in_threadpool(_sync_llm_call, messages)

    # 3. Update short-term memory for this session
    CONVERSATIONS[session].append({"role": "user", "content": question})
    CONVERSATIONS[session].append({"role": "assistant", "content": answer})

    return JSONResponse({"answer": answer, "session": session})

CONVERSATIONS: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))

# -----------------------------------------------------------------------------
# Static mount (add at the very end so it does NOT shadow API routes)
# -----------------------------------------------------------------------------
if FRONT_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONT_DIR), html=True), name="frontend")
else:
    app.mount("/templates", StaticFiles(directory=str(TEMPLATES_DIR)), name="templates")

# -----------------------------------------------------------------------------
# CLI entry (optional)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
