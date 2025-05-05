from __future__ import annotations

"""Simple email/password auth with JWT issued via FastAPI.
This is for demo purposes only – not production-ready."""

import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

SECRET = os.getenv("JWT_SECRET", "change-me")
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 6
DB_PATH = Path(__file__).with_name("users.db")
PWD_CTX = CryptContext(schemes=["bcrypt"], deprecated="auto")

router = APIRouter(prefix="/auth", tags=["auth"])

# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------

def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS users (email TEXT PRIMARY KEY, hash TEXT)"
    )
    return conn

# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------

class UserIn(BaseModel):
    email: EmailStr
    password: str

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@router.post("/signup", status_code=status.HTTP_201_CREATED)
def signup(user: UserIn):
    conn = _conn()
    try:
        conn.execute(
            "INSERT INTO users VALUES (?,?)", (user.email, PWD_CTX.hash(user.password))
        )
        conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(400, "User already exists")
    finally:
        conn.close()
    return {"msg": "signup_success"}

@router.post("/login", response_model=TokenOut)
def login(user: UserIn):
    conn = _conn()
    row = conn.execute(
        "SELECT hash FROM users WHERE email=?", (user.email,)
    ).fetchone()
    conn.close()
    if not row or not PWD_CTX.verify(user.password, row[0]):
        raise HTTPException(401, "Invalid credentials")

    payload = {
        "sub": user.email,
        "exp": datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS),
    }
    token = jwt.encode(payload, SECRET, algorithm=ALGORITHM)
    return TokenOut(access_token=token)

# -----------------------------------------------------------------------------
# Dependency
# -----------------------------------------------------------------------------

security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET, algorithms=[ALGORITHM])
        return str(payload.get("sub"))
    except JWTError:
        raise HTTPException(403, "Invalid token")
