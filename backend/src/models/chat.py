from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ChatRequest(BaseModel):
    message: str
    context_mode: str = "book_wide"  # "book_wide" or "selected_text"
    selected_text: Optional[str] = None
    user_id: Optional[str] = None
    language_preference: str = "en"  # "en" for English, "ur" for Urdu


class ChatResponse(BaseModel):
    response: str
    sources: List[str] = []
    context_used: str = "book_wide"
    timestamp: datetime = Field(default_factory=datetime.now)


class DocumentChunk(BaseModel):
    id: str
    content: str
    metadata: dict
    score: float


class IngestionRequest(BaseModel):
    urls: List[str]
    force_reprocess: bool = False