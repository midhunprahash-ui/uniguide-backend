from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

# Valid categories for document classification
VALID_CATEGORIES = ['rules', 'admissions', 'schedules', 'timetables', 'abhs', 'circulars']

class ChatQuery(BaseModel):
    question: str
    stream: str = "all"  # Stream filter (e.g., "UG", "PG", or "all")
    year: str
    department: str
    category: str = "schedules"  # rules, admissions, schedules, abhs, circulars
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    session_id: str

class AdminLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class DocumentUpload(BaseModel):
    filename: str
    year: str
    department: str
    category: str
    upload_date: datetime
    file_type: str

class DocumentMetadata(BaseModel):
    id: str
    filename: str
    year: str
    department: str
    category: str
    upload_date: str
    chunk_count: int

class CircularMetadata(BaseModel):
    id: str
    filename: str
    year: str
    department: str
    upload_date: str
    summary: Optional[str] = None
    chunk_count: int

class CircularChatQuery(BaseModel):
    question: str
    session_id: Optional[str] = None

class LatestCircularResponse(BaseModel):
    id: str
    filename: str
    year: str
    department: str
    upload_date: str
    summary: Optional[str] = None
    has_circular: bool = True

class CategoryStats(BaseModel):
    category: str
    count: int
    documents: List[DocumentMetadata]

class DocumentsByCategory(BaseModel):
    categories: List[CategoryStats]
    total_documents: int

class AdminStats(BaseModel):
    total_documents: int
    total_chunks: int
    documents_by_category: dict
    total_size_mb: Optional[float] = None

class RenameDocumentRequest(BaseModel):
    new_filename: str

class UpdateDocumentRequest(BaseModel):
    year: Optional[str] = None
    department: Optional[str] = None
    category: Optional[str] = None

