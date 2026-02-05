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
    org_id: str  # REQUIRED: Organization ID for multi-tenant isolation

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    session_id: str


class EventDiscoverQuery(BaseModel):
    question: str
    org_id: str
    max_results: int = 40
    nearby: bool = False
    nearby_location: Optional[str] = None
    nearby_lat: Optional[float] = None
    nearby_lng: Optional[float] = None

class SaveEventRequest(BaseModel):
    name: str
    date: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    location: Optional[str] = None
    cash_prize: Optional[str] = None
    short_description: Optional[str] = None
    url: Optional[str] = None
    source_url: Optional[str] = None
    status: Optional[str] = None
    event_key: Optional[str] = None

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
    stream: str
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
    stream: Optional[str] = None
    semester: Optional[str] = None
