"""
UniGuide Backend API
RAG system for college rules and regulations using Supabase.
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from routes import admin, chat, circular, categories, departments, organizations, streams, years, health
from routes import subjects, notes_admin, notes_chat  # Notes RAG
from routes import deadlines  # Smart Calendar & Deadline Assassin
from services.rate_limiter import limiter
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()


async def background_sync():
    """Run vector store sync in background to avoid blocking startup."""
    from services.vector_store import vector_store

    logger.info("⏳ Starting background vector store sync...")
    try:
        # Run synchronous code in thread pool
        sync_result = await asyncio.to_thread(
            vector_store.sync_with_uploads,
            settings.upload_directory
        )

        logger.info(f"📊 Vector store: {vector_store.get_collection_count()} chunks")

        cloud_only = sync_result.get("cloud_only", [])
        if cloud_only:
            logger.info(f"☁️  {len(cloud_only)} documents stored in Supabase Storage")
        logger.info("✅ Background sync completed")

    except Exception as e:
        logger.error(f"⚠️ Background sync warning: {e}")
        logger.info("📁 Application running (vector store will initialize on first use)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    os.makedirs(settings.upload_directory, exist_ok=True)

    # Initialize and sync vector store in background
    asyncio.create_task(background_sync())

    logger.info("✅ Application started successfully")
    logger.info(f"📁 Upload directory: {settings.upload_directory}")
    logger.info("🗄️  Database: Supabase pgvector")

    yield

    # Shutdown (if needed)
    logger.info("👋 Application shutting down")


# Create FastAPI app with lifespan
app = FastAPI(
    title="UniGuide API",
    description="RAG system for college rules and regulations",
    version="2.0.0",
    lifespan=lifespan
)

# Add rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration - configurable via environment
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:3000",
    "https://uniguide-sjit.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers - Health endpoints (no auth required)
app.include_router(health.router, tags=["Health"])

# Include routers - API endpoints
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])
app.include_router(circular.router, prefix="/api/circular", tags=["Circular"])
app.include_router(deadlines.router, prefix="/api/deadlines", tags=["Deadlines"])
app.include_router(categories.router, prefix="/api/categories", tags=["Categories"])
app.include_router(departments.router, prefix="/api/departments", tags=["Departments"])
app.include_router(organizations.router, prefix="/api/organizations", tags=["Organizations"])
app.include_router(streams.router, prefix="/api/streams", tags=["Streams"])
app.include_router(years.router, prefix="/api/years", tags=["Years"])

# Notes RAG endpoints (isolated from institutional RAG)
app.include_router(subjects.router, prefix="/api/subjects", tags=["Subjects"])
app.include_router(notes_admin.router, prefix="/api/admin/notes", tags=["Notes Admin"])
app.include_router(notes_chat.router, prefix="/api/notes/chat", tags=["Notes Chat"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "UniGuide API",
        "version": "2.0.0",
        "database": "Supabase",
        "endpoints": {
            "student": "/api/chat/query",
            "admin_login": "/api/admin/login",
            "admin_upload": "/api/admin/upload",
            "circular": "/api/circular/latest",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "database": "supabase"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
