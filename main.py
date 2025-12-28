"""
UniGuide Backend API
RAG system for college rules and regulations using Supabase.
"""
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from routes import admin, chat, circular

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title="UniGuide API",
    description="RAG system for college rules and regulations",
    version="2.0.0"
)

# CORS configuration - allow frontend origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
        "https://uniguide-sjit.vercel.app"
    ],
    allow_origin_regex="https://.*\\.vercel\\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])
app.include_router(circular.router, prefix="/api/circular", tags=["Circular"])


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    # Create upload directory
    os.makedirs(settings.upload_directory, exist_ok=True)

    # Initialize and sync vector store in background
    import asyncio
    asyncio.create_task(background_sync())
    
    # Preload ML models in background to avoid timeout on first request
    asyncio.create_task(background_model_preload())
    
    logger.info("✅ Application started successfully")
    logger.info(f"📁 Upload directory: {settings.upload_directory}")
    logger.info("🗄️  Database: Supabase pgvector")


async def background_sync():
    """Run vector store sync in background to avoid blocking startup."""
    from services.vector_store import vector_store
    
    logger.info("⏳ Starting background vector store sync...")
    try:
        # Run synchronous code in thread pool
        import asyncio
        sync_result = await asyncio.to_thread(
            vector_store.sync_with_uploads, 
            settings.upload_directory
        )

        logger.info(f"📊 Vector store: {vector_store.get_collection_count()} chunks")

        removed = sync_result.get("removed", [])
        if removed:
            logger.info(f"🔄 Synced: Removed {len(removed)} orphaned documents")
        logger.info("✅ Background sync completed")
        
    except Exception as e:
        logger.error(f"⚠️ Background sync warning: {e}")
        logger.info("📁 Application running (vector store will initialize on first use)")


async def background_model_preload():
    """Preload ML models in background to avoid timeout on first request."""
    import asyncio
    from services.rag_engine import rag_engine
    
    try:
        # Run model loading in thread pool (it's blocking I/O)
        await asyncio.to_thread(rag_engine.preload_models)
    except Exception as e:
        logger.error(f"⚠️ Model preload error: {e}")


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
