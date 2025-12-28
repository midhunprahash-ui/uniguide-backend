"""
Vector Store implementation using Supabase pgvector.
This replaces the previous Qdrant implementation.

All methods maintain backward compatibility with the existing API.
"""
import logging

from config import get_settings

# Import from the Supabase implementation
from services.supabase_vector_store import SupabaseVectorStore

logger = logging.getLogger(__name__)


class VectorStore(SupabaseVectorStore):
    """
    Vector Store using Supabase pgvector.

    This class extends SupabaseVectorStore and provides any additional
    compatibility methods needed for the existing codebase.
    """

    def __init__(self):
        """Initialize the Supabase-based vector store."""
        super().__init__()
        self.collection_name = "documents"  # For logging compatibility
        self.vector_size = 384  # all-MiniLM-L6-v2 dimension
        logger.info("VectorStore initialized with Supabase pgvector backend")


# Singleton instance
vector_store = VectorStore()
