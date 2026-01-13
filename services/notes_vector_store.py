"""
Notes Vector Store service for Notes RAG subsystem.
Uses note_chunks table, completely isolated from document_chunks (institutional RAG).
"""
import logging
from typing import Optional
from services.supabase_client import get_supabase_admin_client

logger = logging.getLogger(__name__)


class NotesVectorStore:
    """
    Vector store for academic notes.
    Uses note_chunks table with a separate HNSW index from document_chunks.
    """
    
    def __init__(self):
        self._client = None
    
    @property
    def client(self):
        """Lazy load Supabase client."""
        if self._client is None:
            self._client = get_supabase_admin_client()
        return self._client
    
    def add_note(
        self,
        note_id: str,
        org_id: str,
        subject_id: str,
        year_id: str,
        department_id: str,
        stream_id: str,
        unit_id: str,
        chunks: list[dict],
        embeddings: list[list[float]]
    ) -> int:
        """
        Add note chunks with embeddings to the vector store.
        
        Args:
            note_id: The note record ID
            org_id: Organization ID for tenant isolation
            subject_id, year_id, department_id, stream_id, unit_id: Denormalized hierarchy
            chunks: List of dicts with 'content' and optional 'token_count'
            embeddings: List of embedding vectors (768 dimensions)
        
        Returns:
            Number of chunks added
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        chunk_data = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_data.append({
                "org_id": org_id,
                "note_id": note_id,
                "subject_id": subject_id,
                "year_id": year_id,
                "department_id": department_id,
                "stream_id": stream_id,
                "unit_id": unit_id,
                "chunk_number": i + 1,
                "content": chunk["content"],
                "embedding": embedding,
                "token_count": chunk.get("token_count"),
                "metadata": chunk.get("metadata", {})
            })
        
        result = self.client.table("note_chunks").insert(chunk_data).execute()
        
        return len(result.data) if result.data else 0
    
    def delete_note_chunks(self, note_id: str) -> bool:
        """Delete all chunks for a note."""
        self.client.table("note_chunks").delete().eq("note_id", note_id).execute()
        return True
    
    def query(
        self,
        query_embedding: list[float],
        org_id: str,
        year_id: Optional[str] = None,
        subject_id: Optional[str] = None,
        unit_id: Optional[str] = None,
        n_results: int = 5,
        threshold: float = 0.3
    ) -> dict:
        """
        Query similar note chunks using match_notes function.
        
        Args:
            query_embedding: Query vector (768 dimensions)
            org_id: REQUIRED - Organization ID for tenant isolation
            year_id: Optional - Filter by year
            subject_id: Optional - Filter by subject
            unit_id: Optional - Filter by unit
            n_results: Maximum results to return
            threshold: Minimum similarity score
        
        Returns:
            Dict with 'documents', 'metadatas', 'similarities'
        """
        if not org_id:
            raise ValueError("org_id is required for tenant isolation")
        
        result = self.client.rpc(
            "match_notes",
            {
                "query_embedding": query_embedding,
                "filter_org_id": org_id,
                "filter_year_id": year_id,
                "filter_subject_id": subject_id,
                "filter_unit_id": unit_id,
                "match_count": n_results,
                "match_threshold": threshold
            }
        ).execute()
        
        if not result.data:
            return {"documents": [], "metadatas": [], "similarities": []}
        
        return {
            "documents": [r["content"] for r in result.data],
            "metadatas": result.data,
            "similarities": [r["similarity"] for r in result.data]
        }
    
    def get_note_chunks(self, note_id: str) -> list[dict]:
        """Get all chunks for a specific note."""
        result = self.client.table("note_chunks").select("*").eq("note_id", note_id).order("chunk_number").execute()
        return result.data or []
    
    def get_chunk_count(self, org_id: str) -> int:
        """Get total number of note chunks for an organization."""
        result = self.client.table("note_chunks").select("id", count="exact").eq("org_id", org_id).execute()
        return result.count or 0


# Singleton instance
notes_vector_store = NotesVectorStore()
