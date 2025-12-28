"""
Supabase Vector Store implementation.
Replaces Qdrant for vector similarity search using pgvector.
"""
import logging

from services.supabase_client import get_supabase_admin_client

logger = logging.getLogger(__name__)


class SupabaseVectorStore:
    """Vector store using Supabase pgvector for similarity search."""

    def __init__(self):
        """Initialize Supabase vector store client."""
        self._client = None
        
    @property
    def client(self):
        """Lazy load Supabase client."""
        if self._client is None:
            self._client = get_supabase_admin_client()
            logger.info("Supabase Vector Store initialized")
        return self._client

    def add_document(
        self,
        filename: str,
        original_filename: str,
        file_path: str,
        year: str,
        department: str,
        category: str,
        chunks: list[dict],
        file_size_bytes: int | None = None,
        mime_type: str | None = None,
        summaries: dict | None = None,
        uploaded_by: str | None = None
    ) -> str:
        """
        Add a document and its chunks to the vector store.

        Args:
            filename: Stored filename
            original_filename: Original uploaded filename
            file_path: Path to the file
            year: Year filter value
            department: Department filter value
            category: Category (e.g., 'Rules & Regulations')
            chunks: List of dicts with 'text', 'embedding', 'chunk_number'
            file_size_bytes: Optional file size
            mime_type: Optional MIME type
            summaries: Optional dict with 'one_line' and 'brief' keys
            uploaded_by: Optional user ID who uploaded

        Returns:
            Document ID
        """
        # Insert document metadata
        doc_data = {
            "filename": filename,
            "original_filename": original_filename,
            "file_path": file_path,
            "year": year,
            "department": department,
            "category": category,
            "file_size_bytes": file_size_bytes,
            "mime_type": mime_type,
            "one_line_summary": summaries.get("one_line") if summaries else None,
            "brief_summary": summaries.get("brief") if summaries else None,
            "uploaded_by": uploaded_by
        }

        doc_result = self.client.table("documents").insert(doc_data).execute()
        document_id = doc_result.data[0]["id"]
        logger.info(f"Inserted document: {filename} with ID: {document_id}")

        # Insert chunks with embeddings
        if chunks:
            chunk_data = [
                {
                    "document_id": document_id,
                    "chunk_number": chunk.get("chunk_number", i),
                    "content": chunk["text"],
                    "embedding": chunk["embedding"]
                }
                for i, chunk in enumerate(chunks)
            ]

            # Batch insert chunks
            self.client.table("document_chunks").insert(chunk_data).execute()
            logger.info(f"Inserted {len(chunks)} chunks for document: {filename}")

        return document_id

    def add_documents(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        ids: list[str]
    ) -> None:
        """
        Legacy method for compatibility with existing code.
        Adds documents directly as chunks (used during document processing).

        This method groups chunks by source_file and creates document records.
        """
        # Group by source file
        docs_by_source: dict[str, dict] = {}

        for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas, strict=False)):
            source = metadata.get("source_file", f"unknown_{i}")

            if source not in docs_by_source:
                docs_by_source[source] = {
                    "metadata": metadata,
                    "chunks": []
                }

            docs_by_source[source]["chunks"].append({
                "text": text,
                "embedding": embedding,
                "chunk_number": metadata.get("chunk_number", len(docs_by_source[source]["chunks"]))
            })

        # Insert each document
        for source_file, doc_info in docs_by_source.items():
            meta = doc_info["metadata"]

            # Check if document already exists
            existing = self.client.table("documents").select("id").eq("filename", source_file).execute()

            if existing.data:
                document_id = existing.data[0]["id"]
                logger.info(f"Document already exists: {source_file}, adding chunks to existing")
            else:
                # Create new document record
                doc_data = {
                    "filename": source_file,
                    "original_filename": source_file,
                    "file_path": f"uploads/{source_file}",
                    "year": meta.get("year", "all"),
                    "department": meta.get("department", "all"),
                    "category": meta.get("category", "General"),
                }
                doc_result = self.client.table("documents").insert(doc_data).execute()
                document_id = doc_result.data[0]["id"]

            # Insert chunks
            chunk_data = [
                {
                    "document_id": document_id,
                    "chunk_number": chunk["chunk_number"],
                    "content": chunk["text"],
                    "embedding": chunk["embedding"]
                }
                for chunk in doc_info["chunks"]
            ]

            self.client.table("document_chunks").insert(chunk_data).execute()
            logger.info(f"Added {len(chunk_data)} chunks for: {source_file}")

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict | None = None
    ) -> dict:
        """
        Query similar documents using vector similarity search.

        Args:
            query_embedding: Query vector (384 dimensions)
            n_results: Number of results to return
            where: Optional filter dict with 'year', 'department', 'category'
                   Supports ChromaDB-style operators like $and and $in

        Returns:
            Dict with 'documents', 'metadatas', 'ids'
        """
        # Parse filters from where clause (handle $and and $in operators)
        filter_year = None
        filter_department = None
        filter_category = None

        if where:
            # Handle $and operator
            if "$and" in where:
                for condition in where["$and"]:
                    for key, value in condition.items():
                        if key == "category":
                            filter_category = value
                        elif key == "year":
                            # Handle $in operator
                            if isinstance(value, dict) and "$in" in value:
                                # Take the first non-"all" value, or "all"
                                vals = value["$in"]
                                filter_year = next((v for v in vals if v != "all"), vals[0] if vals else None)
                            else:
                                filter_year = value
                        elif key == "department":
                            if isinstance(value, dict) and "$in" in value:
                                vals = value["$in"]
                                filter_department = next((v for v in vals if v != "all"), vals[0] if vals else None)
                            else:
                                filter_department = value
            else:
                # Simple key-value filters
                filter_year = where.get("year")
                filter_department = where.get("department")
                filter_category = where.get("category")

                # Handle $in for simple filters too
                if isinstance(filter_year, dict) and "$in" in filter_year:
                    vals = filter_year["$in"]
                    filter_year = next((v for v in vals if v != "all"), vals[0] if vals else None)
                if isinstance(filter_department, dict) and "$in" in filter_department:
                    vals = filter_department["$in"]
                    filter_department = next((v for v in vals if v != "all"), vals[0] if vals else None)

        # Call the match_documents RPC function
        result = self.client.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_count": n_results,
                "match_threshold": 0.3,
                "filter_year": filter_year,
                "filter_department": filter_department,
                "filter_category": filter_category
            }
        ).execute()

        if not result.data:
            return {
                "documents": [[]],
                "metadatas": [[]],
                "ids": [[]]
            }

        # Format results to match legacy Qdrant/ChromaDB format
        documents = []
        metadatas = []
        ids = []

        for row in result.data:
            documents.append(row["content"])
            metadatas.append({
                "source_file": row["filename"],
                "year": row["year"],
                "department": row["department"],
                "category": row["category"],
                "chunk_number": row["chunk_number"],
                "similarity": row["similarity"]
            })
            ids.append(str(row["id"]))

        return {
            "documents": [documents],
            "metadatas": [metadatas],
            "ids": [ids]
        }

    def delete_by_metadata(self, where: dict) -> None:
        """Delete documents matching metadata filter."""
        # Get documents matching filter
        query = self.client.table("documents").select("id")

        if "source_file" in where:
            query = query.eq("filename", where["source_file"])
        if "year" in where:
            query = query.eq("year", where["year"])
        if "department" in where:
            query = query.eq("department", where["department"])
        if "category" in where:
            query = query.eq("category", where["category"])

        result = query.execute()

        # Delete each document (chunks cascade)
        for doc in result.data:
            self.client.table("documents").delete().eq("id", doc["id"]).execute()
            logger.info(f"Deleted document: {doc['id']}")

    def delete_by_source_file(self, source_file: str) -> bool:
        """Delete all data for a specific source file."""
        result = self.client.table("documents").delete().eq("filename", source_file).execute()
        deleted = len(result.data) > 0 if result.data else False
        logger.info(f"Deleted document by source file: {source_file}, success: {deleted}")
        return deleted

    def get_collection_count(self) -> int:
        """Get total number of chunks in the vector store."""
        result = self.client.table("document_chunks").select("id", count="exact").execute()
        return result.count or 0

    def get_all_source_files(self) -> set:
        """Get all unique source file names."""
        result = self.client.table("documents").select("filename").execute()
        return {doc["filename"] for doc in result.data}

    def get_all_documents_metadata(self) -> list[dict]:
        """Get metadata for all documents."""
        result = self.client.table("documents").select("*").execute()
        return result.data

    def get_document_by_id(self, document_id: str) -> dict | None:
        """Get a document by its ID."""
        result = self.client.table("documents").select("*").eq("id", document_id).single().execute()
        return result.data if result.data else None

    def update_document_metadata(
        self,
        document_id: str,
        year: str | None = None,
        department: str | None = None,
        category: str | None = None,
        filename: str | None = None
    ) -> dict | None:
        """Update document metadata."""
        update_data = {}
        if year is not None:
            update_data["year"] = year
        if department is not None:
            update_data["department"] = department
        if category is not None:
            update_data["category"] = category
        if filename is not None:
            update_data["filename"] = filename
            update_data["original_filename"] = filename

        if not update_data:
            return None

        result = self.client.table("documents").update(update_data).eq("id", document_id).execute()
        return result.data[0] if result.data else None

    def sync_with_uploads(self, upload_directory: str) -> dict:
        """
        Sync vector store with uploads folder.
        NOTE: This is now NON-DESTRUCTIVE. It only logs orphaned documents
        but does NOT delete them, since files are stored in Supabase Storage.
        """
        import os

        # Get all files in uploads directory (local cache only)
        existing_files = set()
        if os.path.exists(upload_directory):
            existing_files = set(os.listdir(upload_directory))

        # Get all documents from database
        stored_docs = self.get_all_source_files()

        # Find documents not in local cache (this is normal for cloud storage)
        not_in_local = stored_docs - existing_files

        if not_in_local:
            logger.info(f"📁 {len(not_in_local)} documents in DB (files in Supabase Storage)")

        return {
            "removed": [],  # We no longer auto-delete
            "kept": list(stored_docs),
            "cloud_only": list(not_in_local)
        }

    def clear_all(self) -> None:
        """Clear all documents and chunks."""
        # Delete all documents (chunks will cascade)
        self.client.table("documents").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
        logger.info("Cleared all documents from vector store")


# Singleton instance
vector_store = SupabaseVectorStore()
