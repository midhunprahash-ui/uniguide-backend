import logging
import os
import uuid

import google.generativeai as genai
import pdfplumber
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError
from PIL import Image

from config import get_settings
from services.rag_engine import rag_engine
from services.semantic_chunker import SemanticChunker, EnhancedChunk
from services.vector_store import vector_store

settings = get_settings()
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        """Initialize document processor with semantic chunking."""
        os.makedirs(settings.upload_directory, exist_ok=True)
        # Configure Gemini
        genai.configure(api_key=settings.gemini_api_key)
        # Prefer newest, fall back for free-tier compatibility
        self.vision_models = [
            genai.GenerativeModel("gemini-2.0-flash"),
            genai.GenerativeModel("gemini-1.5-flash"),
        ]

        # Initialize semantic chunker with optimized settings
        self.chunker = SemanticChunker(
            target_chunk_size=512,   # ~512 tokens per chunk
            max_chunk_size=1024,     # Hard limit
            min_chunk_size=100,      # Skip tiny chunks
            overlap_sentences=2      # Context continuity
        )

        logger.info("DocumentProcessor initialized with Gemini Vision and Semantic Chunker")

    def process_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        text = ""
        try:
            # Try standard text extraction first
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
        except Exception as e:
            print(f"Error processing PDF with pdfplumber: {e}")
            text = ""

        # If text is sparse or empty, it might be a scanned PDF or complex table
        # Use a heuristic: if text length is low, try Vision
        if len(text) < 100:
            print("Low text content in PDF, attempting Vision extraction...")
            text = self.process_pdf_with_vision(file_path)

        return text

    def process_pdf_with_vision(self, file_path: str) -> str:
        """Extract text from PDF using Gemini Vision."""
        try:
            # Explicitly set poppler path for macOS Homebrew install
            poppler_path = '/opt/homebrew/bin' if os.path.exists('/opt/homebrew/bin/pdfinfo') else None
            images = convert_from_path(file_path, poppler_path=poppler_path)
            text = ""
            for i, image in enumerate(images):
                print(f"Processing page {i+1} with Gemini Vision...")
                text += self.process_image_with_gemini(image) + "\n\n"
            return text
        except PDFInfoNotInstalledError:
            raise RuntimeError(
                "Poppler is not installed. Install poppler-utils on Linux or `brew install poppler` on macOS."
            )
        except Exception as e:
            print(f"Error in Vision processing for PDF: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def process_image(self, file_path: str) -> str:
        """Extract text from image using Gemini Vision."""
        try:
            image = Image.open(file_path)
            return self.process_image_with_gemini(image)
        except Exception as e:
            print(f"Error processing image: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def process_image_with_gemini(self, image: Image.Image) -> str:
        """Run Gemini Vision inference on a PIL Image."""
        prompt = """You are an expert at extracting structured information from academic documents.

TASK: Extract ALL text and information from this document image.

CRITICAL RULES FOR TABLES AND SCHEDULES:
1. For tables with EXAM/SCHEDULE or EVENT/DATE columns, output each row as a COMPLETE SENTENCE that combines all columns.
   Example: Instead of separate columns "HALL TICKET ISSUE" and "01st-10th May", output:
   "HALL TICKET ISSUE for all: 01st - 10th May 2026"

2. For calendar/working days tables, output each entry with its month and year:
   Example: "Working Days in January 2026: 2, 3, 5, 6, 7, 8, 9, 10..."

3. For holiday lists, output each holiday with its complete date:
   Example: "New Year Holiday: 1 January 2026"
   Example: "Pongal Holidays: 11-18 January 2026"

4. For events with dates, always combine the event name with its date:
   Example: "Christmas Celebration in College: 20th Dec 2025 (12:00 Noon)"

5. Include the year (2025 or 2026) whenever dates are mentioned to avoid confusion.

6. Group related information together. Each line should be self-contained and searchable.

OUTPUT FORMAT:
- Use clear sections with headers (## Section Name)
- Each piece of information should be complete on its own line
- Preserve all dates, times, and details
- Output in clean Markdown format

Extract everything now:"""

        for model in self.vision_models:
            try:
                response = model.generate_content([prompt, image])
                if response and getattr(response, "text", None):
                    return response.text
            except Exception as e:
                model_name = getattr(model, "model_name", "unknown-model")
                print(f"Error in Gemini Vision inference with {model_name}: {e}")
                continue
        return ""

    def process_text(self, file_path: str) -> str:
        """Read text from plain text file."""
        try:
            with open(file_path, encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading text file: {e}")
            return ""

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
        """
        Split text into semantic chunks.

        Uses sentence-aware chunking that preserves context and respects
        document structure. Falls back to simple splitting for edge cases.

        Args:
            text: The document text to chunk
            chunk_size: Ignored (kept for backwards compatibility)
            overlap: Ignored (kept for backwards compatibility)

        Returns:
            List of chunk text strings
        """
        if not text.strip():
            return []

        # Use semantic chunker for intelligent sentence-aware splitting
        enhanced_chunks = self.chunker.chunk_with_fallback(text)

        if not enhanced_chunks:
            return []

        # Return just the text for backwards compatibility
        return [chunk.text for chunk in enhanced_chunks]

    def chunk_text_enhanced(self, text: str) -> list[EnhancedChunk]:
        """
        Split text into semantic chunks with rich metadata.

        Returns EnhancedChunk objects containing:
        - text: The chunk content
        - chunk_number: Sequential chunk index
        - char_start/char_end: Character positions in original text
        - section_header: Extracted section header if found
        - token_count: Estimated token count

        Args:
            text: The document text to chunk

        Returns:
            List of EnhancedChunk objects with metadata
        """
        if not text.strip():
            return []

        return self.chunker.chunk_with_fallback(text)

    def process_and_store_document(
        self,
        file_path: str,
        filename: str,
        year: str,
        department: str,
        category: str,
        file_type: str,
        org_id: str | None = None,
        stream: str = "all",
        semester: str = "all"
    ) -> dict:
        """
        Process document and store in vector database with semantic chunking.

        Uses intelligent sentence-aware chunking that preserves context and
        extracts rich metadata including section headers and character positions.
        """
        # Extract text based on file type
        if file_type in ['pdf', 'application/pdf']:
            text = self.process_pdf(file_path)
        elif file_type in ['png', 'jpg', 'jpeg', 'image/png', 'image/jpeg']:
            text = self.process_image(file_path)
        elif file_type in ['txt', 'text/plain']:
            text = self.process_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        if not text.strip():
            raise ValueError("No text could be extracted from the document")

        # Use enhanced semantic chunking with metadata
        enhanced_chunks = self.chunk_text_enhanced(text)

        if not enhanced_chunks:
            raise ValueError("No valid chunks created from the document")

        logger.info(f"Created {len(enhanced_chunks)} semantic chunks from {filename}")

        # Generate embeddings for each chunk
        embeddings = []
        metadatas = []
        ids = []
        chunk_texts = []
        doc_id = str(uuid.uuid4())

        for chunk in enhanced_chunks:
            embedding = rag_engine.generate_embedding(chunk.text)
            embeddings.append(embedding)
            chunk_texts.append(chunk.text)

            # Include enhanced metadata from semantic chunker
            metadata = {
                "year": year,
                "department": department,
                "category": category,
                "stream": stream,
                "semester": semester,
                "source_file": filename,
                "chunk_id": chunk.chunk_number,
                "chunk_number": chunk.chunk_number,
                "doc_id": doc_id,
                "org_id": org_id,
                # Enhanced metadata from semantic chunker
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "section_header": chunk.section_header,
                "token_count": chunk.token_count,
            }
            metadatas.append(metadata)
            ids.append(f"{doc_id}_{chunk.chunk_number}")

        # Store in vector database
        vector_store.add_documents(
            texts=chunk_texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        # Calculate total tokens for logging
        total_tokens = sum(c.token_count or 0 for c in enhanced_chunks)
        logger.info(
            f"Stored {len(enhanced_chunks)} chunks ({total_tokens} estimated tokens) "
            f"for document: {filename}"
        )

        return {
            "doc_id": doc_id,
            "filename": filename,
            "chunk_count": len(enhanced_chunks),
            "total_tokens": total_tokens,
            "year": year,
            "department": department,
            "category": category,
            "stream": stream,
            "semester": semester,
            "extracted_text": text  # Include for circular summary generation
        }

    def delete_document(self, doc_id: str):
        """Delete all chunks of a document from vector store."""
        # For Supabase, we delete by document ID which cascades to chunks
        from services.supabase_client import get_supabase_admin_client
        client = get_supabase_admin_client()
        client.table("documents").delete().eq("id", doc_id).execute()

# Singleton instance
document_processor = DocumentProcessor()
