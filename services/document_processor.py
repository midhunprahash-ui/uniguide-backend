import os
import uuid

import google.generativeai as genai
import pdfplumber
from pdf2image import convert_from_path
from PIL import Image

from config import get_settings
from services.rag_engine import rag_engine
from services.vector_store import vector_store

settings = get_settings()

class DocumentProcessor:
    def __init__(self):
        """Initialize document processor."""
        os.makedirs(settings.upload_directory, exist_ok=True)
        # Configure Gemini
        genai.configure(api_key=settings.gemini_api_key)
        self.vision_model = genai.GenerativeModel("gemini-2.0-flash-exp")
        print("DocumentProcessor initialized with Gemini Vision.")

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

            # If text is sparse or empty, it might be a scanned PDF or complex table
            # We can use a heuristic: if text length is low per page, try Vision
            if len(text) < 100:
                print("Low text content in PDF, attempting Vision extraction...")
                text = self.process_pdf_with_vision(file_path)

            return text
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return self.process_pdf_with_vision(file_path)

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
        try:
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

            response = self.vision_model.generate_content([prompt, image])
            return response.text
        except Exception as e:
            print(f"Error in Gemini Vision inference: {e}")
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
        """Split text into overlapping chunks."""
        if not text.strip():
            return []

        chunks = []
        words = text.split()

        if len(words) <= chunk_size:
            return [text]

        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        return chunks

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
        """Process document and store in vector database."""
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

        # Chunk the text
        chunks = self.chunk_text(text)

        if not chunks:
            raise ValueError("No valid chunks created from the document")

        # Generate embeddings for each chunk
        embeddings = []
        metadatas = []
        ids = []
        doc_id = str(uuid.uuid4())

        for i, chunk in enumerate(chunks):
            embedding = rag_engine.generate_embedding(chunk)
            embeddings.append(embedding)

            metadata = {
                "year": year,
                "department": department,
                "category": category,
                "stream": stream,
                "semester": semester,
                "source_file": filename,
                "chunk_id": i,
                "doc_id": doc_id,
                "org_id": org_id
            }
            metadatas.append(metadata)
            ids.append(f"{doc_id}_{i}")

        # Store in vector database
        vector_store.add_documents(
            texts=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        return {
            "doc_id": doc_id,
            "filename": filename,
            "chunk_count": len(chunks),
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

