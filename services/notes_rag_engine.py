"""
Notes RAG Engine for academic notes.
Completely isolated from the institutional RAG (RAGEngine).
Uses match_notes function to query note_chunks table.
"""
import logging
import json

import google.generativeai as genai

from config import get_settings
from services.notes_vector_store import notes_vector_store
from services.supabase_client import get_supabase_admin_client

settings = get_settings()
genai.configure(api_key=settings.gemini_api_key)
logger = logging.getLogger(__name__)


class NotesRAGEngine:
    """
    RAG engine for academic notes.
    
    Key differences from RAGEngine:
    - Uses note_chunks table (via match_notes)
    - Different system prompt (tutoring/teaching focus)
    - Subject-based filtering instead of category-based
    """
    
    def __init__(self):
        self.embedding_model_name = "models/text-embedding-004"
        self.generation_model_name = "gemini-2.0-flash"
        self._generation_model = None
    
    @property
    def generation_model(self):
        """Lazy load generation model."""
        if self._generation_model is None:
            self._generation_model = genai.GenerativeModel(
                self.generation_model_name,
                generation_config=genai.GenerationConfig(
                    temperature=0.7,
                    top_p=0.9,
                    max_output_tokens=2048,
                )
            )
        return self._generation_model
    
    def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding using Gemini API."""
        result = genai.embed_content(
            model=self.embedding_model_name,
            content=text,
            task_type="retrieval_document"
        )
        return result["embedding"]
    
    def generate_query_embedding(self, query: str) -> list[float]:
        """Generate embedding for a query."""
        result = genai.embed_content(
            model=self.embedding_model_name,
            content=query,
            task_type="retrieval_query"
        )
        return result["embedding"]
    
    def get_subject_info(self, subject_id: str) -> dict:
        """Get subject name and code."""
        if not subject_id:
            return {"name": "All Subjects", "code": "ALL"}
        
        try:
            client = get_supabase_admin_client()
            result = client.table("subjects").select("name, code").eq("id", subject_id).single().execute()
            if result.data:
                return result.data
        except Exception as e:
            logger.warning(f"Failed to get subject info: {e}")
        
        return {"name": "Unknown", "code": "UNK"}
    
    def retrieve_context(
        self,
        query: str,
        org_id: str,
        year_id: str = None,
        subject_id: str = None,
        unit_id: str = None,
        n_results: int = 5
    ) -> tuple[list[str], list[str], list[float]]:
        """
        Retrieve relevant context from notes vector store.
        
        Returns:
            Tuple of (chunks, sources, scores)
        """
        query_embedding = self.generate_query_embedding(query)
        
        result = notes_vector_store.query(
            query_embedding=query_embedding,
            org_id=org_id,
            year_id=year_id,
            subject_id=subject_id,
            unit_id=unit_id,
            n_results=n_results
        )
        
        chunks = result["documents"]
        sources = []
        for meta in result["metadatas"]:
            source = f"{meta.get('subject_name', 'Unknown')} - Unit {meta.get('unit_number', '?')}"
            if source not in sources:
                sources.append(source)
        scores = result["similarities"]
        
        return chunks, sources, scores
    
    def _build_system_prompt(self, subject_name: str = None) -> str:
        """Build system prompt for notes-based tutoring."""
        subject_context = f"for the subject '{subject_name}'" if subject_name else "across all subjects"
        
        return f"""You are an intelligent academic tutor helping students understand their class notes {subject_context}.

Your role is to:
1. Answer questions based ONLY on the provided notes content
2. Explain concepts clearly and pedagogically
3. Provide examples when helpful
4. If the notes don't contain enough information, say so honestly
5. Encourage deeper understanding, not just memorization

Guidelines:
- Be encouraging and supportive
- Break down complex topics step by step
- Use analogies when helpful
- If asked about something not in the notes, say "This topic isn't covered in the provided notes. You may want to check your textbook or ask your professor."
- Format your responses with clear structure using headers and bullet points when appropriate

Remember: You are tutoring based on class notes, not general knowledge. Stay grounded in the provided content."""
    
    def generate_answer(
        self,
        query: str,
        context_chunks: list[str],
        sources: list[str],
        subject_name: str = None,
        conversation_history: list[dict] = None
    ) -> str:
        """Generate answer using Gemini."""
        
        # Build context
        if context_chunks:
            context = "\n\n---\n\n".join(context_chunks)
            context_section = f"""
## Relevant Notes Content:

{context}

## Sources:
{', '.join(sources) if sources else 'No specific sources'}
"""
        else:
            context_section = """
## Note:
No relevant content was found in the notes for this question.
"""
        
        # Build conversation context
        history_context = ""
        if conversation_history:
            history_context = "\n## Previous Conversation:\n"
            for msg in conversation_history[-6:]:  # Last 3 exchanges
                role = "Student" if msg["role"] == "user" else "Tutor"
                history_context += f"{role}: {msg['content'][:500]}\n"
        
        # Build prompt
        prompt = f"""
{context_section}
{history_context}

## Student's Question:
{query}

## Your Response:
Provide a helpful, educational response based on the notes content above.
"""
        
        system_prompt = self._build_system_prompt(subject_name)
        
        try:
            chat = self.generation_model.start_chat(history=[])
            response = chat.send_message(
                f"{system_prompt}\n\n{prompt}",
                safety_settings={
                    "harm_category_harassment": "block_none",
                    "harm_category_hate_speech": "block_none",
                    "harm_category_sexually_explicit": "block_none",
                    "harm_category_dangerous_content": "block_none",
                }
            )
            return response.text
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I'm sorry, I encountered an error while generating a response. Please try again."
    
    def query(
        self,
        question: str,
        org_id: str,
        year_id: str = None,
        subject_id: str = None,
        unit_id: str = None,
        conversation_history: list[dict] = None
    ) -> dict:
        """
        Complete Notes RAG pipeline.
        
        Args:
            question: Student's question
            org_id: REQUIRED - Organization ID for tenant isolation
            year_id: Optional - Filter by year
            subject_id: Optional - Filter by subject
            unit_id: Optional - Filter by unit
            conversation_history: Previous messages for context
        
        Returns:
            Dict with 'answer', 'sources', 'chunks_used'
        """
        if not org_id:
            raise ValueError("org_id is required for tenant isolation")
        
        # Get subject info for context
        subject_info = self.get_subject_info(subject_id)
        
        # Retrieve relevant chunks
        chunks, sources, scores = self.retrieve_context(
            query=question,
            org_id=org_id,
            year_id=year_id,
            subject_id=subject_id,
            unit_id=unit_id,
            n_results=5
        )
        
        # Generate answer
        answer = self.generate_answer(
            query=question,
            context_chunks=chunks,
            sources=sources,
            subject_name=subject_info.get("name"),
            conversation_history=conversation_history
        )
        
        return {
            "answer": answer,
            "sources": sources,
            "chunks_used": len(chunks),
            "subject": subject_info
        }
    
    def query_stream(
        self,
        question: str,
        org_id: str,
        year_id: str = None,
        subject_id: str = None,
        unit_id: str = None,
        conversation_history: list[dict] = None
    ):
        """
        Streaming version of query for SSE.
        
        Yields JSON strings for each chunk.
        """
        if not org_id:
            raise ValueError("org_id is required for tenant isolation")
        
        # Get subject info
        subject_info = self.get_subject_info(subject_id)
        
        # Retrieve context
        chunks, sources, scores = self.retrieve_context(
            query=question,
            org_id=org_id,
            year_id=year_id,
            subject_id=subject_id,
            unit_id=unit_id,
            n_results=5
        )
        
        # Yield sources first
        yield json.dumps({
            "type": "sources",
            "data": sources
        })
        
        # Build prompt
        if chunks:
            context = "\n\n---\n\n".join(chunks)
            context_section = f"## Relevant Notes Content:\n\n{context}"
        else:
            context_section = "## Note:\nNo relevant content was found in the notes."
        
        history_context = ""
        if conversation_history:
            history_context = "\n## Previous Conversation:\n"
            for msg in conversation_history[-6:]:
                role = "Student" if msg["role"] == "user" else "Tutor"
                history_context += f"{role}: {msg['content'][:500]}\n"
        
        prompt = f"""
{context_section}
{history_context}

## Student's Question:
{question}

## Your Response:
"""
        
        system_prompt = self._build_system_prompt(subject_info.get("name"))
        
        try:
            chat = self.generation_model.start_chat(history=[])
            response = chat.send_message(
                f"{system_prompt}\n\n{prompt}",
                stream=True,
                safety_settings={
                    "harm_category_harassment": "block_none",
                    "harm_category_hate_speech": "block_none",
                    "harm_category_sexually_explicit": "block_none",
                    "harm_category_dangerous_content": "block_none",
                }
            )
            
            for chunk in response:
                if chunk.text:
                    yield json.dumps({
                        "type": "token",
                        "data": chunk.text
                    })
            
            # Signal completion
            yield json.dumps({
                "type": "done",
                "data": {
                    "chunks_used": len(chunks),
                    "subject": subject_info
                }
            })
            
        except Exception as e:
            import traceback
            logger.error(f"Error in streaming response: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            yield json.dumps({
                "type": "error",
                "data": f"Error generating response: {str(e)}"
            })


# Singleton instance
notes_rag_engine = NotesRAGEngine()
