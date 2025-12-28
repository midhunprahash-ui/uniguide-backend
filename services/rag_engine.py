
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

from config import get_settings
from services.vector_store import vector_store

class RAGEngine:
    def __init__(self):
        """Initialize RAG engine with local embeddings and Gemini generation."""
        # Defer configuration to first usage
        self._embedding_model = None
        self._generation_model = None
        
    @property
    def generation_model(self):
        """Lazy load Gemini model."""
        if self._generation_model is None:
            settings = get_settings()
            genai.configure(api_key=settings.gemini_api_key)
            self._generation_model = genai.GenerativeModel("gemini-2.0-flash-exp")
        return self._generation_model

    @property
    def embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None:
            print("Loading embedding model...")
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedding_model

    def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding using local sentence-transformers model."""
        # Use property access to trigger load
        embedding = self.embedding_model.encode(text, convert_to_tensor=False)
        return embedding.tolist()

    def generate_query_embedding(self, query: str) -> list[float]:
        """Generate embedding for a query using local model."""
        embedding = self.embedding_model.encode(query, convert_to_tensor=False)
        return embedding.tolist()

    def retrieve_context(
        self,
        query: str,
        year: str,
        department: str,
        category: str,
        n_results: int = 5
    ) -> tuple[list[str], list[str]]:
        """Retrieve relevant context from vector store."""
        query_embedding = self.generate_query_embedding(query)

        # Build metadata filter - Qdrant (via VectorStore) handles mapping
        where_filter = None
        conditions = []

        # Category filter (STRICTLY REQUIRED)
        if category:
            conditions.append({"category": category})
        else:
             # Fallback to rules if somehow None, to prevent leaking other categories
             print("WARNING: No category provided, defaulting to 'rules'")
             conditions.append({"category": "rules"})

        if year and year.lower() != "all":
            conditions.append({"year": {"$in": [year, "all"]}})
        if department and department.lower() != "all":
            conditions.append({"department": {"$in": [department, "all"]}})

        # Combine conditions with $and if we have multiple
        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}

        # Query vector store
        results = vector_store.query(
            query_embedding=query_embedding,
            n_results=n_results,
            where=where_filter
        )

        # Extract documents and sources
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        sources = [meta.get('source_file', 'Unknown') for meta in metadatas]

        return documents, sources

    def generate_answer(
        self,
        query: str,
        context_chunks: list[str],
        sources: list[str],
        year: str,
        department: str,
        conversation_history: list[dict[str, str]] = None
    ) -> str:
        """Generate answer using Gemini with retrieved context."""
        from collections import defaultdict
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")

        # Group chunks by source document to prevent confusion
        source_grouped = defaultdict(list)
        for chunk, source in zip(context_chunks, sources, strict=False):
            source_grouped[source].append(chunk)

        # Build context with clear document separation
        context_sections = []
        unique_sources = list(source_grouped.keys())

        for i, source in enumerate(unique_sources, 1):
            chunks = source_grouped[source]
            section_content = "\n\n".join(chunks)
            context_sections.append(
                f"### DOCUMENT {i}: {source}\n{section_content}\n### END OF DOCUMENT {i}"
            )

        context_text = "\n\n" + "=" * 50 + "\n\n".join(context_sections)

        # Format conversation history if present
        history_text = ""
        if conversation_history and len(conversation_history) > 1:  # More than just the current question
            history_items = []
            for msg in conversation_history[:-1]:  # Exclude the current question
                role = "Student" if msg["role"] == "user" else "Assistant"
                history_items.append(f"{role}: {msg['content']}")
            if history_items:
                history_text = "\n\nPrevious Conversation:\n" + "\n".join(history_items) + "\n"

        # Create document list for prompt
        doc_list = "\n".join([f"  - Document {i}: {src}" for i, src in enumerate(unique_sources, 1)])

        prompt = f"""You are a helpful assistant for college students. Based on the following rules and regulations from the institution, answer the student's question accurately and concisely.

Current Date: {current_date}

Student Context:
- Year: {year}
- Department: {department}
{history_text}

IMPORTANT: You have been provided with {len(unique_sources)} document(s):
{doc_list}

{context_text}

Student Question: {query}

Instructions:
1. Answer based ONLY on the information provided in the documents above.
2. If the answer is not found in the provided context, say "I don't have information about this in the current rules and regulations".
3. Be specific and cite relevant rules when applicable.
4. Keep the answer clear and student-friendly.
5. **Do NOT include citation numbers like [1], [2] in your response.**
6. **Temporal Reasoning:**
   - Use the 'Current Date' ({current_date}) as a pivot to identify future events.
   - If the student asks "what's happening next" or about upcoming events, look for dates in the context that are strictly AFTER the Current Date.
   - Ignore past events unless explicitly asked for.
7. **Segregate the response based on the category, Example:**
   - **2nd Year:**
     - Rule 1
     - Exam 1
   - **3rd Year:**
     - Rule 2
     - Exam 2
8. **Format your answer using Markdown:**
   - Use **Bold** for key terms.
   - Use `### Headers` for sections.
   - Use `- Bullet points` for lists.
   - Use `> Blockquotes` for important notes.
   - Use `| Tables |` if data is structured.

**CRITICAL - MULTI-DOCUMENT HANDLING:**
9. **NEVER mix or combine dates/information from different documents.** Each document is a distinct schedule or rule set.
10. When information comes from multiple documents, **clearly indicate which document** the information is from.
11. If documents contain **conflicting information** (e.g., different dates for the same event), present both and mention that there may be multiple versions.
12. For schedule queries, **prioritize the most specific document** that matches the student's context (year/department).
13. When citing specific dates, events, or rules, mention the source document name so the student knows where the information comes from.

Answer:"""

        # Generate response
        response = self.generation_model.generate_content(prompt)
        return response.text

    def generate_circular_summary(self, document_text: str, filename: str) -> dict[str, str]:
        """
        Generate a one-line summary and a brief summary for a circular document.

        Returns:
            Dict with 'one_line' (for sidebar) and 'brief' (for chat header) summaries
        """
        prompt = f"""You are an assistant for a college. A new circular has been uploaded.

Document filename: {filename}
Document content:
{document_text[:4000]}  # Limit to prevent token overflow

Generate two summaries:
1. **One-line summary** (max 60 characters): A very brief headline for the sidebar
2. **Brief summary** (2-3 sentences): A short description for display when students click on it

Format your response EXACTLY like this:
ONE_LINE: [your one-line summary here]
BRIEF: [your brief summary here]

Be specific about dates, events, or actions mentioned in the circular."""

        try:
            response = self.generation_model.generate_content(prompt)
            text = response.text

            # Parse the response
            one_line = ""
            brief = ""

            for line in text.split('\n'):
                if line.startswith('ONE_LINE:'):
                    one_line = line.replace('ONE_LINE:', '').strip()
                elif line.startswith('BRIEF:'):
                    brief = line.replace('BRIEF:', '').strip()

            # Fallback if parsing fails
            if not one_line:
                one_line = f"New Circular: {filename[:40]}..."
            if not brief:
                brief = f"A new circular titled '{filename}' has been uploaded. Click to learn more."

            return {
                "one_line": one_line[:80],  # Limit length
                "brief": brief[:300]  # Limit length
            }
        except Exception as e:
            print(f"Error generating circular summary: {e}")
            return {
                "one_line": f"New Circular: {filename[:40]}",
                "brief": f"A circular titled '{filename}' has been uploaded. Ask questions to learn more."
            }

    def query(
        self,
        question: str,
        year: str,
        department: str,
        category: str,
        conversation_history: list[dict[str, str]] = None
    ) -> dict[str, any]:
        """Complete RAG pipeline: retrieve context and generate answer."""
        # Retrieve relevant context
        context_chunks, sources = self.retrieve_context(question, year, department, category)

        if not context_chunks:
            return {
                "answer": "I don't have any information about this topic for your year and department yet. Please contact the administration or check if documents have been uploaded.",
                "sources": []
            }

        # Generate answer
        # Pass the full list of sources (corresponding to chunks) and conversation history to generate_answer
        answer = self.generate_answer(question, context_chunks, sources, year, department, conversation_history or [])

        # Deduplicate sources
        unique_sources = list(set(sources))

        return {
            "answer": answer,
            "sources": unique_sources
        }

# Singleton instance
rag_engine = RAGEngine()
