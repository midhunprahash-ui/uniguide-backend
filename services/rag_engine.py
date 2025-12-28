
import google.generativeai as genai

from config import get_settings
from services.vector_store import vector_store

settings = get_settings()
genai.configure(api_key=settings.gemini_api_key)

class RAGEngine:
    def __init__(self):
        """Initialize RAG engine with local embeddings and Gemini generation."""
        # Use local embedding model (no API calls!)
        self._embedding_model = None
        # Keep Gemini only for answer generation
        self.generation_model = genai.GenerativeModel("gemini-2.0-flash-exp")

    @property
    def embedding_model(self):
        """Not used for Gemini embeddings but kept for interface compatibility if needed."""
        return None

    def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding using Gemini API."""
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']

    def generate_query_embedding(self, query: str) -> list[float]:
        """Generate embedding for a query using Gemini API."""
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="retrieval_query"
        )
        return result['embedding']

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
        category: str = "rules",
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

        prompt = f"""You are a friendly and intelligent assistant for college students at St. Joseph's Group of Institutions. Your role is to help students with questions about the institution's rules, regulations, schedules, and academic matters.

Current Date: {current_date}

Student Context:
- Year: {year}
- Department: {department}
{history_text}

DOCUMENTS PROVIDED ({len(unique_sources)}):
{doc_list}

{context_text}

Student Question: {query}

=== CONVERSATIONAL HANDLING ===

**CURRENT CATEGORY: {category}**

**GREETINGS (hi, hello, hey, good morning, etc.):**
Respond warmly and offer to help based on the current category. Examples:
- If category is 'rules': "Hello! 👋 How can I help you today? Feel free to ask me about rules & regulations, disciplinary policies, or academic guidelines!"
- If category is 'schedules': "Hello! 👋 I can help you with schedule information! Ask me about exam dates, class timings, or academic calendar."
- If category is 'admissions': "Hello! 👋 I'm here to help with admissions! Ask me about admission procedures, eligibility, fees, or deadlines."
- If category is 'timetables': "Hello! 👋 I can help you with timetable information! Ask me about class schedules, lab timings, or weekly routines."
- If category is 'circulars': "Hello! 👋 I can help you with circulars! Ask me about recent announcements, notices, or updates."

**THANK YOU / GRATITUDE:**
Respond warmly. Example:
"You're welcome! 😊 Feel free to ask if you have any more questions about {category}. I'm here to help!"

**GOODBYE / BYE:**
Respond warmly. Example:
"Goodbye! Have a great day! 👋 Come back anytime you need help with {category} information."

**IRRELEVANT QUESTIONS (not related to college/academics or current category):**
Politely redirect. Example:
"I don't have information about that topic. I'm currently helping you with **{category}**. Feel free to ask me about this, or switch categories for other topics like rules & regulations, exam schedules, fee details, attendance, or admission procedures!"

=== RESPONSE STRATEGY ===

**STEP 1 - CORE ANSWER:**
Start by answering the student's **core question** directly in 1-2 sentences. Do not start with greetings or preamble. Get straight to the point.

**STEP 2 - DETAILED EXPLANATION:**
Provide the complete answer with all relevant details, using proper formatting.

**STEP 3 - ANTICIPATE FOLLOW-UP DOUBTS:**
Think about what related questions the student might have based on their query:
- If asking about fees → mention payment deadlines, late fees, payment modes
- If asking about exams → mention hall tickets, exam schedules, passing criteria
- If asking about attendance → mention minimum requirements, consequences, leave procedures
- If asking about events → mention registration, deadlines, eligibility
Proactively address 2-3 likely follow-up concerns.

=== FORMATTING RULES ===

1. **Bold** all important terms, dates, deadlines, percentages, and key information
2. Use **numbered lists (1, 2, 3)** for sequential steps or procedures
3. Use **bullet points (-)** for non-sequential items or options
4. Use `### Headers` to organize sections when answer is long
5. Use `> Blockquotes` for important warnings or notes
6. Keep paragraphs short (2-3 sentences max)

=== CONTENT RULES ===

1. Answer based ONLY on the provided documents
2. If information is not found, clearly state: "I don't have specific information about this in the current documents."
3. Do NOT include citation numbers like [1], [2]
4. Do NOT mention or reference source document names - just present the information naturally
5. Do NOT mix information from different documents - keep them separate

=== TEMPORAL AWARENESS (IMPORTANT) ===

**Current Date: {current_date}**

When encountering dates in the documents:
- **PAST EVENTS:** Any date BEFORE {current_date} is in the past. Use past tense: "This event **was on** [date]" or "This **has already passed**"
- **FUTURE EVENTS:** Any date AFTER {current_date} is in the future. Use future tense: "This event **is scheduled for** [date]" or "**Upcoming:** [event]"
- **TODAY:** If the date matches {current_date}, say "This is happening **today**!"

When student asks about "upcoming" or "next" events:
- ONLY show events with dates AFTER {current_date}
- Ignore past events unless explicitly asked

When student asks about "past" or "previous" events:
- ONLY show events with dates BEFORE {current_date}

Always indicate whether information is current, past, or upcoming to avoid confusion.

=== SEGREGATION RULE ===

If information varies by year or department, organize like this:
- **For 2nd Year students:**
  - Point 1
  - Point 2
- **For 3rd Year students:**
  - Point 1
  - Point 2

=== EXAMPLE RESPONSE STRUCTURE ===

[1-2 sentence core answer to the question]

### Details
[Detailed explanation with bullets/numbers]

### You Might Also Want to Know
- **Related Point 1:** Brief explanation
- **Related Point 2:** Brief explanation

> **Important:** [Any critical warning or deadline]

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
        answer = self.generate_answer(question, context_chunks, sources, year, department, category, conversation_history or [])

        # Deduplicate sources
        unique_sources = list(set(sources))

        return {
            "answer": answer,
            "sources": unique_sources
        }

# Singleton instance
rag_engine = RAGEngine()
