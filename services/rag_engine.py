import logging

import google.generativeai as genai

from config import get_settings
from services.hybrid_search import hybrid_search
from services.prompt_security import prompt_guard
from services.provider_error_mapper import provider_error_sse_payload

# Enhanced RAG components
from services.query_processor import QueryIntent, query_processor
from services.reranker import reranker
from services.response_cache import response_cache
from services.supabase_client import get_supabase_admin_client
from services.vector_store import vector_store

settings = get_settings()
genai.configure(api_key=settings.gemini_api_key)
logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(
        self,
        use_hybrid_search: bool = True,
        use_reranking: bool = True,
        use_cache: bool = True,
        use_prompt_security: bool = True
    ):
        """
        Initialize enhanced RAG engine with configurable components.

        Args:
            use_hybrid_search: Enable vector + keyword hybrid search
            use_reranking: Enable cross-encoder reranking
            use_cache: Enable response caching for FAQ queries
            use_prompt_security: Enable prompt injection protection
        """
        # Configuration flags
        self.use_hybrid_search = use_hybrid_search
        self.use_reranking = use_reranking
        self.use_cache = use_cache
        self.use_prompt_security = use_prompt_security

        # Keep Gemini for answer generation
        self.generation_model = genai.GenerativeModel("gemini-2.5-flash")

        # Cache for category names
        self._category_cache = {}
        # Cache for stream to year codes mapping (via departments)
        self._stream_years_cache = {}
        # Cache for department to year codes mapping
        self._department_years_cache = {}

        logger.info(
            f"RAGEngine initialized: hybrid_search={use_hybrid_search}, "
            f"reranking={use_reranking}, cache={use_cache}, "
            f"prompt_security={use_prompt_security}"
        )

    def get_category_name(self, category_slug: str) -> str:
        """Look up category name from database by slug."""
        if category_slug in self._category_cache:
            return self._category_cache[category_slug]

        try:
            client = get_supabase_admin_client()
            result = client.table("categories").select("name").eq("slug", category_slug).limit(1).execute()
            if result.data and len(result.data) > 0:
                name = result.data[0]["name"]
                self._category_cache[category_slug] = name
                return name
        except Exception as e:
            print(f"Warning: Could not look up category name: {e}")

        # Fallback to slug title-cased
        return category_slug.replace("-", " ").title()

    def get_year_codes_for_department(self, department_code: str) -> list[str]:
        """Get all year codes that belong to a specific department."""
        if department_code in self._department_years_cache:
            return self._department_years_cache[department_code]

        try:
            client = get_supabase_admin_client()
            # First get the department ID from code
            dept_result = client.table("departments").select("id").eq("code", department_code).eq("is_active", True).limit(1).execute()
            if not dept_result.data:
                return []

            department_id = dept_result.data[0]["id"]

            # Get all year codes for this department
            years_result = client.table("years").select("code").eq("department_id", department_id).eq("is_active", True).execute()
            if years_result.data:
                year_codes = [y["code"] for y in years_result.data]
                self._department_years_cache[department_code] = year_codes
                return year_codes
        except Exception as e:
            print(f"Warning: Could not look up years for department {department_code}: {e}")

        return []

    def get_year_codes_for_stream(self, stream_code: str) -> list[str]:
        """Get all year codes for all departments under a specific stream."""
        if stream_code in self._stream_years_cache:
            return self._stream_years_cache[stream_code]

        try:
            client = get_supabase_admin_client()
            # First get the stream ID from code
            stream_result = client.table("streams").select("id").eq("code", stream_code).eq("is_active", True).limit(1).execute()
            if not stream_result.data:
                return []

            stream_id = stream_result.data[0]["id"]

            # Get all departments for this stream
            depts_result = client.table("departments").select("id").eq("stream_id", stream_id).eq("is_active", True).execute()
            if not depts_result.data:
                return []

            dept_ids = [d["id"] for d in depts_result.data]

            # Get all year codes from all departments
            years_result = client.table("years").select("code").in_("department_id", dept_ids).eq("is_active", True).execute()
            if years_result.data:
                # Deduplicate year codes (same year number might exist in multiple departments)
                year_codes = list(set([y["code"] for y in years_result.data]))
                self._stream_years_cache[stream_code] = year_codes
                return year_codes
        except Exception as e:
            print(f"Warning: Could not look up years for stream {stream_code}: {e}")

        return []

    @property
    def embedding_model(self):
        """Not used for Gemini embeddings but kept for interface compatibility if needed."""
        return None

    def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding using Gemini API."""
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']

    def generate_query_embedding(self, query: str) -> list[float]:
        """Generate embedding for a query using Gemini API."""
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=query,
            task_type="retrieval_query"
        )
        return result['embedding']

    def retrieve_context(
        self,
        query: str,
        stream: str,
        year: str,
        department: str,
        category: str,
        org_id: str,
        n_results: int = 5
    ) -> tuple[list[str], list[str]]:
        """Retrieve relevant context from vector store.

        Args:
            org_id: REQUIRED - Organization ID for multi-tenant isolation.
                    This ensures documents from one org cannot leak to another.
        """
        if not org_id:
            raise ValueError("org_id is required for retrieve_context (tenant isolation)")

        query_embedding = self.generate_query_embedding(query)

        # Build metadata filter - Qdrant (via VectorStore) handles mapping
        where_filter = None
        conditions = []

        # CRITICAL: org_id filter for multi-tenant isolation (ALWAYS FIRST)
        conditions.append({"org_id": org_id})

        # Category filter (STRICTLY REQUIRED)
        if category:
            conditions.append({"category": category})
        else:
             # Fallback to rules if somehow None, to prevent leaking other categories
             print("WARNING: No category provided, defaulting to 'rules'")
             conditions.append({"category": "rules"})

        # CRITICAL: Stream filter - prevents data leakage between streams (UG/PG)
        # This filter is passed directly to match_documents SQL function which handles the logic:
        # - If stream="all", only show documents with stream="all"
        # - If stream="UG", show documents with stream="UG" or stream="all"
        if stream:
            conditions.append({"stream": stream})

        # Handle hierarchy filtering: Stream -> Department -> Year
        # Department filter - ALWAYS add the filter (including 'all' for strict filtering)
        if department:
            if department.lower() != "all":
                # Specific department: show docs for that dept OR docs with dept="all"
                conditions.append({"department": {"$in": [department, "all"]}})
            else:
                # department="all": only show docs with dept="all" (strict filtering)
                conditions.append({"department": "all"})

        # Year filter - ALWAYS add the filter (including 'all' for strict filtering)
        if year:
            if year.lower() != "all":
                # Specific year selected - show docs for that year OR docs with year="all"
                conditions.append({"year": {"$in": [year, "all"]}})
            else:
                # year="all": only show docs with year="all" (strict filtering)
                conditions.append({"year": "all"})

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
        conversation_history: list[dict[str, str]] = None,
        safe_context: str = None,
        safety_preamble: str = ""
    ) -> str:
        """
        Generate answer using Gemini with retrieved context.

        Args:
            query: The user's question
            context_chunks: Retrieved document chunks
            sources: Source document names
            year: Student's year
            department: Student's department
            category: Document category
            conversation_history: Previous messages
            safe_context: Pre-sanitized context from prompt_guard (optional)
            safety_preamble: Safety instructions to prepend (optional)

        Returns:
            Generated answer string
        """
        from collections import defaultdict
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")

        # Look up the human-readable category name
        category_name = self.get_category_name(category)

        # Use pre-sanitized context if provided, otherwise build normally
        if safe_context:
            context_text = safe_context
            unique_sources = list(set(sources))
        else:
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

        # Build prompt with optional safety preamble for prompt injection protection
        prompt = f"""{safety_preamble}You are a friendly and intelligent assistant for college students at St. Joseph's Group of Institutions. Your role is to help students with questions about the institution's rules, regulations, schedules, and academic matters.

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

**CURRENT CATEGORY: {category_name}**

**GREETINGS (hi, hello, hey, good morning, etc.):**
Respond warmly and offer to help based on the current category. Examples:
- If in Rules & Regulations: "Hello! 👋 How can I help you today? Feel free to ask me about rules & regulations, disciplinary policies, or academic guidelines!"
- If in Schedules / Leaves: "Hello! 👋 I can help you with schedule information! Ask me about exam dates, class timings, or academic calendar."
- If in Admissions: "Hello! 👋 I'm here to help with admissions! Ask me about admission procedures, eligibility, fees, or deadlines."
- If in Timetables: "Hello! 👋 I can help you with timetable information! Ask me about class schedules, lab timings, or weekly routines."
- If in Circulars: "Hello! 👋 I can help you with circulars! Ask me about recent announcements, notices, or updates."
- For any other category: "Hello! 👋 I'm here to help with {category_name}! What would you like to know?"

**THANK YOU / GRATITUDE:**
Respond warmly. Example:
"You're welcome! 😊 Feel free to ask if you have any more questions about {category_name}. I'm here to help!"

**GOODBYE / BYE:**
Respond warmly. Example:
"Goodbye! Have a great day! 👋 Come back anytime you need help with {category_name} information."

**ACKNOWLEDGMENTS (okay, ok, alright, got it, I see, etc.):**
Respond naturally and encourage further questions. Vary your response - examples:
- "Great! Is there anything else you'd like to know about {category_name}?"
- "Alright! Feel free to ask if you have more questions."
- "Got it! Let me know if you need any other information about {category_name}."

**GIBBERISH / UNCLEAR INPUT (random letters, typos, or unclear messages):**
Respond helpfully without being robotic. Vary your response - examples:
- "I didn't quite catch that. Could you rephrase your question? I'm here to help with {category_name}! 😊"
- "Hmm, I'm not sure what you meant. Try asking me something like 'What are the exam dates?' or 'What are the attendance rules?'"
- "I couldn't understand that. Would you like to ask something about {category_name}? I'm happy to help!"

**IRRELEVANT QUESTIONS (not related to college/academics or current category):**
Politely redirect with varied responses - examples:
- "I don't have information about that. I'm here to help with **{category_name}** - feel free to ask me anything about it!"
- "That's outside my knowledge area. I specialize in {category_name}. What would you like to know about that?"
- "I can't help with that topic, but I'd love to answer questions about {category_name}! What's on your mind?"

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

    def generate_note_title(self, document_text: str, filename: str, subject_name: str = None, unit_number: int = None) -> str:
        """
        Generate an AI-powered title for an uploaded note based on its content.
        
        Args:
            document_text: The extracted text from the document
            filename: Original filename for fallback
            subject_name: Optional subject name for context
            unit_number: Optional unit number for context
            
        Returns:
            Generated title string (max 100 chars)
        """
        context = ""
        if subject_name:
            context += f"Subject: {subject_name}\n"
        if unit_number:
            context += f"Unit: {unit_number}\n"
            
        prompt = f"""You are an assistant for organizing academic notes. A new note has been uploaded.

{context}
Document filename: {filename}
Document content (first 3000 chars):
{document_text[:3000]}

Generate a clear, concise title for this note that describes its main topic.
- The title should be descriptive and academic
- Maximum 80 characters
- Do not include "Unit X" or subject code - those are shown separately
- Focus on the actual content/topic
- Examples: "Introduction to Data Structures", "Linked Lists and Operations", "SQL Queries and Joins"

Respond with ONLY the title, nothing else."""

        try:
            response = self.generation_model.generate_content(prompt)
            title = response.text.strip()
            # Clean up any quotes or extra formatting
            title = title.strip('\"\'')
            # Limit length
            if len(title) > 100:
                title = title[:97] + "..."
            return title
        except Exception as e:
            logger.warning(f"Failed to generate note title: {e}")
            # Fallback to filename without extension
            return filename.rsplit('.', 1)[0][:80]

    def generate_text(self, prompt: str, max_tokens: int = 100) -> str:
        """
        Generate text using Gemini for simple prompts.
        
        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum tokens in response (approximate)
            
        Returns:
            Generated text string
        """
        try:
            response = self.generation_model.generate_content(prompt)
            text = response.text.strip()
            return text
        except Exception as e:
            logger.warning(f"Failed to generate text: {e}")
            return ""

    def query(
        self,
        question: str,
        stream: str,
        year: str,
        department: str,
        category: str,
        org_id: str,
        conversation_history: list[dict[str, str]] = None
    ) -> dict[str, any]:
        """
        Enhanced RAG pipeline with query processing, hybrid search,
        reranking, prompt security, and response caching.

        Args:
            question: The user's question
            stream: Stream filter (UG/PG/all)
            year: Year filter
            department: Department filter
            category: Document category
            org_id: REQUIRED - Organization ID for multi-tenant isolation
            conversation_history: Previous messages in the conversation

        Returns:
            Dict with 'answer', 'sources', and 'cached' flag
        """
        # Step 1: Process query for intent detection and preprocessing
        processed = query_processor.process(question)
        category_name = self.get_category_name(category)

        # Step 2: Handle conversational queries without retrieval
        if processed.is_conversational:
            return self._handle_conversational(processed, category_name)

        # Step 3: Get the search query (with abbreviation expansion)
        search_query = query_processor.get_search_query(processed)

        # Step 4: Generate query embedding
        query_embedding = self.generate_query_embedding(search_query)

        # Step 5: Check cache for similar queries
        if self.use_cache:
            cached = response_cache.get(
                query=question,
                query_embedding=query_embedding,
                org_id=org_id,
                category=category,
                year=year,
                department=department
            )
            if cached:
                logger.debug(f"Cache hit for query: {question[:50]}...")
                return {
                    "answer": cached.answer,
                    "sources": cached.sources,
                    "cached": True
                }

        # Step 6: Retrieve context using hybrid or vector-only search
        if self.use_hybrid_search:
            context_chunks, sources, scores = self._retrieve_hybrid(
                query=search_query,
                query_embedding=query_embedding,
                stream=stream,
                year=year,
                department=department,
                category=category,
                org_id=org_id,
                n_results=10 if self.use_reranking else 5
            )
        else:
            context_chunks, sources = self.retrieve_context(
                question, stream, year, department, category, org_id
            )
            scores = []

        if not context_chunks:
            return {
                "answer": "I don't have any information about this topic for your year and department yet. Please contact the administration or check if documents have been uploaded.",
                "sources": [],
                "cached": False
            }

        # Step 7: Rerank results if enabled
        if self.use_reranking and len(context_chunks) > 5:
            reranked = reranker.rerank(
                query=question,
                results=[
                    {"content": chunk, "similarity": scores[i] if i < len(scores) else 0.5, "filename": sources[i]}
                    for i, chunk in enumerate(context_chunks)
                ],
                top_k=5
            )
            context_chunks = [r.content for r in reranked]
            sources = [r.metadata.get("filename", "Unknown") for r in reranked]
            logger.debug(f"Reranked {len(reranked)} results")

        # Step 8: Apply prompt security if enabled
        if self.use_prompt_security:
            safe_context = prompt_guard.build_safe_context(context_chunks, sources)
            safety_preamble = prompt_guard.get_safety_preamble()
        else:
            safe_context = None
            safety_preamble = ""

        # Step 9: Generate answer
        answer = self.generate_answer(
            question,
            context_chunks,
            sources,
            year,
            department,
            category,
            conversation_history or [],
            safe_context=safe_context,
            safety_preamble=safety_preamble
        )

        # Step 10: Cache the response
        unique_sources = list(set(sources))
        if self.use_cache:
            response_cache.set(
                query=question,
                query_embedding=query_embedding,
                org_id=org_id,
                category=category,
                year=year,
                department=department,
                answer=answer,
                sources=unique_sources
            )

        return {
            "answer": answer,
            "sources": unique_sources,
            "cached": False
        }

    def _handle_conversational(self, processed, category_name: str) -> dict:
        """Handle greetings, farewells, and other conversational queries."""
        responses = {
            QueryIntent.GREETING: f"Hello! How can I help you today? Feel free to ask me about {category_name}!",
            QueryIntent.FAREWELL: f"Goodbye! Have a great day! Come back anytime you need help with {category_name}.",
            QueryIntent.GRATITUDE: f"You're welcome! Feel free to ask if you have more questions about {category_name}.",
            QueryIntent.ACKNOWLEDGMENT: f"Great! Is there anything else you'd like to know about {category_name}?",
            QueryIntent.UNCLEAR: f"I didn't quite catch that. Could you rephrase? I'm here to help with {category_name}!",
        }

        answer = responses.get(processed.intent, f"How can I help you with {category_name}?")

        return {
            "answer": answer,
            "sources": [],
            "cached": False
        }

    def _retrieve_hybrid(
        self,
        query: str,
        query_embedding: list[float],
        stream: str,
        year: str,
        department: str,
        category: str,
        org_id: str,
        n_results: int = 5
    ) -> tuple[list[str], list[str], list[float]]:
        """
        Retrieve context using hybrid vector + keyword search.

        Returns:
            Tuple of (chunks, sources, scores)
        """
        results = hybrid_search.search(
            query=query,
            query_embedding=query_embedding,
            org_id=org_id,
            category=category,
            stream=stream,
            year=year if year and year.lower() != "all" else None,
            department=department if department and department.lower() != "all" else None,
            n_results=n_results,
            match_threshold=0.25
        )

        if not results:
            return [], [], []

        chunks = [r.content for r in results]
        sources = [r.metadata.get("filename", "Unknown") for r in results]
        scores = [r.combined_score for r in results]

        return chunks, sources, scores


    def generate_answer_stream(
        self,
        query: str,
        context_chunks: list[str],
        sources: list[str],
        year: str,
        department: str,
        category: str = "rules",
        conversation_history: list[dict[str, str]] = None
    ):
        """Generate answer stream using Gemini with retrieved context."""
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

**ACKNOWLEDGMENTS (okay, ok, alright, got it, I see, etc.):**
Respond naturally and encourage further questions. Vary your response - examples:
- "Great! Is there anything else you'd like to know about {category}?"
- "Alright! Feel free to ask if you have more questions."
- "Got it! Let me know if you need any other information about {category}."

**GIBBERISH / UNCLEAR INPUT (random letters, typos, or unclear messages):**
Respond helpfully without being robotic. Vary your response - examples:
- "I didn't quite catch that. Could you rephrase your question? I'm here to help with {category}! 😊"
- "Hmm, I'm not sure what you meant. Try asking me something like 'What are the exam dates?' or 'What are the attendance rules?'"
- "I couldn't understand that. Would you like to ask something about {category}? I'm happy to help!"

**IRRELEVANT QUESTIONS (not related to college/academics or current category):**
Politely redirect with varied responses - examples:
- "I don't have information about that. I'm here to help with **{category}** - feel free to ask me anything about it!"
- "That's outside my knowledge area. I specialize in {category}. What would you like to know about that?"
- "I can't help with that topic, but I'd love to answer questions about {category}! What's on your mind?"

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

        # Generate streaming response
        response = self.generation_model.generate_content(prompt, stream=True)
        for chunk in response:
            try:
                if chunk.text:
                    yield chunk.text
            except ValueError:
                # Skip chunks that don't have text (e.g., finish_reason without content)
                continue

    def query_stream(
        self,
        question: str,
        stream: str,
        year: str,
        department: str,
        category: str,
        org_id: str,
        conversation_history: list[dict[str, str]] = None
    ):
        """Complete RAG pipeline with streaming: retrieve context and stream answer.

        Args:
            org_id: REQUIRED - Organization ID for multi-tenant isolation.
        """
        import json

        # Step 1: Process query for intent detection (matches non-streaming query())
        processed = query_processor.process(question)
        category_name = self.get_category_name(category)

        # Step 2: Handle conversational queries without retrieval
        if processed.is_conversational:
            response = self._handle_conversational(processed, category_name)
            yield json.dumps({"type": "sources", "data": []})
            yield json.dumps({"type": "token", "data": response["answer"]})
            yield json.dumps({"type": "done", "data": ""})
            return

        # Step 3: Retrieve relevant context for actual questions
        context_chunks, sources = self.retrieve_context(question, stream, year, department, category, org_id)

        # Deduplicate sources
        unique_sources = list(set(sources))
        
        # Yield empty list of sources first (to update UI)
        yield json.dumps({"type": "sources", "data": unique_sources})

        if not context_chunks:
            error_msg = "I don't have any information about this topic for your year and department yet. Please contact the administration or check if documents have been uploaded."
            yield json.dumps({"type": "token", "data": error_msg})
            # Also yield error done
            yield json.dumps({"type": "done", "data": ""})
            return

        # Generate answer stream
        history = conversation_history or []
        
        try:
            full_answer = ""
            for token in self.generate_answer_stream(question, context_chunks, sources, year, department, category, history):
                full_answer += token
                # Yield token
                yield json.dumps({"type": "token", "data": token})
            
            # Finally yield done with full answer (optional, or just to signal end)
            yield json.dumps({"type": "done", "data": ""})
            
            return full_answer # Return for caller if needed (e.g. to save to history)
            
        except Exception as e:
            logger.exception("Streaming error in RAG query")
            yield json.dumps(
                provider_error_sse_payload(
                    e,
                    fallback_message=(
                        "We could not generate a response right now. Please try again shortly."
                    ),
                )
            )

# Singleton instance
rag_engine = RAGEngine()
