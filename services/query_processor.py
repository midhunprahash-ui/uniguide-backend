"""
Query Processor for UniGuide RAG.

Preprocesses user queries to improve retrieval quality through:
- Intent detection (greeting, question, etc.)
- Abbreviation expansion
- Keyword extraction
- Query normalization
"""
import re
from dataclasses import dataclass
from enum import Enum


class QueryIntent(Enum):
    """Types of user query intents."""
    GREETING = "greeting"
    FAREWELL = "farewell"
    GRATITUDE = "gratitude"
    ACKNOWLEDGMENT = "acknowledgment"
    QUESTION = "question"
    UNCLEAR = "unclear"


@dataclass
class ProcessedQuery:
    """Result of query processing with extracted metadata."""
    original: str
    normalized: str
    intent: QueryIntent
    keywords: list[str]
    is_conversational: bool
    expanded_query: str | None = None


class QueryProcessor:
    """
    Preprocess and enhance user queries for better retrieval.

    Handles:
    - Intent detection for conversational queries
    - Educational abbreviation expansion
    - Keyword extraction for hybrid search
    - Query normalization
    """

    # Common abbreviations in educational context (Indian colleges)
    ABBREVIATIONS = {
        # Academic terms
        "dept": "department",
        "yr": "year",
        "sem": "semester",
        "exam": "examination",
        "exams": "examinations",
        "prof": "professor",
        "hod": "head of department",
        "cgpa": "cumulative grade point average",
        "gpa": "grade point average",
        "sgpa": "semester grade point average",

        # Attendance
        "att": "attendance",
        "od": "on duty",
        "cl": "casual leave",
        "ml": "medical leave",
        "pl": "previledged leave",

        # Facilities
        "lib": "library",
        "lab": "laboratory",
        "labs": "laboratories",
        "hostel": "hostel accommodation",
        "mess": "mess dining",
        "canteen": "canteen cafeteria",

        # Documents
        "reg": "registration",
        "cert": "certificate",
        "tc": "transfer certificate",
        "bona": "bonafide",
        "bonafide": "bonafide certificate",

        # Exams
        "internals": "internal examinations",
        "externals": "external examinations",
        "supple": "supplementary examination",
        "arrear": "arrear examination",
        "reeval": "revaluation",
        "reval": "revaluation",

        # Programs
        "ug": "undergraduate",
        "pg": "postgraduate",
        "btech": "bachelor of technology",
        "mtech": "master of technology",
        "mba": "master of business administration",
        "mca": "master of computer applications",

        # Departments (common)
        "cse": "computer science engineering",
        "ece": "electronics communication engineering",
        "eee": "electrical electronics engineering",
        "mech": "mechanical engineering",
        "civil": "civil engineering",
        "it": "information technology",
        "aids": "artificial intelligence data science",
        "aiml": "artificial intelligence machine learning",
        "ads": "artificial intelligence data science",

        # Misc
        "timetable": "time table schedule",
        "tt": "time table",
        "syllabus": "syllabus curriculum",
        "fest": "festival event",
        "placement": "campus placement recruitment",
        "intern": "internship",
    }

    # Intent detection patterns
    GREETING_PATTERNS = [
        r'^(hi|hello|hey|hii+|helo+)\b',
        r'^good\s*(morning|afternoon|evening|day)\b',
        r'^(greetings|namaste|vanakkam)\b',
    ]

    FAREWELL_PATTERNS = [
        r'^(bye|goodbye|bye-bye|tata)\b',
        r'^(see\s*you|take\s*care)\b',
        r'^(good\s*night|gn)\b',
    ]

    GRATITUDE_PATTERNS = [
        r'^(thank|thanks|thankyou|thank\s*you)\b',
        r'^(thx|ty|tysm)\b',
        r'^(appreciated?|grateful)\b',
    ]

    ACKNOWLEDGMENT_PATTERNS = [
        r'^(ok|okay|okie|k)\b',
        r'^(alright|all\s*right)\b',
        r'^(got\s*it|understood|i\s*see)\b',
        r'^(cool|great|nice|fine)\b',
        r'^(hmm+|ohh*|ahh*)\b',
    ]

    # Stop words for keyword extraction
    STOP_WORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
        'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'under', 'again', 'further', 'then', 'once',
        'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but',
        'if', 'or', 'because', 'until', 'while', 'what', 'which', 'who',
        'whom', 'this', 'that', 'these', 'those', 'am', 'i', 'me', 'my',
        'myself', 'we', 'our', 'ours', 'you', 'your', 'yours', 'he', 'him',
        'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their',
        'theirs', 'please', 'tell', 'about', 'know', 'want', 'like', 'give',
        'show', 'get', 'find', 'any', 'also', 'really', 'thing', 'things',
    }

    def __init__(self):
        """Initialize compiled regex patterns."""
        self.greeting_re = [re.compile(p, re.IGNORECASE) for p in self.GREETING_PATTERNS]
        self.farewell_re = [re.compile(p, re.IGNORECASE) for p in self.FAREWELL_PATTERNS]
        self.gratitude_re = [re.compile(p, re.IGNORECASE) for p in self.GRATITUDE_PATTERNS]
        self.ack_re = [re.compile(p, re.IGNORECASE) for p in self.ACKNOWLEDGMENT_PATTERNS]

    def _detect_intent(self, text: str) -> QueryIntent:
        """Detect the intent of the query."""
        text_clean = text.lower().strip()

        # Check patterns in order of specificity
        if any(p.search(text_clean) for p in self.greeting_re):
            return QueryIntent.GREETING

        if any(p.search(text_clean) for p in self.farewell_re):
            return QueryIntent.FAREWELL

        if any(p.search(text_clean) for p in self.gratitude_re):
            return QueryIntent.GRATITUDE

        if any(p.search(text_clean) for p in self.ack_re):
            return QueryIntent.ACKNOWLEDGMENT

        # Check if it's unclear (gibberish, too short, no letters)
        if len(text_clean) < 3:
            return QueryIntent.UNCLEAR

        if not re.search(r'[a-z]', text_clean):
            return QueryIntent.UNCLEAR

        # Check for random key mashing
        if re.match(r'^[^aeiou]{5,}$', text_clean.replace(' ', '')):
            return QueryIntent.UNCLEAR

        return QueryIntent.QUESTION

    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations to full forms."""
        words = text.split()
        expanded = []

        for word in words:
            # Preserve original case for non-abbreviations
            lower = word.lower()

            # Check if word is an abbreviation
            if lower in self.ABBREVIATIONS:
                expanded.append(self.ABBREVIATIONS[lower])
            else:
                expanded.append(word)

        return ' '.join(expanded)

    def _extract_keywords(self, text: str) -> list[str]:
        """
        Extract important keywords for hybrid search.

        Removes stop words and short words, preserving order.
        """
        # Extract words (letters only)
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

        # Filter out stop words and short words
        keywords = [
            w for w in words
            if w not in self.STOP_WORDS and len(w) > 2
        ]

        # Preserve order, remove duplicates
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)

        return unique_keywords

    def _normalize(self, text: str) -> str:
        """Normalize query text for consistent processing."""
        # Convert to lowercase
        normalized = text.lower()

        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized)

        # Remove punctuation except question mark
        normalized = re.sub(r'[^\w\s?]', '', normalized)

        return normalized.strip()

    def process(self, query: str) -> ProcessedQuery:
        """
        Process a user query for better retrieval.

        Args:
            query: Raw user query string

        Returns:
            ProcessedQuery with extracted metadata
        """
        original = query.strip()

        if not original:
            return ProcessedQuery(
                original=original,
                normalized="",
                intent=QueryIntent.UNCLEAR,
                keywords=[],
                is_conversational=True,
                expanded_query=None
            )

        # Detect intent
        intent = self._detect_intent(original)

        # Determine if conversational (no retrieval needed)
        is_conversational = intent in [
            QueryIntent.GREETING,
            QueryIntent.FAREWELL,
            QueryIntent.GRATITUDE,
            QueryIntent.ACKNOWLEDGMENT,
            QueryIntent.UNCLEAR
        ]

        # Normalize
        normalized = self._normalize(original)

        # Expand abbreviations
        expanded = self._expand_abbreviations(normalized)

        # Extract keywords
        keywords = self._extract_keywords(expanded)

        return ProcessedQuery(
            original=original,
            normalized=normalized,
            intent=intent,
            keywords=keywords,
            is_conversational=is_conversational,
            expanded_query=expanded if expanded != normalized else None
        )

    def get_search_query(self, processed: ProcessedQuery) -> str:
        """
        Get the best query string for search.

        Uses expanded query if abbreviations were found,
        otherwise uses normalized query.
        """
        return processed.expanded_query or processed.normalized


# Singleton instance
query_processor = QueryProcessor()
