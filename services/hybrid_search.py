"""
Hybrid Search Engine for UniGuide RAG.

Combines vector similarity search with BM25 keyword matching for
better retrieval accuracy. Vector search captures semantic meaning
while BM25 catches exact keyword matches that embeddings might miss.
"""
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass

from services.supabase_client import get_supabase_admin_client

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchResult:
    """A search result with combined scoring from vector and keyword search."""
    chunk_id: str
    content: str
    vector_score: float
    keyword_score: float
    combined_score: float
    metadata: dict


class HybridSearchEngine:
    """
    Combines vector similarity search with BM25 keyword matching.

    The hybrid approach improves retrieval by:
    - Vector search: Captures semantic similarity (understands meaning)
    - BM25 search: Captures exact keyword matches (handles specific terms)

    Default weighting is 70% vector, 30% keyword, but can be adjusted.
    """

    # BM25 parameters (standard values)
    BM25_K1 = 1.5  # Term frequency saturation
    BM25_B = 0.75  # Length normalization

    # Common English stop words for keyword extraction
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
    }

    def __init__(
        self,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3
    ):
        """
        Initialize the hybrid search engine.

        Args:
            vector_weight: Weight for vector similarity score (0-1)
            keyword_weight: Weight for BM25 keyword score (0-1)
        """
        if not (0 <= vector_weight <= 1 and 0 <= keyword_weight <= 1):
            raise ValueError("Weights must be between 0 and 1")

        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self._client = None

    @property
    def client(self):
        """Lazy load Supabase client."""
        if self._client is None:
            self._client = get_supabase_admin_client()
        return self._client

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text for BM25 scoring.

        Converts to lowercase, removes punctuation,
        filters stop words and very short tokens.
        """
        text = text.lower()
        # Keep only alphanumeric and spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        # Split and filter
        tokens = [
            w for w in text.split()
            if w not in self.STOP_WORDS and len(w) > 2
        ]
        return tokens

    def _compute_bm25_score(
        self,
        query_tokens: list[str],
        doc_tokens: list[str],
        avg_doc_len: float,
        doc_freqs: dict[str, int] | None = None,
        total_docs: int = 1
    ) -> float:
        """
        Compute BM25 score for a document.

        Args:
            query_tokens: Tokenized query
            doc_tokens: Tokenized document
            avg_doc_len: Average document length in corpus
            doc_freqs: Document frequencies for IDF (optional)
            total_docs: Total number of documents for IDF

        Returns:
            BM25 score (higher is more relevant)
        """
        if not query_tokens or not doc_tokens:
            return 0.0

        doc_len = len(doc_tokens)
        if doc_len == 0:
            return 0.0

        doc_term_freq = Counter(doc_tokens)
        score = 0.0

        for term in query_tokens:
            if term not in doc_term_freq:
                continue

            tf = doc_term_freq[term]

            # IDF calculation (simplified if no corpus stats)
            if doc_freqs and term in doc_freqs:
                df = doc_freqs[term]
                idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1)
            else:
                # Default IDF when no corpus stats (assume term is somewhat rare)
                idf = 1.0

            # BM25 term score
            numerator = tf * (self.BM25_K1 + 1)
            denominator = tf + self.BM25_K1 * (
                1 - self.BM25_B + self.BM25_B * doc_len / avg_doc_len
            )
            score += idf * (numerator / denominator)

        return score

    def search(
        self,
        query: str,
        query_embedding: list[float],
        org_id: str,
        category: str | None = None,
        stream: str | None = None,
        year: str | None = None,
        department: str | None = None,
        semester: str | None = None,
        n_results: int = 5,
        match_threshold: float = 0.25
    ) -> list[HybridSearchResult]:
        """
        Perform hybrid search combining vector and keyword matching.

        Args:
            query: Original query text (for keyword matching)
            query_embedding: Pre-computed query embedding vector
            org_id: Organization ID (required for tenant isolation)
            category: Optional category filter
            stream: Optional stream filter (UG/PG)
            year: Optional year filter
            department: Optional department filter
            semester: Optional semester filter
            n_results: Number of results to return
            match_threshold: Minimum vector similarity threshold

        Returns:
            List of HybridSearchResult sorted by combined score
        """
        if not org_id:
            raise ValueError("org_id is required for tenant isolation")

        # Get more results from vector search for reranking
        fetch_count = n_results * 3  # Fetch more for hybrid ranking

        # Call the match_documents RPC
        result = self.client.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "filter_org_id": org_id,
                "match_count": fetch_count,
                "match_threshold": match_threshold,
                "filter_category": category,
                "filter_stream": stream,
                "filter_year": year if year and year.lower() != "all" else None,
                "filter_department": department if department and department.lower() != "all" else None,
                "filter_semester": semester if semester and semester.lower() != "all" else None,
            }
        ).execute()

        if not result.data:
            logger.debug(f"No vector search results for query in org {org_id}")
            return []

        # Tokenize query once
        query_tokens = self._tokenize(query)

        if not query_tokens:
            # If query has no meaningful tokens, return vector results only
            return [
                HybridSearchResult(
                    chunk_id=str(row["id"]),
                    content=row["content"],
                    vector_score=row["similarity"],
                    keyword_score=0.0,
                    combined_score=row["similarity"],
                    metadata={
                        "filename": row["filename"],
                        "year": row["year"],
                        "department": row["department"],
                        "category": row["category"],
                        "chunk_number": row["chunk_number"],
                        "stream": row.get("stream"),
                    }
                )
                for row in result.data[:n_results]
            ]

        # Tokenize all documents and compute average length
        doc_tokens_list = [self._tokenize(row["content"]) for row in result.data]
        avg_doc_len = sum(len(t) for t in doc_tokens_list) / len(doc_tokens_list) if doc_tokens_list else 1

        # Compute BM25 scores
        bm25_scores = [
            self._compute_bm25_score(query_tokens, doc_tokens, avg_doc_len)
            for doc_tokens in doc_tokens_list
        ]

        # Normalize BM25 scores to 0-1 range
        max_bm25 = max(bm25_scores) if bm25_scores else 1
        if max_bm25 > 0:
            normalized_bm25 = [s / max_bm25 for s in bm25_scores]
        else:
            normalized_bm25 = [0.0] * len(bm25_scores)

        # Combine scores and create results
        results = []
        for i, row in enumerate(result.data):
            vector_score = row["similarity"]
            keyword_score = normalized_bm25[i]

            # Weighted combination
            combined_score = (
                self.vector_weight * vector_score +
                self.keyword_weight * keyword_score
            )

            results.append(HybridSearchResult(
                chunk_id=str(row["id"]),
                content=row["content"],
                vector_score=vector_score,
                keyword_score=keyword_score,
                combined_score=combined_score,
                metadata={
                    "filename": row["filename"],
                    "year": row["year"],
                    "department": row["department"],
                    "category": row["category"],
                    "chunk_number": row["chunk_number"],
                    "stream": row.get("stream"),
                }
            ))

        # Sort by combined score (descending) and return top n
        results.sort(key=lambda x: x.combined_score, reverse=True)

        logger.debug(
            f"Hybrid search returned {len(results[:n_results])} results "
            f"(top vector: {results[0].vector_score:.3f}, "
            f"top keyword: {results[0].keyword_score:.3f}, "
            f"top combined: {results[0].combined_score:.3f})"
        )

        return results[:n_results]

    def search_with_fallback(
        self,
        query: str,
        query_embedding: list[float],
        org_id: str,
        **kwargs
    ) -> list[HybridSearchResult]:
        """
        Search with automatic fallback to vector-only if hybrid fails.

        Useful for edge cases where BM25 computation might fail.
        """
        try:
            return self.search(query, query_embedding, org_id, **kwargs)
        except Exception as e:
            logger.warning(f"Hybrid search failed, falling back to vector-only: {e}")

            # Fallback to vector-only search
            n_results = kwargs.get("n_results", 5)
            result = self.client.rpc(
                "match_documents",
                {
                    "query_embedding": query_embedding,
                    "filter_org_id": org_id,
                    "match_count": n_results,
                    "match_threshold": kwargs.get("match_threshold", 0.25),
                    "filter_category": kwargs.get("category"),
                    "filter_stream": kwargs.get("stream"),
                    "filter_year": kwargs.get("year"),
                    "filter_department": kwargs.get("department"),
                    "filter_semester": kwargs.get("semester"),
                }
            ).execute()

            if not result.data:
                return []

            return [
                HybridSearchResult(
                    chunk_id=str(row["id"]),
                    content=row["content"],
                    vector_score=row["similarity"],
                    keyword_score=0.0,
                    combined_score=row["similarity"],
                    metadata={
                        "filename": row["filename"],
                        "year": row["year"],
                        "department": row["department"],
                        "category": row["category"],
                        "chunk_number": row["chunk_number"],
                    }
                )
                for row in result.data
            ]

    def get_keyword_boost_terms(self, query: str) -> list[str]:
        """
        Extract important terms from query that should boost keyword matching.

        Useful for understanding why certain documents ranked higher.
        """
        return self._tokenize(query)


# Singleton instance
hybrid_search = HybridSearchEngine()
