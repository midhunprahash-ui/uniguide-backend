"""
Cross-Encoder Reranker for UniGuide RAG.

Uses Gemini to rerank search results based on query-document relevance.
More accurate than embedding similarity alone because it considers
the query and document together (cross-encoding).
"""
import logging
import re
from dataclasses import dataclass

import google.generativeai as genai

from config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class RankedResult:
    """A reranked search result with scoring details."""
    content: str
    original_score: float
    rerank_score: float
    final_score: float
    metadata: dict
    relevance_reason: str | None = None


class GeminiReranker:
    """
    Reranks search results using Gemini for relevance scoring.

    Cross-encoder reranking is more accurate than bi-encoder (embedding)
    similarity because it processes query and document together, allowing
    the model to understand their relationship directly.

    Uses Gemini Flash for fast, cost-effective reranking.
    """

    # Weighting for combining original and rerank scores
    ORIGINAL_WEIGHT = 0.3
    RERANK_WEIGHT = 0.7

    def __init__(self):
        """Initialize the Gemini reranker."""
        settings = get_settings()
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        self._initialized = True

    def _build_rerank_prompt(
        self,
        query: str,
        documents: list[dict],
        max_doc_chars: int = 600
    ) -> str:
        """
        Build the prompt for Gemini to score document relevance.

        Args:
            query: The user's question
            documents: List of documents with 'content' key
            max_doc_chars: Max characters per document (for efficiency)

        Returns:
            Formatted prompt string
        """
        docs_text = "\n\n".join([
            f"[DOC_{i}]\n{doc['content'][:max_doc_chars]}{'...' if len(doc['content']) > max_doc_chars else ''}"
            for i, doc in enumerate(documents)
        ])

        return f"""You are a relevance scoring system for a college information assistant.
Rate how well each document answers or relates to the student's question.

QUESTION: {query}

DOCUMENTS:
{docs_text}

SCORING INSTRUCTIONS:
- Score from 0.0 to 1.0 based on relevance to the question
- 1.0 = Directly answers the question with specific information
- 0.7-0.9 = Contains relevant information that helps answer the question
- 0.4-0.6 = Somewhat related but not directly answering
- 0.1-0.3 = Barely related, might have tangential information
- 0.0 = Completely irrelevant

OUTPUT FORMAT (one per line, nothing else):
DOC_0: 0.XX
DOC_1: 0.XX
...

Output ONLY the scores, no explanations."""

    def _parse_scores(self, response_text: str, expected_count: int) -> list[float]:
        """
        Parse relevance scores from Gemini response.

        Args:
            response_text: Raw response from Gemini
            expected_count: Expected number of scores

        Returns:
            List of scores (0.0-1.0), padded with 0.5 if parsing fails
        """
        scores = []

        for line in response_text.strip().split('\n'):
            # Match patterns like "DOC_0: 0.85" or "DOC_1:0.7"
            match = re.search(r'DOC_\d+\s*:\s*([\d.]+)', line, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    # Clamp to valid range
                    score = max(0.0, min(1.0, score))
                    scores.append(score)
                except ValueError:
                    scores.append(0.5)  # Default on parse error

        # Pad with default if not enough scores
        while len(scores) < expected_count:
            scores.append(0.5)

        return scores[:expected_count]

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int = 5
    ) -> list[RankedResult]:
        """
        Rerank search results using Gemini for relevance scoring.

        Args:
            query: The user's question
            results: List of search results with 'content', 'similarity', etc.
            top_k: Number of top results to return

        Returns:
            List of RankedResult sorted by final score
        """
        if not results:
            return []

        # If we have fewer results than top_k, no need for complex reranking
        if len(results) <= top_k:
            return [
                RankedResult(
                    content=r.get("content", ""),
                    original_score=r.get("similarity", r.get("combined_score", 0)),
                    rerank_score=r.get("similarity", r.get("combined_score", 0)),
                    final_score=r.get("similarity", r.get("combined_score", 0)),
                    metadata=r
                )
                for r in results
            ]

        try:
            # Build and send reranking prompt
            prompt = self._build_rerank_prompt(query, results)

            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,  # Low temperature for consistent scoring
                    max_output_tokens=200,
                )
            )

            rerank_scores = self._parse_scores(response.text, len(results))

        except Exception as e:
            logger.warning(f"Reranking failed, using original scores: {e}")
            # Fallback: use original scores
            rerank_scores = [
                r.get("similarity", r.get("combined_score", 0.5))
                for r in results
            ]

        # Combine scores and create results
        ranked_results = []
        for i, result in enumerate(results):
            original = result.get("similarity", result.get("combined_score", 0.5))
            rerank = rerank_scores[i]

            # Weighted combination
            final = (
                self.ORIGINAL_WEIGHT * original +
                self.RERANK_WEIGHT * rerank
            )

            ranked_results.append(RankedResult(
                content=result.get("content", ""),
                original_score=original,
                rerank_score=rerank,
                final_score=final,
                metadata=result
            ))

        # Sort by final score (descending)
        ranked_results.sort(key=lambda x: x.final_score, reverse=True)

        logger.debug(
            f"Reranked {len(results)} results. "
            f"Top score changed: {results[0].get('similarity', 0):.3f} -> "
            f"{ranked_results[0].final_score:.3f}"
        )

        return ranked_results[:top_k]

    def rerank_with_reasons(
        self,
        query: str,
        results: list[dict],
        top_k: int = 5
    ) -> list[RankedResult]:
        """
        Rerank with explanations for why each document is relevant.

        More expensive but useful for debugging or showing users
        why certain results were selected.
        """
        if not results:
            return []

        # Build prompt that also asks for reasons
        docs_text = "\n\n".join([
            f"[DOC_{i}]\n{doc['content'][:500]}..."
            for i, doc in enumerate(results[:top_k * 2])  # Only score top candidates
        ])

        prompt = f"""You are a relevance scoring system for a college information assistant.
Rate how well each document answers the student's question.

QUESTION: {query}

DOCUMENTS:
{docs_text}

For each document, provide:
1. A relevance score (0.0 to 1.0)
2. A brief reason (10 words max)

OUTPUT FORMAT:
DOC_0: 0.XX | reason here
DOC_1: 0.XX | reason here
...

Output ONLY in the format above."""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=500,
                )
            )

            # Parse scores and reasons
            ranked_results = []
            for i, line in enumerate(response.text.strip().split('\n')):
                if i >= len(results):
                    break

                match = re.search(r'DOC_\d+\s*:\s*([\d.]+)\s*\|\s*(.+)', line)
                if match:
                    score = max(0.0, min(1.0, float(match.group(1))))
                    reason = match.group(2).strip()[:100]
                else:
                    score = 0.5
                    reason = None

                result = results[i]
                original = result.get("similarity", result.get("combined_score", 0.5))
                final = self.ORIGINAL_WEIGHT * original + self.RERANK_WEIGHT * score

                ranked_results.append(RankedResult(
                    content=result.get("content", ""),
                    original_score=original,
                    rerank_score=score,
                    final_score=final,
                    metadata=result,
                    relevance_reason=reason
                ))

            # Sort and return
            ranked_results.sort(key=lambda x: x.final_score, reverse=True)
            return ranked_results[:top_k]

        except Exception as e:
            logger.warning(f"Reranking with reasons failed: {e}")
            # Fallback to simple reranking
            return self.rerank(query, results, top_k)


class NoOpReranker:
    """
    A no-op reranker that returns results as-is.

    Useful for A/B testing or when reranking is disabled.
    """

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int = 5
    ) -> list[RankedResult]:
        """Return results without reranking."""
        return [
            RankedResult(
                content=r.get("content", ""),
                original_score=r.get("similarity", r.get("combined_score", 0)),
                rerank_score=r.get("similarity", r.get("combined_score", 0)),
                final_score=r.get("similarity", r.get("combined_score", 0)),
                metadata=r
            )
            for r in results[:top_k]
        ]


# Singleton instance
reranker = GeminiReranker()
