"""
Semantic Chunker for UniGuide RAG.

Replaces word-based chunking with intelligent sentence-aware chunking
that preserves context and respects document structure.
"""
import re
from dataclasses import dataclass


@dataclass
class EnhancedChunk:
    """A chunk with rich metadata for better retrieval."""
    text: str
    chunk_number: int
    char_start: int
    char_end: int
    page_numbers: list[int] | None = None
    section_header: str | None = None
    token_count: int | None = None


class SemanticChunker:
    """
    Intelligent chunker that:
    - Splits on sentence boundaries (not mid-word)
    - Maintains configurable overlap for context
    - Extracts section headers for metadata
    - Tracks character positions for source highlighting
    """

    def __init__(
        self,
        target_chunk_size: int = 512,   # Target tokens per chunk
        max_chunk_size: int = 1024,     # Maximum tokens before forced split
        min_chunk_size: int = 100,      # Minimum tokens (skip tiny chunks)
        overlap_sentences: int = 2      # Sentences to overlap between chunks
    ):
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_sentences = overlap_sentences

    def _split_into_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences using regex.
        Handles abbreviations, decimals, and common edge cases.
        """
        # Normalize whitespace first
        text = re.sub(r'\s+', ' ', text.strip())

        # Split on sentence boundaries
        # Match period/exclamation/question followed by space and capital letter
        # But not after common abbreviations
        abbreviations = r'(?<!\bDr)(?<!\bMr)(?<!\bMs)(?<!\bMrs)(?<!\bProf)(?<!\bSt)(?<!\bNo)(?<!\bVs)'
        sentence_endings = abbreviations + r'(?<=[.!?])\s+(?=[A-Z])'

        sentences = re.split(sentence_endings, text)

        # Also split on newlines followed by content (likely new paragraphs)
        result = []
        for sentence in sentences:
            # Split on double newlines (paragraph breaks)
            paragraphs = re.split(r'\n\s*\n', sentence)
            for para in paragraphs:
                para = para.strip()
                if para:
                    result.append(para)

        return result

    def _estimate_tokens(self, text: str) -> int:
        """
        Rough token estimate.
        English averages ~4 characters per token for most models.
        """
        return max(1, len(text) // 4)

    def _extract_section_header(self, text: str) -> str | None:
        """
        Extract section header from chunk text.
        Looks for markdown headers or ALL CAPS lines.
        """
        lines = text.split('\n')
        for line in lines[:3]:  # Check first 3 lines
            line = line.strip()
            if not line:
                continue

            # Markdown headers
            if line.startswith('#'):
                return line.lstrip('#').strip()[:100]

            # ALL CAPS headers (common in PDFs)
            if line.isupper() and 5 < len(line) < 100:
                return line

            # Title case with colon (e.g., "Section 1: Introduction")
            if re.match(r'^[A-Z][^.!?]*:\s*$', line) and len(line) < 100:
                return line.rstrip(':').strip()

        return None

    def chunk(self, text: str, page_info: dict[int, int] | None = None) -> list[EnhancedChunk]:
        """
        Create semantic chunks with metadata.

        Args:
            text: The full document text
            page_info: Optional dict mapping character positions to page numbers

        Returns:
            List of EnhancedChunk objects
        """
        if not text or not text.strip():
            return []

        sentences = self._split_into_sentences(text)

        if not sentences:
            return []

        chunks = []
        current_chunk: list[str] = []
        current_tokens = 0
        chunk_start_pos = 0
        current_pos = 0
        chunk_number = 0

        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)

            # If adding this sentence would exceed max size, save current chunk
            if current_tokens + sentence_tokens > self.max_chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunk_end_pos = current_pos

                # Determine page numbers if page_info provided
                page_nums = None
                if page_info:
                    page_nums = self._get_pages_for_range(
                        page_info, chunk_start_pos, chunk_end_pos
                    )

                chunks.append(EnhancedChunk(
                    text=chunk_text,
                    chunk_number=chunk_number,
                    char_start=chunk_start_pos,
                    char_end=chunk_end_pos,
                    page_numbers=page_nums,
                    section_header=self._extract_section_header(chunk_text),
                    token_count=current_tokens
                ))
                chunk_number += 1

                # Keep overlap sentences for context continuity
                overlap_start = max(0, len(current_chunk) - self.overlap_sentences)
                overlap_sentences = current_chunk[overlap_start:]

                # Calculate new start position (approximate)
                overlap_text = ' '.join(overlap_sentences)
                chunk_start_pos = chunk_end_pos - len(overlap_text)

                current_chunk = overlap_sentences.copy()
                current_tokens = sum(self._estimate_tokens(s) for s in current_chunk)

            current_chunk.append(sentence)
            current_tokens += sentence_tokens
            current_pos += len(sentence) + 1  # +1 for space

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)

            # Only add if meets minimum size (avoid tiny trailing chunks)
            if self._estimate_tokens(chunk_text) >= self.min_chunk_size:
                page_nums = None
                if page_info:
                    page_nums = self._get_pages_for_range(
                        page_info, chunk_start_pos, current_pos
                    )

                chunks.append(EnhancedChunk(
                    text=chunk_text,
                    chunk_number=chunk_number,
                    char_start=chunk_start_pos,
                    char_end=current_pos,
                    page_numbers=page_nums,
                    section_header=self._extract_section_header(chunk_text),
                    token_count=current_tokens
                ))
            elif chunks:
                # Append to previous chunk if too small
                prev_chunk = chunks[-1]
                combined_text = prev_chunk.text + ' ' + chunk_text
                chunks[-1] = EnhancedChunk(
                    text=combined_text,
                    chunk_number=prev_chunk.chunk_number,
                    char_start=prev_chunk.char_start,
                    char_end=current_pos,
                    page_numbers=prev_chunk.page_numbers,
                    section_header=prev_chunk.section_header,
                    token_count=self._estimate_tokens(combined_text)
                )

        return chunks

    def _get_pages_for_range(
        self,
        page_info: dict[int, int],
        start: int,
        end: int
    ) -> list[int]:
        """Get all page numbers that overlap with a character range."""
        pages = set()
        for char_pos, page_num in sorted(page_info.items()):
            if char_pos >= end:
                break
            if char_pos >= start:
                pages.add(page_num)
        return sorted(pages) if pages else None

    def chunk_with_fallback(self, text: str) -> list[EnhancedChunk]:
        """
        Chunk text with fallback for edge cases.
        If semantic chunking produces no results, falls back to simple splitting.
        """
        chunks = self.chunk(text)

        if chunks:
            return chunks

        # Fallback: simple character-based splitting
        if not text.strip():
            return []

        # Split into roughly equal parts
        char_chunk_size = self.target_chunk_size * 4  # ~4 chars per token
        fallback_chunks = []

        for i in range(0, len(text), char_chunk_size):
            chunk_text = text[i:i + char_chunk_size].strip()
            if chunk_text:
                fallback_chunks.append(EnhancedChunk(
                    text=chunk_text,
                    chunk_number=len(fallback_chunks),
                    char_start=i,
                    char_end=min(i + char_chunk_size, len(text)),
                    token_count=self._estimate_tokens(chunk_text)
                ))

        return fallback_chunks


# Singleton instance with default settings
semantic_chunker = SemanticChunker()
