"""Text processing utilities for chunking, cleaning, and token counting.

Provides sentence-aware text splitting, HTML stripping, whitespace
normalization, and token counting for context budget management.
"""

import re
from typing import Optional

import tiktoken


# Pre-compiled regex patterns
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_NEWLINE_MULTI_RE = re.compile(r"\n{3,}")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def strip_html(text: str) -> str:
    """Remove HTML tags from text.

    Args:
        text: Input text potentially containing HTML.

    Returns:
        Text with all HTML tags removed.
    """
    return _HTML_TAG_RE.sub(" ", text)


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace: collapse multiple spaces, trim lines.

    Args:
        text: Input text with irregular whitespace.

    Returns:
        Text with normalized whitespace.
    """
    text = _WHITESPACE_RE.sub(" ", text)
    text = _NEWLINE_MULTI_RE.sub("\n\n", text)
    return text.strip()


def clean_text(text: str) -> str:
    """Full text cleaning pipeline: strip HTML, normalize whitespace, lowercase.

    Args:
        text: Raw input text.

    Returns:
        Cleaned and normalized text.
    """
    text = strip_html(text)
    text = normalize_whitespace(text)
    text = text.lower()
    return text


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex-based heuristics.

    Uses punctuation followed by whitespace and uppercase letter as
    sentence boundaries. Falls back to simple period splitting.

    Args:
        text: Input text to split.

    Returns:
        List of sentence strings.
    """
    if not text.strip():
        return []

    sentences = _SENTENCE_SPLIT_RE.split(text)
    # Filter empty strings and strip
    return [s.strip() for s in sentences if s.strip()]


def chunk_text_by_sentences(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    encoding_name: str = "cl100k_base",
) -> list[str]:
    """Split text into chunks respecting sentence boundaries.

    Chunks are built by adding sentences until the token budget is
    reached. Overlap is achieved by including trailing sentences from
    the previous chunk at the start of the next one.

    Args:
        text: Input text to chunk.
        chunk_size: Maximum tokens per chunk.
        overlap: Overlap tokens between consecutive chunks.
        encoding_name: Tiktoken encoding name for token counting.

    Returns:
        List of text chunks.
    """
    if not text.strip():
        return []

    enc = tiktoken.get_encoding(encoding_name)
    sentences = split_sentences(text)

    if not sentences:
        return [text] if text.strip() else []

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(enc.encode(sentence))

        # If a single sentence exceeds chunk_size, split it by tokens
        if sentence_tokens > chunk_size:
            # Flush current chunk
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_tokens = 0

            # Split large sentence into token-based chunks
            tokens = enc.encode(sentence)
            for i in range(0, len(tokens), chunk_size - overlap):
                chunk_tokens = tokens[i : i + chunk_size]
                chunks.append(enc.decode(chunk_tokens))
            continue

        # Check if adding this sentence exceeds the budget
        if current_tokens + sentence_tokens > chunk_size and current_sentences:
            chunks.append(" ".join(current_sentences))

            # Calculate overlap: keep last sentences within overlap budget
            overlap_sentences: list[str] = []
            overlap_tokens = 0
            for s in reversed(current_sentences):
                s_tokens = len(enc.encode(s))
                if overlap_tokens + s_tokens > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_tokens += s_tokens

            current_sentences = overlap_sentences
            current_tokens = overlap_tokens

        current_sentences.append(sentence)
        current_tokens += sentence_tokens

    # Flush remaining
    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count the number of tokens in text using tiktoken.

    Args:
        text: Input text.
        encoding_name: Tiktoken encoding name (default: cl100k_base for GPT-4).

    Returns:
        Token count.
    """
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))


def truncate_to_tokens(
    text: str,
    max_tokens: int,
    encoding_name: str = "cl100k_base",
) -> str:
    """Truncate text to a maximum number of tokens.

    Args:
        text: Input text.
        max_tokens: Maximum token count.
        encoding_name: Tiktoken encoding name.

    Returns:
        Truncated text.
    """
    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


def extract_section_title(text: str) -> Optional[str]:
    """Extract a section title from markdown-style headers.

    Looks for lines starting with # or ## or bold text (**title**).

    Args:
        text: Text to search for a section title.

    Returns:
        The section title if found, None otherwise.
    """
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
        if line.startswith("**") and line.endswith("**"):
            return line.strip("*").strip()
    return None


def combine_incident_text(
    short_description: str,
    description: str,
    work_notes: str = "",
    resolution_notes: str = "",
) -> str:
    """Combine incident text fields into a single document.

    Args:
        short_description: Incident short description.
        description: Full incident description.
        work_notes: Work notes history.
        resolution_notes: Resolution notes.

    Returns:
        Combined text with section markers.
    """
    parts = [f"TITLE: {short_description}"]

    if description:
        parts.append(f"DESCRIPTION: {description}")
    if work_notes:
        parts.append(f"WORK NOTES: {work_notes}")
    if resolution_notes:
        parts.append(f"RESOLUTION: {resolution_notes}")

    return "\n\n".join(parts)
