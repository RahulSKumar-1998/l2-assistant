"""KB article processor for chunking knowledge base articles.

Splits KB articles by section headers, strips HTML, and creates
``TextChunk`` objects ready for embedding and vector store indexing.
Sections are further split into sentence-aware sub-chunks when
they exceed the token budget.
"""

from __future__ import annotations

import re
from typing import Optional

import structlog

from app.models.incident import (
    ChunkType,
    KBArticle,
    SourceType,
    TextChunk,
)
from app.utils.text_utils import (
    chunk_text_by_sentences,
    clean_text,
    normalize_whitespace,
    strip_html,
)

logger = structlog.get_logger(__name__)


# ── Section Header Patterns ──────────────────────────────────────────────────

# HTML header tags (h1-h6)
_HTML_HEADER_RE = re.compile(
    r"<h([1-6])[^>]*>(.*?)</h\1>",
    re.IGNORECASE | re.DOTALL,
)

# Markdown headers
_MD_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# Bold text headers (**Title**)
_BOLD_HEADER_RE = re.compile(r"^\*\*(.+?)\*\*\s*$", re.MULTILINE)


# ── Data Models ──────────────────────────────────────────────────────────────


class ArticleSection:
    """A single section extracted from a KB article.

    Attributes:
        title: Section header text (or "Overview" for untitled preamble).
        content: Section body text (HTML stripped).
        level: Header level (1-6).
    """

    __slots__ = ("title", "content", "level")

    def __init__(self, title: str, content: str, level: int = 2) -> None:
        self.title = title
        self.content = content
        self.level = level

    def __repr__(self) -> str:
        return f"ArticleSection(title={self.title!r}, len={len(self.content)})"


# ── KBArticleProcessor ──────────────────────────────────────────────────────


class KBArticleProcessor:
    """Processes KB articles into chunked ``TextChunk`` objects.

    The processing pipeline:
        1. Split article HTML into sections by header tags
        2. Strip HTML from each section
        3. Further chunk large sections by sentence boundaries
        4. Create ``TextChunk`` objects with metadata

    Example:
        >>> processor = KBArticleProcessor(chunk_size=512)
        >>> chunks = processor.process(kb_article)
        >>> len(chunks)  # e.g., 4
        >>> chunks[0].metadata["section_title"]
        'Overview'
    """

    def __init__(
        self,
        *,
        chunk_size: int = 512,
        overlap: int = 50,
        min_section_length: int = 30,
    ) -> None:
        """Initialize the KB article processor.

        Args:
            chunk_size: Maximum tokens per text chunk.
            overlap: Token overlap between consecutive chunks.
            min_section_length: Minimum character length for a section
                to be included (filters noise).
        """
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._min_section_length = min_section_length
        self._log = logger.bind(component="kb_article_processor")

    def process(self, article: KBArticle) -> list[TextChunk]:
        """Process a KB article into text chunks.

        Args:
            article: KB article from ServiceNow.

        Returns:
            List of TextChunk objects ready for embedding.
        """
        self._log.info(
            "processing_kb_article",
            number=article.number,
            title=article.short_description,
        )

        # 1. Extract sections from HTML content
        sections = self._split_into_sections(article.text)
        self._log.debug(
            "sections_extracted",
            number=article.number,
            section_count=len(sections),
        )

        # 2. Create chunks from sections
        chunks: list[TextChunk] = []
        chunk_index = 0

        for section in sections:
            section_chunks = self._chunk_section(
                section=section,
                article=article,
                start_index=chunk_index,
            )
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        self._log.info(
            "kb_article_processed",
            number=article.number,
            total_chunks=len(chunks),
            sections=len(sections),
        )
        return chunks

    def process_batch(self, articles: list[KBArticle]) -> list[TextChunk]:
        """Process multiple KB articles into text chunks.

        Args:
            articles: List of KB articles.

        Returns:
            Combined list of TextChunk objects from all articles.
        """
        all_chunks: list[TextChunk] = []
        for article in articles:
            try:
                chunks = self.process(article)
                all_chunks.extend(chunks)
            except Exception as exc:
                self._log.error(
                    "kb_article_processing_failed",
                    number=article.number,
                    error=str(exc),
                )
        self._log.info(
            "batch_processing_complete",
            articles_processed=len(articles),
            total_chunks=len(all_chunks),
        )
        return all_chunks

    def _split_into_sections(self, html_text: str) -> list[ArticleSection]:
        """Split HTML content into sections by header tags.

        Handles HTML headers (<h1>-<h6>), Markdown headers (#-######),
        and bold text headers (**Title**).

        Args:
            html_text: Raw HTML article content.

        Returns:
            List of ArticleSection objects.
        """
        if not html_text or not html_text.strip():
            return []

        sections: list[ArticleSection] = []

        # Find all HTML headers with their positions
        headers: list[tuple[int, int, str, int]] = []  # (start, end, title, level)

        for m in _HTML_HEADER_RE.finditer(html_text):
            level = int(m.group(1))
            title = strip_html(m.group(2)).strip()
            headers.append((m.start(), m.end(), title, level))

        if not headers:
            # Try Markdown headers
            stripped = strip_html(html_text)
            for m in _MD_HEADER_RE.finditer(stripped):
                level = len(m.group(1))
                title = m.group(2).strip()
                headers.append((m.start(), m.end(), title, level))

        if not headers:
            # No headers found — treat entire content as one section
            content = strip_html(html_text)
            content = normalize_whitespace(content)
            if len(content) >= self._min_section_length:
                sections.append(ArticleSection(
                    title="Content",
                    content=content,
                    level=1,
                ))
            return sections

        # Extract preamble (content before first header)
        if headers[0][0] > 0:
            preamble = html_text[:headers[0][0]]
            preamble_clean = strip_html(preamble)
            preamble_clean = normalize_whitespace(preamble_clean)
            if len(preamble_clean) >= self._min_section_length:
                sections.append(ArticleSection(
                    title="Overview",
                    content=preamble_clean,
                    level=1,
                ))

        # Extract content between headers
        for i, (start, end, title, level) in enumerate(headers):
            # Content runs from end of this header to start of next header
            if i + 1 < len(headers):
                content = html_text[end:headers[i + 1][0]]
            else:
                content = html_text[end:]

            content_clean = strip_html(content)
            content_clean = normalize_whitespace(content_clean)

            if len(content_clean) < self._min_section_length:
                continue

            # Prepend section title as context
            full_content = f"{title}: {content_clean}"

            sections.append(ArticleSection(
                title=title,
                content=full_content,
                level=level,
            ))

        return sections

    def _chunk_section(
        self,
        section: ArticleSection,
        article: KBArticle,
        start_index: int = 0,
    ) -> list[TextChunk]:
        """Chunk a single section into TextChunk objects.

        If the section fits within the chunk size budget, it becomes
        a single chunk. Otherwise, it is split by sentence boundaries.

        Args:
            section: The article section.
            article: The parent KB article (for metadata).
            start_index: Starting chunk index.

        Returns:
            List of TextChunk objects.
        """
        metadata = {
            "source_id": article.number,
            "source_type": "kb_article",
            "article_title": article.short_description,
            "section_title": section.title,
            "section_level": section.level,
            "category": article.category,
            "workflow_state": article.workflow_state,
        }

        # Clean the section content
        cleaned = clean_text(section.content)

        # Chunk by sentences
        text_chunks = chunk_text_by_sentences(
            cleaned,
            chunk_size=self._chunk_size,
            overlap=self._overlap,
        )

        if not text_chunks:
            return []

        chunks: list[TextChunk] = []
        for i, text in enumerate(text_chunks):
            chunks.append(TextChunk(
                chunk_text=text,
                chunk_type=ChunkType.KB_ARTICLE,
                source_id=article.number,
                source_type=SourceType.KB_ARTICLE,
                metadata=metadata,
                chunk_index=start_index + i,
            ))

        return chunks
