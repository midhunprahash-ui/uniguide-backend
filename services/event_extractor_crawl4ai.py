"""
Crawl4AI-backed extraction for event discovery.

The service is intentionally optional: if Crawl4AI is unavailable or fails,
callers can fall back to Tavily/Gemini extraction.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

try:
    from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
except Exception as e:  # pragma: no cover - validated at runtime
    AsyncWebCrawler = None
    CrawlerRunConfig = None
    CacheMode = None
    logger.warning("Crawl4AI import failed: %s", e)


HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
SCRIPT_STYLE_PATTERN = re.compile(
    r"<(script|style)\b[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass
class CrawlExtractedContent:
    url: str
    text: str
    links: list[str]
    status_code: int | None = None


def _normalize_whitespace(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text or "").strip()


def _html_to_text(cleaned_html: str) -> str:
    without_scripts = SCRIPT_STYLE_PATTERN.sub(" ", cleaned_html or "")
    without_tags = HTML_TAG_PATTERN.sub(" ", without_scripts)
    return _normalize_whitespace(without_tags)


def _extract_markdown(markdown_obj: Any) -> str:
    if not markdown_obj:
        return ""
    if isinstance(markdown_obj, str):
        return markdown_obj

    for field in ("fit_markdown", "markdown_with_citations", "raw_markdown"):
        value = getattr(markdown_obj, field, None)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _extract_links(links_payload: Any) -> list[str]:
    if not isinstance(links_payload, dict):
        return []

    seen: set[str] = set()
    ordered: list[str] = []

    for bucket in ("internal", "external"):
        entries = links_payload.get(bucket) or []
        if not isinstance(entries, list):
            continue

        for entry in entries:
            if isinstance(entry, str):
                candidate = entry.strip()
            elif isinstance(entry, dict):
                candidate = str(entry.get("href") or entry.get("url") or "").strip()
            else:
                candidate = ""

            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            ordered.append(candidate)

    return ordered


class Crawl4AIExtractor:
    def __init__(
        self,
        *,
        timeout_ms: int = 12000,
        max_chars: int = 18000,
        check_robots_txt: bool = True,
        max_workers: int = 2,
    ) -> None:
        self.timeout_ms = max(4000, int(timeout_ms))
        self.max_chars = max(3000, int(max_chars))
        self.check_robots_txt = check_robots_txt
        self._available = bool(AsyncWebCrawler and CrawlerRunConfig)
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)),
            thread_name_prefix="event-crawl4ai",
        )
        self._consecutive_failures = 0
        self._disabled_until_ts = 0.0

    @property
    def available(self) -> bool:
        if not self._available:
            return False
        return time.time() >= self._disabled_until_ts

    def _record_failure(self, error: Exception | str) -> None:
        self._consecutive_failures += 1
        logger.warning("Crawl4AI extraction failed: %s", error)

        # Backoff window to avoid repeated expensive failures on hosts
        # that don't have browser dependencies available.
        if self._consecutive_failures >= 3:
            self._disabled_until_ts = time.time() + 300
            logger.warning("Temporarily disabling Crawl4AI extraction for 300s")

    def _record_success(self) -> None:
        self._consecutive_failures = 0
        self._disabled_until_ts = 0.0

    def extract_page(self, url: str) -> CrawlExtractedContent | None:
        if not self.available:
            return None

        timeout_seconds = max(6, int(self.timeout_ms / 1000) + 3)
        future = self._executor.submit(self._extract_sync, url)

        try:
            result = future.result(timeout=timeout_seconds)
        except Exception as e:
            future.cancel()
            self._record_failure(e)
            return None

        if not result or not result.text:
            self._record_failure("empty extraction payload")
            return None

        self._record_success()
        return result

    def _build_run_config(self) -> Any | None:
        if CrawlerRunConfig is None:
            return None

        kwargs: dict[str, Any] = {}
        param_names: set[str]

        try:
            signature = inspect.signature(CrawlerRunConfig)
            param_names = set(signature.parameters.keys())
        except Exception:
            param_names = set()

        if "cache_mode" in param_names and CacheMode is not None:
            cache_mode = getattr(CacheMode, "BYPASS", None) or getattr(CacheMode, "DISABLED", None)
            if cache_mode is not None:
                kwargs["cache_mode"] = cache_mode

        if "page_timeout" in param_names:
            kwargs["page_timeout"] = self.timeout_ms
        if "check_robots_txt" in param_names:
            kwargs["check_robots_txt"] = self.check_robots_txt
        if "remove_overlay_elements" in param_names:
            kwargs["remove_overlay_elements"] = True
        if "word_count_threshold" in param_names:
            kwargs["word_count_threshold"] = 12
        if "excluded_tags" in param_names:
            kwargs["excluded_tags"] = ["script", "style", "noscript", "svg"]
        if "process_iframes" in param_names:
            kwargs["process_iframes"] = False
        if "exclude_external_links" in param_names:
            kwargs["exclude_external_links"] = False
        if "verbose" in param_names:
            kwargs["verbose"] = False

        try:
            return CrawlerRunConfig(**kwargs)
        except Exception as e:
            logger.warning("Failed to build Crawl4AI run config, using defaults: %s", e)
            try:
                return CrawlerRunConfig()
            except Exception:
                return None

    def _extract_sync(self, url: str) -> CrawlExtractedContent | None:
        return asyncio.run(self._extract_async(url))

    async def _extract_async(self, url: str) -> CrawlExtractedContent | None:
        if AsyncWebCrawler is None:
            return None

        run_config = self._build_run_config()
        crawl_kwargs: dict[str, Any] = {"url": url}
        if run_config is not None:
            crawl_kwargs["config"] = run_config

        async with AsyncWebCrawler(verbose=False) as crawler:
            result = await crawler.arun(**crawl_kwargs)

        if not getattr(result, "success", False):
            error = getattr(result, "error_message", "unknown Crawl4AI error")
            raise RuntimeError(str(error))

        markdown_text = _normalize_whitespace(_extract_markdown(getattr(result, "markdown", None)))
        html_text = _html_to_text(str(getattr(result, "cleaned_html", "") or ""))

        merged_text = _normalize_whitespace("\n".join(part for part in [markdown_text, html_text] if part))
        if not merged_text:
            return None

        links = _extract_links(getattr(result, "links", None))
        status_code = getattr(result, "status_code", None)
        if not isinstance(status_code, int):
            status_code = None

        return CrawlExtractedContent(
            url=url,
            text=merged_text[: self.max_chars],
            links=links,
            status_code=status_code,
        )
