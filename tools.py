"""
tools.py
--------
Implements the `search_web` tool that the AI research agent calls during its
agentic loop. This module is intentionally framework-agnostic so it can be
unit-tested or reused outside of FastAPI.

Behavior:
1. Primary backend: Tavily (`tavily-python`). Tavily is a search API tuned for
   LLM agents and returns structured JSON results.
2. Fallback backend: DuckDuckGo HTML scrape via `httpx`. We hit the
   `html.duckduckgo.com` lite endpoint and parse the result list with regex.
   This means the agent still works even if Tavily is unreachable, rate
   limited, or the API key is missing.

The function always returns a *single string* because that string is what gets
fed back into Claude as the `tool_result.content`. Returning a string (rather
than structured JSON) keeps the contract with the Anthropic SDK trivial.

All errors are caught and converted into a human-readable string so that the
agentic loop never crashes mid-conversation.
"""

from __future__ import annotations

import html
import os
import re
from typing import List, Dict, Any

import httpx
from dotenv import load_dotenv

load_dotenv()


TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()

DUCKDUCKGO_ENDPOINT = "https://html.duckduckgo.com/html/"


def _format_results(results: List[Dict[str, Any]]) -> str:
    """
    Convert a list of result dicts (with keys `title`, `url`, `snippet`) into
    a single newline-delimited string that is easy for the LLM to read.

    We number the results and include a clear separator so the model can refer
    to specific sources when writing the final report.
    """
    if not results:
        return "No results found."

    lines: List[str] = []
    for i, r in enumerate(results, start=1):
        title = (r.get("title") or "Untitled").strip()
        url = (r.get("url") or "").strip()
        snippet = (r.get("snippet") or "").strip()
        lines.append(f"[{i}] {title}\n    URL: {url}\n    {snippet}")
    return "\n\n".join(lines)


async def _search_tavily(query: str) -> List[Dict[str, Any]]:
    """
    Run a Tavily search and normalize the response to our common shape.

    Tavily's Python client is synchronous, so we instantiate it lazily and call
    it directly. The call itself is fast (one HTTP request) and FastAPI runs
    our endpoint in a worker, so wrapping it in `run_in_executor` is overkill
    for this project.

    Raises:
        Any underlying tavily/httpx exception, so the caller can fall back.
    """
    if not TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY is not set")

    # Imported lazily so a missing optional dependency doesn't break the
    # module-level import (the DuckDuckGo fallback still works).
    from tavily import TavilyClient

    client = TavilyClient(api_key=TAVILY_API_KEY)
    response = client.search(query=query, max_results=5)

    raw_results = response.get("results", []) if isinstance(response, dict) else []
    normalized: List[Dict[str, Any]] = []
    for item in raw_results:
        normalized.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                # Tavily returns the page snippet under `content`.
                "snippet": item.get("content", ""),
            }
        )
    return normalized


# Pre-compiled regexes for the DuckDuckGo HTML fallback. DuckDuckGo's lite
# HTML page is small and stable; we only need three fields per result.
_DDG_RESULT_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>'
    r'.*?<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode entities from a fragment of DDG markup."""
    return html.unescape(_HTML_TAG_RE.sub("", text)).strip()


async def _search_duckduckgo(query: str) -> List[Dict[str, Any]]:
    """
    Fallback search via DuckDuckGo's HTML endpoint.

    We use the lite HTML endpoint because it does not require JavaScript and
    is friendly to simple regex parsing. We still send a real-looking
    User-Agent header to avoid being blocked outright.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        resp = await client.post(
            DUCKDUCKGO_ENDPOINT,
            data={"q": query},
            headers=headers,
        )
        resp.raise_for_status()
        body = resp.text

    matches = _DDG_RESULT_RE.findall(body)
    results: List[Dict[str, Any]] = []
    for href, title_html, snippet_html in matches[:5]:
        results.append(
            {
                "title": _strip_html(title_html),
                "url": href.strip(),
                "snippet": _strip_html(snippet_html),
            }
        )
    return results


async def search_web(query: str) -> str:
    """
    Public tool entrypoint exposed to the LLM. Always returns a string.

    The function tries Tavily first (higher-quality, ranked results) and falls
    back to DuckDuckGo if Tavily raises *any* exception (missing key, network
    failure, rate limit, etc.). If both fail, we return a descriptive error
    string instead of raising — the agent can still continue the loop and try
    a different query.
    """
    query = (query or "").strip()
    if not query:
        return "Search failed: empty query."

    # Attempt #1: Tavily.
    try:
        results = await _search_tavily(query)
        if results:
            return _format_results(results)
        # Empty result set — fall through to DDG so the agent still gets data.
    except Exception as tavily_err:  # noqa: BLE001 — we intentionally catch all
        tavily_error_msg = f"{type(tavily_err).__name__}: {tavily_err}"
    else:
        tavily_error_msg = "Tavily returned no results"

    # Attempt #2: DuckDuckGo scrape.
    try:
        results = await _search_duckduckgo(query)
        if results:
            return _format_results(results)
        return f"Search failed: no results from Tavily ({tavily_error_msg}) or DuckDuckGo."
    except Exception as ddg_err:  # noqa: BLE001
        return (
            f"Search failed: Tavily error ({tavily_error_msg}); "
            f"DuckDuckGo error ({type(ddg_err).__name__}: {ddg_err})."
        )
