"""
main.py
-------
FastAPI entry point. Serves the single-page frontend at `/` and exposes a
streaming `/research` endpoint that pipes events from the research agent to
the browser as Server-Sent Events.

Run with:
    python main.py
or:
    uvicorn main:app --reload
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from agent import run_agent

load_dotenv()


BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML = BASE_DIR / "index.html"

app = FastAPI(
    title="AI Research Agent",
    description="An autonomous research agent that streams live progress.",
    version="1.0.0",
)


class ResearchRequest(BaseModel):
    """Request body for POST /research."""

    query: str = Field(..., min_length=1, description="Topic to research.")


@app.get("/")
async def root() -> FileResponse:
    """Serve the single-page frontend."""
    if not INDEX_HTML.exists():
        # Should not happen in normal use, but handle gracefully just in case.
        return JSONResponse(
            status_code=500,
            content={"error": "index.html is missing from the project root."},
        )
    return FileResponse(INDEX_HTML, media_type="text/html")


@app.get("/healthz")
async def healthz() -> JSONResponse:
    """Lightweight health check, handy for deployment platforms."""
    return JSONResponse(
        {
            "ok": True,
            "anthropic_key_set": bool(os.getenv("ANTHROPIC_API_KEY")),
            "tavily_key_set": bool(os.getenv("TAVILY_API_KEY")),
        }
    )


def _sse_format(event: dict) -> str:
    """
    Encode a dict as a single SSE `data:` frame.

    We JSON-encode the *entire* event payload (including the `type` field) so
    the client only has to do one parse per frame. A blank line terminates the
    SSE event per the spec.
    """
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


async def _event_stream(
    query: str,
    request: Request,
) -> AsyncGenerator[str, None]:
    """
    Bridge between `run_agent` (yields dicts) and the SSE wire format (yields
    strings). Also handles client disconnects so we don't keep generating
    tokens for a browser that has already navigated away.
    """
    try:
        # Initial comment frame — keeps some proxies (nginx) from buffering
        # the first real event until additional data is flushed.
        yield ": stream-open\n\n"

        async for event in run_agent(query):
            if await request.is_disconnected():
                # Stop spending tokens if the client closed the tab.
                break
            yield _sse_format(event)
    except Exception as exc:  # noqa: BLE001
        # Last-ditch safety net: never let an exception escape the stream,
        # because that would close the connection without telling the user
        # what happened.
        yield _sse_format(
            {"type": "error", "content": f"Server error: {type(exc).__name__}: {exc}"}
        )


@app.post("/research")
async def research(body: ResearchRequest, request: Request) -> StreamingResponse:
    """
    Kick off a research run and stream the agent's events back to the client.

    The response uses the `text/event-stream` content type so a fetch-based
    reader on the client can parse it incrementally without any extra
    library.
    """
    headers = {
        # Disable any kind of buffering between us and the client.
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        _event_stream(body.query, request),
        media_type="text/event-stream",
        headers=headers,
    )


if __name__ == "__main__":
    # Run via `python main.py` for the simplest possible developer workflow.
    # `reload=False` because uvicorn's reloader requires the app passed as a
    # string, and we don't need it in production-style runs.
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
        log_level="info",
    )
