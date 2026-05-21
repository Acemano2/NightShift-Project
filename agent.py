"""
agent.py
--------
The actual research agent. Implements `run_agent`, an async generator that
drives a Claude tool-use loop and yields structured events that the FastAPI
layer streams to the browser as Server-Sent Events.

Event shapes (always JSON-serializable dicts):
    {"type": "step",   "content": "<human readable progress message>"}
    {"type": "report", "content": "<final markdown report>"}
    {"type": "done",   "content": ""}
    {"type": "error",  "content": "<error message>"}

Design notes:
- We use the official `anthropic` Python SDK in *async* mode so the entire
  request lifecycle is non-blocking and plays well with FastAPI.
- The model is given a single tool, `search_web`. We loop until Claude returns
  an `end_turn` stop reason with text content — that text is the final report.
- A hard cap (`MAX_ITERATIONS`) prevents runaway loops in pathological cases.
- All exceptions are converted into "error" events so the UI can surface them
  cleanly instead of dropping the SSE connection.
"""

from __future__ import annotations

import os
from typing import AsyncGenerator, Any, Dict, List, Optional

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from tools import search_web

load_dotenv()


ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
MODEL_NAME = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = (
    "You are a research agent. To answer the user's question, search the web "
    "multiple times (at least 4 searches). Search for different angles: "
    "facts, recent news, expert opinions, counterarguments. After gathering "
    "enough info, write a comprehensive structured report.\n\n"
    "When you write the final report, use Markdown with clear section "
    "headings (##), bullet points where appropriate, and cite sources inline "
    "using their URLs. Do not call any more tools after you start writing the "
    "report."
)

# The single tool we expose to Claude. The JSON schema mirrors the signature
# of `search_web` in tools.py.
TOOLS: List[Dict[str, Any]] = [
    {
        "name": "search_web",
        "description": (
            "Search the public web for information about a topic. Returns the "
            "top results as a numbered list with title, URL, and snippet. "
            "Call this multiple times with different queries to gather diverse "
            "perspectives before writing the report."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to run.",
                }
            },
            "required": ["query"],
        },
    }
]

# Safety net so the loop can't spin forever if something goes wrong upstream.
MAX_ITERATIONS = 12
MAX_TOKENS_PER_TURN = 4096


def _extract_text(content_blocks: List[Any]) -> str:
    """Concatenate all `text` blocks from an Anthropic message response."""
    parts: List[str] = []
    for block in content_blocks:
        # The SDK returns block objects with a `.type` attribute.
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", "") or "")
    return "".join(parts).strip()


def _serialize_assistant_content(content_blocks: List[Any]) -> List[Dict[str, Any]]:
    """
    Convert the SDK's content block objects back into the dict form that the
    Anthropic API expects when we echo them in `messages`. We keep only the
    fields the API actually requires for each block type.
    """
    serialized: List[Dict[str, Any]] = []
    for block in content_blocks:
        btype = getattr(block, "type", None)
        if btype == "text":
            serialized.append({"type": "text", "text": getattr(block, "text", "")})
        elif btype == "tool_use":
            serialized.append(
                {
                    "type": "tool_use",
                    "id": getattr(block, "id", ""),
                    "name": getattr(block, "name", ""),
                    "input": getattr(block, "input", {}) or {},
                }
            )
    return serialized


async def run_agent(
    query: str,
    history: Optional[List[Dict[str, Any]]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Run the research agent on `query` and yield streaming progress events.

    Args:
        query: The user's research topic / question.
        history: Optional prior chat turns (used for follow-up questions on
                 the same session). Each element should already be in
                 Anthropic message format: {"role": "...", "content": ...}.

    Yields:
        Dicts shaped like the event contract documented at the top of this
        file. The generator always finishes with either a {"type": "done"}
        event or a {"type": "error"} event.
    """
    query = (query or "").strip()
    if not query:
        yield {"type": "error", "content": "Empty query."}
        return

    if not ANTHROPIC_API_KEY:
        yield {
            "type": "error",
            "content": (
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and "
                "fill in your key."
            ),
        }
        return

    client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    # Seed the conversation with any prior history plus the new user turn.
    messages: List[Dict[str, Any]] = list(history or [])
    messages.append({"role": "user", "content": query})

    yield {"type": "step", "content": f"🧠 Planning research for: {query}"}

    try:
        for iteration in range(MAX_ITERATIONS):
            response = await client.messages.create(
                model=MODEL_NAME,
                max_tokens=MAX_TOKENS_PER_TURN,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            stop_reason = response.stop_reason
            content_blocks = response.content or []

            # Collect every tool_use block in this assistant turn — Claude can
            # in principle ask for multiple parallel tool calls.
            tool_uses = [b for b in content_blocks if getattr(b, "type", None) == "tool_use"]

            if stop_reason == "tool_use" and tool_uses:
                # Echo the assistant turn back into the message history so the
                # API can correlate `tool_result` blocks with `tool_use` ids.
                messages.append(
                    {
                        "role": "assistant",
                        "content": _serialize_assistant_content(content_blocks),
                    }
                )

                tool_results_content: List[Dict[str, Any]] = []
                for tu in tool_uses:
                    tool_name = getattr(tu, "name", "")
                    tool_input = getattr(tu, "input", {}) or {}
                    tool_use_id = getattr(tu, "id", "")

                    if tool_name == "search_web":
                        sub_query = str(tool_input.get("query", "")).strip()
                        yield {
                            "type": "step",
                            "content": f"🔍 Searching: {sub_query}",
                        }
                        result_text = await search_web(sub_query)
                        result_count = result_text.count("\n[") + (
                            1 if result_text.startswith("[") else 0
                        )
                        # If the helper returned an error string, surface 0.
                        if result_text.startswith("Search failed") or result_text.startswith(
                            "No results"
                        ):
                            result_count = 0
                        yield {
                            "type": "step",
                            "content": f"📄 Got {result_count} results",
                        }
                    else:
                        # Defensive: Claude shouldn't invent tools, but if it
                        # does we report the error back so it can recover.
                        result_text = f"Unknown tool: {tool_name}"
                        yield {
                            "type": "step",
                            "content": f"⚠️ Unknown tool requested: {tool_name}",
                        }

                    tool_results_content.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": result_text,
                        }
                    )

                messages.append({"role": "user", "content": tool_results_content})
                # Loop again — Claude will either request more searches or
                # produce the final report.
                continue

            # No more tool calls: the assistant's text *is* the final report.
            report_text = _extract_text(content_blocks)
            if not report_text:
                yield {
                    "type": "error",
                    "content": (
                        "The model returned no text content. Stop reason was "
                        f"'{stop_reason}'."
                    ),
                }
                return

            yield {"type": "step", "content": "✍️ Writing report..."}
            yield {"type": "report", "content": report_text}
            yield {"type": "done", "content": ""}
            return

        # If we exhausted MAX_ITERATIONS without a final report, tell the UI.
        yield {
            "type": "error",
            "content": (
                f"Agent stopped after {MAX_ITERATIONS} iterations without "
                "producing a final report."
            ),
        }
    except Exception as exc:  # noqa: BLE001 — convert *any* failure to an event
        yield {
            "type": "error",
            "content": f"Agent error: {type(exc).__name__}: {exc}",
        }
