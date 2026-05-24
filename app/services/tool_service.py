"""
Lightweight tool-use layer for the hosted assistant.
Tools are implemented as plain Python functions and exposed to the model
via Groq's function-calling API (OpenAI-compatible).

Tools available:
  - web_search (DuckDuckGo, no API key required)
  - calculator
  - get_current_datetime
"""
from __future__ import annotations

import json
import datetime
from typing import Any

from app.utils.logger import get_logger

logger = get_logger(__name__)

# ─── Tool Definitions (OpenAI function-call schema) ───────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information. Use when asked about recent events, facts you're unsure about, or real-time data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression. Use for arithmetic, percentages, and basic math.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A safe mathematical expression e.g. '2 + 2', '15% of 200', 'sqrt(144)'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "Get the current date and time.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
]

# ─── Tool Implementations ─────────────────────────────────────────────────────

def web_search(query: str) -> str:
    """DuckDuckGo instant answer search — no API key required."""
    try:
        import urllib.request
        import urllib.parse
        encoded = urllib.parse.quote(query)
        url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1&skip_disambig=1"
        req = urllib.request.Request(url, headers={"User-Agent": "OlliveAI/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())

        parts = []
        if data.get("AbstractText"):
            parts.append(data["AbstractText"])
        if data.get("Answer"):
            parts.append(f"Answer: {data['Answer']}")
        for r in data.get("RelatedTopics", [])[:3]:
            if isinstance(r, dict) and r.get("Text"):
                parts.append(r["Text"])

        return "\n".join(parts) if parts else f"No instant results for '{query}'. Please consult a search engine directly."
    except Exception as e:
        logger.warning("web_search_failed", query=query, error=str(e))
        return f"Search unavailable: {e}"


def calculator(expression: str) -> str:
    """Safe math evaluator — only allows numbers and math operators."""
    import re, math
    # Whitelist: digits, operators, spaces, dots, parentheses, common functions
    if not re.match(r'^[\d\s\+\-\*\/\(\)\.\^%sqrtpowlogabsceilfloorround]+$',
                    expression.lower()):
        return "Error: expression contains disallowed characters."
    try:
        # Replace ^ with ** for power
        expr = expression.replace("^", "**").replace("sqrt", "math.sqrt") \
                         .replace("log", "math.log").replace("abs", "abs") \
                         .replace("ceil", "math.ceil").replace("floor", "math.floor")
        result = eval(expr, {"__builtins__": {}, "math": math, "abs": abs})
        return str(round(result, 10))
    except Exception as e:
        return f"Calculation error: {e}"


def get_current_datetime() -> str:
    now = datetime.datetime.now()
    return now.strftime("%A, %B %d, %Y at %I:%M %p")


# ─── Dispatcher ───────────────────────────────────────────────────────────────

TOOL_MAP = {
    "web_search": web_search,
    "calculator": calculator,
    "get_current_datetime": get_current_datetime,
}


def dispatch_tool(name: str, arguments: dict[str, Any]) -> str:
    """Execute a tool call and return the result as a string."""
    fn = TOOL_MAP.get(name)
    if not fn:
        return f"Unknown tool: {name}"
    try:
        result = fn(**arguments)
        logger.info("tool_dispatched", tool=name, args=str(arguments)[:100])
        return str(result)
    except Exception as e:
        logger.error("tool_dispatch_error", tool=name, error=str(e))
        return f"Tool error: {e}"


class ToolEnabledHostedAssistant:
    """
    Wraps HostedAssistant to add function-calling support.
    Falls back to plain generation if tools aren't triggered.
    """

    def __init__(self, service) -> None:
        self.service = service
        self._client = None

    def chat_with_tools(self, user_message: str, history: list) -> tuple[str, bool]:
        """
        Returns (response_text, used_tools: bool).
        Runs one agentic loop: generate → maybe call tool → generate final response.
        """
        from app.models.base_assistant import Message, SYSTEM_PROMPT
        from app.utils.config import get_config
        config = get_config()

        if not self._client:
            from groq import Groq
            self._client = Groq(api_key=config.gemini_api_key)

        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in history:
            msgs.append({"role": m.role, "content": m.content})
        msgs.append({"role": "user", "content": user_message})

        # First pass — let model decide if it wants tools
        response = self._client.chat.completions.create(
            model=config.hosted_model_name,
            messages=msgs,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
        )

        choice = response.choices[0]

        # If model wants to call a tool
        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            used_tools = True
            msgs.append(choice.message)  # add assistant tool-call message

            for tool_call in choice.message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments or "{}")
                result = dispatch_tool(name, args)

                msgs.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

            # Second pass — generate final answer with tool results
            final = self._client.chat.completions.create(
                model=config.hosted_model_name,
                messages=msgs,
                max_tokens=config.max_new_tokens,
                temperature=config.temperature,
            )
            return final.choices[0].message.content or "", used_tools

        # No tool calls — return direct response
        return choice.message.content or "", False
