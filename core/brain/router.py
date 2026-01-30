"""LLM Router - routes requests to local Ollama or Claude Code CLI.
Supports tool calling: LLM can invoke integrations (email, calendar, servers, memory).
Complex queries hand off to Claude Code CLI (uses Max subscription, no API costs)."""

import asyncio
import json
import logging
import re
import shutil
from enum import Enum
from typing import AsyncGenerator

import ollama as ollama_client

from config.settings import settings
from core.tools.registry import get_tools_prompt, parse_tool_call, execute_tool

logger = logging.getLogger(__name__)


def get_memory_context(query: str) -> str:
    """Automatically recall relevant memories based on the query.

    This runs before every response so Alfred always has personal context.
    """
    try:
        from core.memory.store import get_collection

        memories = []
        # Search across all memory categories
        for category in ["personal", "business", "general", "financial"]:
            try:
                coll = get_collection(f"memory_{category}")
                if coll.count() > 0:
                    results = coll.query(
                        query_texts=[query],
                        n_results=min(3, coll.count()),
                    )
                    for i, doc in enumerate(results["documents"][0]):
                        distance = results["distances"][0][i] if results["distances"] else 1.0
                        # Only include relevant memories (lower distance = more relevant)
                        if distance < 1.5:
                            memories.append(f"[{category}] {doc}")
            except Exception as e:
                logger.debug(f"Memory search in {category} failed: {e}")
                continue

        if memories:
            return "Your memory about the user:\n" + "\n".join(memories)
        return ""
    except Exception as e:
        logger.debug(f"Memory context retrieval failed: {e}")
        return ""


class ModelTier(str, Enum):
    LOCAL = "local"
    CLAUDE_CODE = "claude-code"


# Queries matching these patterns get routed to Claude Code
CLAUDE_CODE_TRIGGERS = [
    "explain", "write code", "debug", "plan",
    "compare", "evaluate", "strategy", "research",
    "financial", "legal", "complex", "detailed",
]

# Queries matching these patterns MUST stay local (they need tools)
LOCAL_TOOL_TRIGGERS = [
    "crm", "contact", "email", "calendar", "schedule", "meeting",
    "server", "upload", "create", "add", "send", "check",
    "reminder", "task", "opportunity", "company", "person",
    "import", "export", "csv", "document",
    "knowledge", "remember", "recall", "what did", "what was",
    "find in", "search for", "look up",
]


def classify_query(query: str) -> ModelTier:
    """Determine if a query needs local Ollama or Claude Code."""
    query_lower = query.lower().strip()

    # "claude" prefix always goes to Claude Code
    if query_lower.startswith("claude ") or query_lower.startswith("claude,"):
        return ModelTier.CLAUDE_CODE

    # Tool-related queries MUST stay local (Claude Code doesn't have tools)
    for trigger in LOCAL_TOOL_TRIGGERS:
        if trigger in query_lower:
            return ModelTier.LOCAL

    # Long queries without tool triggers go to Claude Code
    if len(query) > 500:
        return ModelTier.CLAUDE_CODE

    for trigger in CLAUDE_CODE_TRIGGERS:
        if trigger in query_lower:
            return ModelTier.CLAUDE_CODE

    return ModelTier.LOCAL


def get_system_prompt(query: str = None) -> str:
    """Build the full system prompt including available tools and memory context.

    If query is provided, only includes relevant tools to reduce context size.
    Also automatically retrieves relevant memories about the user.
    """
    tools_section = get_tools_prompt(query)
    memory_section = get_memory_context(query) if query else ""

    prompt = SYSTEM_PROMPT_BASE
    if memory_section:
        prompt += "\n\n" + memory_section
    if tools_section:
        prompt += "\n\n" + tools_section
    return prompt


async def query_local(messages: list[dict], model: str | None = None) -> str:
    model = model or settings.ollama_model
    logger.info(f"Querying local model: {model}")
    response = ollama_client.chat(model=model, messages=messages)
    return response["message"]["content"]


async def stream_local(messages: list[dict], model: str | None = None) -> AsyncGenerator[str, None]:
    model = model or settings.ollama_model
    logger.info(f"Streaming local model: {model}")
    stream = ollama_client.chat(model=model, messages=messages, stream=True)
    for chunk in stream:
        content = chunk["message"]["content"]
        if content:
            yield content


async def query_claude_vision(query: str, image_data: str, media_type: str = "image/jpeg") -> str:
    """Query Claude API with an image for vision analysis.

    Args:
        query: The user's question about the image
        image_data: Base64-encoded image data
        media_type: MIME type (image/jpeg, image/png, etc.)
    """
    import anthropic
    if not settings.anthropic_api_key or settings.anthropic_api_key == "sk-ant-CHANGEME":
        return "Vision analysis requires Anthropic API key. Please configure it in settings."

    logger.info(f"Querying Claude vision: {query[:100]}...")

    try:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        message = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": query or "Describe this image in detail.",
                        },
                    ],
                }
            ],
        )
        return message.content[0].text
    except Exception as e:
        logger.error(f"Claude vision error: {e}")
        return f"Vision analysis failed: {e}"


async def query_claude_code(query: str, allow_tools: bool = False) -> str:
    """Hand off a query to Claude Code CLI (uses Max subscription).

    Args:
        query: The user's query
        allow_tools: If True, enables file editing and tool use (for "fix alfred" type tasks)
    """
    claude_path = shutil.which("claude")
    if not claude_path:
        logger.warning("Claude Code CLI not found, falling back to local")
        return await query_local([
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": query},
        ])

    # Strip the "claude" prefix if present
    clean_query = query.strip()
    for prefix in ["claude,", "claude "]:
        if clean_query.lower().startswith(prefix):
            clean_query = clean_query[len(prefix):].strip()
            break

    # Check if this is a "fix alfred" type request that needs tools
    fix_keywords = ["fix", "edit", "update", "modify", "change", "add", "implement", "refactor", "debug"]
    alfred_keywords = ["alfred", "this code", "the code", "this file", "codebase"]
    query_lower = clean_query.lower()
    needs_tools = any(f in query_lower for f in fix_keywords) and any(a in query_lower for a in alfred_keywords)

    if needs_tools or allow_tools:
        logger.info(f"Handing off to Claude Code WITH TOOLS: {clean_query[:100]}...")
        # Use --dangerously-skip-permissions for full tool access
        cmd = [claude_path, "--dangerously-skip-permissions", "-p", clean_query]
    else:
        logger.info(f"Handing off to Claude Code (read-only): {clean_query[:100]}...")
        cmd = [claude_path, "-p", clean_query]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd="/home/aialfred/alfred",
            env={
                **__import__("os").environ,
                "PATH": "/home/aialfred/.nvm/versions/node/v20.12.2/bin:" + __import__("os").environ.get("PATH", ""),
            },
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)  # 5 min for complex tasks
        response = stdout.decode("utf-8", errors="replace").strip()
        if not response and stderr:
            response = f"Claude Code error: {stderr.decode('utf-8', errors='replace').strip()}"
        return response or "Claude Code returned no output."
    except asyncio.TimeoutError:
        logger.error("Claude Code timed out after 300s")
        return "Claude Code timed out. The task may be too complex for non-interactive mode."
    except Exception as e:
        logger.error(f"Claude Code failed: {e}")
        return f"Claude Code error: {e}"


async def ask(
    query: str,
    messages: list[dict] | None = None,
    tier: ModelTier | None = None,
    stream: bool = False,
):
    """Main entry point - ask Alfred a question.

    Routes to local Ollama or Claude Code CLI.
    If the local LLM response contains a tool call, executes it
    and feeds the result back for a final answer.
    """
    tier = tier or classify_query(query)
    logger.info(f"Routing to {tier.value}")

    # Claude Code handles its own context and tools via MCP
    if tier == ModelTier.CLAUDE_CODE:
        return await query_claude_code(query)

    if messages is None:
        messages = [
            {"role": "system", "content": get_system_prompt(query)},
            {"role": "user", "content": query},
        ]

    # Ensure system prompt has tools (filtered by query)
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] = get_system_prompt(query)

    if stream:
        return stream_local(messages)

    # Non-streaming: support tool calling loop
    response = await query_local(messages)
    generated_images = []
    ui_action = None

    # Check if LLM wants to call a tool
    tool_call = parse_tool_call(response)
    if tool_call:
        tool_name, tool_args = tool_call
        logger.info(f"LLM requested tool: {tool_name}")
        tool_result = await execute_tool(tool_name, tool_args)

        # Capture UI actions (for frontend to execute)
        if isinstance(tool_result, dict) and tool_result.get("ui_action"):
            ui_action = {
                "action": tool_result["ui_action"],
                "value": tool_result.get("value"),
            }
            # Use the message from the tool as the response
            response = tool_result.get("message", "Done.")
            # Don't need to query LLM again for UI actions
        # Capture generated images
        elif tool_name == "generate_image" and tool_result.get("success") and tool_result.get("base64"):
            generated_images.append({
                "base64": tool_result["base64"],
                "filename": tool_result.get("filename", "generated.png"),
                "download_url": tool_result.get("download_url", ""),
            })
            # Don't include base64 in LLM context (too large)
            tool_result_for_llm = {k: v for k, v in tool_result.items() if k != "base64"}
            # Feed tool result back to LLM for final response
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"[Tool result for {tool_name}]:\n```json\n{json.dumps(tool_result_for_llm, indent=2, default=str)}\n```\nNow provide a natural language response to the user based on this data.",
            })
            response = await query_local(messages)
        else:
            tool_result_for_llm = tool_result
            # Feed tool result back to LLM for final response
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"[Tool result for {tool_name}]:\n```json\n{json.dumps(tool_result_for_llm, indent=2, default=str)}\n```\nNow provide a natural language response to the user based on this data.",
            })
            response = await query_local(messages)

    # Safety net: strip any leaked JSON tool blocks from the final response
    response = _strip_tool_json(response)

    return {"response": response, "images": generated_images if generated_images else None, "ui_action": ui_action}


def _strip_tool_json(text: str) -> str:
    """Remove any {"tool": ...} JSON blocks that leaked into user-facing text."""
    # Remove fenced code blocks containing tool JSON
    text = re.sub(r'```(?:json)?\s*\{[^`]*"tool"\s*:[^`]*\}\s*```', '', text, flags=re.DOTALL)
    # Remove inline {"tool": ...} JSON objects
    text = re.sub(r'\{\s*"tool"\s*:\s*"[^"]*"\s*,\s*"args"\s*:\s*\{[^}]*\}\s*\}', '', text)
    # Clean up leftover whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


SYSTEM_PROMPT_BASE = """You are Alfred, a personal AI assistant for Bruce Johnson, owner of Ground Rush Inc.

Your role:
- Manage business operations (Base CRM, servers, deployments)
- Handle personal life management (family, health, calendar)
- Provide financial oversight
- Be proactive with reminders and suggestions
- Speak naturally and professionally

Response style:
- Always respond in natural, conversational language as if speaking aloud.
- Never use markdown tables, code blocks, or bullet-point lists.
- Present information in flowing sentences and short paragraphs.
- For schedules, say things like "You have a 9:30 AM meeting with Mike Johnson for the AdMaster demo, then..." rather than tables.
- Keep responses concise and voice-friendly.
- Your responses are read aloud by a voice engine so write the way a British butler would speak, not the way a computer would print.

IMPORTANT - Memory instructions:
- When the user says "remember" followed by personal information (names, preferences, dates), you MUST use the "remember" tool to store it.
- Categories: "personal" for family/preferences, "business" for work, "financial" for money matters, "general" for everything else.
- Example: "Remember my wife's name is Sarah" → use remember tool with text="Wife's name is Sarah" and category="personal"

Be concise, helpful, and proactive. Address Bruce as "sir" or by name.
When you don't know something, say so. When you need more information, ask.

CRITICAL tool-calling rule: When you need to use a tool, your ENTIRE response must be ONLY the JSON object and absolutely nothing else. No words before it, no words after it, no explanation. Just the raw JSON. Example of a correct tool-call response:
{"tool": "create_event", "args": {"summary": "Meeting", "start_time": "2026-01-28T09:00:00"}}
Never mix a spoken answer with a tool call in the same response. Either speak OR call a tool, never both."""
