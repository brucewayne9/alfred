"""LLM Router - routes requests to Claude API, local Ollama, or Claude Code CLI.
Supports native tool calling via Claude API for reliable tool execution.
Complex queries hand off to Claude Code CLI (uses Max subscription, no API costs)."""

import asyncio
import json
import logging
import re
import shutil
from enum import Enum
from typing import AsyncGenerator

import anthropic
import ollama as ollama_client

from config.settings import settings
from core.tools.registry import get_tools_prompt, parse_tool_call, execute_tool, get_tools, _tools

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
    CLOUD = "cloud"  # Claude API with native tool calling
    CLAUDE_CODE = "claude-code"


# Queries matching these patterns get routed to Claude Code CLI
CLAUDE_CODE_TRIGGERS = [
    "explain", "write code", "debug", "plan",
    "compare", "evaluate", "strategy", "research",
    "legal", "complex", "detailed",
]

# Queries matching these patterns use Claude API with tool calling
TOOL_TRIGGERS = [
    "crm", "contact", "email", "calendar", "schedule", "meeting",
    "server", "upload", "create", "add", "send", "check",
    "reminder", "task", "opportunity", "company", "person",
    "import", "export", "csv", "document",
    "knowledge", "remember", "recall", "what did", "what was",
    "find in", "search for", "look up",
    # Meta Ads triggers
    "meta", "facebook", "instagram", "campaign", "ad ", "ads",
    "ctr", "cpc", "impressions", "conversions", "spend", "roas",
    # Stripe triggers
    "stripe", "payment", "invoice", "subscription", "charge", "refund",
    # Radio triggers
    "radio", "station", "playlist", "song", "dj", "stream",
    # Nextcloud triggers
    "nextcloud", "cloud storage",
    # n8n triggers
    "n8n", "workflow", "automation",
]


# Meta Ads queries go to Claude API for reliable native tool calling
META_ADS_TRIGGERS = [
    "meta", "facebook", "instagram", "meta ads", "facebook ads", "instagram ads",
    "campaign", "ad set", "ad performance", "ctr", "cpc", "cpm", "roas",
    "impressions", "conversions", "ad spend", "pause ad", "enable ad",
    "budget", "underperform", "ads manager",
]


def classify_query(query: str) -> ModelTier:
    """Determine if a query needs local Ollama (tools), Claude API, or Claude Code."""
    query_lower = query.lower().strip()

    # "claude" prefix always goes to Claude Code CLI
    if query_lower.startswith("claude ") or query_lower.startswith("claude,"):
        return ModelTier.CLAUDE_CODE

    # Meta Ads queries go to Claude API (native tool calling is more reliable)
    for trigger in META_ADS_TRIGGERS:
        if trigger in query_lower:
            return ModelTier.CLOUD

    # Other tool-related queries stay LOCAL
    for trigger in TOOL_TRIGGERS:
        if trigger in query_lower:
            return ModelTier.LOCAL

    # Long queries without tool triggers go to Claude Code
    if len(query) > 500:
        return ModelTier.CLAUDE_CODE

    for trigger in CLAUDE_CODE_TRIGGERS:
        if trigger in query_lower:
            return ModelTier.CLAUDE_CODE

    # Simple queries stay local (cheap, fast)
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


def _get_anthropic_tools(query: str = None) -> list[dict]:
    """Convert our tool definitions to Anthropic's tool format."""
    from core.tools.registry import get_relevant_tools

    # Get relevant tool names for this query
    if query:
        relevant_names = get_relevant_tools(query)
    else:
        relevant_names = list(_tools.keys())

    tools = []
    for name, tool in _tools.items():
        if name not in relevant_names:
            continue
        # Convert to Anthropic format
        properties = {}
        required = []
        for param_name, param_info in tool.get("parameters", {}).items():
            if isinstance(param_info, dict):
                properties[param_name] = {
                    "type": param_info.get("type", "string"),
                    "description": param_info.get("description", ""),
                }
                if param_info.get("required", False):
                    required.append(param_name)
            else:
                # Simple parameter definition
                properties[param_name] = {"type": "string", "description": str(param_info)}

        tools.append({
            "name": name,
            "description": tool["description"],
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        })
    return tools


async def query_claude_tools(query: str, messages: list[dict] | None = None) -> dict:
    """Query Claude API with native tool calling support.

    This is the reliable way to handle tool calls - Claude returns structured
    tool_use blocks instead of hoping the LLM outputs valid JSON text.
    """
    if not settings.anthropic_api_key or settings.anthropic_api_key == "sk-ant-CHANGEME":
        logger.warning("Claude API key not configured, falling back to local")
        return await query_local([
            {"role": "system", "content": get_system_prompt(query)},
            {"role": "user", "content": query},
        ])

    logger.info(f"Querying Claude API with tools: {query[:100]}...")

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    tools = _get_anthropic_tools(query)

    # Build messages for Claude (without system role in messages)
    memory_context = get_memory_context(query)
    system_content = SYSTEM_PROMPT_BASE
    if memory_context:
        system_content += "\n\n" + memory_context

    # Convert messages format (remove system messages, they go in system param)
    claude_messages = []
    if messages:
        for msg in messages:
            if msg["role"] != "system":
                claude_messages.append({"role": msg["role"], "content": msg["content"]})

    # Ensure we have the current user query
    if not claude_messages or claude_messages[-1]["content"] != query:
        claude_messages.append({"role": "user", "content": query})

    generated_images = []
    ui_action = None
    max_iterations = 5  # Prevent infinite tool loops

    for iteration in range(max_iterations):
        try:
            response = client.messages.create(
                model=settings.anthropic_model,
                max_tokens=4096,
                system=system_content,
                tools=tools if tools else None,
                messages=claude_messages,
            )
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return {"response": f"I encountered an error: {e}", "images": None, "ui_action": None}

        # Process response content blocks
        text_response = ""
        tool_uses = []

        for block in response.content:
            if block.type == "text":
                text_response += block.text
            elif block.type == "tool_use":
                tool_uses.append(block)

        # If no tool calls, we're done
        if not tool_uses:
            return {"response": text_response.strip(), "images": generated_images or None, "ui_action": ui_action}

        # Execute tool calls
        tool_results = []
        for tool_use in tool_uses:
            tool_name = tool_use.name
            tool_args = tool_use.input
            logger.info(f"Claude requested tool: {tool_name} with args: {tool_args}")

            tool_result = await execute_tool(tool_name, tool_args)

            # Handle special tool results
            if isinstance(tool_result, dict):
                if tool_result.get("ui_action"):
                    ui_action = {
                        "action": tool_result["ui_action"],
                        "value": tool_result.get("value"),
                    }

                if tool_name == "generate_image" and tool_result.get("success") and tool_result.get("base64"):
                    generated_images.append({
                        "base64": tool_result["base64"],
                        "filename": tool_result.get("filename", "generated.png"),
                        "download_url": tool_result.get("download_url", ""),
                    })
                    # Don't include base64 in tool result (too large)
                    tool_result = {k: v for k, v in tool_result.items() if k != "base64"}

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": json.dumps(tool_result, default=str) if not isinstance(tool_result, str) else tool_result,
            })

        # Add assistant response with tool uses and tool results
        claude_messages.append({"role": "assistant", "content": response.content})
        claude_messages.append({"role": "user", "content": tool_results})

        # If stop reason is end_turn, we're done after processing tools
        if response.stop_reason == "end_turn":
            # One more call to get final response
            continue

    # Max iterations reached
    return {"response": text_response.strip() or "I completed the requested actions.", "images": generated_images or None, "ui_action": ui_action}


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

    Routes to Claude API (with tools), local Ollama, or Claude Code CLI.
    """
    tier = tier or classify_query(query)
    logger.info(f"Routing to {tier.value}")

    # Claude Code handles its own context and tools via MCP
    if tier == ModelTier.CLAUDE_CODE:
        return await query_claude_code(query)

    # Claude API with native tool calling - most reliable for tool queries
    if tier == ModelTier.CLOUD:
        return await query_claude_tools(query, messages)

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
    """Remove any {"tool": ...} JSON blocks and tool narration from user-facing text."""
    # Remove fenced code blocks containing tool JSON
    text = re.sub(r'```(?:json)?\s*\{[^`]*"tool"\s*:[^`]*\}\s*```', '', text, flags=re.DOTALL)
    # Remove inline {"tool": ...} JSON objects
    text = re.sub(r'\{\s*"tool"\s*:\s*"[^"]*"\s*,\s*"args"\s*:\s*\{[^}]*\}\s*\}', '', text)
    # Remove tool narration patterns the LLM sometimes outputs
    narration_patterns = [
        r'We (?:must|need to|should|will) (?:call|use|fetch|invoke) (?:tool |the )?[\w_]+\.?\s*',
        r'We need to wait for (?:the )?result\.?\s*',
        r'User will provide (?:the )?result\.?\s*',
        r'(?:Let me |I\'ll |I will )(?:call|use|fetch|check) (?:the )?(?:tool )?[\w_]+\.?\s*',
        r'Calling (?:tool )?[\w_]+\.\.\.?\s*',
        r'Fetching (?:data|results?)\.\.\.?\s*',
    ]
    for pattern in narration_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    # Clean up leftover whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


SYSTEM_PROMPT_BASE = """You are Alfred, a personal AI assistant for Mike Johnson, owner of GrondRush Inc | GroundRush Labs.

Your role:
- Manage business operations (Base CRM, servers, deployments)
- Handle personal life management (family, health, calendar)
- Provide financial oversight
- Be proactive with reminders and suggestions
- Speak naturally and professionally

IMPORTANT - Actions and permissions:
- When the user ASKS you to do something (like "pause the ads", "do it", "yes"), that IS permission - execute the action.
- NEVER take actions on your own initiative without being asked (don't send messages to team members, don't make changes unprompted).
- Do not delegate tasks to team members - if Mike asks you to do something, use your tools to do it directly.
- You CAN and SHOULD make Meta Ads changes (pause ads, update budgets) when Mike asks you to.

Response style:
- Always respond in natural, conversational language as if speaking aloud.
- Never use markdown tables, code blocks, or bullet-point lists.
- Present information in flowing sentences and short paragraphs.
- For schedules, say things like "You have a 9:30 AM meeting with Mike Johnson for the AdMaster demo, then..." rather than tables.
- Keep responses concise and voice-friendly.
- Your responses are read aloud by a voice engine so write the way a British butler would speak, not the way a computer would print.
- NEVER add email-style signatures like "Kind regards, Alfred" or "Best, Alfred" to your responses. This is a chat interface, not email.
- NEVER use filler phrases like "One moment sir", "Right away sir", "Certainly sir", "Let me check", "Allow me to", etc. Go straight to the answer.

IMPORTANT - Memory instructions:
- When the user says "remember" followed by personal information (names, preferences, dates), you MUST use the "remember" tool to store it.
- Categories: "personal" for family/preferences, "business" for work, "financial" for money matters, "general" for everything else.
- Example: "Remember my wife's name is Sarah" → use remember tool with text="Wife's name is Sarah" and category="personal"

Be concise, helpful, and proactive. Address Mike as "sir" or by name.
When you don't know something, say so. When you need more information, ask.

CRITICAL tool-calling rules:
1. When you need data, USE THE TOOL IMMEDIATELY. Do not say "I need to fetch" or "User will provide" - just call the tool.
2. When calling a tool, your ENTIRE response must be ONLY the JSON object. No words before or after. Example:
{"tool": "create_event", "args": {"summary": "Meeting", "start_time": "2026-01-28T09:00:00"}}
3. NEVER say "I don't have that data" if a tool exists that can get it. Use the tool.
4. NEVER narrate what you're about to do. Just do it.
5. Either speak OR call a tool, never both in the same response."""
