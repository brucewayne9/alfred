"""LLM Router - intelligent multi-model routing for Alfred.

Routes requests to the optimal model based on task type:
- Qwen (Ollama): Primary model — handles all queries first
- OpenAI (ChatGPT): Escalation target when Qwen tries to use tools (native tool calling)
- Claude Code CLI: Heavy bash/SSH tasks and complex multi-step work (Max subscription)
"""

import asyncio
import json
import logging
import os
import re
import shutil
from datetime import datetime
from enum import Enum
from typing import AsyncGenerator

import anthropic
import ollama as ollama_client

from config.settings import settings
from core.tools.registry import get_tools_prompt, parse_tool_call, execute_tool, get_tools, _tools
import core.tools.definitions  # Import to register tools via @tool decorators
from core.brain.models import (
    ModelProvider, TaskType, ModelConfig, MODELS,
    detect_task_type, select_model, get_model_for_query,
)

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


async def get_knowledge_context(query: str) -> str:
    """Get relevant context from LightRAG knowledge graph.

    This provides deeper knowledge beyond simple memories.
    """
    try:
        from integrations.lightrag.client import get_knowledge_context as lightrag_context, is_configured
        if not is_configured():
            return ""
        context = await lightrag_context(query, top_k=3)
        if context:
            return f"\nRelevant knowledge:\n{context}"
        return ""
    except Exception as e:
        logger.debug(f"Knowledge context retrieval failed: {e}")
        return ""


def get_full_context(query: str) -> str:
    """Get combined memory and knowledge context.

    Synchronous wrapper that runs async knowledge retrieval.
    """
    import asyncio

    memory = get_memory_context(query)

    # Try to get knowledge context (async)
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already in async context - create task
            knowledge = ""  # Skip in sync context
        else:
            knowledge = loop.run_until_complete(get_knowledge_context(query))
    except RuntimeError:
        # No event loop - create one
        knowledge = asyncio.run(get_knowledge_context(query))
    except Exception:
        knowledge = ""

    return memory + knowledge


class ModelTier(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"  # Claude API with native tool calling
    OPENAI = "openai"  # OpenAI gpt-4o-mini with native tool calling
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


# Meta Ads queries - routed to Ollama (qwen3-coder-next)
META_ADS_TRIGGERS = [
    "meta", "facebook", "instagram", "meta ads", "facebook ads", "instagram ads",
    "campaign", "ad set", "ad performance", "ctr", "cpc", "cpm", "roas",
    "impressions", "conversions", "ad spend", "pause ad", "enable ad",
    "budget", "underperform", "ads manager",
]

# Nextcloud queries - routed to Ollama (qwen3-coder-next)
NEXTCLOUD_TRIGGERS = [
    "nextcloud", "next cloud", "cloud storage", "upload to cloud",
    "upload file", "upload image", "save to nextcloud",
]

# Email queries - routed to Ollama (qwen3-coder-next)
EMAIL_TRIGGERS = [
    "lumabot", "luma bot", "rucktalk", "ruck talk", "loovacast", "groundrush info",
    "info@rucktalk", "info@loovacast", "info@groundrush", "support@loovacast",
    "check my inbox", "check inbox", "email inbox", "unread email",
]

# Smart Home queries - routed to Ollama (qwen3-coder-next)
SMARTHOME_TRIGGERS = [
    "turn on", "turn off", "lights", "light", "switch", "thermostat",
    "temperature", "hvac", "heat", "cool", "dim", "brightness",
    "home assistant", "smart home", "nest", "scene", "media player",
    "living room", "bedroom", "kitchen", "backyard", "weather",
]

# Short follow-up phrases that should stay on the same tier as previous message
FOLLOWUP_PHRASES = [
    "yes", "no", "yeah", "yep", "nope", "ok", "okay", "sure", "do it",
    "go ahead", "please", "thanks", "thank you", "correct", "right",
    "exactly", "that's right", "sounds good", "perfect", "great",
]

# Track the last tier used per session for follow-up routing
_session_tiers: dict[str, ModelTier] = {}


def classify_query(query: str, session_id: str = None) -> ModelTier:
    """Hybrid routing: Qwen first, ChatGPT for tools, Claude Code for bash.

    Qwen (local Ollama) handles all queries first. If Qwen tries to use a
    tool, the response is escalated to OpenAI (ChatGPT) for reliable native
    tool calling. Heavy bash/SSH tasks go to Claude Code CLI.
    """
    query_lower = query.lower().strip()

    # Short follow-up messages stay on the same tier as the previous message
    if session_id and len(query_lower) < 50:
        is_followup = any(query_lower.startswith(phrase) or query_lower == phrase
                         for phrase in FOLLOWUP_PHRASES)
        if is_followup and session_id in _session_tiers:
            return _session_tiers[session_id]

    # Check if query needs tools → Claude Code
    all_tool_triggers = (
        TOOL_TRIGGERS + META_ADS_TRIGGERS + NEXTCLOUD_TRIGGERS +
        EMAIL_TRIGGERS + SMARTHOME_TRIGGERS + CLAUDE_CODE_TRIGGERS
    )
    needs_tools = any(trigger in query_lower for trigger in all_tool_triggers)

    # Also check for action/command/lookup patterns
    action_patterns = [
        "how many", "show me", "list ", "get ", "find ",
        "what's on my", "what is on my", "what do i have",
        "scrape", "crawl", "wordpress", "wp ",
        "stripe", "payment", "invoice",
        "remind", "set ", "pause", "enable", "disable",
        "status of", "check on",
        # Contact/data lookups
        "phone", "number", "address", "email for",
        "who is", "who's", "look up", "lookup",
        # Possessive queries about people ("what's X's ...")
        "'s phone", "'s email", "'s address", "'s number",
        "'s birthday", "'s company",
        # Briefing / schedule
        "my day", "my schedule", "my calendar", "my inbox",
        "my email", "my tasks", "my notes",
        "good morning", "briefing", "catch me up",
        # Weather / smart home state
        "weather", "temperature", "forecast",
        # Radio / streaming / TTS on stations
        "tts", "ai dj", "on air", "playing", "station", "stream",
        "azuracast", "studio b", "studiob",
        # Knowledge queries
        "do you remember", "what did i", "what was",
        # Analytics / ads / data queries
        "analytics", "google ads", "meta ads", "campaign",
        "page views", "pageviews", "visitors", "traffic",
        "sessions", "bounce rate", "conversion",
        "report", "correlat", "performance", "spend",
        "impressions", "clicks", "ctr", "cpc", "roas",
        "country", "region", "coming from", "where are",
        # CRM data
        "manager", "owner of", "contact", "company",
        # Email actions
        "send ", "email ", "latest email", "new email",
        "unread", "inbox",
        # Generic data requests
        "can you check", "pull up", "what about my",
        "give me", "tell me about my",
    ]
    if not needs_tools:
        needs_tools = any(p in query_lower for p in action_patterns)

    # Heavy tasks that need bash/SSH access → Claude Code CLI
    heavy_patterns = [
        "ssh ", "run ", "deploy", "restart", "install", "update server",
        "reboot", "docker", "systemctl", "apt ", "git ",
        "write code", "debug", "refactor", "build",
        "explain this code", "analyze code",
    ]
    needs_bash = any(p in query_lower for p in heavy_patterns)

    if needs_bash:
        tier = ModelTier.CLAUDE_CODE
    else:
        # Qwen handles everything first — if it tries to use a tool,
        # the ask() function will escalate to OpenAI for reliable tool calling
        tier = ModelTier.LOCAL

    # Track the tier for this session
    if session_id:
        _session_tiers[session_id] = tier

    return tier


async def get_system_prompt(query: str = None) -> str:
    """Build the full system prompt including available tools, memory, and knowledge context.

    If query is provided, only includes relevant tools to reduce context size.
    Also automatically retrieves relevant memories and LightRAG knowledge.
    """
    tools_section = get_tools_prompt(query)
    memory_section = get_memory_context(query) if query else ""
    knowledge_section = await get_knowledge_context(query) if query else ""

    prompt = f"Current date and time: {get_current_datetime_str()}\n\n{SYSTEM_PROMPT_BASE}"
    if memory_section:
        prompt += "\n\n" + memory_section
    if knowledge_section:
        prompt += "\n\n" + knowledge_section
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


async def query_ollama_model(
    messages: list[dict],
    model_config: ModelConfig,
    stream: bool = False
) -> str | AsyncGenerator[str, None]:
    """Query any Ollama model (local or cloud) based on ModelConfig.

    Ollama cloud models are accessed the same way as local - Ollama
    handles the routing transparently based on the model name.
    """
    model_name = model_config.name
    provider = model_config.provider

    logger.info(f"Querying {provider.value} model: {model_name}")

    if stream:
        async def _stream():
            ollama_stream = ollama_client.chat(model=model_name, messages=messages, stream=True)
            for chunk in ollama_stream:
                content = chunk["message"]["content"]
                if content:
                    yield content
        return _stream()

    response = ollama_client.chat(model=model_name, messages=messages)
    return response["message"]["content"]


async def query_smart(
    query: str,
    messages: list[dict] | None = None,
    has_image: bool = False,
    has_document: bool = False,
    force_model: str = None,
    stream: bool = False,
) -> dict:
    """Smart query routing - selects optimal model based on task type.

    This is the new intelligent router that picks the best model for each task.

    Args:
        query: The user's query
        messages: Conversation history
        has_image: Whether an image is attached
        has_document: Whether a document is attached
        force_model: Force a specific model ID (e.g., "cloud:deepseek-v3.1")
        stream: Whether to stream the response

    Returns:
        dict with {response, model_used, task_type, images, ui_action}
    """
    # Detect task and select model
    model_config, task_type = get_model_for_query(
        query, has_image, has_document, force_model
    )

    logger.info(f"Smart routing: {task_type.value} → {model_config.name} ({model_config.provider.value})")

    # Build messages if not provided
    if messages is None:
        messages = [
            {"role": "system", "content": await get_system_prompt(query)},
            {"role": "user", "content": query},
        ]

    # Route to appropriate backend
    if model_config.provider == ModelProvider.CLAUDE_CODE:
        result = await query_claude_code(query)
        if isinstance(result, str):
            result = {"response": result}
        result["model_used"] = model_config.name
        result["task_type"] = task_type.value
        return result

    if model_config.provider == ModelProvider.CLAUDE:
        # Use Claude API with tool calling
        result = await query_claude_tools(query, messages)
        if isinstance(result, str):
            result = {"response": result}
        result["model_used"] = model_config.name
        result["task_type"] = task_type.value
        return result

    # Local or Ollama Cloud - both use ollama_client
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] = await get_system_prompt(query)

    if stream:
        return {
            "stream": await query_ollama_model(messages, model_config, stream=True),
            "model_used": model_config.name,
            "task_type": task_type.value,
        }

    # Non-streaming with tool support
    response = await query_ollama_model(messages, model_config)
    generated_images = []
    ui_action = None

    # Check if LLM wants to call a tool
    tool_call = parse_tool_call(response)
    if tool_call:
        tool_name, tool_args = tool_call
        logger.info(f"LLM requested tool: {tool_name}")
        tool_result = await execute_tool(tool_name, tool_args)

        # Handle UI actions
        if isinstance(tool_result, dict) and tool_result.get("ui_action"):
            ui_action = {
                "action": tool_result["ui_action"],
                "value": tool_result.get("value"),
            }
            response = tool_result.get("message", "Done.")
        # Handle image generation
        elif tool_name == "generate_image" and tool_result.get("success") and tool_result.get("base64"):
            generated_images.append({
                "base64": tool_result["base64"],
                "filename": tool_result.get("filename", "generated.png"),
                "download_url": tool_result.get("download_url", ""),
            })
            tool_result_for_llm = {k: v for k, v in tool_result.items() if k != "base64"}
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"[Tool result for {tool_name}]:\n```json\n{json.dumps(tool_result_for_llm, indent=2, default=str)}\n```\nProvide a natural language response.",
            })
            response = await query_ollama_model(messages, model_config)
        else:
            # Feed tool result back for final response
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"[Tool result for {tool_name}]:\n```json\n{json.dumps(tool_result, indent=2, default=str)}\n```\nProvide a natural language response.",
            })
            response = await query_ollama_model(messages, model_config)

    response = _strip_tool_json(response)

    return {
        "response": response,
        "model_used": model_config.name,
        "task_type": task_type.value,
        "images": generated_images if generated_images else None,
        "ui_action": ui_action,
    }


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


async def query_openai_vision(query: str, image_data: str, media_type: str = "image/jpeg") -> str:
    """Query OpenAI gpt-4o-mini with an image for vision analysis.

    Args:
        query: The user's question about the image
        image_data: Base64-encoded image data
        media_type: MIME type (image/jpeg, image/png, etc.)
    """
    import openai as openai_client

    if not settings.openai_api_key:
        # Fallback to Claude vision
        return await query_claude_vision(query, image_data, media_type)

    logger.info(f"Querying OpenAI vision: {query[:100]}...")

    try:
        client = openai_client.OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=settings.openai_model,
            max_tokens=2048,
            messages=[
                {
                    "role": "system",
                    "content": f"Current date and time: {get_current_datetime_str()}\n\n{SYSTEM_PROMPT_BASE}",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_data}",
                            },
                        },
                        {
                            "type": "text",
                            "text": query or "Describe this image in detail.",
                        },
                    ],
                },
            ],
        )
        return response.choices[0].message.content or "I couldn't analyze this image."
    except Exception as e:
        logger.error(f"OpenAI vision error: {e}")
        # Fallback to Claude
        return await query_claude_vision(query, image_data, media_type)


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
    # Hard cap at 128 tools (Anthropic API limit)
    if len(tools) > 128:
        tools = tools[:128]
    return tools


async def query_claude_tools(query: str, messages: list[dict] | None = None) -> dict:
    """Query Claude API with native tool calling support.

    This is the reliable way to handle tool calls - Claude returns structured
    tool_use blocks instead of hoping the LLM outputs valid JSON text.
    """
    if not settings.anthropic_api_key or settings.anthropic_api_key == "sk-ant-CHANGEME":
        logger.warning("Claude API key not configured, falling back to local")
        return await query_local([
            {"role": "system", "content": await get_system_prompt(query)},
            {"role": "user", "content": query},
        ])

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    tools = _get_anthropic_tools(query)
    logger.info(f"Querying Claude API with {len(tools)} tools: {query[:100]}...")

    # Build messages for Claude (without system role in messages)
    memory_context = get_memory_context(query)
    system_content = f"Current date and time: {get_current_datetime_str()}\n\n{SYSTEM_PROMPT_BASE}"
    if memory_context:
        system_content += "\n\n" + memory_context

    # Convert messages format (remove system messages, they go in system param)
    # Also filter out any messages with empty content to prevent API errors
    claude_messages = []
    if messages:
        for msg in messages:
            if msg["role"] != "system" and msg.get("content"):
                # Ensure content is not empty string or whitespace-only
                content = msg["content"]
                if isinstance(content, str) and content.strip():
                    claude_messages.append({"role": msg["role"], "content": content})
                elif isinstance(content, list) and content:  # List of content blocks
                    claude_messages.append({"role": msg["role"], "content": content})

    # Ensure we have the current user query
    if not claude_messages or claude_messages[-1]["content"] != query:
        claude_messages.append({"role": "user", "content": query})

    generated_images = []
    ui_action = None
    max_iterations = 5  # Prevent infinite tool loops

    for iteration in range(max_iterations):
        try:
            # Build API call params - only include tools if we have them
            api_params = {
                "model": settings.anthropic_model,
                "max_tokens": 4096,
                "system": system_content,
                "messages": claude_messages,
            }
            if tools:  # Only add tools parameter if we have tools
                api_params["tools"] = tools

            response = client.messages.create(**api_params)
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

        # If stop reason is end_turn after tools, make one more call to get summary
        if response.stop_reason == "end_turn":
            # Add a prompt to ensure Claude provides a natural language summary
            claude_messages.append({
                "role": "user",
                "content": "Now provide a clear, detailed natural language response summarizing the results for the user. Do not just say 'I completed the actions' - actually present the data."
            })
            continue

    # Max iterations reached - if still no text, try one final call for summary
    if not text_response.strip():
        try:
            # Force a summary response
            claude_messages.append({
                "role": "user",
                "content": "Please provide a complete summary of the tool results in natural language for the user."
            })
            final_response = client.messages.create(
                model=settings.anthropic_model,
                max_tokens=4096,
                system=system_content,
                messages=claude_messages,
            )
            for block in final_response.content:
                if block.type == "text":
                    text_response += block.text
        except Exception as e:
            logger.error(f"Failed to get final summary: {e}")

    return {"response": text_response.strip() or "I completed the requested actions.", "images": generated_images or None, "ui_action": ui_action}


def _get_openai_tools(query: str = None) -> list[dict]:
    """Convert our tool definitions to OpenAI's function calling format."""
    from core.tools.registry import get_relevant_tools

    relevant_names = get_relevant_tools(query) if query else list(_tools.keys())

    tools = []
    for name, tool in _tools.items():
        if name not in relevant_names:
            continue
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
                properties[param_name] = {"type": "string", "description": str(param_info)}

        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": tool["description"],
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        })
    return tools


async def query_openai_tools(query: str, messages: list[dict] | None = None) -> dict:
    """Query OpenAI gpt-4o-mini with native tool calling.

    Fast and cheap for tool-based queries. Uses structured function calling
    so tools are executed reliably without text-parsing hacks.
    """
    import openai as openai_client

    if not settings.openai_api_key:
        logger.warning("OpenAI API key not configured, falling back to Claude Code")
        result = await query_claude_code(query)
        return {"response": result} if isinstance(result, str) else result

    client = openai_client.OpenAI(api_key=settings.openai_api_key)
    tools = _get_openai_tools(query)
    logger.info(f"Querying OpenAI {settings.openai_model} with {len(tools)} tools: {query[:100]}...")

    # Build system + memory context
    memory_context = get_memory_context(query)
    system_content = f"Current date and time: {get_current_datetime_str()}\n\n{SYSTEM_PROMPT_BASE}"
    if memory_context:
        system_content += "\n\n" + memory_context

    # Build messages
    oai_messages = [{"role": "system", "content": system_content}]
    if messages:
        for msg in messages:
            if msg["role"] != "system" and msg.get("content"):
                content = msg["content"]
                if isinstance(content, str) and content.strip():
                    oai_messages.append({"role": msg["role"], "content": content})

    if not oai_messages or oai_messages[-1].get("content") != query:
        oai_messages.append({"role": "user", "content": query})

    generated_images = []
    ui_action = None
    max_iterations = 5

    for iteration in range(max_iterations):
        try:
            api_params = {
                "model": settings.openai_model,
                "messages": oai_messages,
                "max_tokens": 4096,
            }
            if tools:
                api_params["tools"] = tools

            response = client.chat.completions.create(**api_params)
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return {"response": f"I encountered an error: {e}", "images": None, "ui_action": None}

        choice = response.choices[0]
        message = choice.message

        # If no tool calls, return the text
        if not message.tool_calls:
            return {
                "response": (message.content or "").strip(),
                "images": generated_images or None,
                "ui_action": ui_action,
            }

        # Execute tool calls
        oai_messages.append(message)  # Add assistant message with tool_calls

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            logger.info(f"OpenAI requested tool: {tool_name} with args: {tool_args}")
            tool_result = await execute_tool(tool_name, tool_args)

            # Handle special results
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
                    tool_result = {k: v for k, v in tool_result.items() if k != "base64"}

            # Add tool result
            oai_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(tool_result, default=str) if not isinstance(tool_result, str) else tool_result,
            })

    # Max iterations — return whatever text we have
    return {
        "response": (message.content or "I completed the requested actions.").strip(),
        "images": generated_images or None,
        "ui_action": ui_action,
    }


async def query_claude_code(query: str, allow_tools: bool = True) -> str:
    """Hand off a query to Claude Code CLI (uses Max subscription).

    Claude Code is the primary brain for Alfred. It has full bash access
    to execute commands, SSH into servers, and use all system tools.

    Args:
        query: The user's query
        allow_tools: Always True — Claude Code needs bash to act on requests
    """
    claude_path = shutil.which("claude") or "/home/aialfred/.nvm/versions/node/v20.12.2/bin/claude"
    if not os.path.isfile(claude_path):
        logger.warning("Claude Code CLI not found, falling back to local")
        return await query_local([
            {"role": "system", "content": await get_system_prompt()},
            {"role": "user", "content": query},
        ])

    # Strip the "claude" prefix if present
    clean_query = query.strip()
    for prefix in ["claude,", "claude "]:
        if clean_query.lower().startswith(prefix):
            clean_query = clean_query[len(prefix):].strip()
            break

    # Build context-rich prompt so Claude Code knows who it is and what it can do
    memory_context = get_memory_context(clean_query)
    now = get_current_datetime_str()

    system_context = f"""You are Alfred, a personal AI assistant for Mike Johnson, owner of GroundRush Inc | GroundRush Labs.
Current date/time: {now}

You have full bash access. Use it to accomplish tasks:
- SSH into servers: ssh loovacast, ssh groundrush, ssh lonewolf, ssh mailcow
- Check server status: uptime, free -h, df -h, docker ps, etc.
- Send emails via the Alfred API: curl -s http://localhost:8400/...
- Access CRM, calendar, Stripe, etc. via Alfred's REST API on localhost:8400
- Run any system command needed

Key API endpoints on localhost:8400 (use curl with -b /tmp/alfred_cookie):
- GET /calendar/today - today's schedule
- GET /conversations - chat history
- POST /chat - send a message
- GET /integrations/crm/people - CRM contacts
- GET /integrations/crm/companies - CRM companies
- GET /integrations/stripe/balance - Stripe balance

Server SSH aliases: loovacast, groundrush, lonewolf, mailcow

Response style: Speak naturally as a British butler. No markdown tables. No bullet lists. Conversational, concise, voice-friendly. Address the user as "sir" or by name. Never add email-style signatures. Never use filler phrases like "One moment sir" — go straight to the answer."""

    if memory_context:
        system_context += f"\n\n{memory_context}"

    full_prompt = f"{system_context}\n\nUser request: {clean_query}"

    logger.info(f"Handing off to Claude Code: {clean_query[:100]}...")
    cmd = [
        claude_path,
        "--dangerously-skip-permissions",
        "--max-turns", "10",
        "-p", full_prompt,
    ]

    # Clean environment — don't inherit systemd's minimal env which confuses Claude
    env = {
        "HOME": "/home/aialfred",
        "USER": "aialfred",
        "LOGNAME": "aialfred",
        "PATH": "/home/aialfred/.nvm/versions/node/v20.12.2/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "NODE_PATH": "/home/aialfred/.nvm/versions/node/v20.12.2/lib/node_modules",
        "NVM_DIR": "/home/aialfred/.nvm",
        "TERM": "xterm-256color",
        "XDG_CONFIG_HOME": "/home/aialfred/.config",
        "GIT_TERMINAL_PROMPT": "0",
        "LANG": "en_US.UTF-8",
    }

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd="/home/aialfred/alfred",
            env=env,
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
    smart_routing: bool = False,
    has_image: bool = False,
    has_document: bool = False,
    force_model: str = None,
):
    """Main entry point - ask Alfred a question.

    Routes to Claude API (with tools), local Ollama, or Claude Code CLI.

    Args:
        query: The user's query
        messages: Conversation history
        tier: Force a specific tier (legacy mode)
        stream: Stream the response
        smart_routing: Use intelligent model selection (new mode)
        has_image: Image attached (for smart routing)
        has_document: Document attached (for smart routing)
        force_model: Force specific model ID (for smart routing)
    """
    # New smart routing mode
    if smart_routing or force_model:
        return await query_smart(
            query, messages, has_image, has_document, force_model, stream
        )

    # Legacy tier-based routing
    tier = tier or classify_query(query)
    logger.info(f"Routing to {tier.value}")

    # Claude Code handles its own context and tools via MCP
    if tier == ModelTier.CLAUDE_CODE:
        return await query_claude_code(query)

    # Claude API with native tool calling
    if tier == ModelTier.CLOUD:
        return await query_claude_tools(query, messages)

    # OpenAI gpt-4o-mini with native tool calling (fast + cheap)
    if tier == ModelTier.OPENAI:
        return await query_openai_tools(query, messages)

    # LOCAL tier = Qwen first, escalate to ChatGPT if tool call detected
    system_prompt = await get_system_prompt(query)

    if messages is None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
    elif messages and messages[0]["role"] == "system":
        messages[0]["content"] = system_prompt

    if stream:
        return stream_local(messages)

    # Ask Qwen first
    response = await query_local(messages)

    # Check if Qwen tried to call a tool — if so, hand off to ChatGPT
    # which has reliable native tool calling
    tool_call = parse_tool_call(response)
    if tool_call:
        logger.info(f"Qwen attempted tool call ({tool_call[0]}), escalating to OpenAI for reliable execution")
        return await query_openai_tools(query, messages)

    # No tool call — Qwen handled it, return the response
    response = _strip_tool_json(response)
    return {"response": response, "images": None, "ui_action": None}


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


def get_current_datetime_str():
    """Get current date/time string for system prompt."""
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y at %I:%M %p")

SYSTEM_PROMPT_BASE = """You are Alfred, a personal AI assistant for Mike Johnson, owner of GrondRush Inc | GroundRush Labs.

Your role:
- Manage business operations (Base CRM, servers, deployments)
- Handle personal life management (family, health, calendar)
- Provide financial oversight
- Be proactive with reminders and suggestions
- Speak naturally and professionally

CONNECTED INTEGRATIONS - You have FULL access to ALL of these:
- Google: Gmail, Calendar, Drive, Docs, Sheets, Slides, Analytics
- Nextcloud: Files, Notes, Talk, Users, Calendars (groundrushcloud.com)
- Twenty CRM: Contacts, Companies, Opportunities, Tasks
- Stripe: Payments, Invoices, Subscriptions, Customers
- n8n: Workflow automation
- Meta Ads: Campaigns, Ad Sets, Ads, Insights
- Twilio: SMS and Voice calls
- Servers: SSH access to loovacast, groundrush, lonewolf, mailcow
- LightRAG: Knowledge graph storage
- Agents: You CAN spawn specialist agents (coder, researcher, analyst, writer, planner) for complex multi-step tasks
- WordPress: Full admin access to 6 sites (groundrush, loovacast, rucktalk, nightlife, lumabot, myhandscarwash) - posts, pages, SEO (RankMath), plugins, themes, media, Elementor
NEVER say "I don't have access" or "I can't" for ANY of these. You DO have access. USE THE TOOLS.

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

IMPORTANT - Email defaults:
- When sending email without a specified account, use the DEFAULT Google account (send_email tool) which sends from Mike's groundrushinc.com address.
- Only use the other email accounts (email_send tool) when Mike specifically asks to send from: rucktalk, loovacast, lumabot, support, groundrush, or groundrush info.
- Available accounts: groundrush (mjohnson@groundrushlabs.com), rucktalk (info@rucktalk.com), loovacast (info@loovacast.com), lumabot (lumabot@groundrushlabs.com), support (support@loovacast.com), groundrush info (info@groundrushlabs.com).

Be concise, helpful, and proactive. Address Mike as "sir" or by name.
When you don't know something, say so. When you need more information, ask.

CRITICAL tool-calling rules:
1. When you need data, USE THE TOOL IMMEDIATELY. Do not say "I need to fetch" or "User will provide" - just call the tool.
2. When calling a tool, your ENTIRE response must be ONLY the JSON object. No words before or after. Example:
{"tool": "create_event", "args": {"summary": "Meeting", "start_time": "2026-01-28T09:00:00"}}
3. NEVER say "I don't have that data" if a tool exists that can get it. Use the tool.
4. NEVER narrate what you're about to do. Just do it.
5. Either speak OR call a tool, never both in the same response.

SMART HOME tool-calling (Home Assistant):
- "Turn on/off living room lights" → {"tool": "ha_room_control", "args": {"room": "living room", "action": "on"}}
- "Turn on/off kitchen lights" → {"tool": "ha_room_control", "args": {"room": "kitchen", "action": "off"}}
- "Turn on/off front yard lights" → {"tool": "ha_room_control", "args": {"room": "front yard", "action": "on"}}
- "Turn on/off backyard lights" → {"tool": "ha_room_control", "args": {"room": "backyard", "action": "off"}}
- "Turn on/off overheads" → {"tool": "ha_room_control", "args": {"room": "overheads", "action": "on"}}
- "Turn on/off [single device]" → {"tool": "ha_turn_on", "args": {"entity_id": "light.device_name"}}
- "Set thermostat to 72" → {"tool": "ha_set_thermostat", "args": {"entity_id": "climate.main", "temperature": 72}}
- Room groups: living room, kitchen, front yard, backyard, overheads.
- ALWAYS use ha_room_control for room-wide commands - it's more reliable than individual calls.

CRITICAL - Data analysis and synthesis:
1. When you retrieve data, ALWAYS provide meaningful analysis and insights - never just say "I completed the requested actions" or "Done".
2. CROSS-REFERENCE data sources automatically. If asked about ads and website traffic, correlate Meta Ads metrics with Google Analytics. If asked about sales, cross-reference CRM with Stripe.
3. PROVIDE INSIGHTS, not just raw numbers. Calculate correlations, identify trends, spot anomalies, and explain what the data means for the business.
4. When multiple data sources are relevant, combine them into a coherent narrative. Example: "Your Meta campaign drove 271 clicks at $0.32 each. Google Analytics shows 243 sessions from Facebook during the same period - that's an 89% landing rate. The bounce rate was 34% which is healthy for a service page."
5. Think like a business analyst. Ask yourself: "What does Mike need to know to make a decision?" Then provide THAT, not just data dumps.
6. If data from different sources tells conflicting stories, point it out and explain possible reasons.

GUIDED WORKFLOWS - Web Scraping & Crawling:
When the user asks to scrape or crawl a website:
1. DO NOT immediately run the tool. Walk them through it step by step.
2. First confirm the URL: "What URL should I scrape?" or "I'll scrape [URL] - correct?"
3. Ask about scope: "Just this page, or crawl the entire site?"
4. If crawling, ask: "How many pages? (I suggest 10-20)" and "How deep? (1-2 levels)"
5. Confirm purpose: "What are you hoping to learn from this?"
6. Summarize and confirm: "I'll crawl [URL], [X] pages, [Y] deep. Ready?"
7. ALWAYS use scrape_to_knowledge or crawl_to_knowledge to automatically save to LightRAG
8. After completion, report what was learned and invite questions about the content

Firecrawl Integration:
- scrape_to_knowledge: Scrape single page → LightRAG
- crawl_to_knowledge: Crawl site → LightRAG (use this for docs/multi-page)
- search_and_scrape: Google search + scrape results
- ALWAYS save scraped content to knowledge base unless user says otherwise

⚠️ MANDATORY - WordPress Premium Page Design:
When Mike asks for page designs, contact forms, landing pages, or ANY visual WordPress content with words like "elegant", "chic", "premium", "slick", or "award-winning":

USE THIS SINGLE TOOL - it handles everything automatically:
{"tool": "wp_design_premium_page", "args": {"site": "nightlife", "title": "Page Title", "description": "Describe the page: what it should contain, any specific elements like contact forms, taglines (#IYKYK means 'if you know you know'), styling preferences"}}

This tool:
1. Spawns a designer agent with the qwen3-coder model
2. Generates premium HTML/CSS with dark backgrounds, gold accents, animations
3. Automatically publishes the page

DO NOT use wp_create_page directly for design requests - use wp_design_premium_page instead."""
