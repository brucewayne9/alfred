"""Model configuration and task routing for Alfred's multi-model architecture.

This module defines which models handle which tasks, enabling intelligent
routing between local Ollama, Ollama cloud, and Claude API models.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ModelProvider(str, Enum):
    """Model hosting provider."""
    LOCAL = "local"           # Local Ollama (RTX 4070)
    OLLAMA_CLOUD = "ollama"   # Ollama cloud models
    CLAUDE = "claude"         # Anthropic Claude API
    CLAUDE_CODE = "claude-code"  # Claude Code CLI


class TaskType(str, Enum):
    """Types of tasks Alfred can handle."""
    # Simple/Fast tasks (local)
    EMBEDDING = "embedding"
    CLASSIFICATION = "classification"
    SIMPLE_CHAT = "simple_chat"

    # Code tasks
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    CODE_EXPLANATION = "code_explanation"

    # Reasoning tasks
    COMPLEX_REASONING = "complex_reasoning"
    ANALYSIS = "analysis"
    RESEARCH = "research"
    PLANNING = "planning"

    # Tool/Integration tasks
    TOOL_CALLING = "tool_calling"
    INTEGRATION = "integration"

    # Agentic tasks
    ORCHESTRATION = "orchestration"
    MULTI_STEP = "multi_step"
    AGENT_SPAWN = "agent_spawn"

    # Multimodal
    VISION = "vision"
    DOCUMENT_ANALYSIS = "document_analysis"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str                    # Ollama model name or API identifier
    provider: ModelProvider
    context_window: int = 8192
    supports_tools: bool = False
    supports_vision: bool = False
    supports_thinking: bool = False
    cost_tier: int = 1           # 1=free, 2=cheap, 3=moderate, 4=expensive
    speed_tier: int = 2          # 1=fast, 2=medium, 3=slow
    quality_tier: int = 2        # 1=basic, 2=good, 3=excellent
    description: str = ""


# ==================== MODEL REGISTRY ====================

MODELS = {
    # LOCAL MODELS (RTX 4070 - 12GB VRAM)
    "local:nomic-embed": ModelConfig(
        name="nomic-embed-text",
        provider=ModelProvider.LOCAL,
        context_window=8192,
        cost_tier=1, speed_tier=1, quality_tier=2,
        description="Fast local embeddings for RAG"
    ),
    "local:llama3.2:3b": ModelConfig(
        name="llama3.2:3b",
        provider=ModelProvider.LOCAL,
        context_window=8192,
        cost_tier=1, speed_tier=1, quality_tier=1,
        description="Ultra-fast for simple tasks"
    ),
    "local:mistral:7b": ModelConfig(
        name="mistral:7b-instruct",
        provider=ModelProvider.LOCAL,
        context_window=32768,
        supports_tools=True,
        cost_tier=1, speed_tier=1, quality_tier=2,
        description="Fast routing and classification"
    ),
    "local:qwen2.5-coder:7b": ModelConfig(
        name="qwen2.5-coder:7b",
        provider=ModelProvider.LOCAL,
        context_window=32768,
        supports_tools=True,
        cost_tier=1, speed_tier=1, quality_tier=2,
        description="Local code generation"
    ),

    # OLLAMA CLOUD MODELS
    "cloud:ministral-3b": ModelConfig(
        name="ministral-3:3b",
        provider=ModelProvider.OLLAMA_CLOUD,
        context_window=32768,
        supports_tools=True,
        cost_tier=2, speed_tier=1, quality_tier=2,
        description="Fast general tasks"
    ),
    "cloud:ministral-8b": ModelConfig(
        name="ministral-3:8b",
        provider=ModelProvider.OLLAMA_CLOUD,
        context_window=32768,
        supports_tools=True,
        supports_vision=True,
        cost_tier=2, speed_tier=1, quality_tier=2,
        description="Fast with vision support"
    ),
    "cloud:qwen3-coder": ModelConfig(
        name="qwen3-coder",
        provider=ModelProvider.OLLAMA_CLOUD,
        context_window=32768,
        supports_tools=True,
        cost_tier=2, speed_tier=2, quality_tier=3,
        description="Code specialist"
    ),
    "cloud:devstral-24b": ModelConfig(
        name="devstral-small-2:24b",
        provider=ModelProvider.OLLAMA_CLOUD,
        context_window=32768,
        supports_tools=True,
        supports_vision=True,
        cost_tier=2, speed_tier=2, quality_tier=3,
        description="Code agent with file editing"
    ),
    "cloud:devstral-123b": ModelConfig(
        name="devstral-2:123b",
        provider=ModelProvider.OLLAMA_CLOUD,
        context_window=32768,
        supports_tools=True,
        cost_tier=3, speed_tier=3, quality_tier=3,
        description="Advanced code agent"
    ),
    "cloud:nemotron-30b": ModelConfig(
        name="nemotron-3-nano:30b",
        provider=ModelProvider.OLLAMA_CLOUD,
        context_window=32768,
        supports_tools=True,
        supports_thinking=True,
        cost_tier=2, speed_tier=2, quality_tier=3,
        description="Efficient agentic model"
    ),
    "cloud:deepseek-v3.1": ModelConfig(
        name="deepseek-v3.1",
        provider=ModelProvider.OLLAMA_CLOUD,
        context_window=65536,
        supports_tools=True,
        supports_thinking=True,
        cost_tier=3, speed_tier=3, quality_tier=3,
        description="Complex reasoning with thinking mode"
    ),
    "cloud:deepseek-v3.2": ModelConfig(
        name="deepseek-v3.2",
        provider=ModelProvider.OLLAMA_CLOUD,
        context_window=65536,
        supports_tools=True,
        cost_tier=3, speed_tier=2, quality_tier=3,
        description="Latest DeepSeek - high efficiency"
    ),
    "cloud:qwen3-vl": ModelConfig(
        name="qwen3-vl:32b",
        provider=ModelProvider.OLLAMA_CLOUD,
        context_window=32768,
        supports_tools=True,
        supports_vision=True,
        supports_thinking=True,
        cost_tier=2, speed_tier=2, quality_tier=3,
        description="Vision + language + thinking"
    ),
    "cloud:kimi-k2.5": ModelConfig(
        name="kimi-k2.5",
        provider=ModelProvider.OLLAMA_CLOUD,
        context_window=32768,
        supports_tools=True,
        supports_vision=True,
        cost_tier=3, speed_tier=2, quality_tier=3,
        description="Multimodal agentic model"
    ),
    "cloud:gpt-oss-120b": ModelConfig(
        name="gpt-oss:120b-cloud",
        provider=ModelProvider.OLLAMA_CLOUD,
        context_window=32768,
        supports_tools=True,
        supports_thinking=True,
        cost_tier=3, speed_tier=3, quality_tier=3,
        description="OpenAI open model - strong reasoning"
    ),
    "cloud:gemini-3-flash": ModelConfig(
        name="gemini-3-flash-preview",
        provider=ModelProvider.OLLAMA_CLOUD,
        context_window=65536,
        cost_tier=2, speed_tier=1, quality_tier=2,
        description="Fast Gemini for speed"
    ),
    "cloud:gemini-3-pro": ModelConfig(
        name="gemini-3-pro-preview",
        provider=ModelProvider.OLLAMA_CLOUD,
        context_window=65536,
        cost_tier=3, speed_tier=2, quality_tier=3,
        description="Gemini Pro - strong reasoning"
    ),

    # CLAUDE MODELS
    "claude:sonnet": ModelConfig(
        name="claude-sonnet-4-20250514",
        provider=ModelProvider.CLAUDE,
        context_window=200000,
        supports_tools=True,
        supports_vision=True,
        cost_tier=4, speed_tier=2, quality_tier=3,
        description="Claude Sonnet - orchestration & quality"
    ),
    "claude:opus": ModelConfig(
        name="claude-opus-4-5-20251101",
        provider=ModelProvider.CLAUDE,
        context_window=200000,
        supports_tools=True,
        supports_vision=True,
        supports_thinking=True,
        cost_tier=4, speed_tier=3, quality_tier=3,
        description="Claude Opus - highest quality"
    ),
}


# ==================== TASK ROUTING ====================

# Maps task types to preferred models (in order of preference)
TASK_ROUTING: dict[TaskType, list[str]] = {
    # Simple/Fast - use local
    TaskType.EMBEDDING: ["local:nomic-embed"],
    TaskType.CLASSIFICATION: ["local:mistral:7b", "cloud:ministral-3b"],
    TaskType.SIMPLE_CHAT: ["local:llama3.2:3b", "cloud:ministral-3b", "local:mistral:7b"],

    # Code tasks - use code specialists
    TaskType.CODE_GENERATION: ["cloud:devstral-24b", "cloud:qwen3-coder", "local:qwen2.5-coder:7b"],
    TaskType.CODE_REVIEW: ["cloud:devstral-24b", "cloud:qwen3-coder", "claude:sonnet"],
    TaskType.CODE_EXPLANATION: ["cloud:devstral-24b", "local:qwen2.5-coder:7b"],

    # Reasoning tasks - use thinking models
    TaskType.COMPLEX_REASONING: ["cloud:deepseek-v3.1", "cloud:deepseek-v3.2", "claude:sonnet"],
    TaskType.ANALYSIS: ["cloud:deepseek-v3.2", "cloud:nemotron-30b", "claude:sonnet"],
    TaskType.RESEARCH: ["cloud:deepseek-v3.1", "cloud:gpt-oss-120b", "claude:sonnet"],
    TaskType.PLANNING: ["claude:sonnet", "cloud:deepseek-v3.1", "cloud:nemotron-30b"],

    # Tool/Integration tasks - need tool support
    TaskType.TOOL_CALLING: ["claude:sonnet", "cloud:nemotron-30b", "cloud:devstral-24b"],
    TaskType.INTEGRATION: ["claude:sonnet", "cloud:kimi-k2.5"],

    # Agentic tasks - orchestration models
    TaskType.ORCHESTRATION: ["claude:sonnet", "cloud:kimi-k2.5", "cloud:nemotron-30b"],
    TaskType.MULTI_STEP: ["claude:sonnet", "cloud:deepseek-v3.1", "cloud:nemotron-30b"],
    TaskType.AGENT_SPAWN: ["claude:sonnet"],  # Only Claude spawns agents

    # Multimodal
    TaskType.VISION: ["cloud:qwen3-vl", "cloud:kimi-k2.5", "claude:sonnet"],
    TaskType.DOCUMENT_ANALYSIS: ["cloud:qwen3-vl", "claude:sonnet", "cloud:deepseek-v3.2"],
}


# ==================== TASK DETECTION ====================

# Keywords that indicate specific task types
TASK_KEYWORDS: dict[TaskType, list[str]] = {
    TaskType.CODE_GENERATION: [
        "write code", "create function", "implement", "code for",
        "script to", "program that", "build a", "develop",
    ],
    TaskType.CODE_REVIEW: [
        "review code", "check this code", "improve this", "refactor",
        "optimize", "what's wrong with", "fix this code",
    ],
    TaskType.CODE_EXPLANATION: [
        "explain this code", "what does this", "how does this work",
        "walk me through", "understand this",
    ],
    TaskType.COMPLEX_REASONING: [
        "analyze", "compare", "evaluate", "trade-offs", "pros and cons",
        "implications", "consequences", "reasoning", "logic",
    ],
    TaskType.ANALYSIS: [
        "analyze", "breakdown", "examine", "investigate", "deep dive",
        "detailed look", "assessment",
    ],
    TaskType.RESEARCH: [
        "research", "find out", "look into", "investigate",
        "what do you know about", "tell me about",
    ],
    TaskType.PLANNING: [
        "plan", "strategy", "roadmap", "approach", "how should i",
        "steps to", "process for", "workflow",
    ],
    TaskType.VISION: [
        "this image", "this picture", "what do you see", "describe this",
        "in this photo", "looking at",
    ],
    TaskType.DOCUMENT_ANALYSIS: [
        "this document", "this pdf", "this file", "analyze document",
        "read this", "summarize this doc",
    ],
    TaskType.MULTI_STEP: [
        "step by step", "multiple steps", "process", "workflow",
        "then", "after that", "next",
    ],
}

# Tool-related keywords (routes to TOOL_CALLING)
TOOL_KEYWORDS = [
    # CRM & contacts
    "crm", "contact", "lead", "opportunity", "company", "person", "deal",
    # Email
    "email", "inbox", "unread", "mail", "message",
    # Calendar
    "calendar", "schedule", "meeting", "appointment", "event", "today's schedule",
    # Actions
    "server", "upload", "create", "add", "send", "check", "pause", "resume",
    "stop", "start", "update", "delete", "remove", "set", "change", "modify",
    # Reminders & tasks
    "reminder", "remind me", "task", "todo", "to-do",
    # Finance
    "stripe", "payment", "invoice", "subscription", "charge", "refund",
    # Media
    "radio", "station", "playlist", "song", "azuracast",
    # Cloud storage
    "nextcloud", "cloud storage", "file", "document",
    # Automation
    "n8n", "workflow", "automate", "automation",
    # Ads & marketing
    "meta", "facebook", "instagram", "campaign", "ads", "ad ", "budget",
    # Analytics
    "analytics", "traffic", "google analytics", "visitors", "pageviews",
    # Memory
    "remember", "recall", "what do you know about",
    # Briefing
    "briefing", "brief me", "summary of", "status report",
]


def detect_task_type(query: str, has_image: bool = False, has_document: bool = False) -> TaskType:
    """Detect the task type from a query.

    Args:
        query: The user's query text
        has_image: Whether an image is attached
        has_document: Whether a document is attached

    Returns:
        The detected TaskType
    """
    query_lower = query.lower()

    # Check for multimodal first
    if has_image:
        return TaskType.VISION
    if has_document:
        return TaskType.DOCUMENT_ANALYSIS

    # Check for tool keywords (these need tool calling)
    if any(kw in query_lower for kw in TOOL_KEYWORDS):
        return TaskType.TOOL_CALLING

    # Check task-specific keywords
    for task_type, keywords in TASK_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            return task_type

    # Check query length - long queries likely need reasoning
    if len(query) > 500:
        return TaskType.COMPLEX_REASONING

    # Default to simple chat
    return TaskType.SIMPLE_CHAT


def select_model(
    task_type: TaskType,
    require_tools: bool = False,
    require_vision: bool = False,
    prefer_speed: bool = False,
    prefer_quality: bool = False,
    prefer_cheap: bool = True,
) -> ModelConfig:
    """Select the best model for a given task.

    Args:
        task_type: The type of task to perform
        require_tools: Must support tool calling
        require_vision: Must support vision
        prefer_speed: Prefer faster models
        prefer_quality: Prefer higher quality models
        prefer_cheap: Prefer cheaper models (default True)

    Returns:
        The selected ModelConfig
    """
    candidates = TASK_ROUTING.get(task_type, ["local:mistral:7b"])

    for model_id in candidates:
        config = MODELS.get(model_id)
        if not config:
            continue

        # Check requirements
        if require_tools and not config.supports_tools:
            continue
        if require_vision and not config.supports_vision:
            continue

        # First valid candidate wins (they're ordered by preference)
        return config

    # Fallback to local mistral
    return MODELS["local:mistral:7b"]


def get_model_for_query(
    query: str,
    has_image: bool = False,
    has_document: bool = False,
    force_model: str = None,
) -> tuple[ModelConfig, TaskType]:
    """Get the best model for a query.

    Args:
        query: The user's query
        has_image: Whether an image is attached
        has_document: Whether a document is attached
        force_model: Force a specific model (optional)

    Returns:
        Tuple of (ModelConfig, TaskType)
    """
    if force_model and force_model in MODELS:
        task_type = detect_task_type(query, has_image, has_document)
        return MODELS[force_model], task_type

    task_type = detect_task_type(query, has_image, has_document)

    model = select_model(
        task_type,
        require_tools=(task_type == TaskType.TOOL_CALLING),
        require_vision=(task_type == TaskType.VISION),
    )

    logger.info(f"Task: {task_type.value} → Model: {model.name} ({model.provider.value})")

    return model, task_type
