# Architecture

**Analysis Date:** 2026-02-20

## Pattern Overview

**Overall:** Layered multi-model AI orchestration platform with FastAPI backend, React frontend, and modular integration system.

**Key Characteristics:**
- Multi-model LLM routing (local Ollama + cloud providers + Claude Code CLI)
- Tool-based LLM autonomy with registry-driven execution
- Memory-augmented agents with ChromaDB vector store
- Async-first WebSocket support for real-time interactions
- Modular integration architecture with per-service clients

## Layers

**HTTP API Layer:**
- Purpose: RESTful endpoints for auth, chat, integrations, knowledge management, and agent orchestration
- Location: `core/api/main.py`
- Contains: 60+ endpoints handling requests, WebSocket connections, file uploads, streaming responses
- Depends on: All core modules, integrations, security, notifications
- Used by: React frontend, mobile clients, external webhooks

**Brain/LLM Router Layer:**
- Purpose: Intelligent routing of queries to optimal models based on task type and capability
- Location: `core/brain/router.py`, `core/brain/models.py`
- Contains: Task type detection, model selection logic, context injection (memory + knowledge graph)
- Depends on: Tool registry, memory store, LightRAG client, Ollama, Anthropic API
- Used by: Chat endpoints, agent orchestration

**Tool Calling & Execution Layer:**
- Purpose: Manages 150+ LLM-callable tools grouped by category (email, CRM, smart home, etc.)
- Location: `core/tools/registry.py`, `core/tools/definitions.py`
- Contains: Tool registration decorator, tool lookup, parameter parsing, execution dispatch
- Depends on: Integration clients (gmail, stripe, homeassistant, etc.)
- Used by: Brain router for tool execution, LLM context generation

**Integration Layer:**
- Purpose: Adapters for external services (email, CRM, payment, smart home, etc.)
- Location: `integrations/` (26 integration modules)
- Contains: Service-specific clients with API methods (e.g., `base_crm/client.py`, `stripe/client.py`)
- Pattern: Each integration has `client.py` with authenticated requests, error handling, data transformation
- Depends on: HTTP clients (requests), external APIs, environment settings
- Used by: Tool definitions to execute commands

**Memory & Persistence Layer:**
- Purpose: Semantic memory (ChromaDB), conversation history (SQLite), user state (JSON)
- Location: `core/memory/store.py` (ChromaDB), `core/memory/conversations.py` (SQLite)
- Contains: Vector store operations, FTS5 search, project/reference management, message history
- Depends on: ChromaDB, SQLite, filesystem
- Used by: Router for memory context, API for conversation retrieval

**Authentication & Security Layer:**
- Purpose: User management, JWT tokens, TOTP 2FA, WebAuthn passkeys, Google OAuth
- Location: `core/security/auth.py`, `core/security/google_oauth.py`
- Contains: Password hashing, token creation/validation, passkey registration/verification, QR code generation
- Depends on: pyotp, passlib, python-jose, WebAuthn libraries
- Used by: API middleware for route protection, login endpoints

**Notification & WebSocket Layer:**
- Purpose: Real-time event broadcasting to clients (agent status, alerts, task updates)
- Location: `core/notifications/manager.py`
- Contains: WebSocket connection management, notification types, VAPID push setup
- Depends on: FastAPI WebSocket, asyncio
- Used by: Agent orchestration, long-running operations

**Agent Orchestration Layer:**
- Purpose: Multi-agent spawning and task coordination for complex workflows
- Location: `core/orchestration/agents.py`
- Contains: Agent pool initialization, task queueing, result aggregation
- Depends on: Brain router, notification manager, memory store
- Used by: Agent spawn endpoints, complex multi-step operations

**Interface Layer:**
- Purpose: Multi-channel access points (Telegram, WhatsApp, voice, MCP)
- Location: `interfaces/` (telegram, whatsapp, voice, mcp, mobile)
- Contains: Channel-specific handlers, protocol adapters, event processing
- Depends on: Core API functions, external channel SDKs
- Used by: External services to invoke Alfred

**Frontend/UI Layer:**
- Purpose: React-based interactive web interface with real-time features
- Location: `frontend/src/`
- Contains: Components, state management (Zustand stores), API client, Tailwind styling
- Depends on: Core API endpoints via HTTP/WebSocket
- Used by: Users accessing Alfred via web browser

## Data Flow

**Chat Query with Tool Calling:**

1. User sends message via React frontend → `/chat` endpoint
2. API classifies query task type (simple_chat, tool_calling, complex_reasoning, etc.)
3. Brain router retrieves memory context via `get_memory_context()` and knowledge context via `get_knowledge_context()`
4. System selects optimal model based on task type:
   - Simple queries → Local Ollama (Qwen3-coder-next)
   - Tool-heavy queries → OpenAI GPT-4 (native tool support) or Claude
   - Complex reasoning → Claude Sonnet via Anthropic API
5. Tool-enabled models receive full tool list via `get_tools_prompt()` (alphabetically sorted, under 128 token budget)
6. Model generates response with optional tool calls (JSON-formatted)
7. API parses tool calls via `parse_tool_call()` → looks up tool in registry → validates parameters
8. Tool executes via integration client (e.g., `stripe_create_customer` → `stripe/client.py`)
9. Tool result fed back to model as context continuation
10. Model generates final response combining reasoning + tool results
11. Message pair stored in SQLite (`conversations.db`) via `add_message()`
12. Response streamed to frontend via Server-Sent Events or returned as JSON
13. Frontend updates chat store and UI

**Memory Injection Flow:**

1. Router automatically calls `get_memory_context(query)` before every LLM call
2. Searches ChromaDB collections: `memory_personal`, `memory_business`, `memory_general`, `memory_financial`
3. Returns top 3 matches with cosine distance < 1.5 (relevance threshold)
4. Memory context injected into system prompt as: "Your memory about the user:\n[memory1]\n[memory2]\n[memory3]"
5. LLM has full context without explicit user recall requests

**Knowledge Graph Flow (LightRAG):**

1. Router calls `get_knowledge_context(query)` for additional context beyond simple memories
2. LightRAG client queries knowledge graph via `/query` endpoint
3. Returns entity relationships and contextual information (top k=3)
4. Injected into prompt as: "Relevant knowledge:\n[graph_context]"
5. Provides deeper semantic understanding vs. raw memory vectors

**Agent Orchestration Flow:**

1. User calls `/agents/spawn` with task description
2. API creates agent instance and appends to `agent_tasks` list in memory
3. Agent runs in background, invoking LLM repeatedly for multi-step task
4. On each agent LLM call, tool results accumulated in agent's context
5. When agent completes/fails, `NotificationManager` broadcasts `AGENT_COMPLETED` or `AGENT_FAILED`
6. Frontend receives notification via WebSocket and updates UI
7. User polls `/agents/tasks/{task_id}` to retrieve final result

**State Management:**

- **User Auth State:** JWT token stored in HTTP-only cookie (`alfred_token`), validated on every protected route via `get_current_user()`
- **Conversation State:** SQLite `conversations` table with indexed lookup by ID/updated_at, FTS5 search on `messages` table
- **Memory State:** ChromaDB collections with HNSW indexing, vector embeddings stored permanently
- **Agent State:** In-memory task queue with optional persistence to session files
- **UI State:** React Zustand stores (`authStore`, `chatStore`, `sidebarStore`, `voiceStore`, `notificationStore`)

## Key Abstractions

**Model:**
- Purpose: Encapsulates LLM provider and capabilities (local vs. cloud, tool support, vision, etc.)
- Examples: `ModelConfig` in `core/brain/models.py`, MODELS registry maps "local:qwen3" to capabilities
- Pattern: Dataclass-based configuration with provider enum, context window, feature flags, cost/speed/quality tiers

**Tool:**
- Purpose: Makes a Python function available to LLMs as an autonomous action
- Examples: `@tool` decorator in `core/tools/definitions.py` wraps 150+ functions
- Pattern: Decorator registers function with name, description, and JSON schema parameters; LLM generates `{"tool_name": "...", "parameters": {...}}` which API executes via `execute_tool()`

**Integration Client:**
- Purpose: Single source of truth for external service API interactions
- Examples: `integrations/stripe/client.py`, `integrations/gmail/client.py`
- Pattern: Python module with authenticated HTTP methods (_get, _post, _patch, _delete), environment config validation, data parsing

**Memory Collection:**
- Purpose: Named vector store for semantic search across a category
- Examples: `memory_personal`, `memory_business`, `memory_conversations`
- Pattern: ChromaDB collection with cosine similarity, optional metadata filtering

**Notification:**
- Purpose: Typed event broadcast to WebSocket clients
- Examples: `AGENT_COMPLETED`, `SYSTEM_ALERT`, `TASK_UPDATE`
- Pattern: Dataclass with type enum, JSON serialization, timestamp

## Entry Points

**Web API Server:**
- Location: `core/api/main.py`
- Triggers: `python -m uvicorn core.api.main:app --host 0.0.0.0 --port 8400`
- Responsibilities: Initialize app (load tools, setup auth DB, warmup voice models), register all routes, middleware setup (CORS, rate limiting), serve embedded HTML if frontend not built

**React Frontend:**
- Location: `frontend/src/main.tsx`
- Triggers: `npm run dev` (Vite dev server on 5173) or `npm run build` (static dist served from API `/`)
- Responsibilities: Load stores, check auth, render login or main app layout, manage WebSocket connection

**Task Queue Worker (Agent Orchestration):**
- Location: `core/orchestration/agents.py`
- Triggers: Initialized at API startup via `initialize_agent_pool()` in lifespan context
- Responsibilities: Spawn 3 worker agents, pull tasks from queue, invoke LLM, store results, emit notifications

**Scheduled Jobs (Learning & Briefing):**
- Location: `core/learning/` and `core/briefing/`
- Triggers: Cron jobs or manual API calls
- Responsibilities: Analyze user interaction patterns, generate daily briefings, recommend next actions

## Error Handling

**Strategy:** Try-catch at integration boundaries; graceful degradation for non-critical failures (memory lookup, knowledge graph); propagate auth/validation errors as HTTP exceptions.

**Patterns:**
- Integration failures (API timeout, auth error) logged and returned as error dict to LLM: `{"error": "..."}`, LLM explains failure to user
- Memory/knowledge lookup exceptions caught, empty context returned, chat continues without context
- Tool execution errors caught by router, formatted as tool_result with error message
- Validation errors (bad parameters, auth failure) raised as `HTTPException` with status code and detail
- Unhandled exceptions at endpoint level logged and return 500, frontend shows generic error message
- Database errors (SQLite lock, ChromaDB connection) retried with exponential backoff

## Cross-Cutting Concerns

**Logging:**
- All modules use Python `logging` with module-scoped loggers
- Base logger "alfred" configured in API startup to INFO level with timestamp formatting
- Tool execution, integration calls, auth events, WebSocket connections logged
- Location: Logs print to stdout (captured by systemd or container logs)

**Validation:**
- Path parameters validated via Pydantic models (ChatRequest, LoginRequest, etc.)
- Integration APIs validate API keys in environment at startup via `config/settings.py`
- Tool parameters validated against JSON schema before execution
- User auth validated on protected routes via `Depends(get_current_user)` or `Depends(require_auth)`

**Authentication:**
- JWT token generation/validation via `jose` library (HS256 algorithm, 480-minute expiry)
- Token stored in HTTP-only secure cookie set on login
- TOTP 2FA optional, stored as per-user secret in `config/users.json`
- Passkey (WebAuthn) optional, credential registration/login via `webauthn` library
- Google OAuth optional, callback redirects to `/auth/google/callback`
- IP-based auto-login for local network via `/auth/auto` endpoint

---

*Architecture analysis: 2026-02-20*
