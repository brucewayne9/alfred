# Codebase Structure

**Analysis Date:** 2026-02-20

## Directory Layout

```
/home/aialfred/alfred/
├── core/                    # Core application logic (LLM, memory, security, tools)
│   ├── api/                 # FastAPI server (~6800 lines, all endpoints + embedded UI)
│   ├── brain/               # LLM routing and model selection
│   ├── tools/               # Tool registry and definitions (150+ LLM-callable functions)
│   ├── memory/              # Semantic memory (ChromaDB) and conversation history (SQLite)
│   ├── security/            # Authentication, JWT, 2FA, passkeys, OAuth
│   ├── notifications/       # WebSocket + VAPID push notifications
│   ├── orchestration/       # Multi-agent task coordination
│   ├── learning/            # User interaction pattern analysis
│   └── briefing/            # Daily briefing generation
├── frontend/                # React + Vite + TypeScript + Zustand + Tailwind
│   ├── src/
│   │   ├── components/      # React components (auth, chat, layout, settings)
│   │   ├── stores/          # Zustand state management
│   │   ├── api/             # HTTP client layer
│   │   ├── hooks/           # Custom React hooks
│   │   └── lib/             # Utilities
│   ├── dist/                # Built production bundle (served from API root)
│   └── public/              # Static assets, manifest, service worker
├── integrations/            # External service adapters (26 modules)
│   ├── gmail/               # Google Workspace email
│   ├── base_crm/            # Twenty CRM (people, companies, opportunities, tasks)
│   ├── calendar/            # Google Calendar
│   ├── stripe/              # Payment processing
│   ├── homeassistant/       # Smart home control
│   ├── lightrag/            # Knowledge graph
│   ├── wordpress/           # WordPress site management
│   ├── n8n/                 # Workflow automation
│   ├── twilio/              # SMS/voice
│   ├── google_docs/         # Document editing
│   ├── google_sheets/       # Spreadsheets
│   ├── google_slides/       # Presentations
│   ├── google_drive/        # File storage
│   ├── google_ads/          # Ad campaign management
│   ├── google_analytics/    # Website analytics
│   ├── meta_ads/            # Meta Ad Manager
│   ├── nextcloud/           # File sync
│   ├── azuracast/           # Radio station
│   ├── firecrawl/           # Web scraping
│   ├── comfyui/             # Image generation
│   ├── email/               # Multi-account email (IMAP/SMTP)
│   ├── finance/             # Financial calculations
│   ├── servers/             # Server management (SSH, Docker)
│   └── __init__.py
├── interfaces/              # Multi-channel access (Telegram, WhatsApp, voice, MCP)
│   ├── telegram/            # Telegram bot handler
│   ├── whatsapp/            # WhatsApp bot (WhatsApp Web)
│   ├── voice/               # STT/TTS (Whisper, Kokoro)
│   ├── mcp/                 # Model Context Protocol
│   └── web/                 # Web chat UI (legacy, superseded by frontend/)
├── config/                  # Configuration and settings
│   ├── settings.py          # Pydantic settings from .env
│   ├── users.json           # User database (passwords, TOTP, passkeys)
│   ├── .env                 # Environment variables (NEVER COMMITTED)
│   └── google_analytics_credentials.json  # GA service account key
├── data/                    # Persistent data storage
│   ├── chromadb/            # Vector store (semantic memory, embeddings)
│   ├── conversations.db     # SQLite message history + projects/references
│   ├── conversations.db-wal # WAL checkpoint for concurrent access
│   ├── audio/               # TTS cache and voice files
│   ├── uploads/             # User file uploads (documents, images)
│   ├── backups/             # Database backups
│   ├── static/              # Static assets
│   ├── personal/            # Personal knowledge domain folder
│   ├── business/            # Business knowledge domain folder
│   ├── financial/           # Financial knowledge domain folder
│   └── generated/           # Generated content (briefings, reports)
├── scripts/                 # Utility scripts
│   └── alfred_claw_monitor.py  # Health monitoring + escalation bridge
├── .planning/               # GSD mapping and planning documents
│   └── codebase/            # Architecture/structure/conventions analysis
├── .claude/                 # Claude Code CLI context
├── .git/                    # Git version control
├── requirements.txt         # Python dependencies
├── run.sh                   # Startup script (builds frontend, starts API)
├── README.md                # Project documentation
├── TOOLS.md                 # Available tools reference (20k char limit)
└── .gitignore               # Git exclusions (.env, node_modules, __pycache__)
```

## Directory Purposes

**core/:**
- Purpose: All brain, logic, persistence, and security code
- Contains: FastAPI server, LLM routing, tools, memory, auth
- Key files: `api/main.py` (6800 lines), `tools/definitions.py` (205k), `brain/router.py`

**core/api/:**
- Purpose: HTTP server and all REST endpoints
- Contains: FastAPI app initialization, 60+ route handlers, request/response models
- Key files: `main.py`

**core/brain/:**
- Purpose: LLM selection and multi-model routing logic
- Contains: TaskType enum, ModelProvider enum, task detection, model selection, context injection
- Key files: `router.py` (ask, classify_query), `models.py` (MODELS registry, ModelConfig)

**core/tools/:**
- Purpose: Manage 150+ LLM-callable tools and execution
- Contains: Tool registry decorator, tool lookup, parameter parsing, execution dispatcher
- Key files: `registry.py` (tool decorator, get_tools), `definitions.py` (all tool wrappers)

**core/memory/:**
- Purpose: Long-term memory and conversation history persistence
- Contains: ChromaDB semantic vector store, SQLite conversation DB, project/reference management
- Key files: `store.py` (ChromaDB wrapper), `conversations.py` (SQLite schema, queries)

**core/security/:**
- Purpose: Authentication, authorization, and cryptographic operations
- Contains: User creation/verification, JWT tokens, TOTP 2FA, WebAuthn passkeys, OAuth
- Key files: `auth.py` (main auth logic), `google_oauth.py` (OAuth flow)

**core/notifications/:**
- Purpose: Real-time event broadcasting to connected clients
- Contains: WebSocket connection management, notification types, VAPID push
- Key files: `manager.py` (NotificationManager class)

**core/orchestration/:**
- Purpose: Multi-agent task coordination and parallel work
- Contains: Agent pool, task queue, result aggregation
- Key files: `agents.py` (initialize_agent_pool, agent workers)

**core/learning/:**
- Purpose: Pattern analysis and personalization
- Contains: Interaction feedback, preference learning, suggestion generation
- Key files: Python modules for feedback storage and analysis

**core/briefing/:**
- Purpose: Automated daily summary generation
- Contains: Briefing template, content aggregation, delivery scheduling
- Key files: Briefing generation logic

**frontend/:**
- Purpose: React web interface
- Contains: React components, Zustand stores, API client, Tailwind styling, PWA manifest
- Key files: `src/main.tsx`, `src/App.tsx`, `src/stores/*`, `public/manifest.webmanifest`

**frontend/src/:**
- Purpose: Source code for React app
- Contains: Organized by feature (components, stores, api, lib, hooks)
- Accessible at: Dev mode on localhost:5173, production from API root `/`

**frontend/dist/:**
- Purpose: Built production bundle
- Generated: `npm run build` in `frontend/`
- Served: From API root `/` when exists, else embedded HTML fallback

**integrations/:**
- Purpose: Adapter layer for external services
- Contains: 26 modules, each with `client.py` and auth/API methods
- Pattern: Each module is independent, exposes high-level functions to tool definitions

**integrations/{service}/:**
- Purpose: Single service integration
- Structure: `client.py` (main client class/functions), `__init__.py` (exports)
- Example: `stripe/client.py` has stripe_create_customer, stripe_list_invoices, etc.

**interfaces/:**
- Purpose: Multi-channel entry points (not the web UI — that's frontend/)
- Contains: Telegram bot handler, WhatsApp bot, voice interface, MCP server
- Key files: `telegram/handler.py`, `whatsapp/`, `voice/stt.py`, `voice/tts.py`

**config/:**
- Purpose: Application configuration and secrets
- Contains: Pydantic settings, user database, external credentials
- Key files: `settings.py` (loads from .env), `users.json`, `.env` (never committed)

**data/:**
- Purpose: All persistent data storage (databases, uploads, cache)
- Contains: ChromaDB vector index, SQLite databases, audio cache, user files
- Important: `conversations.db` is primary record, `chromadb/` contains semantic index

**scripts/:**
- Purpose: Utility and administrative scripts
- Contains: Monitoring, maintenance, deployment helpers
- Key files: `alfred_claw_monitor.py` (health checks, escalation)

## Key File Locations

**Entry Points:**

- `core/api/main.py`: FastAPI server — all HTTP routes and WebSocket handlers
- `frontend/src/main.tsx`: React app entry point
- `core/orchestration/agents.py`: Agent pool initialization on startup
- `interfaces/telegram/handler.py`: Telegram bot entry point
- `run.sh`: Deployment startup script

**Configuration:**

- `config/settings.py`: Pydantic BaseSettings, loads from `config/.env`
- `config/.env`: Environment variables (API keys, database URLs, etc.) — NOT COMMITTED
- `config/users.json`: User database (usernames, password hashes, TOTP, passkeys)
- `frontend/vite.config.ts`: Vite build configuration
- `frontend/tailwind.config.ts`: Tailwind CSS configuration

**Core Logic:**

- `core/api/main.py`: All endpoint implementations, chat logic, streaming
- `core/brain/router.py`: LLM routing decision logic (ask, classify_query)
- `core/tools/registry.py`: Tool registration and execution
- `core/tools/definitions.py`: 150+ tool wrappers (email, CRM, stripe, etc.)
- `core/memory/store.py`: ChromaDB semantic memory operations
- `core/memory/conversations.py`: SQLite conversation history and projects

**Database Schemas:**

- `core/memory/conversations.py` (lines 26-105): CREATE TABLE for conversations, messages, projects, references with indexes and FTS5
- `core/memory/store.py`: ChromaDB collections (memory_personal, memory_business, memory_conversations, etc.)

**Authentication:**

- `core/security/auth.py`: User management, JWT, TOTP, passkey verification
- `core/security/google_oauth.py`: OAuth 2.0 flow for Google Workspace

**UI State:**

- `frontend/src/stores/authStore.ts`: User auth state, login/logout, token management
- `frontend/src/stores/chatStore.ts`: Conversation state, message history, current session
- `frontend/src/stores/sidebarStore.ts`: Sidebar expansion, navigation state
- `frontend/src/stores/voiceStore.ts`: Voice input/output state, STT/TTS settings
- `frontend/src/stores/notificationStore.ts`: Toast notifications, system alerts

**Testing:**

- `tests/` directory not present — no test files detected in current structure
- Testing is manual or via CI integration

## Naming Conventions

**Files:**

- `main.py`: Entry point files or significant modules (core/api/main.py, core/brain/main.py, etc.)
- `client.py`: Integration service clients (stripe/client.py, gmail/client.py)
- `router.py`: LLM routing logic, request dispatching
- `definitions.py`: Declarative definitions (tool definitions, model registry)
- `settings.py`: Configuration management
- `store.py`: Data persistence layer
- Suffixes: `.test.ts` or `.spec.ts` for tests (not present), `.d.ts` for TypeScript types

**Directories:**

- Lowercase with underscores: `core/`, `integrations/`, `smart_home/`, `email/`
- Feature-based grouping: `/integrations/{service}/` (gmail, stripe, etc.)
- Layer-based grouping: `/core/{layer}/` (api, brain, memory, etc.)
- Domain-based grouping: `/frontend/{domain}/` (components, stores, api, hooks)

**Functions:**

- camelCase (Python snake_case): `def create_event()`, `def get_inbox()`
- Verbs: `get_`, `create_`, `update_`, `delete_`, `list_`, `search_`
- Private/internal: `_get()`, `_parse()` (leading underscore)
- Async: `async def ask()`, `async def recall()`

**Classes:**

- PascalCase: `ModelConfig`, `TaskType`, `NotificationManager`, `Notification`
- Enums: `ModelProvider`, `NotificationType`, `TaskType`
- Pydantic models: `ChatRequest`, `LoginRequest`, `ChatResponse`

**Variables:**

- snake_case: `user_id`, `message_count`, `api_key`
- Constants: `UPPERCASE` (MODELS, TOOL_CATEGORIES, RP_ID, BASE_URL)
- Boolean flags: `is_authenticated`, `supports_tools`, `should_escalate`

**TypeScript/React:**

- Component files: PascalCase (LoginOverlay.tsx, AppLayout.tsx)
- Hooks: camelCase with `use` prefix (useAuthStore.ts, custom hooks)
- Store files: camelCase with `Store` suffix (authStore.ts, chatStore.ts)
- Constants: CONSTANT_CASE or camelCase depending on scope

## Where to Add New Code

**New LLM Tool (e.g., send SMS):**

1. Create tool wrapper in `core/tools/definitions.py`:
   ```python
   @tool(
       name="send_sms",
       description="Send an SMS message",
       parameters={"phone": "string", "message": "string"}
   )
   def send_sms(phone: str, message: str) -> dict:
       from integrations.twilio.client import send_sms as _send
       return _send(phone, message)
   ```

2. If integration doesn't exist, create `integrations/{service}/client.py` with implementation

3. Register category in `core/tools/registry.py` TOOL_CATEGORIES if new domain

4. Add test tool calls to verify in `/chat` endpoint

**New External Integration (e.g., Slack):**

1. Create directory: `integrations/slack/`

2. Create `integrations/slack/client.py` with authenticated API methods:
   ```python
   from config.settings import settings

   BASE_URL = "https://slack.com/api"
   API_KEY = settings.slack_api_key

   def _headers() -> dict:
       return {"Authorization": f"Bearer {API_KEY}"}

   def send_message(channel: str, text: str) -> dict:
       # Implementation
   ```

3. Create tool wrappers in `core/tools/definitions.py` using `@tool` decorator

4. Add Slack tools to TOOL_CATEGORIES in `core/tools/registry.py`

5. Update `config/settings.py` to add slack_api_key setting

6. Load API key from `config/.env`

**New React Component:**

1. Create file in `frontend/src/components/{domain}/{ComponentName}.tsx`

2. Use existing component patterns (imports, hooks, Tailwind classes)

3. Connect to state via Zustand stores if needed:
   ```typescript
   import { useAuthStore } from '../../stores/authStore'
   const { user } = useAuthStore()
   ```

4. API calls via `src/api/client.ts` wrapper

5. Styling with Tailwind + Alfred color tokens (#0a0a0a bg, #f97316 accent)

**New API Endpoint:**

1. Add route handler in `core/api/main.py`:
   ```python
   @app.post("/my/endpoint")
   async def my_endpoint(req: MyRequest, user: dict = Depends(require_auth)):
       # Implementation
       return {"result": ...}
   ```

2. Define request/response models as Pydantic BaseModel above routes

3. Use `Depends(require_auth)` for protected routes, `Depends(get_current_user)` for optional

4. Return JSONResponse or dict for JSON, StreamingResponse for streaming

5. Log important operations with logger.info(), errors with logger.error()

**New Zustand Store:**

1. Create file `frontend/src/stores/{featureName}Store.ts`

2. Use Zustand pattern with TypeScript:
   ```typescript
   import { create } from 'zustand'

   interface State {
     value: string
     setValue: (v: string) => void
   }

   export const useMyStore = create<State>((set) => ({
     value: '',
     setValue: (v) => set({ value: v }),
   }))
   ```

3. Export hook as default export, use in components

**New Memory Category:**

1. Add to memory context searches in `core/brain/router.py`:
   ```python
   for category in ["personal", "business", "general", "financial", "new_category"]:
       coll = get_collection(f"memory_{new_category}")
   ```

2. Create ChromaDB collection via `store_memory(..., category="new_category")`

3. Collection auto-created on first write via `get_or_create_collection()`

## Special Directories

**data/chromadb/:**
- Purpose: Vector database for semantic memory
- Generated: Automatically created by ChromaDB on first write
- Committed: No (excluded in .gitignore)
- Structure: UUIDs for collection IDs, internal HNSW index files

**data/conversations.db:**
- Purpose: Primary SQLite database for message history, projects, references
- Generated: Initialized on API startup via `init_db()`
- Committed: No (excluded in .gitignore, backed up separately)
- WAL files: `conversations.db-wal`, `conversations.db-shm` for concurrent access

**data/uploads/:**
- Purpose: User file uploads from `/upload/document` and `/upload/image` endpoints
- Generated: On demand when users upload
- Committed: No (excluded in .gitignore)
- Structure: UUID-named subdirectories for organization

**data/audio/:**
- Purpose: TTS audio cache and pre-generated voice files
- Generated: `audio/tts/` for Kokoro/Qwen3 cache, `audio_cache/` for other models
- Committed: No (excluded in .gitignore)
- Pre-generation: `pregenerate_static_phrases()` in startup

**config/.env:**
- Purpose: Environment variables (API keys, database URLs, secrets)
- Generated: Manually created (template in docs)
- Committed: No (in .gitignore for security)
- Contents: ANTHROPIC_API_KEY, STRIPE_API_KEY, database credentials, etc.

**frontend/dist/:**
- Purpose: Production-built React bundle
- Generated: `npm run build` in frontend/
- Committed: No (excluded in .gitignore)
- Served: From API root `/` via StaticFiles mount
- Chunks: Split vendor, markdown, app for optimal caching

**.planning/codebase/:**
- Purpose: GSD mapping documents (architecture, structure, conventions, testing, concerns)
- Generated: By `/gsd:map-codebase` command
- Committed: Yes (part of repo)
- Contents: ARCHITECTURE.md, STRUCTURE.md, CONVENTIONS.md, TESTING.md, CONCERNS.md

---

*Structure analysis: 2026-02-20*
