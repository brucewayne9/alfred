# External Integrations

**Analysis Date:** 2026-02-20

## APIs & External Services

**LLM/AI Services:**
- Anthropic Claude API - Primary cloud LLM for complex reasoning tasks
  - SDK/Client: `anthropic` package (0.76.0)
  - Auth: `anthropic_api_key` (env var)
  - Model: `claude-sonnet-4-20250514` (configurable in `config/settings.py`)
  - Usage: Escalation target for complex multi-step tasks

- OpenAI ChatGPT - Secondary cloud LLM for tool-calling tasks
  - SDK/Client: `openai` package (2.15.0)
  - Auth: `openai_api_key` (env var)
  - Model: `gpt-4o-mini` (configurable in `config/settings.py`)
  - Usage: Escalation when Ollama attempts to use tools

**Web Scraping & Crawling:**
- Firecrawl API - Web scraping, crawling, and structured data extraction
  - Client: `integrations/firecrawl/client.py`
  - Auth: `FIRECRAWL_API_KEY` (env var)
  - Base URL: `https://api.firecrawl.dev/v1`
  - Features: Single page scrape, site crawl, Google search scraping

**Knowledge Graph:**
- LightRAG - Document memory and knowledge graph queries
  - Client: `integrations/lightrag/client.py`
  - Auth: `LIGHTRAG_URL`, `LIGHTRAG_USER`, `LIGHTRAG_PASS` (env vars)
  - Default URL: `http://75.43.156.117:9621` (Lonewolf 117)
  - Features: Knowledge graph queries, document memory, semantic search
  - Circuit breaker: Auto-disables for 1 hour after 2 consecutive failures

**Communication:**
- Twilio SMS/Voice - SMS and voice communications
  - Client: `integrations/twilio/client.py`
  - Auth: `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_PHONE_NUMBER` (env vars)
  - Features: Send SMS, make voice calls, receive webhooks

**Content Management:**
- WordPress REST API - Multi-site WordPress management
  - Client: `integrations/wordpress/client.py`
  - Auth: Base64 Basic Auth (username/password)
  - Config: `WP_SITE_<name>_URL`, `WP_SITE_<name>_USER`, `WP_SITE_<name>_PASS` (env vars)
  - Features: Posts, pages, media, RankMath SEO, Elementor

**Workflow Automation:**
- n8n - Workflow automation and task scheduling
  - Client: `integrations/n8n/client.py`
  - Auth: `X-N8N-API-KEY` header
  - Config: `n8n_url`, `n8n_api_key` (from `config/settings.py`)
  - API: REST API at `/api/v1/*` endpoints

**Cloud Storage & Sync:**
- Nextcloud - File storage and sync
  - Client: `integrations/nextcloud/client.py`
  - Auth: Basic Auth (username/password)
  - Config: `nextcloud_url`, `nextcloud_username`, `nextcloud_password` (from `config/settings.py`)

**Advertising & Analytics:**
- Meta Ads (Facebook/Instagram) - Campaign insights and advertising
  - Client: `integrations/meta_ads/client.py`
  - Auth: Access token
  - API Version: v21.0
  - Config: `meta_access_token`, `meta_ad_account_id` (from `config/settings.py`)
  - Features: Campaign metrics, ad account insights, spend tracking

- Google Ads - Search and display advertising management
  - Client: `integrations/google_ads/client.py`
  - Auth: OAuth token (saved in `config/google_token.json`)
  - Config: `GOOGLE_ADS_DEVELOPER_TOKEN`, `GOOGLE_ADS_LOGIN_CUSTOMER_ID`, `GOOGLE_ADS_CUSTOMER_ID` (env vars)
  - Features: Campaign management, metrics, keyword insights

- Google Analytics - GA4 analytics and reporting
  - Client: `integrations/google_analytics/client.py`
  - Auth: Service account credentials from `config/google_analytics_credentials.json`
  - Features: Real-time and historical analytics, 14 tracked properties (RuckTalk, Rod Wave, LensSniper, etc.)
  - API: BetaAnalyticsDataClient (google-analytics-data-v1beta)

**Smart Home:**
- Home Assistant - Smart home device control and automation
  - Client: `integrations/homeassistant/client.py`
  - Auth: Bearer token
  - Config: `HA_URL`, `HA_API_TOKEN` (env vars)
  - Default URL: `https://home.groundrushlabs.com`
  - Note: SSL cert is broken on server (TLS issue on 75.43.156.104)

**Radio Broadcasting:**
- AzuraCast Radio - Radio station management and streaming
  - Client: `integrations/azuracast/client.py`
  - Auth: `X-API-Key` header
  - Config: `azuracast_url`, `azuracast_api_key` (from `config/settings.py`)
  - Default: `https://studiob.loovacast.com`
  - Station ID: 22 (News Mews Radio)

**Finance & Payments:**
- Stripe - Payment processing and subscription management
  - Client: `integrations/stripe/client.py`
  - Auth: Bearer token (API key)
  - Config: `stripe_api_key` (from `config/settings.py`)
  - Base URL: `https://api.stripe.com/v1`

## Data Storage

**Databases:**
- PostgreSQL 13+
  - Primary database: `alfred_main`
  - Vault database: `alfred_vault` (for sensitive data)
  - Connection: `postgresql://user:password@localhost:5432/alfred_main`
  - Client: psycopg2-binary (via SQLAlchemy ORM)

- SQLite
  - Conversation history: `/data/conversations.db` (WAL mode, FTS5 search enabled)
  - Schema includes: conversations, messages, projects
  - Features: Foreign key constraints, full-text search

**Vector Database:**
- ChromaDB (Vector store for RAG)
  - Location: `data/chromadb/` (persistent)
  - Configuration: `chromadb.PersistentClient()`
  - Collections: `memory_personal`, `memory_business`, `memory_general`, `memory_financial`
  - Metric: Cosine similarity
  - Used for: Long-term memory recall, semantic search

**File Storage:**
- Local filesystem
  - Uploads: `data/uploads/`
  - Chat data: `data/conversations.db*` files
  - Audio: `data/audio/`
  - Backups: `data/backups/`
  - No cloud storage integration detected

**Caching:**
- Redis 7.1.0
  - Default: `redis://localhost:6379`
  - Purpose: Session caching, real-time data, broadcast channels
  - Features: WebSocket broadcaster (fastapi-websockets), session store

## Authentication & Identity

**Auth Provider:**
- Custom (in-house implementation)
  - JWT tokens with `HS256` algorithm
  - Cookie-based: `alfred_token` httpOnly cookie
  - Token expiry: 480 minutes (8 hours)
  - Implementation: `core/security/auth.py`

**WebAuthn/Passkeys:**
- FIDO2 passkey support
  - RP ID: `aialfred.groundrushcloud.com`
  - RP Name: `Alfred AI Assistant`
  - Passkey storage: User profiles with registered credentials

**Multi-Factor Authentication (2FA):**
- TOTP (Time-based One-Time Password)
  - Library: `pyotp`
  - QR code generation: `qrcode`
  - Optional per-user setting

**OAuth Integration:**
- Google OAuth 2.0
  - Client ID/Secret: From `config/.env`
  - Redirect URI: `https://aialfred.groundrushcloud.com/auth/google/callback`
  - Scopes: Gmail, Sheets, Drive, Calendar, Analytics APIs
  - Implementation: `core/security/google_oauth.py`

**Default User:**
- Username: `bruce`
- Location: `config/users.json`
- Role: Admin (can manage all users)

## Monitoring & Observability

**Error Tracking:**
- Not detected (no Sentry, DataDog, or similar)

**Logs:**
- Loguru logging framework with file and console handlers
- Log output: stdout and `data/*.log` files
- Structured logging via `logging` module

**Notifications:**
- WebSocket-based real-time notifications
  - Manager: `core/notifications/manager.py`
  - Types: Agent completion, system alerts, task updates, long processing
  - Clients: Connected WebSocket subscribers

**Web Push Notifications:**
- VAPID protocol (RFC 8292)
  - Config: `vapid_private_key`, `vapid_public_key`, `vapid_contact_email` (from settings)
  - Client: `pywebpush` library
  - Purpose: Background notifications to PWA clients

## CI/CD & Deployment

**Hosting:**
- Linux server at 75.43.156.105 (Alfred Labs)
- Reverse proxy: Traefik (managed by Dokploy on 75.43.156.117)
- Domain: `aialfred.groundrushcloud.com`

**CI Pipeline:**
- Not detected (no GitHub Actions, GitLab CI, Jenkins found)

**Deployment Process:**
- Manual or script-based (likely via `run.sh` script)
- Frontend: Built with `npm run build`, dist served from `frontend/dist/`
- Backend: FastAPI server started via Uvicorn on port 8400

## Environment Configuration

**Required env vars (critical):**
- `SECRET_KEY` - JWT signing key (auto-generated if missing)
- `ANTHROPIC_API_KEY` - Claude API key
- `OPENAI_API_KEY` - ChatGPT API key
- `db_password` - PostgreSQL password
- `anthropic_model` - Claude model identifier
- `openai_model` - ChatGPT model identifier

**Optional integration env vars:**
- `STRIPE_API_KEY` - Stripe API key
- `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_PHONE_NUMBER` - Twilio SMS/Voice
- `FIRECRAWL_API_KEY` - Firecrawl web scraping
- `LIGHTRAG_URL`, `LIGHTRAG_USER`, `LIGHTRAG_PASS` - LightRAG knowledge graph
- `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` - Google OAuth
- `META_ACCESS_TOKEN`, `META_AD_ACCOUNT_ID` - Meta Ads
- `GOOGLE_ADS_*` - Google Ads API credentials
- `HA_URL`, `HA_API_TOKEN` - Home Assistant
- `AZURACAST_URL`, `AZURACAST_API_KEY` - AzuraCast
- `N8N_URL`, `N8N_API_KEY` - n8n automation
- `NEXTCLOUD_URL`, `NEXTCLOUD_USERNAME`, `NEXTCLOUD_PASSWORD` - Nextcloud
- `WP_SITES`, `WP_SITE_*_*` - WordPress multi-site config
- `VAPID_PRIVATE_KEY`, `VAPID_PUBLIC_KEY`, `VAPID_CONTACT_EMAIL` - Web Push

**Secrets location:**
- Environment variables in `config/.env` (not committed, contains secrets)
- Service account credentials: `config/google_analytics_credentials.json`
- OAuth tokens: `config/google_token.json`
- Users: `config/users.json`

## Webhooks & Callbacks

**Incoming:**
- Twilio webhooks - SMS/voice message callbacks
  - Endpoint: `/webhooks/twilio` (inferred, not explicitly found)
  - Validation: RequestValidator (Twilio signature verification)

- Google OAuth callback
  - Endpoint: `/auth/google/callback`
  - Handles code exchange and token storage

**Outgoing:**
- Home Assistant webhook triggers (commands sent to HA)
- n8n workflow triggers (via API calls)
- Stripe event subscriptions (via webhook configuration)
- WordPress REST API calls (for post updates, media management)

---

*Integration audit: 2026-02-20*
