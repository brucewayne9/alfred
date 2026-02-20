# Codebase Concerns

**Analysis Date:** 2026-02-20

## Tech Debt

**Monolithic API file:**
- Issue: `core/api/main.py` is 6910 lines containing all endpoints, lifespan, auth, webhooks, and integrations checks
- Files: `core/api/main.py`
- Impact: Difficult to navigate, test, and maintain. Every change risks affecting unrelated endpoints. Module load time increases.
- Fix approach: Extract endpoint groups into separate routers (`core/api/routes/auth.py`, `core/api/routes/chat.py`, `core/api/routes/integrations.py`, etc.) and mount via `app.include_router()`. Consolidate lifespan and middleware setup.

**Monolithic tool definitions:**
- Issue: `core/tools/definitions.py` is 5573 lines with 200+ tool definitions in a single file
- Files: `core/tools/definitions.py`
- Impact: Slow to load, hard to find specific tools, difficult to organize by domain. Every tool import happens at startup.
- Fix approach: Split by domain (email, crm, calendar, memory, server, etc.) into `core/tools/email.py`, `core/tools/crm.py`, etc. Use dynamic registration pattern with lazy imports in `register_all()`.

**Bare exception handlers with silent failures:**
- Issue: Multiple `except Exception: pass` blocks in critical paths
- Files: `core/api/main.py:111, 625, 633, 644, 652, 660, 673, 684`, and in startup/shutdown
- Impact: Failures in integrations status check, voice warmup, agent pool initialization silently suppressed. Difficult to diagnose issues.
- Fix approach: Replace `pass` with specific logging: `logger.error(f"Integration {name} check failed: {e}", exc_info=True)`. Log stack traces for debugging.

**Unvalidated integration status checks:**
- Issue: `/integrations/status` endpoint catches all exceptions and returns false without specificity
- Files: `core/api/main.py:614-753`
- Impact: Unclear why integrations fail (network vs. credentials vs. misconfiguration). Users see "not connected" without remediation path.
- Fix approach: Catch specific exceptions (ConnectionError, HTTPError, TimeoutError) and return detailed status: `{"connected": false, "reason": "timeout", "last_check": "..."}`.

**Global mutable state in integrations:**
- Issue: `integrations/lightrag/client.py` uses module-level `_circuit_breaker` and `_token_cache` dictionaries as global state
- Files: `integrations/lightrag/client.py:22-26`
- Impact: Not thread-safe. Multiple concurrent requests can race on circuit breaker state. Token refresh can be triggered simultaneously.
- Fix approach: Use asyncio.Lock for state mutations. Consider moving to a StatefulClient class or using contextvars.

**Database connection threading issue:**
- Issue: `core/memory/conversations.py:19` uses `check_same_thread=False` to allow multi-threaded access to SQLite
- Files: `core/memory/conversations.py`
- Impact: WAL mode helps, but not officially supported for concurrent writes. Risk of database corruption under high concurrency.
- Fix approach: Replace with connection pooling (e.g., sqlalchemy with thread pool) or migrate to PostgreSQL for true concurrent access.

**Tool execution lacks error context:**
- Issue: `core/tools/registry.py:371-373` catches all exceptions and returns `{"error": str(e)}` without tool name in error dict
- Files: `core/tools/registry.py`
- Impact: LLM receives minimal context about which tool failed. Cannot distinguish between parameter errors vs. service unavailability vs. auth failures.
- Fix approach: Return `{"error": str(e), "tool": name, "error_type": type(e).__name__, "recoverable": <bool>}` so LLM can decide to retry.

## Known Bugs

**Home Assistant SSL/TLS misconfiguration:**
- Symptoms: Home Assistant integration fails silently on HTTPS connections
- Files: `integrations/homeassistant/client.py:33, 44`
- Trigger: Connecting to `HA_URL="https://home.groundrushlabs.com"` with broken SSL cert (server-side TLS issue on 75.43.156.104)
- Workaround: Currently requests library will fail silently in try/except. No SSL verification bypass in place, but also no explicit error reporting.

**LightRAG circuit breaker timeout calculation:**
- Symptoms: After 2 failures, all LightRAG requests blocked for 1 hour
- Files: `integrations/lightrag/client.py:27-28, 34-35`
- Trigger: Network outage or LightRAG server down. Circuit remains open even after server recovers.
- Workaround: Manual server restart or 1-hour wait. No admin endpoint to reset circuit breaker.

**Tool parameter validation missing:**
- Symptoms: LLM can pass invalid argument types or missing required params to tools
- Files: `core/tools/registry.py:366`, `core/tools/definitions.py` (all tools)
- Trigger: LLM generates `{"tool": "send_email", "args": {"to": null}}` - no validation before func(**args) call
- Workaround: Each tool individually validates inputs (inconsistently). Some tools crash, others return error dicts.

## Security Considerations

**No rate limiting on auth endpoints:**
- Risk: Brute force attack on `/auth/login`, `/auth/passkey/login/begin` endpoints
- Files: `core/api/main.py:332-334, 536-538` (marked with @app decorators but no limiter applied)
- Current mitigation: Server-side rate limiting via slowapi is configured but not applied to auth routes explicitly
- Recommendations: Add `@limiter.limit("5/minute")` to login endpoints. Implement account lockout after 5 failed attempts.

**JWT stored in cookies with explicit http-only:**
- Risk: Moderate. Mitigated but could be stronger.
- Files: `core/security/auth.py`, responses set `secure=True, httponly=True, samesite="strict"`
- Current mitigation: Flags set correctly in auth flow
- Recommendations: Add CSRF token validation for state-changing operations. Implement token rotation on each request.

**Plaintext password in error messages:**
- Risk: Low but present
- Files: `core/security/auth.py` and database store use bcrypt hashing, passwords never logged
- Current mitigation: Good
- Recommendations: Ensure all integration credentials (API keys, tokens) are never logged. Audit integrations/*/client.py for logger calls with sensitive data.

**Google OAuth flow missing code validation:**
- Risk: Moderate. Code could be reused.
- Files: `core/api/main.py:594-612`
- Current mitigation: Google's servers validate
- Recommendations: Store used auth codes in a set with TTL and reject replays. Add state parameter validation.

**CORS allows localhost:5173 for dev:**
- Risk: Low in development. Removes same-origin protection.
- Files: `core/api/main.py:121-127` hardcodes `allow_origins=["https://aialfred.groundrushcloud.com", "http://localhost:5173"]`
- Current mitigation: Production deployment uses strict HTTPS origin
- Recommendations: Load CORS origins from environment config. Never hardcode localhost in production builds.

## Performance Bottlenecks

**Integration status endpoint checks 15+ services serially:**
- Problem: `/integrations/status` makes sequential requests to CRM, n8n, Nextcloud, Stripe, LightRAG, Twilio, etc. If any timeout, entire endpoint hangs.
- Files: `core/api/main.py:614-753`
- Cause: No parallelization. Each try/except block is sequential.
- Improvement path: Use `asyncio.gather(*[check_crm(), check_n8n(), check_stripe(), ...], return_exceptions=True)` to check all integrations concurrently with configurable timeouts.

**ChromaDB embedding collection grows unbounded:**
- Problem: `data/chromadb/chroma.sqlite3` is 13MB and grows with every conversation/memory stored
- Files: `core/memory/store.py`, `core/memory/conversations.py`
- Cause: No cleanup policy. Old conversations stored indefinitely. Embeddings never pruned.
- Improvement path: Implement conversation archival (move old convos to separate read-only collection after 90 days). Add `store_conversation()` cleanup job that compacts embeddings.

**Tool registration at startup scans all 200+ tool definitions:**
- Problem: `register_all()` called in `core/api/main.py:58` imports entire definitions module, parsing all functions even if not used
- Files: `core/tools/definitions.py`, `core/api/main.py:56-58`
- Cause: Single import() of full module. No lazy loading.
- Improvement path: Use dynamic import in `execute_tool()` based on tool name. Load only requested tools.

**SQLite WAL file accumulation:**
- Problem: `data/conversations.db-wal` is 4.1MB (should checkpoint regularly)
- Files: `core/memory/conversations.py:21` sets WAL mode but doesn't configure checkpointing
- Cause: WAL mode enabled but `PRAGMA wal_autocheckpoint` not set. Checkpoints only on connection close.
- Improvement path: Add `_conn.execute("PRAGMA wal_autocheckpoint=1000")` to flush WAL after 1000 pages.

## Fragile Areas

**Wordpress Elementor JSON generation:**
- Files: `integrations/wordpress/elementor.py` (3417 lines)
- Why fragile: Complex nested JSON structure for Elementor widgets. Random ID generation with `_generate_id()` is low entropy (7 hex chars = 268M possibilities, high collision risk). One malformed widget breaks entire page.
- Safe modification: Add UUID-based ID generation instead of random hex. Add JSON schema validation before posting to WordPress. Add unit tests for each widget type with roundtrip validation.
- Test coverage: No tests found for elementor.py. No validation of generated JSON against Elementor's schema.

**Tool call parsing with regex:**
- Files: `core/tools/registry.py:376-410`
- Why fragile: Regex patterns for JSON extraction assume specific formatting. LLM can generate JSON blocks in unexpected ways (extra spaces, multiline, escaped quotes).
- Safe modification: Use JSON decoder with `allow_nan=False` and strict parsing. Consider requiring LLM to use structured output format if available.
- Test coverage: No tests for malformed JSON, nested JSON in responses, or edge cases.

**WebSocket connection lifecycle in chat:**
- Files: `core/api/main.py` (WebSocket handler exists but not shown in excerpt, likely around line 1800+)
- Why fragile: WebSocket can disconnect mid-message. No reconnection backoff. If network drops, user loses context.
- Safe modification: Implement connection heartbeat. Buffer messages during disconnect. Provide client-side reconnection logic.
- Test coverage: No WebSocket tests visible.

**Integration credential loading at module level:**
- Files: All `integrations/*/client.py` load `os.getenv()` at import time
- Why fragile: Credentials loaded once at startup. Credential rotation or updates require restart. Missing credentials crash on import.
- Safe modification: Load credentials on first use via lazy initialization. Implement credential reload endpoint for zero-downtime updates.
- Test coverage: No tests for missing credentials gracefully.

## Scaling Limits

**Single SQLite database for conversations:**
- Current capacity: 1.3MB (conversations.db), 116KB (learning.db). Based on growth rate, will exceed 100MB within 6 months.
- Limit: SQLite performs poorly with files >1GB. Writes become slow around 500MB.
- Scaling path: Migrate to PostgreSQL. Implement conversation archival (monthly export to S3, delete local entries >1 year old).

**LightRAG circuit breaker one-hour cooldown:**
- Current capacity: Blocks all LightRAG requests for 1 hour after 2 failures
- Limit: Entire knowledge graph unavailable during cooldown
- Scaling path: Implement exponential backoff (10s, 30s, 2min, 10min) instead of fixed 1 hour. Add manual reset endpoint.

**Tool registry scales linearly with tool count:**
- Current capacity: 200+ tools registered, each with description and parameters
- Limit: No hard limit but adding 300+ tools will increase startup time and LLM context size
- Scaling path: Implement category-based tool filtering. Send only relevant tools to LLM based on query intent. Use smart tool selection instead of sending all 200+.

## Dependencies at Risk

**Chromadb dependency fragile:**
- Risk: Large library (13MB database), complex embedding dependencies, periodic breaking API changes
- Files: `core/memory/store.py` uses chromadb
- Impact: Embedding model updates may change output, breaking existing collections. Library major version changes require migration.
- Migration plan: Add abstraction layer (`core/memory/embeddings.py`) so embedding provider can be swapped. Keep last 2 major chromadb versions supported.

**Ollama local model availability:**
- Risk: Model `settings.ollama_model` (default: phi2) may not be installed or may download on first request
- Files: `core/brain/router.py` calls ollama via HTTP
- Impact: Chat requests fail if Ollama is offline or model not present. No fallback.
- Migration plan: Pre-check model availability in `core/api/main.py` lifespan. Warn if model missing. Fallback to Anthropic cloud model if local unavailable.

**PassKey (WebAuthn) library stability:**
- Risk: `webauthn` library may not be actively maintained
- Files: `core/security/auth.py` uses `webauthn` for passkey registration/verification
- Impact: Security issues in passkey validation could compromise auth
- Migration plan: Audit `webauthn` library security. Plan migration to `py_webauthn` (actively maintained) or built-in browser APIs with fallback.

## Missing Critical Features

**No backup/recovery strategy:**
- Problem: Conversations database, ChromaDB embeddings, and learned knowledge have no automated backup
- Files: `data/conversations.db`, `data/chromadb/chroma.sqlite3`, `data/learning.db`
- Blocks: Cannot recover from data corruption or accidental deletion. No point-in-time recovery.
- Recommendation: Implement daily S3 backup of data/ directory with retention policy. Add restoration endpoint for admins.

**No conversation pagination:**
- Problem: List conversations endpoint may load all conversations at once
- Files: `core/memory/conversations.py` likely missing LIMIT/OFFSET
- Blocks: UI will freeze loading 1000+ conversations
- Recommendation: Add `skip` and `limit` parameters to list endpoints. Implement cursor-based pagination.

**No tool audit trail:**
- Problem: No logging of which tools were called, with what parameters, and what results
- Files: `core/tools/registry.py:363` logs tool name but not args/results
- Blocks: Cannot audit for misuse, debug user issues, or analyze tool usage patterns
- Recommendation: Add `tool_audit` table to conversations.db. Log all tool invocations with args, result summary, and execution time.

**No integration health monitoring:**
- Problem: Integrations only checked on-demand when user visits status page
- Files: `core/api/main.py:614-753`
- Blocks: Silent failures go unnoticed until user tries to use feature. No alerting.
- Recommendation: Implement background health check job that tests each integration every 5 minutes. Alert admin if status changes.

## Test Coverage Gaps

**No tests for error paths:**
- What's not tested: Timeout handling, credential failures, malformed responses, rate limit handling
- Files: `core/tools/definitions.py` (200+ tools with no test coverage), `core/api/main.py` (webhook endpoints untested)
- Risk: Tool failures, auth bypasses, and webhook injection not caught
- Priority: High

**No integration tests:**
- What's not tested: End-to-end flows like "send email then create CRM task". Cross-integration interactions.
- Files: All integration client files lack roundtrip tests
- Risk: Integration incompatibilities surface in production
- Priority: High

**No database schema migration tests:**
- What's not tested: ALTER TABLE statements, PRAGMA changes, FTS5 trigger behavior under concurrent load
- Files: `core/memory/conversations.py:26-101` has no migration test suite
- Risk: Schema changes fail silently on some databases, corrupt data
- Priority: Medium

**No WebSocket reconnection tests:**
- What's not tested: Client disconnect/reconnect, message buffering, state consistency
- Files: WebSocket handler in `core/api/main.py` (likely untested)
- Risk: Lost messages, duplicate messages, state desynchronization
- Priority: Medium

**No LLM routing tests:**
- What's not tested: ModelTier selection logic, fallback behavior, token counting accuracy
- Files: `core/brain/router.py` (1232 lines with no visible tests)
- Risk: LLM selection misbehavior, unexpected cloud charges, poor performance
- Priority: High

---

*Concerns audit: 2026-02-20*
