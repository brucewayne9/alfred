# Coding Conventions

**Analysis Date:** 2026-02-20

## Naming Patterns

**Files:**
- Python modules: `snake_case.py` (e.g., `auth.py`, `chat_store.py`)
- TypeScript components: `PascalCase.tsx` for React components (e.g., `ChatInput.tsx`, `MessageBubble.tsx`)
- TypeScript utilities/hooks: `camelCase.ts` (e.g., `chatStore.ts`, `useChat.ts`)
- Integration clients follow pattern: `integration_name/client.py` (e.g., `stripe/client.py`, `gmail/client.py`)

**Functions:**
- Python: `snake_case` (e.g., `get_balance()`, `create_user()`, `store_memory()`)
- TypeScript: `camelCase` for functions (e.g., `handleSend()`, `stripToolJson()`, `apiFetch()`)
- React hooks: `useCamelCase` (e.g., `useChat()`, `useVoice()`)
- Private functions prefixed with underscore: `_get_token()`, `_record_failure()`, `_ensure_secret_key()`

**Variables:**
- Python: `snake_case` (e.g., `message_counter`, `current_file`, `api_key`)
- TypeScript: `camelCase` (e.g., `messageCounter`, `currentFile`, `apiKey`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `API_KEY`, `BASE_URL`, `CIRCUIT_BREAKER_THRESHOLD`)
- React state variables: `camelCase` (e.g., `isThinking`, `currentConversationId`)

**Types:**
- TypeScript interfaces: `PascalCase` (e.g., `ChatMessage`, `UploadedFile`, `ChatResponse`)
- Python type hints: Lowercase with union via `|` (e.g., `dict | None`, `str | int`)
- Enum classes: `PascalCase` (e.g., `ModelTier`, `TaskType`, `ModelProvider`)

## Code Style

**Formatting:**
- Python: 4-space indentation (implicit - following PEP 8)
- TypeScript: 2-space indentation (observed in frontend code)
- Line length: No explicit enforcer detected, but generally under 100 characters for readability
- Module docstrings: All Python modules start with triple-quote docstrings describing purpose

**Linting:**
- No explicit linter configuration found (.eslintrc/.eslintignore not in main project root)
- No .prettierrc file detected in root; using Tailwind/PostCSS formatting for CSS
- TypeScript strict mode enabled in `tsconfig.json` with `"strict": true`
- Type checking: `noFallthroughCasesInSwitch: true` enforced

## Import Organization

**Order:**
1. Standard library imports (`import sys`, `import os`, `from datetime import datetime`)
2. Third-party imports (`import requests`, `from pydantic import BaseModel`)
3. Local imports (`from config.settings import settings`, `from core.brain.router import ask`)

**Path Aliases:**
- TypeScript: `@/*` resolves to `src/*` (defined in `tsconfig.json`)
- Python: Root project added to path via `sys.path.insert(0, str(Path(__file__).parent.parent.parent))`
- Relative imports used within modules (e.g., `from ../../api/client` in React)

**Examples:**
- Python: `from core.security.auth import create_user, verify_user` (absolute imports)
- TypeScript: `import { useChatStore } from '../../stores/chatStore'` (relative) or `import { apiFetch } from '@/api/client'` (alias)

## Error Handling

**Patterns:**

**Python:**
- Try-except with specific exception types when possible:
  ```python
  try:
      coll = get_collection(f"memory_{category}")
      results = coll.query(...)
  except Exception as e:
      logger.debug(f"Memory search in {category} failed: {e}")
      continue
  ```
- Generic `Exception` catch with logging for integration calls
- HTTPException from FastAPI for API errors: `raise HTTPException(status_code=401, detail="Not authenticated")`
- Circuit breaker pattern for external services (LightRAG): track failures and cooldown periods
- `requests.raise_for_status()` for HTTP response validation in integrations

**TypeScript:**
- Try-catch with generic fallbacks:
  ```typescript
  try {
    await chatApi.stream(...)
  } catch {
    // Fallback to non-streaming
    const res = await chatApi.send(...)
  }
  ```
- Custom error class: `ApiError extends Error` with status code property
- Catch-all with type-safe checks: `catch (e: unknown)` with `e instanceof Error` guards
- Silent error swallowing for non-critical operations: `catch { /* ignore TTS errors */ }`

## Logging

**Framework:** Python uses `logging` module, TypeScript/React uses `console` (if needed)

**Patterns:**

**Python:**
- Logger initialization: `logger = logging.getLogger(__name__)` at module level
- Log levels used:
  - `logger.info()`: Startup messages, successful operations
  - `logger.debug()`: Detailed diagnostic info, non-critical failures
  - `logger.warning()`: Degraded functionality, circuit breaker state changes
  - `logger.error()`: Significant failures (rare in observed code)
- Formatted messages include context: `logger.warning(f"LightRAG circuit breaker OPEN after {cb['failures']} failures")`

**TypeScript:**
- No explicit logging framework in frontend; error logging is inline with try-catch
- API errors propagated to store/state for UI display

## Comments

**When to Comment:**
- Always include module-level docstrings explaining purpose and scope
- Explain WHY, not WHAT: "Circuit breaker prevents chatting on unavailable service" vs. "Set failures += 1"
- Complex logic that isn't self-evident (rare in observed code)
- Integration-specific quirks: `# Token typically expires in 24h, refresh at 23h`
- State machine comments: `# Open circuit after 2 consecutive failures`

**JSDoc/TSDoc:**
- Python docstrings for functions follow pattern:
  ```python
  def get_balance() -> dict:
      """Get account balance."""
  ```
- Single-line docstrings for simple functions
- Multi-line for complex functions with Args/Returns sections (observed in `lightrag/client.py`)
- React components have brief inline comments for complex JSX sections: `{/* Tier badge */}`

## Function Design

**Size:** Functions kept short, typically 20-50 lines for integration calls. Longer functions decomposed into helpers.

**Parameters:**
- Required parameters first, then optional with defaults
- Use type hints consistently: `def create_user(username: str, password: str, role: str = "admin") -> bool:`
- Keyword arguments preferred for multiple optional params
- Default values for common operations: `max_results: int = 5`, `timeout: float = 5`

**Return Values:**
- Type hints always present: `-> dict`, `-> list[dict]`, `-> bool`, `-> str | None`
- Return `None` implicitly for void operations (logging, state updates)
- Return dict for data responses, matching API response structures
- Union types used: `dict | None` for optional returns

## Module Design

**Exports:**
- Python: Functions decorated with `@tool` decorator are registered; others accessed via direct import
- TypeScript: Explicit `export` on components, hooks, store creators
- Barrel files: Not observed in this codebase
- Integration clients export main client class or standalone functions

**Barrel Files:**
- Not used; imports are direct from module files (e.g., `from core.security.auth import create_user`)
- Integration modules have `__init__.py` but are typically empty

---

*Convention analysis: 2026-02-20*
