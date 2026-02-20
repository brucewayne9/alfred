# Testing Patterns

**Analysis Date:** 2026-02-20

## Test Framework

**Runner:**
- No test runner detected in main codebase (no pytest.ini, conftest.py, or test scripts)
- Frontend: No test framework configured (no Jest, Vitest, or Playwright configs)
- Testing infrastructure: NOT IMPLEMENTED

**Assertion Library:**
- N/A - no testing framework in place

**Run Commands:**
- No test execution commands defined
- `npm run build` and `npm run dev` for frontend (no test variants)
- No Python test scripts in `scripts/` directory

## Test File Organization

**Location:**
- NOT OBSERVED - No test files in main codebase
- Tests exist in external dependencies (`node_modules/`, `venv/lib/`) but not in project code
- No co-located tests (e.g., `chat.test.ts` alongside `chat.ts`)
- No separate `tests/` or `__tests__/` directories

**Naming:**
- N/A - no tests to name

**Structure:**
- N/A - no tests to structure

## Test Structure

**Suite Organization:**
- NOT IMPLEMENTED - No test suites observed

**Patterns:**
- N/A - no testing patterns present
- Setup/teardown: N/A
- Assertion patterns: N/A

## Mocking

**Framework:**
- NOT IMPLEMENTED - No mocking library detected

**Patterns:**
- Code uses integration clients directly (e.g., `from integrations.gmail.client import get_inbox`)
- No mock implementations or test doubles observed
- Tool registry and execution tested only by live calls to integrations

**What to Mock (if testing is added):**
- External integrations (Gmail, Stripe, Home Assistant, etc.)
- LLM API calls (Ollama, Anthropic Claude)
- ChromaDB/LightRAG calls
- HTTP requests to external services
- WebSocket connections in notifications

**What NOT to Mock (if testing is added):**
- Core logic: memory storage, conversation history, token validation
- Local data structures and utilities
- Authentication flows (can use test users in config/users.json)
- State management (Zustand stores are synchronous and testable)

## Fixtures and Factories

**Test Data:**
- NOT OBSERVED - No fixture files or factories

**Location:**
- N/A - no test infrastructure

## Coverage

**Requirements:**
- No coverage targets enforced
- No CI/CD pipeline for testing detected
- No coverage reports generated

**View Coverage:**
```bash
# NOT CONFIGURED - No test execution path
```

## Test Types

**Unit Tests:**
- NOT IMPLEMENTED
- Recommended scope would be:
  - Tool registry and execution (`core/tools/registry.py`)
  - Authentication functions (`core/security/auth.py`)
  - Model selection logic (`core/brain/models.py`)
  - Integration client methods (e.g., `integrations/stripe/client.py`)

**Integration Tests:**
- NOT IMPLEMENTED
- Recommended scope would be:
  - Tool calling flow (from LLM → registry → execution)
  - Memory store operations (ChromaDB interactions)
  - LLM routing decisions (`core/brain/router.py`)
  - API endpoints (FastAPI handlers in `core/api/main.py`)

**E2E Tests:**
- NOT IMPLEMENTED
- Could test:
  - Chat message flow (frontend → API → LLM → response)
  - Multi-turn conversations
  - Tool execution in live chat
  - Authentication flows (login, 2FA, passkeys)

## Testing Gaps & Risks

**Critical Gaps:**

1. **No Integration Tests:**
   - Tool execution untested: Changes to tool registry or parameter handling can break live integrations
   - LLM routing untested: ModelTier decisions and fallback logic not validated
   - API endpoints untested: No verification that endpoints match frontend expectations

2. **No Unit Tests:**
   - Security functions (`auth.py`) without tests: Password hashing, JWT creation, token validation are critical
   - Memory store operations (`store.py`) untested: ChromaDB queries and storage could silently fail
   - Tool definitions (`definitions.py`) unmaintained: Parameter validation not checked

3. **No Frontend Tests:**
   - State management (`chatStore.ts`) untested: Message ordering, conversation loading could be broken
   - API client error handling (`client.ts`) unmaintested: Network failures may not be handled correctly
   - React components untested: UI logic for streaming, file uploads, markdown rendering not validated

4. **No E2E Tests:**
   - Chat flow untested: No verification that frontend ↔ backend ↔ LLM pipeline works end-to-end
   - Authentication flows untested: Login, passkey registration, 2FA could break
   - Tool calling untested: No validation of LLM → tool → result flow in realistic scenario

**Recommended Priority:**
1. Add pytest for Python integration tests (tool execution, API endpoints)
2. Add Vitest for TypeScript unit tests (stores, client functions)
3. Add E2E tests for critical flows (chat, auth, tools)

## Common Patterns to Test (if testing is added)

**Async Testing:**

Python (pytest-asyncio):
```python
@pytest.mark.asyncio
async def test_get_knowledge_context():
    result = await get_knowledge_context("test query")
    assert isinstance(result, str)
```

TypeScript (Vitest):
```typescript
it('should handle streaming', async () => {
  const chunks: string[] = []
  await apiStream('/chat', {}, chunk => chunks.push(chunk))
  expect(chunks.length).toBeGreaterThan(0)
})
```

**Error Testing:**

Python:
```python
def test_circuit_breaker_opens_on_threshold():
    for _ in range(CIRCUIT_BREAKER_THRESHOLD):
        _record_failure()
    assert _circuit_is_open() == True
```

TypeScript:
```typescript
it('should handle API errors gracefully', async () => {
  const error = new ApiError(401, 'Unauthorized')
  expect(error.status).toBe(401)
  expect(error.message).toBe('Unauthorized')
})
```

**Mocking External Services:**

Python (unittest.mock):
```python
@patch('integrations.gmail.client.get_inbox')
def test_check_email(mock_get_inbox):
    mock_get_inbox.return_value = [{'id': '1', 'subject': 'Test'}]
    result = check_email()
    assert len(result) == 1
```

TypeScript (Vitest):
```typescript
vi.mock('../api/chat', () => ({
  chatApi: { stream: vi.fn(), send: vi.fn() }
}))
```

---

*Testing analysis: 2026-02-20*

## Summary

**Current State:** Testing framework and test coverage do NOT exist in this codebase.

**Risk:** Live integrations, critical auth functions, and multi-turn chat flows are untested. Changes risk silent failures in production.

**Recommendation:** Establish testing infrastructure before adding new features. Start with:
1. Unit tests for `core/security/auth.py` and `core/tools/registry.py`
2. Integration tests for tool execution and API endpoints
3. E2E tests for chat flow and authentication
