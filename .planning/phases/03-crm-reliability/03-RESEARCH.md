# Phase 3: CRM Reliability - Research

**Researched:** 2026-02-21
**Domain:** Twenty CRM REST/GraphQL API, Python CLI integration on Alfred Claw (101)
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Contact search uses partial match (contains) — searching "Sarah" finds "Sarah Johnson", "Sarah Smith", etc.
- When multiple contacts match, Alfred asks user to pick from a numbered list (no auto-select)
- If zero results, Alfred offers to create a new contact rather than reporting "not found"

### Claude's Discretion
- Rollback strategy: how many retries before cleanup, immediate vs delayed rollback
- Failure messaging: detail level, tone, and format of CRM error messages to Mike via Telegram
- Error visibility: whether rollback actions are mentioned in failure messages
- Search field implementation: first+last separate vs full name combined — whatever Twenty CRM API supports best

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CRM-01 | Note/task linking uses rollback on second-step failure (no orphaned records) | Twenty CRM's REST API requires two separate HTTP calls to link a note/task to a contact. The API explicitly rejects inline target creation (HTTP 400). Rollback = DELETE the first-step record if the second step fails. Both DELETE endpoints confirmed working. |
| CRM-02 | Contact search uses server-side filter instead of 100-record Python fuzzy match | `search_people()` already uses GraphQL server-side filtering — but is capped at `first: 50`, which misses results when total matches exceed 50. Increasing to `first: 500` resolves the issue. No new library needed. |
</phase_requirements>

---

## Summary

Phase 3 targets two bugs in `/home/brucewayne9/.openclaw/workspace/scripts/integrations/crm.py` on Alfred Claw (101). Both bugs were verified through live API testing against `https://crm.groundrushlabs.com`.

**CRM-01 (Orphaned records):** Twenty CRM's REST API cannot link a note/task to a contact in a single request — the API returns HTTP 400 with "One-to-many relation field does not support write operations" when you include `noteTargets` or `taskTargets` inline. Linking always requires two steps: (1) create the record, (2) create the target. Currently, `crm.py` has no CLI commands that perform both steps. The `create_note()` function is also broken (uses a `body` field that the API no longer accepts). The fix is to add `create_linked_note()` and `create_linked_task()` functions that perform the two-step flow with rollback: if step 2 fails, step 1 is deleted immediately (no retry). Both the note DELETE and task DELETE endpoints are confirmed working.

**CRM-02 (Truncated search results):** The `search_people()` function already uses server-side GraphQL filtering with `LIKE` pattern matching — it is NOT doing a 100-record Python fuzzy match. However, it fetches at most `first: 50` results from the GQL query. For broad searches ("son" matches 88 total contacts, only 50 are returned), the correct contact may be in the unseen 38. Increasing `first` to 500 resolves this with no other changes needed.

**Primary recommendation:** Extend `crm.py` with atomic two-step linked creation functions and increase the GQL `first` cap from 50 to 500. No new libraries or services needed — all fixes are pure Python edits to the existing script and TOOLS.md.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `urllib.request` | stdlib | HTTP calls to Twenty CRM REST and GraphQL | Already used throughout crm.py; no deps to add |
| `json` | stdlib | Serialize/deserialize API payloads | Already used throughout crm.py |

### Supporting
None needed — this is a pure Python script edit, no new packages.

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `urllib.request` | `requests` library | `requests` is simpler but requires installation. `urllib` is already in use — consistency wins. |
| LIKE filter | `searchVector.search` (TSVector full-text) | searchVector IS available (`__type` introspection confirmed `TSVectorFilter` with `search` field). But it searches ALL indexed text (names AND email addresses), returning 23 "Johnson" matches vs 6 from name-only LIKE. For the use case (find contacts BY NAME), LIKE on name fields is more precise. searchVector is useful as a fallback when LIKE returns 0 results. |

---

## Architecture Patterns

### Recommended Project Structure
No structural changes. All edits go to a single file:
```
~/.openclaw/workspace/scripts/integrations/
└── crm.py          # Add create_linked_note(), create_linked_task(); fix search cap
~/.openclaw/workspace/
└── TOOLS.md        # Add create-linked-note, create-linked-task command docs
```

### Pattern 1: Two-Step Atomic Linked Creation (with Rollback)

**What:** Create a note or task (step 1), then create the `noteTarget`/`taskTarget` linking record (step 2). If step 2 fails, immediately delete the step-1 record.

**When to use:** Any time an agent wants to add a note or task to a specific contact.

**Verified API behavior:**
- `POST /rest/notes` with `{"title": "...", "bodyV2": {"markdown": "..."}}` returns `{"data": {"createNote": {"id": "<uuid>", ...}}}`
- `POST /rest/noteTargets` with `{"noteId": "<uuid>", "personId": "<uuid>"}` returns `{"data": {"createNoteTarget": {"id": "<uuid>", ...}}}`
- `DELETE /rest/notes/<uuid>` returns `{"data": {"deleteNote": {"id": "<uuid>"}}}`
- `DELETE /rest/noteTargets/<uuid>` returns `{"data": {"deleteNoteTarget": {"id": "<uuid>"}}}`
- Same pattern applies to tasks: `POST /rest/tasks`, `POST /rest/taskTargets`, `DELETE /rest/tasks/<uuid>`
- Deleting a note does NOT cascade-delete its noteTargets (confirmed by live test: noteTarget remains with `noteId: null` after note deleted)

**Example implementation pattern:**
```python
def create_linked_note(person_id, body_markdown, title=""):
    """Create a note linked to a contact. Rollback on link failure (no orphans)."""
    # Step 1: Create the note
    data = {"bodyV2": {"markdown": body_markdown}}
    if title:
        data["title"] = title
    note_resp = _request("notes", "POST", data)
    if "error" in note_resp:
        return {"error": f"Note creation failed: {note_resp['error']}"}

    note_id = note_resp.get("data", {}).get("createNote", {}).get("id")
    if not note_id:
        return {"error": "Note creation returned no ID"}

    # Step 2: Link note to contact
    target_resp = _request("noteTargets", "POST", {"noteId": note_id, "personId": person_id})
    if "error" in target_resp:
        # Rollback: delete the orphaned note
        _request(f"notes/{note_id}", "DELETE")
        return {"error": f"Note created but link to contact failed — note deleted (no orphan). Detail: {target_resp['error']}"}

    target_id = target_resp.get("data", {}).get("createNoteTarget", {}).get("id")
    return {"success": True, "noteId": note_id, "targetId": target_id}
```

### Pattern 2: Server-Side GQL Filter with Adequate Page Size

**What:** Use GraphQL LIKE filters on name fields with `first: 500` to capture the full result set for typical name searches.

**When to use:** Contact search by name.

**Verified behavior:**
- `first: 500` is accepted by the API (no server-side rejection seen at 500 or 1000)
- For broad single-character searches ("a"), total matches = 1489, `first: 500` still truncates but names are scored and returned in DB order
- For realistic name searches (a first or last name), total matches are typically <50 in this CRM
- The `totalCount` field in GQL response always shows the true total regardless of `first` value
- If `count == first`, the response is truncated — worth surfacing as a warning

**Example change:**
```python
# BEFORE (in search_people):
"first": 50,

# AFTER:
"first": 500,
```

### Pattern 3: Multiple-Match Disambiguation (Agent Behavior)

**What:** When search returns multiple contacts, present a numbered list and ask Mike to pick.

**When to use:** Locked user decision — always when multiple matches exist.

**Implementation note:** This is agent behavior, not script behavior. The script should return all matches cleanly. The agent (OpenClaw) reads the numbered list and asks Mike to pick. TOOLS.md should document that search returns multiple results when ambiguous.

**Zero-match behavior (locked user decision):** When search returns 0 contacts, the script should return `{"people": [], "count": 0, ...}` — the agent then offers to create a new contact. No script-level change needed for this; it's agent behavior driven by SOUL.md/AGENTS.md instructions.

### Anti-Patterns to Avoid

- **Retrying step 2 before rollback:** The CONTEXT.md delegates retry strategy to Claude's discretion. Recommendation: no retries on step 2 before rollback. Retrying a failed API call that got an HTTP 400 or 500 is unlikely to succeed and creates a window for partial state. Roll back immediately on first failure.
- **Mentioning rollback in success messages:** Don't say "I created the note and linked it (rollback mechanism active)." Only mention rollback in failure messages.
- **Increasing first to unlimited:** Don't paginate through all 2544 contacts for a name search. 500 is a practical cap that covers any realistic name search result set.
- **Using searchVector as primary search:** It searches email addresses and other indexed fields alongside names, returning false positives for name-based searches. Keep LIKE on name fields as primary.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Two-step atomic note creation | Custom transaction manager | Simple try/except rollback in Python | The failure mode is simple: HTTP error on step 2. A synchronous rollback DELETE is sufficient. No distributed transaction needed. |
| Contact deduplication | Fuzzy match algorithm | Server-side LIKE filter + scoring | Already implemented in `search_people()`. The scoring function already handles partial/exact name matching. |
| Full-text search | Custom tokenizer | `searchVector.search` TSVector filter | Built into Twenty CRM's GQL API as `TSVectorFilter.search` — available as fallback if needed |

---

## Common Pitfalls

### Pitfall 1: create_note() Is Already Broken (Silent Failure)
**What goes wrong:** Calls to `create_note(person_id, body)` fail silently — the API no longer has a `body` field on the `note` object. The `_request()` function catches the HTTP 400 and returns `{"error": "HTTP 400: ..."}`, which the agent may or may not surface clearly.
**Why it happens:** Twenty CRM migrated to `bodyV2` format. The old `body` field was removed.
**How to avoid:** Replace `create_note()` with `create_linked_note()` using `bodyV2: {markdown: ...}` format. Keep backward-compat by aliasing or removing the broken command from TOOLS.md.
**Warning signs:** Calls to `crm.py create-note` return error JSON instead of a note ID.

### Pitfall 2: Orphaned Records Are Invisible to the User
**What goes wrong:** If step 2 fails and no rollback is done, the note/task sits unlinked in Twenty CRM forever. Mike won't see it when browsing a contact, only when viewing all notes/tasks globally.
**Why it happens:** Twenty CRM has no automatic cascade-delete when a note target's `noteId` becomes null.
**How to avoid:** Rollback (DELETE step-1 record) immediately on step-2 failure. Verified: `DELETE /rest/notes/<id>` and `DELETE /rest/tasks/<id>` both work and return the deleted ID.

### Pitfall 3: noteTargets Orphans From Task Deletion
**What goes wrong:** When a task is deleted but its taskTarget is not, the taskTarget row persists with `taskId: null`. This was observed in live testing during research.
**Why it happens:** No cascade delete on the target table side.
**How to avoid:** When performing rollback of step 1 (delete note/task), also delete the target if one was created. In normal rollback flow, if step 2 FAILED, no target was created, so only step 1 needs deleting. But if step 1 is deleted for other reasons while a target exists, the target becomes orphaned.

### Pitfall 4: TOOLS.md Character Limit
**What goes wrong:** TOOLS.md is injected into OpenClaw's context at each session. The limit is 20,000 characters. Current file is 6,458 characters — there is ample headroom.
**Why it happens:** N/A for this phase — not a concern.
**How to avoid:** Adding 2-3 new command lines to TOOLS.md section 1 will use ~100 chars. No risk of hitting the limit.

### Pitfall 5: search_people Returns Duplicated Data
**What goes wrong:** `search_people()` returns both `"people"` and `"results"` keys with identical data. This is redundant.
**Why it happens:** Legacy key name (`results`) was kept alongside the new `people` key.
**How to avoid:** This is a pre-existing issue. Do not change the output structure during Phase 3 — changing it could break agent parsing patterns. Note it as tech debt.

---

## Code Examples

### Verified: Note ID Extraction from Create Response
```python
# Source: live API test against https://crm.groundrushlabs.com
note_resp = _request("notes", "POST", {"bodyV2": {"markdown": "test"}, "title": "My Note"})
# Response structure:
# {"data": {"createNote": {"id": "5daa680e-...", "title": "My Note", ...}}}
note_id = note_resp.get("data", {}).get("createNote", {}).get("id")
```

### Verified: NoteTarget Creation
```python
# Source: live API test
target_resp = _request("noteTargets", "POST", {"noteId": note_id, "personId": person_id})
# Response structure:
# {"data": {"createNoteTarget": {"id": "9d2334b8-...", "noteId": "...", "personId": "..."}}}
target_id = target_resp.get("data", {}).get("createNoteTarget", {}).get("id")
```

### Verified: Note Rollback (Delete)
```python
# Source: live API test
del_resp = _request(f"notes/{note_id}", "DELETE")
# Response: {"data": {"deleteNote": {"id": "5daa680e-..."}}}
```

### Verified: Task Creation Response Structure
```python
# Source: live API test
task_resp = _request("tasks", "POST", {"title": "Follow up", "status": "TODO"})
# Response: {"data": {"createTask": {"id": "7671a51f-...", "title": "Follow up", ...}}}
task_id = task_resp.get("data", {}).get("createTask", {}).get("id")
```

### Verified: TaskTarget Creation
```python
# Source: live API test
target_resp = _request("taskTargets", "POST", {"taskId": task_id, "personId": person_id})
# Response: {"data": {"createTaskTarget": {"id": "02cdb23c-...", "taskId": "...", "personId": "..."}}}
```

### Verified: GQL Search with Increased Cap
```python
# Source: live API test - first:500 accepted, returns all matches
result = _gql("""
query($filter: PersonFilterInput, $first: Int) {
  people(filter: $filter, first: $first) {
    edges {
      node {
        id
        name { firstName lastName }
        emails { primaryEmail additionalEmails }
        phones { primaryPhoneNumber primaryPhoneCountryCode }
        jobTitle
        city
        companyId
      }
    }
    totalCount
  }
}
""", {
    "filter": {"or": filters},
    "first": 500,  # Changed from 50
})
```

### Verified: API Rejection of Inline Target Creation
```python
# Source: live API test - HTTP 400 confirmed
# {"statusCode":400,"error":"BadRequestException",
#  "messages":["One-to-many relation noteTargets field does not support write operations."]}
# Same applies to taskTargets
# Conclusion: ALWAYS use two-step creation pattern
```

---

## State of the Art

| Old Approach | Current Approach | Status | Impact |
|--------------|------------------|--------|--------|
| `create_note(person_id, body)` using `body` field | Must use `bodyV2: {markdown: "..."}` | Broken in crm.py — API no longer accepts `body` field | create-note CLI command fails silently |
| list_people(100) + Python fuzzy match | GQL server-side LIKE filter | Already fixed in current crm.py | Good, but capped at first:50 |
| Inline target setting on note/task create | Separate `noteTargets`/`taskTargets` REST endpoint | API enforces this — no inline writes allowed | Two-step is required, rollback must be explicit |

---

## Open Questions

1. **Should create_linked_note also support linking to a company (companyId) instead of person?**
   - What we know: `noteTargets` has `companyId` and `opportunityId` fields in addition to `personId`
   - What's unclear: Whether Mike needs company-linked notes in addition to person-linked notes
   - Recommendation: Implement person-only linking for Phase 3 (matches the requirements). Adding company/opportunity support can be Phase 3 scope extension if needed — the pattern is identical.

2. **Should the rollback error message mention the failed step to Mike?**
   - What we know: CONTEXT.md says Claude's discretion on "whether rollback actions are mentioned in failure messages"
   - Recommendation: Keep error messages concise. Example: "I couldn't link the note to [Contact Name] and removed the draft note to keep things clean. Try again or let me know if the issue persists." Do NOT expose raw API errors.

3. **Is `first: 500` sufficient long-term as the CRM grows?**
   - What we know: Current CRM has 2,544 contacts. Name searches return <50 results even for common names. Single-character searches can match 1,489+ contacts.
   - What's unclear: Growth trajectory of the contact database
   - Recommendation: Use `first: 500` for Phase 3. If `count == 500`, log a truncation warning in the response (`"truncated": True`). Full cursor-based pagination is a future optimization if needed.

---

## Sources

### Primary (HIGH confidence)
- Live API testing against `https://crm.groundrushlabs.com` via SSH to Alfred Claw 101 — all API behaviors described in this document were directly verified
- `crm.py` source code at `/home/brucewayne9/.openclaw/workspace/scripts/integrations/crm.py` — current implementation analyzed
- GraphQL introspection of `PersonFilterInput` type — confirmed available filter fields and `TSVectorFilter.search` capability
- TOOLS.md at `/home/brucewayne9/.openclaw/workspace/TOOLS.md` — current CLI surface and character count (6,458 chars of 20,000 limit)

### Secondary (MEDIUM confidence)
None needed — all claims verified through direct API testing.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new libraries, all stdlib; existing patterns verified
- Architecture: HIGH — all API behaviors confirmed through live testing
- Pitfalls: HIGH — orphan behavior and cascade-delete absence confirmed empirically

**Research date:** 2026-02-21
**Valid until:** 2026-03-21 (stable — Twenty CRM REST API unlikely to change materially in 30 days)

### Key Numbers (Reference for Planner)
- CRM contact count: 2,544 total
- TOOLS.md current size: 6,458 chars (limit: 20,000 — 13,542 chars of headroom)
- GQL `first: 500` confirmed working: returns correct results, no server rejection
- Rollback tested: note DELETE ✓, noteTarget DELETE ✓, task DELETE ✓, taskTarget DELETE ✓
- Two-step linking confirmed required for: notes (noteTargets), tasks (taskTargets)
- Inline creation rejected with HTTP 400 for both noteTargets and taskTargets
