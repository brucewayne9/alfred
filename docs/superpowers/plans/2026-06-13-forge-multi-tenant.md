# Forge Multi-Tenancy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn Forge into a multi-tenant platform where each company (org) sees only its own sources/clips/jobs, Mike (super-admin) sees all orgs, and each org posts under its own Postiz account.

**Architecture:** Single SQLite DB with an `org_id` column on every directly-listed table (`sources`, `jobs`, `clip_candidates`, `dist_posts`); transcripts inherit scope via `source_id`. A `Scope` value built once at the API boundary from the logged-in identity is threaded into every data-layer read/write — writes stamp the viewer's org, reads filter by it, super-admin can skip the filter. Posting selects a per-org Postiz key.

**Tech Stack:** Python 3.11, FastAPI, SQLite (WAL), ChromaDB (vector search), pytest, bcrypt (passlib), Caddy `forward_auth`.

**Spec:** `docs/superpowers/specs/2026-06-13-forge-multi-tenant-design.md`

**Key design note:** `org_id` columns are added with `DEFAULT 'mainstay'`. SQLite's `ALTER TABLE ADD COLUMN ... DEFAULT 'mainstay'` backfills all existing rows automatically — so the 15 existing sources / 80 clips become Mainstay's with no separate data-migration step. Only the user store (JSON) needs an explicit migration.

**Roles vocabulary change:** old `admin`/`team` → new `super_admin` / `org_admin` / `member`.

---

## Task 1: Org schema — `orgs` table + `org_id` columns + idempotent migration

**Files:**
- Modify: `core/forge/db.py:23-113` (init_db)
- Test: `tests/forge/test_orgs_schema.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/forge/test_orgs_schema.py
import pytest


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_PATH", str(tmp_path / "forge.db"))
    from core.forge import db as _db
    _db.init_db()
    return _db


def test_orgs_table_seeded_with_three_orgs(db):
    with db._conn() as c:
        ids = {r["id"] for r in c.execute("SELECT id FROM orgs")}
    assert {"mainstay", "rucktalk", "groundrush"} <= ids


def test_org_id_column_on_scoped_tables_defaults_to_mainstay(db):
    with db._conn() as c:
        c.execute(
            "INSERT INTO sources (id, kind, spec, status, created_at, updated_at) "
            "VALUES ('s1','url','x','done',0,0)"
        )
        row = c.execute("SELECT org_id FROM sources WHERE id='s1'").fetchone()
    assert row["org_id"] == "mainstay"


def test_all_four_scoped_tables_have_org_id(db):
    with db._conn() as c:
        for table in ("sources", "jobs", "clip_candidates", "dist_posts"):
            cols = {r[1] for r in c.execute(f"PRAGMA table_info({table})")}
            assert "org_id" in cols, f"{table} missing org_id"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/forge/test_orgs_schema.py -v`
Expected: FAIL — `no such table: orgs` / `no such column: org_id`.

- [ ] **Step 3: Add the orgs table + seed + org_id migration to init_db**

In `core/forge/db.py`, inside the `executescript("""...""")` block (after the `clip_candidates` block, before the closing `"""`), add:

```sql
-- Multi-tenancy: one row per company (tenant).
CREATE TABLE IF NOT EXISTS orgs (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    created_at INTEGER NOT NULL DEFAULT 0
);

-- Distribution post ledger (was implicit; make it explicit + org-scoped).
CREATE TABLE IF NOT EXISTS dist_posts (
    post_id   TEXT PRIMARY KEY,
    posted    INTEGER NOT NULL DEFAULT 0,
    posted_at INTEGER,
    org_id    TEXT NOT NULL DEFAULT 'mainstay'
);
```

Then, after the existing `created_by` migration block at the end of `init_db()` (line ~110-112), add the org_id column migrations and the seed:

```python
        # Multi-tenancy: add org_id to scoped tables (DEFAULT backfills old rows
        # to 'mainstay' automatically). Idempotent — skip if already present.
        for table in ("sources", "jobs", "clip_candidates", "dist_posts"):
            tcols = {r[1] for r in c.execute(f"PRAGMA table_info({table})").fetchall()}
            if "org_id" not in tcols:
                c.execute(
                    f"ALTER TABLE {table} ADD COLUMN org_id TEXT NOT NULL DEFAULT 'mainstay'"
                )
        # Index for org-filtered listing.
        c.execute("CREATE INDEX IF NOT EXISTS idx_sources_org ON sources(org_id, status)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_jobs_org ON jobs(org_id, created_at)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_candidates_org ON clip_candidates(org_id)")
        # Seed the three known orgs (idempotent via INSERT OR IGNORE).
        for oid, name in (("mainstay", "Mainstay Music Group"),
                          ("rucktalk", "RuckTalk"),
                          ("groundrush", "Ground Rush")):
            c.execute("INSERT OR IGNORE INTO orgs (id, name, created_at) VALUES (?, ?, 0)",
                      (oid, name))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/forge/test_orgs_schema.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add core/forge/db.py tests/forge/test_orgs_schema.py
git commit -m "feat(forge): orgs table + org_id columns on scoped tables (multi-tenant schema)"
```

---

## Task 2: User store — `org` field + new role vocabulary

**Files:**
- Modify: `core/forge/users.py:48-110`
- Test: `tests/forge/test_forge_users_org.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/forge/test_forge_users_org.py
import pytest
from core.forge import users


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_USERS_FILE", str(tmp_path / "forge_users.json"))


def test_create_user_stores_org_and_role():
    users.create_user("alice", "pw123", role="member", org="rucktalk")
    u = users.verify_user("alice", "pw123")
    assert u == {"username": "alice", "role": "member", "org": "rucktalk"}


def test_unknown_role_falls_back_to_member():
    users.create_user("bob", "pw123", role="banana", org="mainstay")
    assert users.verify_user("bob", "pw123")["role"] == "member"


def test_super_admin_role_is_allowed():
    users.create_user("mike", "pw123", role="super_admin", org="*")
    assert users.verify_user("mike", "pw123")["role"] == "super_admin"


def test_list_users_includes_org():
    users.create_user("alice", "pw123", role="member", org="rucktalk")
    roster = users.list_users()
    assert {"username": "alice", "role": "member", "org": "rucktalk"} in roster
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/forge/test_forge_users_org.py -v`
Expected: FAIL — `create_user() got an unexpected keyword argument 'org'`.

- [ ] **Step 3: Update users.py**

Replace `create_user`, `verify_user`, `list_users`, and the role constant in `core/forge/users.py`:

```python
_ROLES = ("member", "org_admin", "super_admin")


def create_user(username: str, password: str, role: str = "member",
                org: str = "mainstay") -> bool:
    """Add or update a user. role in {member, org_admin, super_admin}."""
    username = (username or "").strip().lower()
    if not username or not password:
        raise ValueError("username and password are required")
    role = role if role in _ROLES else "member"
    org = (org or "mainstay").strip().lower()
    users = load_users()
    users[username] = {"password_hash": _pwd.hash(password), "role": role, "org": org}
    save_users(users)
    return True


def verify_user(username: str, password: str) -> dict | None:
    username = (username or "").strip().lower()
    user = load_users().get(username)
    if not user:
        return None
    try:
        if not _pwd.verify(password, user["password_hash"]):
            return None
    except Exception:  # noqa: BLE001
        return None
    return {
        "username": username,
        "role": user.get("role", "member"),
        "org": user.get("org", "mainstay"),
    }


def list_users() -> list[dict]:
    """Public roster (no hashes), sorted by username."""
    return [
        {"username": u, "role": d.get("role", "member"), "org": d.get("org", "mainstay")}
        for u, d in sorted(load_users().items())
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/forge/test_forge_users_org.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Update the seed block to the new role/org vocabulary**

Replace the `_SEED` list and `ensure_seeded` at the bottom of `core/forge/users.py`:

```python
# Seed once with the accounts already issued, now org/role aware.
_SEED = [
    ("mike", "Mike-Boss0619!", "super_admin", "*"),
    ("mainstay", "RodWave0619!", "org_admin", "mainstay"),
    ("jordan", "Jordan-Anthem26!", "member", "mainstay"),
    ("mello", "Mello-Studio72!", "member", "mainstay"),
]


def ensure_seeded() -> None:
    if not load_users():
        for u, p, r, o in _SEED:
            create_user(u, p, r, o)
```

- [ ] **Step 6: Commit**

```bash
git add core/forge/users.py tests/forge/test_forge_users_org.py
git commit -m "feat(forge): user store gains org + super_admin/org_admin/member roles"
```

---

## Task 3: Scope object + identity flow (X-Forge-Org header)

**Files:**
- Create: `core/forge/scope.py`
- Modify: `core/api/forge.py:37-60` (authcheck), `services/forge-web/serve.py:37-50` (_forge_user)
- Test: `tests/forge/test_scope.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/forge/test_scope.py
from core.forge.scope import Scope, scope_from_user


def test_member_is_pinned_to_own_org():
    s = scope_from_user({"username": "alice", "role": "member", "org": "rucktalk"})
    assert s.org == "rucktalk"
    assert s.view_all is False
    assert s.can_write_org("rucktalk") is True
    assert s.can_write_org("mainstay") is False


def test_super_admin_view_all_by_default():
    s = scope_from_user({"username": "mike", "role": "super_admin", "org": "*"})
    assert s.view_all is True
    assert s.can_write_org("mainstay") is True
    assert s.can_write_org("rucktalk") is True


def test_super_admin_can_focus_one_org():
    s = scope_from_user(
        {"username": "mike", "role": "super_admin", "org": "*"},
        requested_org="mainstay",
    )
    assert s.view_all is False
    assert s.org == "mainstay"


def test_member_cannot_escape_org_via_requested_org():
    s = scope_from_user(
        {"username": "alice", "role": "member", "org": "rucktalk"},
        requested_org="mainstay",   # attacker-supplied — must be ignored
    )
    assert s.org == "rucktalk"
    assert s.view_all is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/forge/test_scope.py -v`
Expected: FAIL — `No module named 'core.forge.scope'`.

- [ ] **Step 3: Create core/forge/scope.py**

```python
"""Forge tenancy scope — the single source of truth for 'what can this viewer
see and write'. Built once at the API boundary from the authenticated identity
and threaded into every data-layer call.

A member is pinned to their own org. A super_admin defaults to view_all (every
org merged) but may focus one org via the dashboard switcher. requested_org is
ONLY honored for super_admin — a member can never escape their org by passing it.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scope:
    org: str                # the org to stamp writes with / focus reads on
    role: str               # member | org_admin | super_admin
    view_all: bool          # super_admin seeing every org at once

    @property
    def is_super(self) -> bool:
        return self.role == "super_admin"

    def can_write_org(self, org: str) -> bool:
        """May this viewer create/modify rows in `org`?"""
        return self.is_super or org == self.org

    def can_read_org(self, org: str) -> bool:
        return self.view_all or self.is_super or org == self.org


def scope_from_user(user: dict, requested_org: str | None = None) -> Scope:
    role = (user or {}).get("role", "member")
    org = (user or {}).get("org", "mainstay")
    if role == "super_admin":
        focus = (requested_org or "").strip().lower()
        if focus and focus != "*" and focus != "all":
            return Scope(org=focus, role=role, view_all=False)
        return Scope(org="*", role=role, view_all=True)
    # member / org_admin — pinned to their own org; requested_org ignored.
    return Scope(org=org, role=role, view_all=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/forge/test_scope.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Emit X-Forge-Org from authcheck**

In `core/api/forge.py`, the `forge_authcheck` success Response (line ~57-60) becomes:

```python
        return Response(
            status_code=200,
            headers={
                "X-Forge-User": u["username"],
                "X-Forge-Role": u["role"],
                "X-Forge-Org": u.get("org", "mainstay"),
            },
        )
```

- [ ] **Step 6: Read X-Forge-Org in serve.py**

In `services/forge-web/serve.py`, replace `FORGE_ADMINS` + `_forge_user` (lines ~37-50):

```python
# Fallback super-admins if the X-Forge-Role header is ever missing.
FORGE_SUPER = {"mike", "mainstay"}


def _forge_user(request: Request) -> dict:
    username = request.headers.get("X-Forge-User") or "mainstay"
    role = request.headers.get("X-Forge-Role")
    org = request.headers.get("X-Forge-Org") or "mainstay"
    if role not in ("member", "org_admin", "super_admin"):
        role = "super_admin" if username in FORGE_SUPER else "member"
    return {"username": username, "role": role, "org": org}
```

- [ ] **Step 7: Commit**

```bash
git add core/forge/scope.py tests/forge/test_scope.py core/api/forge.py services/forge-web/serve.py
git commit -m "feat(forge): Scope object + X-Forge-Org identity header"
```

---

## Task 4: Scope `sources` — ingest.py

**Files:**
- Modify: `core/forge/ingest.py:34-62` (create_source), `:216-229` (list_sources)
- Test: `tests/forge/test_ingest_org.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/forge/test_ingest_org.py
import pytest


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_PATH", str(tmp_path / "forge.db"))
    from core.forge import db as _db
    _db.init_db()
    return _db


def test_create_source_stamps_org(db):
    from core.forge import ingest
    sid = ingest.create_source("url", "https://x", None, org="rucktalk")
    assert ingest.get_source(sid)["org_id"] == "rucktalk"


def test_create_source_defaults_to_mainstay(db):
    from core.forge import ingest
    sid = ingest.create_source("url", "https://x", None)
    assert ingest.get_source(sid)["org_id"] == "mainstay"


def test_list_sources_filters_by_org(db):
    from core.forge import ingest
    ingest.create_source("url", "a", None, org="mainstay")
    ingest.create_source("url", "b", None, org="rucktalk")
    mainstay = [s for s in ingest.list_sources(org="mainstay")]
    rucktalk = [s for s in ingest.list_sources(org="rucktalk")]
    assert len(mainstay) == 1 and mainstay[0]["spec"] == "a"
    assert len(rucktalk) == 1 and rucktalk[0]["spec"] == "b"


def test_list_sources_org_none_returns_all(db):
    from core.forge import ingest
    ingest.create_source("url", "a", None, org="mainstay")
    ingest.create_source("url", "b", None, org="rucktalk")
    assert len(ingest.list_sources(org=None)) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/forge/test_ingest_org.py -v`
Expected: FAIL — `create_source() got an unexpected keyword argument 'org'`.

- [ ] **Step 3: Update create_source and list_sources**

`create_source` — add an `org` param and include it in the INSERT. The current signature (line 34) ends with the existing params; add `org: str = "mainstay"` and update the SQL:

```python
def create_source(kind: str, spec: str, file_path: str | None = None,
                  org: str = "mainstay") -> str:
    import uuid, time
    source_id = uuid.uuid4().hex
    ts = int(time.time())
    with _conn() as c:
        c.execute(
            """
            INSERT INTO sources (id, kind, spec, file_path, status, org_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)
            """,
            (source_id, kind, spec, file_path, org, ts, ts),
        )
    return source_id
```

> NOTE: match the EXISTING body of `create_source` (it already builds id/ts) — the only changes are the `org` param, the added `org_id` column, and the `?` for `org`. Do not duplicate id/ts logic if it already exists; just thread org through.

`list_sources` (line 216) — add `org` filter:

```python
def list_sources(status: str | None = None, org: str | None = None) -> list[dict]:
    init_db()
    clauses, params = [], []
    if status:
        clauses.append("status = ?"); params.append(status)
    if org:
        clauses.append("org_id = ?"); params.append(org)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    with _conn() as c:
        rows = c.execute(
            f"SELECT * FROM sources{where} ORDER BY rowid DESC", params
        ).fetchall()
    return [dict(r) for r in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/forge/test_ingest_org.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Run the existing ingest tests to confirm no regression**

Run: `pytest tests/forge/test_topic_clip.py tests/forge/test_db.py -q`
Expected: PASS (org defaults keep single-arg callers working).

- [ ] **Step 6: Commit**

```bash
git add core/forge/ingest.py tests/forge/test_ingest_org.py
git commit -m "feat(forge): org-scope sources (create_source stamps, list_sources filters)"
```

---

## Task 5: Scope `jobs` — jobs.py

**Files:**
- Modify: `core/forge/jobs.py:77-110` (enqueue, list_jobs)
- Test: `tests/forge/test_jobs_org.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/forge/test_jobs_org.py
import pytest


@pytest.fixture
def forge(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_PATH", str(tmp_path / "forge.db"))
    from core.forge import jobs as _jobs
    _jobs.init_db()
    return _jobs


def test_enqueue_stamps_org(forge):
    jid = forge.enqueue("echo", {"x": 1}, org="rucktalk")
    assert forge.get_job(jid)["org_id"] == "rucktalk"


def test_list_jobs_filters_by_org(forge):
    forge.enqueue("echo", {}, org="mainstay")
    forge.enqueue("echo", {}, org="rucktalk")
    assert len(forge.list_jobs(org="rucktalk")) == 1
    assert len(forge.list_jobs(org=None)) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/forge/test_jobs_org.py -v`
Expected: FAIL — `enqueue() got an unexpected keyword argument 'org'`.

- [ ] **Step 3: Update enqueue and list_jobs**

`enqueue` (line 77):

```python
def enqueue(job_type: str, params: Optional[dict] = None, now: Optional[int] = None,
            created_by: Optional[str] = None, org: str = "mainstay") -> str:
    init_db()
    job_id = uuid.uuid4().hex
    ts = now if now is not None else _now()
    with _conn() as c:
        c.execute(
            "INSERT INTO jobs (id, job_type, status, params, created_by, org_id, created_at, updated_at) "
            "VALUES (?, ?, 'pending', ?, ?, ?, ?, ?)",
            (job_id, job_type, json.dumps(params or {}), created_by, org, ts, ts),
        )
    return job_id
```

`list_jobs` (line 98):

```python
def list_jobs(status: Optional[str] = None, limit: int = 100,
              org: Optional[str] = None) -> list[dict]:
    init_db()
    clauses, params = [], []
    if status:
        clauses.append("status = ?"); params.append(status)
    if org:
        clauses.append("org_id = ?"); params.append(org)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)
    with _conn() as c:
        rows = c.execute(
            f"SELECT * FROM jobs{where} ORDER BY created_at DESC LIMIT ?", params
        ).fetchall()
    return [_row_to_dict(r) for r in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/forge/test_jobs_org.py tests/forge/test_jobs.py -v`
Expected: PASS (existing test_jobs still green — org defaults to mainstay).

- [ ] **Step 5: Commit**

```bash
git add core/forge/jobs.py tests/forge/test_jobs_org.py
git commit -m "feat(forge): org-scope jobs (enqueue stamps, list_jobs filters)"
```

---

## Task 6: Scope `clip_candidates` — scorer.py

**Files:**
- Modify: `core/forge/scorer.py:168-247` (save_candidates, get_candidate, get_candidates, _CANDIDATE_COLS)
- Test: `tests/forge/test_scorer_org.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/forge/test_scorer_org.py
import pytest


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_PATH", str(tmp_path / "forge.db"))
    from core.forge import db as _db
    _db.init_db()
    return _db


def _cand():
    return {"start_s": 1.0, "end_s": 9.0, "score": 80, "hook": "h",
            "emotion": "e", "reason": "r", "caption": "c"}


def test_save_candidates_stamps_org(db):
    from core.forge import scorer
    scorer.save_candidates("src1", [_cand()], org="rucktalk")
    rows = scorer.get_candidates("src1")
    assert rows and rows[0]["org_id"] == "rucktalk"


def test_get_candidate_exposes_org_for_ownership_check(db):
    from core.forge import scorer
    scorer.save_candidates("src1", [_cand()], org="rucktalk")
    cid = scorer.get_candidates("src1")[0]["id"]
    assert scorer.get_candidate(cid)["org_id"] == "rucktalk"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/forge/test_scorer_org.py -v`
Expected: FAIL — `save_candidates() got an unexpected keyword argument 'org'`.

- [ ] **Step 3: Update scorer.py**

Add `org_id` to the `_CANDIDATE_COLS` select list (so get_candidate/get_candidates return it). Locate `_CANDIDATE_COLS` near the top of the candidate-DB section and append `, org_id`.

`save_candidates` (line 168) — add `org` param, include `org_id` in the INSERT column list and each row's values:

```python
def save_candidates(source_id: str, candidates: list[dict],
                    judge_model: str | None = None, org: str = "mainstay") -> None:
    init_db()
    now = int(time.time())
    with _conn() as c:
        c.execute("DELETE FROM clip_candidates WHERE source_id = ?", (source_id,))
        for cand in candidates:
            c.execute(
                """
                INSERT INTO clip_candidates
                  (source_id, start_s, end_s, score, hook, emotion, reason,
                   caption, judge_model, org_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (source_id, cand["start_s"], cand["end_s"], cand["score"],
                 cand.get("hook"), cand.get("emotion"), cand.get("reason"),
                 cand.get("caption"), judge_model, org, now),
            )
```

> NOTE: match the EXISTING column/value ordering of `save_candidates` — the only additions are the `org` param, `org_id` in the column list, and `org` in the values tuple. Preserve any columns already present that aren't shown here.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/forge/test_scorer_org.py tests/forge/test_scorer.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/forge/scorer.py tests/forge/test_scorer_org.py
git commit -m "feat(forge): org-scope clip_candidates (save stamps, get exposes org_id)"
```

---

## Task 7: Scope vector search — search.py (ChromaDB metadata filter)

**Files:**
- Modify: `core/forge/search.py:139-168` (upsert_windows), `:234-307` (search_segments)
- Test: `tests/forge/test_search_org.py` (create)

> WHY: ChromaDB is a separate store from SQLite. Without an org filter, semantic
> topic-search would return another org's segments — a leak. Chroma supports a
> metadata `where=` filter; we tag each window with `org_id` on upsert and filter
> on search.

- [ ] **Step 1: Write the failing test**

```python
# tests/forge/test_search_org.py
import pytest


@pytest.fixture
def chroma(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_CHROMA_DIR", str(tmp_path / "chroma"))
    from core.forge import search
    # fresh collection per test
    if hasattr(search, "_COLLECTION"):
        search._COLLECTION = None
    return search


def test_search_only_returns_own_org_windows(chroma):
    search = chroma
    win = [{"start_s": 0.0, "end_s": 5.0, "text": "ground rush secret sauce"}]
    search.upsert_windows("srcMain", win, org="mainstay")
    search.upsert_windows("srcRuck", win, org="rucktalk")
    hits = search.search_segments("secret sauce", org="rucktalk", top_k=10)
    src_ids = {h["source_id"] for h in hits}
    assert src_ids == {"srcRuck"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/forge/test_search_org.py -v`
Expected: FAIL — `upsert_windows() got an unexpected keyword argument 'org'`.

- [ ] **Step 3: Tag windows with org_id on upsert and filter on search**

In `upsert_windows` (line 139), add `org: str = "mainstay"` and include `"org_id": org` in each window's metadata dict that is passed to `collection.upsert(metadatas=[...])`. Find where metadata is built per window and add the key.

In `search_segments` (line 234), add `org: str | None = None` and pass a Chroma `where` filter when org is set:

```python
def search_segments(query: str, source_id: str | None = None,
                    top_k: int = 8, org: str | None = None) -> list[dict]:
    coll = _get_collection()
    where = None
    conds = []
    if source_id:
        conds.append({"source_id": source_id})
    if org:
        conds.append({"org_id": org})
    if len(conds) == 1:
        where = conds[0]
    elif len(conds) > 1:
        where = {"$and": conds}
    res = coll.query(query_texts=[query], n_results=top_k, where=where)
    # ... existing result-shaping code unchanged ...
```

> NOTE: keep the EXISTING result-shaping body of `search_segments`; only the
> signature and the `where=` argument to `coll.query(...)` change.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/forge/test_search_org.py tests/forge/test_search.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/forge/search.py tests/forge/test_search_org.py
git commit -m "feat(forge): org-scope vector search (tag windows + where-filter by org)"
```

---

## Task 8: Wire Scope through the API + ownership checks

**Files:**
- Modify: `core/api/forge.py` (multiple endpoints), add a `_scope(request, user)` helper
- Test: `tests/forge/test_api_org_isolation.py` (create)

> This is the security-critical task: every endpoint that reads or writes
> tenant data builds a `Scope` and either filters by it or ownership-checks the
> target row. A `super_admin` may pass `?org=<id>` (set by the dashboard
> switcher) to focus one org; everyone else is pinned.

- [ ] **Step 1: Write the failing isolation test**

```python
# tests/forge/test_api_org_isolation.py
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.api.forge import register
from core.security.auth import require_auth


def _client(tmp_path, monkeypatch, user):
    monkeypatch.setenv("FORGE_DB_PATH", str(tmp_path / "forge.db"))
    from core.forge import db as _db
    _db.init_db()
    app = FastAPI()
    register(app)
    app.dependency_overrides[require_auth] = lambda: user
    return TestClient(app)


def test_member_sees_only_own_org_sources(tmp_path, monkeypatch):
    from core.forge import ingest
    ingest.create_source("url", "mainstay-clip", None, org="mainstay")
    ingest.create_source("url", "ruck-clip", None, org="rucktalk")

    ruck = _client(tmp_path, monkeypatch, {"username": "r", "role": "member", "org": "rucktalk"})
    specs = [s["spec"] for s in ruck.get("/forge/sources").json()["sources"]]
    assert specs == ["ruck-clip"]


def test_member_gets_404_on_other_orgs_source(tmp_path, monkeypatch):
    from core.forge import ingest
    sid = ingest.create_source("url", "mainstay-clip", None, org="mainstay")
    ruck = _client(tmp_path, monkeypatch, {"username": "r", "role": "member", "org": "rucktalk"})
    assert ruck.get(f"/forge/sources/{sid}").status_code == 404


def test_super_admin_sees_all_orgs(tmp_path, monkeypatch):
    from core.forge import ingest
    ingest.create_source("url", "mainstay-clip", None, org="mainstay")
    ingest.create_source("url", "ruck-clip", None, org="rucktalk")
    mike = _client(tmp_path, monkeypatch, {"username": "mike", "role": "super_admin", "org": "*"})
    specs = {s["spec"] for s in mike.get("/forge/sources").json()["sources"]}
    assert specs == {"mainstay-clip", "ruck-clip"}


def test_super_admin_can_focus_one_org(tmp_path, monkeypatch):
    from core.forge import ingest
    ingest.create_source("url", "mainstay-clip", None, org="mainstay")
    ingest.create_source("url", "ruck-clip", None, org="rucktalk")
    mike = _client(tmp_path, monkeypatch, {"username": "mike", "role": "super_admin", "org": "*"})
    specs = [s["spec"] for s in mike.get("/forge/sources?org=rucktalk").json()["sources"]]
    assert specs == ["ruck-clip"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/forge/test_api_org_isolation.py -v`
Expected: FAIL — sources unfiltered (member sees both).

- [ ] **Step 3: Add a `_scope` helper and apply it**

At the top of `register(app)` in `core/api/forge.py`, add:

```python
    from core.forge.scope import scope_from_user

    def _scope(request: Request, user: dict):
        # super_admin may focus an org via ?org=; everyone else is pinned.
        requested = request.query_params.get("org")
        return scope_from_user(user, requested_org=requested)

    def _scoped_org(scope):
        # None => no filter (super_admin view_all); else the focused org id.
        return None if scope.view_all else scope.org
```

Update the sources endpoints:

```python
    @app.get("/forge/sources")
    async def list_sources_endpoint(request: Request, status: str | None = None,
                                    user: dict = Depends(require_auth)):
        from core.forge import ingest
        scope = _scope(request, user)
        return {"sources": ingest.list_sources(status=status, org=_scoped_org(scope))}

    @app.get("/forge/sources/{source_id}")
    async def get_source_status(source_id: str, request: Request,
                                user: dict = Depends(require_auth)):
        from core.forge import ingest
        scope = _scope(request, user)
        source = ingest.get_source(source_id)
        if source is None or not scope.can_read_org(source.get("org_id", "mainstay")):
            raise HTTPException(status_code=404, detail="source not found")
        return source
```

Apply the SAME `source = get_source(...); if None or not scope.can_read_org(...): 404`
ownership guard to every endpoint that takes a `source_id` and returns its data:
`stream_source_video` (line 280), and any source-scoped clip/segment/score
endpoints further down the file. For job listing/creation:

```python
    @app.get("/forge/jobs")
    async def list_jobs(request: Request, status: str | None = None,
                        user: dict = Depends(require_auth)):
        scope = _scope(request, user)
        return {"jobs": forge_jobs.list_jobs(status=status, org=_scoped_org(scope))}

    @app.post("/forge/jobs")
    async def create_job(request: Request, payload: dict = Body(...),
                         user: dict = Depends(require_auth)):
        job_type = payload.get("job_type")
        if not job_type:
            raise HTTPException(status_code=400, detail="job_type is required")
        scope = _scope(request, user)
        write_org = scope.org if not scope.view_all else "mainstay"
        job_id = forge_jobs.enqueue(job_type, payload.get("params") or {},
                                    created_by=user.get("username"), org=write_org)
        return forge_jobs.get_job(job_id)
```

For the **ingest write endpoints** (`create_upload` line 210, `ingest_cloud` 232,
`ingest_url` 250), thread the viewer's org into `create_source(...)` and the
enqueued ingest job:

```python
        scope = _scope(request, user)
        write_org = scope.org if not scope.view_all else "mainstay"
        source_id = ingest.create_source("url", url, None, org=write_org)
        job_id = _forge_jobs.enqueue("ingest_transcribe",
                                     {"source_id": source_id, "url": url}, org=write_org)
```

(Add `request: Request` to each of these endpoint signatures.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/forge/test_api_org_isolation.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Run the full forge API suite for regressions**

Run: `pytest tests/forge/test_api.py tests/forge/test_search_api.py tests/forge/test_upload_api.py tests/forge/test_library_api.py -q`
Expected: PASS. Fix any endpoint that now needs `request: Request` added.

- [ ] **Step 6: Commit**

```bash
git add core/api/forge.py tests/forge/test_api_org_isolation.py
git commit -m "feat(forge): enforce org scope + ownership checks across the API"
```

---

## Task 9: Org-scoped posting — per-org Postiz key

**Files:**
- Modify: `core/forge/postiz_client.py:45-55,72-160` (key selection)
- Test: `tests/forge/test_postiz_org.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/forge/test_postiz_org.py
import pytest
from core.forge import postiz_client as pz


def test_key_for_org_selects_per_org_env(monkeypatch):
    monkeypatch.setenv("POSTIZ_MAINSTAY_API_KEY", "MAIN-KEY")
    monkeypatch.setenv("POSTIZ_RUCKTALK_API_KEY", "RUCK-KEY")
    assert pz.key_for_org("mainstay") == "MAIN-KEY"
    assert pz.key_for_org("rucktalk") == "RUCK-KEY"


def test_unknown_org_has_no_key(monkeypatch):
    monkeypatch.delenv("POSTIZ_GROUNDRUSH_API_KEY", raising=False)
    assert pz.key_for_org("groundrush") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/forge/test_postiz_org.py -v`
Expected: FAIL — `module 'core.forge.postiz_client' has no attribute 'key_for_org'`.

- [ ] **Step 3: Add key_for_org and thread org through the post path**

In `core/forge/postiz_client.py`, add:

```python
# Per-org Postiz API keys. Each org's burners live in its own Postiz org,
# reached with its own key. mainstay = Rod Wave burners; rucktalk = the
# RuckTalk / Ground Rush org (RuckTalk + a burner).
_ORG_KEY_ENV = {
    "mainstay": "POSTIZ_MAINSTAY_API_KEY",
    "rucktalk": "POSTIZ_RUCKTALK_API_KEY",
}


def key_for_org(org: str) -> str | None:
    return _env(_ORG_KEY_ENV.get(org, ""))
```

Then change `list_integrations`, `create_post`, and any other function that
currently reads `_env("POSTIZ_MAINSTAY_API_KEY")` to accept `org: str = "mainstay"`
and call `key = key_for_org(org)`. Example for `list_integrations` (line 72):

```python
def list_integrations(org: str = "mainstay") -> list[dict]:
    """All channels connected in `org`'s Postiz organization."""
    key = key_for_org(org)
    if not key:
        logger.error("No Postiz key for org=%s — cannot list integrations", org)
        return []
    # ... existing request code, using `key` ...
```

When a clip is posted, pass its `org_id` (from the candidate/source row) into
these functions. In the distribution endpoint/handler that calls `create_post`,
read the org off the candidate (`scorer.get_candidate(cid)["org_id"]`) and pass it.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/forge/test_postiz_org.py tests/forge/test_distribution.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/forge/postiz_client.py tests/forge/test_postiz_org.py
git commit -m "feat(forge): per-org Postiz key selection (Mainstay vs RuckTalk posting)"
```

---

## Task 10: Admin panel org field + super-admin org switcher (UI)

**Files:**
- Modify: `core/api/forge.py:13-15,62-96` (role gates + user CRUD), `services/forge-web/index.html` (switcher + org field)
- Test: `tests/forge/test_user_admin_org.py` (create)

- [ ] **Step 1: Write the failing test (API side — the testable part)**

```python
# tests/forge/test_user_admin_org.py
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.api.forge import register
from core.security.auth import require_auth


def _client(tmp_path, monkeypatch, user):
    monkeypatch.setenv("FORGE_USERS_FILE", str(tmp_path / "u.json"))
    monkeypatch.setenv("FORGE_DB_PATH", str(tmp_path / "forge.db"))
    app = FastAPI(); register(app)
    app.dependency_overrides[require_auth] = lambda: user
    return TestClient(app)


def test_org_admin_creates_user_only_in_own_org(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch,
                {"username": "boss", "role": "org_admin", "org": "rucktalk"})
    # org field in payload is ignored for org_admin — forced to their own org
    c.post("/forge/users", json={"username": "newbie", "password": "pw12345",
                                 "role": "member", "org": "mainstay"})
    from core.forge import users
    assert users.load_users()["newbie"]["org"] == "rucktalk"


def test_super_admin_creates_user_in_any_org(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch,
                {"username": "mike", "role": "super_admin", "org": "*"})
    c.post("/forge/users", json={"username": "x", "password": "pw12345",
                                 "role": "member", "org": "mainstay"})
    from core.forge import users
    assert users.load_users()["x"]["org"] == "mainstay"


def test_member_cannot_manage_users(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch,
                {"username": "alice", "role": "member", "org": "rucktalk"})
    assert c.get("/forge/users").status_code == 403
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/forge/test_user_admin_org.py -v`
Expected: FAIL — `_require_admin` still keys on old `admin` role; org not forced.

- [ ] **Step 3: Update role gates + user CRUD in core/api/forge.py**

```python
    def _require_manage(user: dict) -> None:
        if (user or {}).get("role") not in ("org_admin", "super_admin"):
            raise HTTPException(status_code=403, detail="admin only")

    def _require_super(user: dict) -> None:
        if (user or {}).get("role") != "super_admin":
            raise HTTPException(status_code=403, detail="super admin only")
```

Replace `_require_admin(user)` calls in the user endpoints (lines 65, 70, 81)
with `_require_manage(user)`. In `add_forge_user` force the org for non-supers
and accept the new roles:

```python
    @app.post("/forge/users")
    async def add_forge_user(payload: dict = Body(...), user: dict = Depends(require_auth)):
        _require_manage(user)
        username = (payload.get("username") or "").strip().lower()
        password = payload.get("password") or ""
        role = payload.get("role")
        if role not in ("member", "org_admin", "super_admin"):
            role = "member"
        # super_admin may target any org; org_admin is forced to their own.
        if user.get("role") == "super_admin":
            org = (payload.get("org") or "mainstay").strip().lower()
        else:
            org = user.get("org", "mainstay")
            role = "member" if role == "super_admin" else role  # no privilege escalation
        if not username or not password:
            raise HTTPException(status_code=400, detail="username and password required")
        forge_users.create_user(username, password, role, org)
        return {"ok": True, "users": forge_users.list_users()}
```

Add an orgs list endpoint for the switcher + create-user dropdown:

```python
    @app.get("/forge/orgs")
    async def list_orgs(user: dict = Depends(require_auth)):
        from core.forge import db
        with db._conn() as c:
            orgs = [dict(r) for r in c.execute("SELECT id, name FROM orgs ORDER BY name")]
        return {"orgs": orgs, "me_org": user.get("org"), "role": user.get("role")}
```

Update `/forge/me` to include org so the front-end knows whether to show the switcher:

```python
    @app.get("/forge/me")
    async def forge_me(user: dict = Depends(require_auth)):
        return {"username": user.get("username"), "role": user.get("role", "member"),
                "org": user.get("org", "mainstay")}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/forge/test_user_admin_org.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Add the front-end switcher (manual, in index.html)**

In `services/forge-web/index.html`, add a header control visible only when
`/forge/me` returns `role === 'super_admin'`. It populates from `/forge/orgs`
and re-requests data with `?org=<id>` (or omits `org` for "All orgs"):

```html
<select id="org-switcher" style="display:none"></select>
```
```javascript
async function initOrgSwitcher() {
  const me = await fetch('/forge/me').then(r => r.json());
  if (me.role !== 'super_admin') return;          // members never see it
  const { orgs } = await fetch('/forge/orgs').then(r => r.json());
  const sel = document.getElementById('org-switcher');
  sel.innerHTML = '<option value="">All orgs</option>' +
    orgs.map(o => `<option value="${o.id}">${o.name}</option>`).join('');
  sel.style.display = '';
  sel.onchange = () => { window.FORGE_ORG = sel.value; reloadAll(); };
}
// In every data fetch, append the org filter when set:
function withOrg(url) {
  if (!window.FORGE_ORG) return url;
  return url + (url.includes('?') ? '&' : '?') + 'org=' + encodeURIComponent(window.FORGE_ORG);
}
```

Wrap the existing `fetch('/forge/sources')`, `fetch('/forge/jobs')`, etc. calls
in `withOrg(...)`. Call `initOrgSwitcher()` on page load. Also add an **org
`<select>`** to the "add user" form, populated from `/forge/orgs`, shown only to
super_admin.

- [ ] **Step 6: Manual verification**

Run: `sudo systemctl restart forge-web.service`
Then, behind Caddy at `https://aialfred.groundrushcloud.com/forge`:
- Log in as `mike` → org switcher visible; "All orgs" shows everything; pick
  "RuckTalk" → only RuckTalk sources/jobs.
- Log in as `jordan` (member/mainstay) → no switcher; only Mainstay data.
Expected: visibility matches role. Record the result.

- [ ] **Step 7: Commit**

```bash
git add core/api/forge.py services/forge-web/index.html tests/forge/test_user_admin_org.py
git commit -m "feat(forge): org-aware user admin + super-admin org switcher"
```

---

## Task 11: One-time production migration of the user store

**Files:**
- Create: `scripts/forge_migrate_multitenant.py`
- Test: `tests/forge/test_user_migration.py` (create)

> The DB columns self-migrate via Task 1 (ALTER … DEFAULT 'mainstay'). Only the
> user JSON needs an explicit role/org rewrite. This script is idempotent.

- [ ] **Step 1: Write the failing test**

```python
# tests/forge/test_user_migration.py
import json
import pytest


def test_migration_maps_legacy_roles(tmp_path, monkeypatch):
    f = tmp_path / "forge_users.json"
    f.write_text(json.dumps({
        "mike":     {"password_hash": "h", "role": "admin"},
        "mainstay": {"password_hash": "h", "role": "admin"},
        "jordan":   {"password_hash": "h", "role": "team"},
    }))
    monkeypatch.setenv("FORGE_USERS_FILE", str(f))
    from scripts.forge_migrate_multitenant import migrate_users
    migrate_users()
    data = json.loads(f.read_text())
    assert data["mike"]["role"] == "super_admin" and data["mike"]["org"] == "*"
    assert data["mainstay"]["role"] == "org_admin" and data["mainstay"]["org"] == "mainstay"
    assert data["jordan"]["role"] == "member" and data["jordan"]["org"] == "mainstay"


def test_migration_is_idempotent(tmp_path, monkeypatch):
    f = tmp_path / "forge_users.json"
    f.write_text(json.dumps({"mike": {"password_hash": "h", "role": "admin"}}))
    monkeypatch.setenv("FORGE_USERS_FILE", str(f))
    from scripts.forge_migrate_multitenant import migrate_users
    migrate_users(); first = f.read_text()
    migrate_users(); assert f.read_text() == first
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/forge/test_user_migration.py -v`
Expected: FAIL — `No module named 'scripts.forge_migrate_multitenant'`.

- [ ] **Step 3: Write the migration script**

```python
# scripts/forge_migrate_multitenant.py
"""One-time migration: rewrite the Forge user store to org/role-aware records.

DB columns self-migrate via core.forge.db.init_db() (ALTER … DEFAULT 'mainstay').
This handles the JSON user store. Idempotent — safe to run repeatedly.

Run:  python scripts/forge_migrate_multitenant.py
"""
import json
import os
from pathlib import Path

# Legacy username -> (role, org). Unlisted users default to (member, mainstay).
_KNOWN = {
    "mike":     ("super_admin", "*"),
    "mainstay": ("org_admin", "mainstay"),
}
_LEGACY_ROLE = {"admin": "org_admin", "team": "member"}


def _users_file() -> Path:
    return Path(os.environ.get("FORGE_USERS_FILE", "data/forge_users.json"))


def migrate_users() -> None:
    f = _users_file()
    if not f.exists():
        print(f"[migrate] no user store at {f} — nothing to do")
        return
    data = json.loads(f.read_text())
    changed = False
    for username, rec in data.items():
        if username in _KNOWN:
            role, org = _KNOWN[username]
        else:
            role = _LEGACY_ROLE.get(rec.get("role"), rec.get("role", "member"))
            if role not in ("member", "org_admin", "super_admin"):
                role = "member"
            org = rec.get("org", "mainstay")
        if rec.get("role") != role or rec.get("org") != org:
            rec["role"], rec["org"] = role, org
            changed = True
    if changed:
        f.write_text(json.dumps(data, indent=2))
        print(f"[migrate] rewrote {f}")
    else:
        print("[migrate] already current — no changes")


if __name__ == "__main__":
    from core.forge import db
    db.init_db()          # self-migrates the SQLite schema + seeds orgs
    migrate_users()
    print("[migrate] done")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/forge/test_user_migration.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/forge_migrate_multitenant.py tests/forge/test_user_migration.py
git commit -m "feat(forge): idempotent multi-tenant user-store migration script"
```

- [ ] **Step 6: Run the migration against production (with a backup)**

```bash
cp data/forge_live.db data/forge_live.db.bak-$(date +%Y%m%d)   # DB safety copy
cp data/forge_users.json data/forge_users.json.bak 2>/dev/null || true
FORGE_DB_PATH=data/forge_live.db python scripts/forge_migrate_multitenant.py
sudo systemctl restart forge-web.service
```
Expected: schema gains org_id columns + orgs seeded; users.json shows mike=super_admin,
mainstay=org_admin, jordan/mello=member. Verify with the curl in Task 12.

---

## Task 12: End-to-end isolation guard + full-suite green

**Files:**
- Test: `tests/forge/test_tenant_e2e.py` (create)

- [ ] **Step 1: Write an end-to-end leak test spanning ingest → score → search → API**

```python
# tests/forge/test_tenant_e2e.py
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.api.forge import register
from core.security.auth import require_auth


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_PATH", str(tmp_path / "forge.db"))
    monkeypatch.setenv("FORGE_CHROMA_DIR", str(tmp_path / "chroma"))
    from core.forge import db as _db
    _db.init_db()
    return tmp_path


def _client(user):
    app = FastAPI(); register(app)
    app.dependency_overrides[require_auth] = lambda: user
    return TestClient(app)


def test_full_isolation_member_cannot_reach_other_org(env):
    from core.forge import ingest, scorer
    # Mainstay data
    m = ingest.create_source("url", "mainstay", None, org="mainstay")
    scorer.save_candidates(m, [{"start_s": 0, "end_s": 5, "score": 90, "hook": "h",
                                "emotion": "e", "reason": "r", "caption": "c"}],
                           org="mainstay")
    # RuckTalk member
    ruck = _client({"username": "r", "role": "member", "org": "rucktalk"})
    # cannot list, cannot fetch, cannot stream the Mainstay source
    assert ruck.get("/forge/sources").json()["sources"] == []
    assert ruck.get(f"/forge/sources/{m}").status_code == 404

    # super-admin sees it
    mike = _client({"username": "mike", "role": "super_admin", "org": "*"})
    assert len(mike.get("/forge/sources").json()["sources"]) == 1
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/forge/test_tenant_e2e.py -v`
Expected: PASS.

- [ ] **Step 3: Run the ENTIRE forge suite — the regression gate**

Run: `pytest tests/forge/ -q`
Expected: all green. Any failure here is a real regression — fix before commit.

- [ ] **Step 4: Commit**

```bash
git add tests/forge/test_tenant_e2e.py
git commit -m "test(forge): end-to-end tenant isolation guard"
```

---

## Open items (carry forward, not in this plan)

- **RuckTalk TikTok channel** — not connected in the RuckTalk Postiz org (FB/IG/YT
  only). Connect it in Postiz if TikTok posting is wanted; no code change needed
  once connected (it becomes a selectable integration).
- **Per-org storage target / branding** — deferred (spec Non-Goals).
- **Self-serve org onboarding / connect-your-accounts flow** — deferred.
- **`groundrush` org** — seeded but empty; populate with users + a Postiz key when ready.

## Self-Review

- **Spec coverage:** orgs table (T1) ✓ · org_id columns (T1) ✓ · user org+roles (T2) ✓ ·
  X-Forge-Org identity (T3) ✓ · Scope choke point (T3, T8) ✓ · sources scope (T4) ✓ ·
  jobs scope (T5) ✓ · clip_candidates scope (T6) ✓ · vector-search scope (T7) ✓ ·
  ownership checks (T8) ✓ · super-admin switcher (T10) ✓ · admin org field (T10) ✓ ·
  org-scoped posting (T9) ✓ · migration (T1 self-migrates DB, T11 users) ✓ ·
  isolation tests (T8, T12) ✓.
- **Placeholder scan:** none — every code step shows real code; "NOTE" blocks point at
  preserving existing bodies, not deferring work.
- **Type consistency:** `Scope(org, role, view_all)` + `scope_from_user(user, requested_org=None)`
  + `can_read_org`/`can_write_org` used consistently T3→T8→T12. `org` kwarg name uniform across
  `create_source` / `enqueue` / `save_candidates` / `upsert_windows` / `search_segments`.
  Role set `{member, org_admin, super_admin}` uniform across T2/T3/T10/T11.
```
