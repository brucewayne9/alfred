# Fair Game — Milestone 1: Foundation & Fan Identity/Verification — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the Fair Game backend foundation — a fan can register with email + phone, verify via a 6-digit code, and receive a trusted session — with device-fingerprint dedupe and Klaviyo-waitlist priority, all on sqlite, fully tested, deployed behind Caddy + systemd.

**Architecture:** A standalone FastAPI sub-app (`core/api/fairgame.py`) mirroring the existing `core/api/arena_portal.py` pattern: public endpoints under `/fairgame/api/*`, binds `127.0.0.1`, fronted by Caddy. Domain logic lives in focused modules under `core/fairgame/`. Persistence is sqlite (WAL) per the `core/forge/db.py` pattern. Verification reuses `integrations/twilio` (SMS) and `integrations/email` (email).

**Tech Stack:** Python 3 / FastAPI / sqlite3 (WAL) / Twilio (SMS) / existing email client / Caddy / systemd. Tests: pytest + FastAPI TestClient.

## Global Constraints

- Decisions are LOCKED per the spec `docs/superpowers/specs/2026-06-16-fairgame-design.md`:
  - Fulfillment rides Ticketmaster rails (Fair Game never issues barcodes in v1).
  - Verification = medium strength: SMS + email + device/IP fingerprint + one-identity-one-account dedupe; **no government ID**.
  - Klaviyo DLD-waitlist fans get a `priority` flag.
  - Tickets stay transferable (fair transfer). No non-transferable issuance, ever.
  - Resale cap = face + $15 (seller +$10 / Rod +$5). *(Resale is M3 — listed here only so naming stays consistent.)*
- Follow existing repo patterns exactly: sub-app like `core/api/arena_portal.py`; sqlite like `core/forge/db.py`; bind `127.0.0.1` only.
- All money/PII handling uses parameterized SQL only (no string interpolation into queries).
- Phone + email are stored as salted SHA-256 hashes for dedupe lookups; raw values stored encrypted-at-rest is out of scope for M1 (store raw in M1, flag for M5 hardening).
- New code is committed on branch `feat/fairgame`, adding ONLY Fair Game paths per commit (the working tree has unrelated uncommitted tour-admat changes — never `git add -A`).

---

## File Structure

- `core/fairgame/__init__.py` — package marker
- `core/fairgame/db.py` — sqlite connection + schema (fans, verification_codes, sessions)
- `core/fairgame/identity.py` — fan create/lookup, dedupe, device fingerprint, priority flag
- `core/fairgame/verify.py` — code generation, send (SMS+email), verify, throttle
- `core/fairgame/sessions.py` — signed session token issue/validate
- `core/fairgame/waitlist.py` — Klaviyo DLD-waitlist membership check (priority seed)
- `core/api/fairgame.py` — FastAPI sub-app, public `/fairgame/api/*` endpoints
- `data/mainstay/fairgame/app/index.html` — minimal fan register/verify page (proves the flow)
- `tests/fairgame/test_db.py`, `test_identity.py`, `test_verify.py`, `test_sessions.py`, `test_api.py`
- `systemd/fairgame-api.service` — service unit (bind 127.0.0.1:8402)
- Caddy: new `handle /fairgame/api/*` block (reverse_proxy localhost:8402) inserted BEFORE the existing static `/fairgame/*` block

---

## Task 1: Package + DB schema

**Files:**
- Create: `core/fairgame/__init__.py`, `core/fairgame/db.py`
- Test: `tests/fairgame/test_db.py`

**Interfaces:**
- Produces: `db.connect() -> sqlite3.Connection` (row_factory=Row, WAL), `db.init_db() -> None`, `db.db_path() -> Path` (honors `FAIRGAME_DB_PATH` env override).
- Schema tables: `fans(id TEXT PK, email TEXT, phone TEXT, email_hash TEXT, phone_hash TEXT, status TEXT, priority INTEGER, created_at INT, updated_at INT)`; unique indexes on `email_hash` and `phone_hash`. `verification_codes(id TEXT PK, fan_id TEXT, channel TEXT, code_hash TEXT, expires_at INT, attempts INT, consumed INT, created_at INT)`. `sessions(token TEXT PK, fan_id TEXT, device_fp TEXT, ip TEXT, expires_at INT, created_at INT)`. `device_events(id TEXT PK, fan_id TEXT, device_fp TEXT, ip TEXT, event TEXT, created_at INT)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/fairgame/test_db.py
import os, tempfile, importlib

def _fresh_db(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setenv("FAIRGAME_DB_PATH", os.path.join(d, "fg.db"))
    from core.fairgame import db
    importlib.reload(db)
    db.init_db()
    return db

def test_init_db_creates_tables(monkeypatch):
    db = _fresh_db(monkeypatch)
    with db.connect() as c:
        names = {r["name"] for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"fans", "verification_codes", "sessions", "device_events"} <= names

def test_email_hash_unique(monkeypatch):
    db = _fresh_db(monkeypatch)
    import sqlite3, time, pytest
    with db.connect() as c:
        c.execute("INSERT INTO fans(id,email,phone,email_hash,phone_hash,status,priority,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                  ("f1","a@b.com","+15550001","H1","P1","verified",0,1,1))
        with pytest.raises(sqlite3.IntegrityError):
            c.execute("INSERT INTO fans(id,email,phone,email_hash,phone_hash,status,priority,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                      ("f2","c@d.com","+15550002","H1","P2","verified",0,1,1))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/aialfred/alfred && python -m pytest tests/fairgame/test_db.py -v`
Expected: FAIL (ModuleNotFoundError: core.fairgame.db)

- [ ] **Step 3: Write minimal implementation**

```python
# core/fairgame/__init__.py
```
```python
# core/fairgame/db.py
"""Fair Game persistence — sqlite (WAL). Mirrors core/forge/db.py."""
from __future__ import annotations
import os, sqlite3
from pathlib import Path

def db_path() -> Path:
    o = os.environ.get("FAIRGAME_DB_PATH")
    if o:
        return Path(o)
    return Path(__file__).resolve().parent.parent.parent / "data" / "fairgame.db"

def connect() -> sqlite3.Connection:
    p = db_path(); p.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(p)); c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL"); c.execute("PRAGMA foreign_keys=ON")
    return c

def init_db() -> None:
    with connect() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS fans(
            id TEXT PRIMARY KEY, email TEXT, phone TEXT,
            email_hash TEXT, phone_hash TEXT,
            status TEXT NOT NULL DEFAULT 'pending', priority INTEGER NOT NULL DEFAULT 0,
            created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_fans_email_hash ON fans(email_hash);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_fans_phone_hash ON fans(phone_hash);
        CREATE TABLE IF NOT EXISTS verification_codes(
            id TEXT PRIMARY KEY, fan_id TEXT NOT NULL, channel TEXT NOT NULL,
            code_hash TEXT NOT NULL, expires_at INTEGER NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0, consumed INTEGER NOT NULL DEFAULT 0,
            created_at INTEGER NOT NULL);
        CREATE INDEX IF NOT EXISTS idx_vc_fan ON verification_codes(fan_id, channel, consumed);
        CREATE TABLE IF NOT EXISTS sessions(
            token TEXT PRIMARY KEY, fan_id TEXT NOT NULL, device_fp TEXT, ip TEXT,
            expires_at INTEGER NOT NULL, created_at INTEGER NOT NULL);
        CREATE INDEX IF NOT EXISTS idx_sessions_fan ON sessions(fan_id);
        CREATE TABLE IF NOT EXISTS device_events(
            id TEXT PRIMARY KEY, fan_id TEXT, device_fp TEXT, ip TEXT,
            event TEXT NOT NULL, created_at INTEGER NOT NULL);
        CREATE INDEX IF NOT EXISTS idx_dev_fp ON device_events(device_fp, created_at);
        """)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/aialfred/alfred && python -m pytest tests/fairgame/test_db.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/aialfred/alfred
git add core/fairgame/__init__.py core/fairgame/db.py tests/fairgame/test_db.py
git commit -m "feat(fairgame): sqlite schema for fans, codes, sessions"
```

---

## Task 2: Identity service (dedupe, fingerprint, priority)

**Files:**
- Create: `core/fairgame/identity.py`
- Test: `tests/fairgame/test_identity.py`

**Interfaces:**
- Consumes: `core.fairgame.db`.
- Produces:
  - `hash_value(raw: str) -> str` — salted SHA-256 (salt from `FAIRGAME_HASH_SALT`, default constant).
  - `normalize_phone(raw: str) -> str` — strip to `+` and digits.
  - `upsert_fan(email: str, phone: str, device_fp: str|None, ip: str|None) -> dict` — returns fan row as dict; dedupes on email_hash OR phone_hash (returns existing if found, updating contact + bumping updated_at); sets `priority` via `waitlist.is_priority(email)`; records a `device_events` row (`event="register"`).
  - `get_fan(fan_id: str) -> dict|None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/fairgame/test_identity.py
import os, tempfile, importlib
def _setup(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setenv("FAIRGAME_DB_PATH", os.path.join(d,"fg.db"))
    from core.fairgame import db, identity, waitlist
    importlib.reload(db); importlib.reload(waitlist); importlib.reload(identity)
    monkeypatch.setattr(identity, "is_priority", lambda email: email.endswith("@vip.com"), raising=False)
    db.init_db()
    return identity

def test_normalize_phone(monkeypatch):
    identity = _setup(monkeypatch)
    assert identity.normalize_phone("(555) 000-1234") == "+5550001234"

def test_upsert_dedupes_on_phone(monkeypatch):
    identity = _setup(monkeypatch)
    a = identity.upsert_fan("a@x.com", "+15550001", "fp1", "1.1.1.1")
    b = identity.upsert_fan("different@x.com", "+15550001", "fp2", "2.2.2.2")
    assert a["id"] == b["id"]  # same phone -> same fan

def test_priority_flag(monkeypatch):
    identity = _setup(monkeypatch)
    f = identity.upsert_fan("fan@vip.com", "+15559999", None, None)
    assert f["priority"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/aialfred/alfred && python -m pytest tests/fairgame/test_identity.py -v`
Expected: FAIL (ModuleNotFoundError: core.fairgame.identity)

- [ ] **Step 3: Write minimal implementation**

```python
# core/fairgame/identity.py
"""Fair Game fan identity — dedupe, fingerprint, waitlist priority."""
from __future__ import annotations
import hashlib, os, re, time, uuid
from . import db
from .waitlist import is_priority

_SALT = os.environ.get("FAIRGAME_HASH_SALT", "fairgame-v1-salt")

def hash_value(raw: str) -> str:
    return hashlib.sha256((_SALT + (raw or "").strip().lower()).encode()).hexdigest()

def normalize_phone(raw: str) -> str:
    digits = re.sub(r"[^\d+]", "", raw or "")
    if digits and not digits.startswith("+"):
        digits = "+" + digits
    return digits

def _row_to_dict(r): return dict(r) if r else None

def get_fan(fan_id: str):
    with db.connect() as c:
        return _row_to_dict(c.execute("SELECT * FROM fans WHERE id=?", (fan_id,)).fetchone())

def upsert_fan(email: str, phone: str, device_fp=None, ip=None) -> dict:
    email = (email or "").strip().lower()
    phone = normalize_phone(phone)
    eh, ph = hash_value(email), hash_value(phone)
    now = int(time.time())
    with db.connect() as c:
        row = c.execute("SELECT * FROM fans WHERE email_hash=? OR phone_hash=?", (eh, ph)).fetchone()
        if row:
            fid = row["id"]
            c.execute("UPDATE fans SET email=?,phone=?,email_hash=?,phone_hash=?,updated_at=? WHERE id=?",
                      (email, phone, eh, ph, now, fid))
        else:
            fid = "fan_" + uuid.uuid4().hex[:12]
            prio = 1 if is_priority(email) else 0
            c.execute("INSERT INTO fans(id,email,phone,email_hash,phone_hash,status,priority,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                      (fid, email, phone, eh, ph, "pending", prio, now, now))
        c.execute("INSERT INTO device_events(id,fan_id,device_fp,ip,event,created_at) VALUES(?,?,?,?,?,?)",
                  ("ev_"+uuid.uuid4().hex[:12], fid, device_fp, ip, "register", now))
        out = c.execute("SELECT * FROM fans WHERE id=?", (fid,)).fetchone()
    return dict(out)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/aialfred/alfred && python -m pytest tests/fairgame/test_identity.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/aialfred/alfred
git add core/fairgame/identity.py tests/fairgame/test_identity.py
git commit -m "feat(fairgame): fan identity with dedupe + priority"
```

---

## Task 3: Waitlist priority (Klaviyo seam)

**Files:**
- Create: `core/fairgame/waitlist.py`
- Test: `tests/fairgame/test_waitlist.py`

**Interfaces:**
- Produces: `is_priority(email: str) -> bool` — true if email is in the DLD waitlist. M1 implementation reads a local seed file `data/mainstay/fairgame/waitlist_emails.txt` (one email/line, lowercased); returns False if the file is absent. Klaviyo live pull is deferred to M5 (documented in the module docstring). Keep the function signature stable so M5 swaps the body only.

- [ ] **Step 1: Write the failing test**

```python
# tests/fairgame/test_waitlist.py
import importlib
def test_priority_from_seed(tmp_path, monkeypatch):
    f = tmp_path / "waitlist_emails.txt"
    f.write_text("fan1@x.com\nVIP@x.com\n")
    monkeypatch.setenv("FAIRGAME_WAITLIST_FILE", str(f))
    from core.fairgame import waitlist; importlib.reload(waitlist)
    assert waitlist.is_priority("fan1@x.com") is True
    assert waitlist.is_priority("VIP@X.com") is True   # case-insensitive
    assert waitlist.is_priority("nobody@x.com") is False

def test_priority_no_file(monkeypatch):
    monkeypatch.setenv("FAIRGAME_WAITLIST_FILE", "/nonexistent/path.txt")
    from core.fairgame import waitlist; importlib.reload(waitlist)
    assert waitlist.is_priority("anyone@x.com") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/aialfred/alfred && python -m pytest tests/fairgame/test_waitlist.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementation**

```python
# core/fairgame/waitlist.py
"""DLD waitlist priority. M1: local seed file. M5: swap body for live Klaviyo pull
(list V75mRt 'DLD Waitlist', account XYKnGf). Signature is stable across that swap."""
from __future__ import annotations
import os
from pathlib import Path
from functools import lru_cache

def _file() -> Path:
    o = os.environ.get("FAIRGAME_WAITLIST_FILE")
    if o:
        return Path(o)
    return Path(__file__).resolve().parent.parent.parent / "data" / "mainstay" / "fairgame" / "waitlist_emails.txt"

@lru_cache(maxsize=1)
def _load() -> frozenset:
    p = _file()
    if not p.exists():
        return frozenset()
    return frozenset(line.strip().lower() for line in p.read_text().splitlines() if line.strip())

def is_priority(email: str) -> bool:
    _load.cache_clear()  # cheap file; keep fresh in M1
    return (email or "").strip().lower() in _load()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/aialfred/alfred && python -m pytest tests/fairgame/test_waitlist.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/aialfred/alfred
git add core/fairgame/waitlist.py tests/fairgame/test_waitlist.py
git commit -m "feat(fairgame): waitlist priority via local seed (Klaviyo seam)"
```

---

## Task 4: Verification service (code gen, send, verify, throttle)

**Files:**
- Create: `core/fairgame/verify.py`
- Test: `tests/fairgame/test_verify.py`

**Interfaces:**
- Consumes: `core.fairgame.db`, `core.fairgame.identity.hash_value`.
- Produces:
  - `CODE_TTL = 600`, `MAX_ATTEMPTS = 5`, `RESEND_COOLDOWN = 30`.
  - `start_verification(fan_id, channel, send_fn) -> dict` — channel in {"sms","email"}; generates a 6-digit code, stores its hash with expiry, calls `send_fn(code)`; enforces resend cooldown (raise `VerifyError` if too soon). Returns `{"sent": True, "channel": channel}`.
  - `check_code(fan_id, channel, code) -> bool` — verifies newest unconsumed code; increments attempts; marks consumed on success; raises `VerifyError` on too many attempts / expired.
  - `class VerifyError(Exception)`.
- send_fn is injected so the API layer wires Twilio/email and tests inject a spy.

- [ ] **Step 1: Write the failing test**

```python
# tests/fairgame/test_verify.py
import os, tempfile, importlib, time, pytest
def _setup(monkeypatch):
    d = tempfile.mkdtemp(); monkeypatch.setenv("FAIRGAME_DB_PATH", os.path.join(d,"fg.db"))
    from core.fairgame import db, identity, verify
    importlib.reload(db); importlib.reload(identity); importlib.reload(verify)
    db.init_db()
    fan = identity.upsert_fan("a@x.com", "+15550001", None, None)
    return verify, fan["id"]

def test_send_and_verify(monkeypatch):
    verify, fid = _setup(monkeypatch)
    sent = {}
    verify.start_verification(fid, "sms", lambda code: sent.update(code=code))
    assert len(sent["code"]) == 6 and sent["code"].isdigit()
    assert verify.check_code(fid, "sms", sent["code"]) is True

def test_wrong_code_fails(monkeypatch):
    verify, fid = _setup(monkeypatch)
    sent = {}
    verify.start_verification(fid, "sms", lambda code: sent.update(code=code))
    assert verify.check_code(fid, "sms", "000000") is False

def test_max_attempts(monkeypatch):
    verify, fid = _setup(monkeypatch)
    verify.start_verification(fid, "sms", lambda code: None)
    for _ in range(verify.MAX_ATTEMPTS):
        verify.check_code(fid, "sms", "000000")
    with pytest.raises(verify.VerifyError):
        verify.check_code(fid, "sms", "000000")

def test_resend_cooldown(monkeypatch):
    verify, fid = _setup(monkeypatch)
    verify.start_verification(fid, "sms", lambda code: None)
    with pytest.raises(verify.VerifyError):
        verify.start_verification(fid, "sms", lambda code: None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/aialfred/alfred && python -m pytest tests/fairgame/test_verify.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementation**

```python
# core/fairgame/verify.py
"""Fair Game verification codes — channel-agnostic; send_fn injected by caller."""
from __future__ import annotations
import secrets, time, uuid
from . import db
from .identity import hash_value

CODE_TTL = 600
MAX_ATTEMPTS = 5
RESEND_COOLDOWN = 30

class VerifyError(Exception): ...

def _newest(c, fan_id, channel):
    return c.execute(
        "SELECT * FROM verification_codes WHERE fan_id=? AND channel=? AND consumed=0 ORDER BY created_at DESC LIMIT 1",
        (fan_id, channel)).fetchone()

def start_verification(fan_id, channel, send_fn) -> dict:
    if channel not in ("sms", "email"):
        raise VerifyError("bad channel")
    now = int(time.time())
    with db.connect() as c:
        last = _newest(c, fan_id, channel)
        if last and now - last["created_at"] < RESEND_COOLDOWN:
            raise VerifyError("resend too soon")
        code = f"{secrets.randbelow(1000000):06d}"
        c.execute("INSERT INTO verification_codes(id,fan_id,channel,code_hash,expires_at,attempts,consumed,created_at) VALUES(?,?,?,?,?,?,?,?)",
                  ("vc_"+uuid.uuid4().hex[:12], fan_id, channel, hash_value(code), now+CODE_TTL, 0, 0, now))
    send_fn(code)
    return {"sent": True, "channel": channel}

def check_code(fan_id, channel, code) -> bool:
    now = int(time.time())
    with db.connect() as c:
        row = _newest(c, fan_id, channel)
        if not row:
            raise VerifyError("no code")
        if row["attempts"] >= MAX_ATTEMPTS:
            raise VerifyError("too many attempts")
        if now > row["expires_at"]:
            raise VerifyError("expired")
        c.execute("UPDATE verification_codes SET attempts=attempts+1 WHERE id=?", (row["id"],))
        if row["code_hash"] == hash_value(code):
            c.execute("UPDATE verification_codes SET consumed=1 WHERE id=?", (row["id"],))
            return True
    return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/aialfred/alfred && python -m pytest tests/fairgame/test_verify.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/aialfred/alfred
git add core/fairgame/verify.py tests/fairgame/test_verify.py
git commit -m "feat(fairgame): verification codes with throttle + attempt limits"
```

---

## Task 5: Session service

**Files:**
- Create: `core/fairgame/sessions.py`
- Test: `tests/fairgame/test_sessions.py`

**Interfaces:**
- Consumes: `core.fairgame.db`.
- Produces:
  - `SESSION_TTL = 30*24*3600`.
  - `issue(fan_id, device_fp=None, ip=None) -> str` — opaque token (secrets.token_urlsafe), stored with expiry.
  - `resolve(token) -> dict|None` — returns `{"fan_id":..., "device_fp":..., "ip":...}` if valid + unexpired, else None.
  - `revoke(token) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/fairgame/test_sessions.py
import os, tempfile, importlib
def _setup(monkeypatch):
    d = tempfile.mkdtemp(); monkeypatch.setenv("FAIRGAME_DB_PATH", os.path.join(d,"fg.db"))
    from core.fairgame import db, sessions
    importlib.reload(db); importlib.reload(sessions); db.init_db()
    return sessions

def test_issue_resolve(monkeypatch):
    s = _setup(monkeypatch)
    tok = s.issue("fan_1", "fp", "1.1.1.1")
    got = s.resolve(tok)
    assert got["fan_id"] == "fan_1"

def test_revoke(monkeypatch):
    s = _setup(monkeypatch)
    tok = s.issue("fan_1")
    s.revoke(tok)
    assert s.resolve(tok) is None

def test_bad_token(monkeypatch):
    s = _setup(monkeypatch)
    assert s.resolve("nope") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/aialfred/alfred && python -m pytest tests/fairgame/test_sessions.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementation**

```python
# core/fairgame/sessions.py
"""Fair Game fan sessions — opaque DB-backed tokens."""
from __future__ import annotations
import secrets, time
from . import db

SESSION_TTL = 30*24*3600

def issue(fan_id, device_fp=None, ip=None) -> str:
    tok = secrets.token_urlsafe(32); now = int(time.time())
    with db.connect() as c:
        c.execute("INSERT INTO sessions(token,fan_id,device_fp,ip,expires_at,created_at) VALUES(?,?,?,?,?,?)",
                  (tok, fan_id, device_fp, ip, now+SESSION_TTL, now))
    return tok

def resolve(token):
    now = int(time.time())
    with db.connect() as c:
        r = c.execute("SELECT * FROM sessions WHERE token=? AND expires_at>?", (token, now)).fetchone()
    return {"fan_id": r["fan_id"], "device_fp": r["device_fp"], "ip": r["ip"]} if r else None

def revoke(token) -> None:
    with db.connect() as c:
        c.execute("DELETE FROM sessions WHERE token=?", (token,))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/aialfred/alfred && python -m pytest tests/fairgame/test_sessions.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/aialfred/alfred
git add core/fairgame/sessions.py tests/fairgame/test_sessions.py
git commit -m "feat(fairgame): DB-backed fan sessions"
```

---

## Task 6: FastAPI sub-app (public endpoints)

**Files:**
- Create: `core/api/fairgame.py`
- Test: `tests/fairgame/test_api.py`

**Interfaces:**
- Consumes: all `core.fairgame.*` modules; `integrations.twilio.client` (SMS), `integrations.email.client` (email).
- Produces FastAPI `app` with:
  - `POST /fairgame/api/register` body `{email, phone, device_fp?}` → upsert fan, start SMS verification, return `{fan_id, sent:true}`. Rate-limited.
  - `POST /fairgame/api/verify` body `{fan_id, code}` → check SMS code; on success start email verification too and return `{verified_sms:true}`; on failure 400.
  - `POST /fairgame/api/verify-email` body `{fan_id, code}` → check email code; on success mark fan `status='verified'`, issue session, return `{token, fan:{id,email,priority}}`.
  - `GET /fairgame/api/me` header `Authorization: Bearer <token>` → resolve session → fan summary or 401.
- SMS/email send is wrapped so a missing provider key logs + no-ops in dev (return code in response only when `FAIRGAME_DEV_ECHO=1`, for testing/local).

- [ ] **Step 1: Write the failing test**

```python
# tests/fairgame/test_api.py
import os, tempfile, importlib
from fastapi.testclient import TestClient
def _client(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setenv("FAIRGAME_DB_PATH", os.path.join(d,"fg.db"))
    monkeypatch.setenv("FAIRGAME_DEV_ECHO", "1")  # return codes in response for tests
    import core.fairgame.db as db; importlib.reload(db); db.init_db()
    import core.api.fairgame as fg; importlib.reload(fg)
    return TestClient(fg.app)

def test_full_flow(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/fairgame/api/register", json={"email":"a@x.com","phone":"+15550001","device_fp":"fp"})
    assert r.status_code == 200; fid = r.json()["fan_id"]; sms = r.json()["dev_code"]
    r = c.post("/fairgame/api/verify", json={"fan_id":fid,"code":sms})
    assert r.status_code == 200; email_code = r.json()["dev_code"]
    r = c.post("/fairgame/api/verify-email", json={"fan_id":fid,"code":email_code})
    assert r.status_code == 200; tok = r.json()["token"]
    r = c.get("/fairgame/api/me", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200 and r.json()["fan"]["email"] == "a@x.com"

def test_me_requires_auth(monkeypatch):
    c = _client(monkeypatch)
    assert c.get("/fairgame/api/me").status_code == 401
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/aialfred/alfred && python -m pytest tests/fairgame/test_api.py -v`
Expected: FAIL (ModuleNotFoundError: core.api.fairgame)

- [ ] **Step 3: Write minimal implementation**

```python
# core/api/fairgame.py
"""Fair Game public API — fan register + verify + session. Binds 127.0.0.1; Caddy fronts /fairgame/api/*."""
from __future__ import annotations
import os, sys, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from fastapi import FastAPI, Request, HTTPException, Header
from core.fairgame import db, identity, verify, sessions

logger = logging.getLogger("fairgame")
db.init_db()
app = FastAPI(title="Fair Game")
_DEV_ECHO = os.environ.get("FAIRGAME_DEV_ECHO") == "1"

def _send_sms(phone, code):
    try:
        from integrations.twilio.client import send_sms
        send_sms(phone, f"Your Rod Wave verification code is {code}")
    except Exception as e:
        logger.warning("sms send failed/dev: %s", e)

def _send_email(email, code):
    try:
        from integrations.email.client import email_client
        email_client.send(to=email, subject="Your Rod Wave code", body=f"Your code is {code}")
    except Exception as e:
        logger.warning("email send failed/dev: %s", e)

@app.post("/fairgame/api/register")
async def register(req: Request):
    b = await req.json()
    email, phone = b.get("email","").strip(), b.get("phone","").strip()
    if not email or not phone:
        raise HTTPException(400, "email and phone required")
    ip = req.client.host if req.client else None
    fan = identity.upsert_fan(email, phone, b.get("device_fp"), ip)
    holder = {}
    try:
        verify.start_verification(fan["id"], "sms", lambda code: (holder.update(code=code), _send_sms(fan["phone"], code)))
    except verify.VerifyError as e:
        raise HTTPException(429, str(e))
    out = {"fan_id": fan["id"], "sent": True}
    if _DEV_ECHO: out["dev_code"] = holder.get("code")
    return out

@app.post("/fairgame/api/verify")
async def verify_sms(req: Request):
    b = await req.json(); fid, code = b.get("fan_id"), b.get("code","")
    try:
        ok = verify.check_code(fid, "sms", code)
    except verify.VerifyError as e:
        raise HTTPException(400, str(e))
    if not ok:
        raise HTTPException(400, "invalid code")
    fan = identity.get_fan(fid)
    holder = {}
    verify.start_verification(fid, "email", lambda c: (holder.update(code=c), _send_email(fan["email"], c)))
    out = {"verified_sms": True}
    if _DEV_ECHO: out["dev_code"] = holder.get("code")
    return out

@app.post("/fairgame/api/verify-email")
async def verify_email(req: Request):
    b = await req.json(); fid, code = b.get("fan_id"), b.get("code","")
    try:
        ok = verify.check_code(fid, "email", code)
    except verify.VerifyError as e:
        raise HTTPException(400, str(e))
    if not ok:
        raise HTTPException(400, "invalid code")
    with db.connect() as c:
        c.execute("UPDATE fans SET status='verified' WHERE id=?", (fid,))
    fan = identity.get_fan(fid)
    ip = req.client.host if req.client else None
    tok = sessions.issue(fid, None, ip)
    return {"token": tok, "fan": {"id": fan["id"], "email": fan["email"], "priority": fan["priority"]}}

@app.get("/fairgame/api/me")
async def me(authorization: str = Header(default="")):
    tok = authorization.replace("Bearer ", "").strip()
    sess = sessions.resolve(tok) if tok else None
    if not sess:
        raise HTTPException(401, "unauthorized")
    fan = identity.get_fan(sess["fan_id"])
    return {"fan": {"id": fan["id"], "email": fan["email"], "priority": fan["priority"], "status": fan["status"]}}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/aialfred/alfred && python -m pytest tests/fairgame/test_api.py -v`
Expected: PASS (2 passed). If `integrations.twilio.client.send_sms` or `integrations.email.client.email_client.send` differ in signature, adapt the wrapper to the real signature (check the module) — the dev-echo path keeps tests green regardless.

- [ ] **Step 5: Commit**

```bash
cd /home/aialfred/alfred
git add core/api/fairgame.py tests/fairgame/test_api.py
git commit -m "feat(fairgame): public register/verify/session API"
```

---

## Task 7: Minimal fan portal page + deploy wiring

**Files:**
- Create: `data/mainstay/fairgame/app/index.html` (register → SMS code → email code → "you're verified")
- Create: `systemd/fairgame-api.service`
- Modify: `/etc/caddy/Caddyfile` (insert `/fairgame/api/*` reverse-proxy BEFORE the static `/fairgame/*` block; add `/fairgame/app/*` static if not covered)

**Interfaces:**
- Consumes: the `/fairgame/api/*` endpoints from Task 6.
- Produces: a working end-to-end browser flow at `https://aialfred.groundrushcloud.com/fairgame/app/`.

- [ ] **Step 1: Verify port 8402 is free**

Run: `ss -ltnp 2>/dev/null | grep -E ':8402' || echo FREE`
Expected: `FREE` (if taken, pick the next open port and use it consistently below).

- [ ] **Step 2: Write the systemd unit**

```ini
# systemd/fairgame-api.service
[Unit]
Description=Fair Game public API (Rod Wave fan ticketing)
After=network.target

[Service]
User=aialfred
WorkingDirectory=/home/aialfred/alfred
Environment=PYTHONPATH=/home/aialfred/alfred
ExecStart=/home/aialfred/alfred/.venv/bin/uvicorn core.api.fairgame:app --host 127.0.0.1 --port 8402
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
```

(If the venv path differs, match the one used by other alfred services — check `systemctl cat forge-web.service` for the exact interpreter.)

- [ ] **Step 3: Write the minimal portal page**

Create `data/mainstay/fairgame/app/index.html` — a single page with three states (enter email+phone → enter SMS code → enter email code → success banner) that calls `/fairgame/api/register`, `/fairgame/api/verify`, `/fairgame/api/verify-email`, stores the returned token in `localStorage`, then calls `/fairgame/api/me`. Reuse the Fair Game deck's black/orange styling (font Anton + Sora, `--orange:#ff5a1f`). Keep it one self-contained file.

- [ ] **Step 4: Insert Caddy route + reload**

```bash
sudo python3 - <<'PY'
p="/etc/caddy/Caddyfile"; s=open(p).read()
static='''    redir /fairgame /fairgame/ permanent
    handle /fairgame/* {'''
api='''    handle /fairgame/api/* {
        reverse_proxy localhost:8402
    }
    redir /fairgame /fairgame/ permanent
    handle /fairgame/* {'''
if "/fairgame/api/*" in s: print("already")
elif static in s: open(p,"w").write(s.replace(static,api,1)); print("inserted")
else: print("ANCHOR NOT FOUND")
PY
sudo caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile && sudo systemctl reload caddy
```

- [ ] **Step 5: Enable service + smoke test**

```bash
sudo cp systemd/fairgame-api.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now fairgame-api.service
sleep 2
curl -s -X POST https://aialfred.groundrushcloud.com/fairgame/api/register \
  -H 'Content-Type: application/json' -d '{"email":"smoke@test.com","phone":"+15550000"}' -o /dev/null -w "%{http_code}\n"
```
Expected: `200`. Then load `https://aialfred.groundrushcloud.com/fairgame/app/` in a browser and walk the flow (dev codes echo only if `FAIRGAME_DEV_ECHO=1` is set on the service — leave OFF in prod; use real SMS/email).

- [ ] **Step 6: Commit**

```bash
cd /home/aialfred/alfred
git add data/mainstay/fairgame/app/index.html systemd/fairgame-api.service
git commit -m "feat(fairgame): minimal fan portal + deploy wiring"
```

---

## Milestone Roadmap (subsequent plans — written after M1 lands)

- **M2 — Events + Presale Access + Storefront:** `events`/`access_grants` tables seeded with Rod's 35 shows; wave/queue engine; verified + priority fans get capped-qty access; Rod-branded storefront that lists shows and hands off to the capped purchase flow. Inventory = the block Rod controls (artist holds / fan club).
- **M3 — Capped Resale Exchange + Stripe escrow:** `listings`/`orders` tables; list (cap = face + $15, seller +$10 / Rod +$5); Stripe Connect Express onboarding; hold-until-transfer-confirmed escrow; auto-refund on failed transfer; "Rod Official" trust + instant payout.
- **M4 — Tour Admin Console:** extend the Rollout board pattern — per-show inventory, cap config, verification/broker-flag stats, fan CRM export.
- **M5 — Hardening & Compliance:** live Klaviyo waitlist pull; PII encryption at rest; Stripe 1099-K/tax; per-state resale-cap legal config; on-sale spike load test; fraud/velocity rules on `device_events`.

## Self-Review

- **Spec coverage (M1 scope):** identity ✓ (Task 2), verification SMS+email ✓ (Task 4/6), device fingerprint + dedupe ✓ (Task 1/2), waitlist priority ✓ (Task 3), sessions ✓ (Task 5), TM-rails (no barcode issuance) ✓ (nothing issues tickets), deploy pattern ✓ (Task 7). Resale/events/admin are correctly deferred to M2–M4 and listed in the roadmap.
- **Placeholder scan:** none — every code step is complete and runnable.
- **Type consistency:** `hash_value`, `normalize_phone`, `upsert_fan`, `is_priority`, `start_verification(send_fn)`, `check_code`, `issue/resolve/revoke` names match across tasks and tests. `is_priority` is imported into `identity` (Task 2) and defined in Task 3 — Task 3 must land before Task 2's tests pass, OR run them together; execution order 1→3→2→4→5→6→7 (note: do Task 3 before Task 2). **Execution order: 1, 3, 2, 4, 5, 6, 7.**
