# Forge Multi-Tenancy — Design Spec

**Date:** 2026-06-13
**Status:** Approved for planning
**Author:** Alfred (with Mike Johnson)

## Problem

Forge is single-tenant. Today there is one shared surface: a per-person login
store exists, but every user effectively sees every source/clip/job, and all
posting goes through one hard-wired Postiz account (Mainstay). Mike wants Forge
to become a **user-based, multi-company platform**: each company (org) is a
tenant, users log in and see only their own org's work, and Mike (super-admin)
sees everything across all orgs. This turns Forge from "a tool we copy per
company" into "a platform we onboard companies onto" — and is strictly less work
than maintaining duplicate instances on separate servers.

## Goals

- Each user belongs to exactly **one org** and sees **all** of that org's work.
- Three roles: `member`, `org_admin`, `super_admin`.
- `super_admin` (Mike) sees/manages every org, with an org switcher
  ("All orgs" + each org by name).
- Hard data isolation between orgs — a Mainstay user cannot read or write
  RuckTalk data, and vice versa — enforced in the data layer and proven by tests.
- **Org-scoped posting** (in v1): a clip posts under its own org's Postiz
  account. Mainstay → Mainstay org; RuckTalk → RuckTalk/Ground Rush org.

## Non-Goals (YAGNI — deferred)

- **Users spanning multiple orgs.** One org per user. Super-admin is the only
  cross-org identity.
- **Self-serve "connect your accounts" onboarding flow** for brand-new orgs.
  v1 wires the two orgs whose Postiz already exists (Mainstay + RuckTalk). A
  general connect flow is a later phase.
- **Per-org storage target / branding / billing.** Storage stays as-is for now.
  Branding and storage become per-org in a later phase.
- **Strict per-channel allow-lists inside an org.** Mike confirmed everything in
  the RuckTalk org is his (RuckTalk + a burner), so org → Postiz key is the unit;
  channel choice happens at post time as today.

## Decision: Approach A — single DB, `org_id` columns, scoping in the data layer

Rejected alternatives:
- **B. Query-rewriting / per-org SQLite views** — SQLite has no real row-level
  security; fragile and surprising to debug.
- **C. One DB file per org** — good isolation but turns the super-admin
  "see everything merged" view into an N-database fan-out + merge, and runs every
  schema migration N times. Fights the headline requirement.

Approach A is the standard SaaS pattern and makes the cross-org super-admin view
a one-line difference (skip the `WHERE org_id` filter) instead of an architecture.

## Current state (verified 2026-06-13)

- **User store:** `core/forge/users.py` → `data/forge_users.json`,
  `{username: {password_hash (bcrypt), role}}`. Roles today: `admin` | `team`.
  Seeded: mike (admin), mainstay (admin), jordan (team), mello (team).
- **Identity flow:** Caddy `forward_auth` → `/forge/authcheck` validates the
  login and returns `X-Forge-User` + `X-Forge-Role`; Caddy copies them onto the
  proxied request. `serve.py::_forge_user()` reads them into `{username, role}`.
- **DB** (`data/forge_live.db`): `sources`, `jobs` (has `created_by`),
  `clip_candidates`, `transcript_segments`, `dist_posts`, `intel_*`, `trash`.
  No `org_id` anywhere. Everything hangs off a `source` via `source_id`.
- **Posting:** `core/forge/postiz_client.py` is hard-wired to
  `POSTIZ_MAINSTAY_API_KEY` and assumes the Mainstay org.
- **RuckTalk Postiz key:** verified live 2026-06-13 (HTTP 200, 10 channels).
  Stored as `POSTIZ_RUCKTALK_API_KEY` in `config/.env` (chmod 600). Confirmed
  connected: Ruck Talk Facebook, Instagram, YouTube. **No TikTok channel found**
  in that org — open item below.

## Design

### 1. Data model

**New `orgs` table:**
```
orgs(id TEXT PRIMARY KEY, name TEXT, created_at INTEGER)
```
Seed three rows: `mainstay` ("Mainstay Music Group"), `rucktalk` ("RuckTalk"),
`groundrush` ("Ground Rush"). Adding a customer later = one row.

**User store (`forge_users.json`) gains `org` and a new role vocabulary:**
```
{username: {password_hash, role, org}}
```
- Roles: `member` (sees all of own org), `org_admin` (same visibility + manages
  users in own org), `super_admin` (all orgs, all management).
- Migration of existing accounts:
  - `mike` → `super_admin`, org `*` (or null; treated as all-orgs)
  - `mainstay` → `org_admin`, org `mainstay`
  - `jordan`, `mello` → `member`, org `mainstay`

**`org_id` column added to the directly-listed tables:**
`sources`, `jobs`, `clip_candidates`, `dist_posts`.
`transcript_segments` is only ever fetched by `source_id`, so it inherits scope
for free (no column needed). Backfill all existing rows to `org_id = 'mainstay'`.

### 2. Identity flow

Add **one header**: `X-Forge-Org`. `/forge/authcheck` looks up the user's org
from the store at validation time and emits `X-Forge-User` + `X-Forge-Role` +
`X-Forge-Org`; Caddy copies all three. `serve.py::_forge_user()` returns
`{username, role, org}`. No new login screen, no Caddy reload, existing
credentials keep working.

### 3. Scope object + enforcement (the core work)

Introduce a small `Scope` value built once in the API layer from the identity:
```
Scope(org: str, role: str, view_all: bool)
```
- `view_all` is true only for `super_admin` who has selected "All orgs".
- Threaded down into every read/write in `core/forge`
  (`db.py`, `clips.py`, `search.py`, `jobs.py`, `scorer.py`, `ingest.py`).
- **Writes** stamp `org_id` from `Scope.org` — never from request input. A member
  physically cannot create rows in another org.
- **Reads** add `WHERE org_id = ?` unless `Scope.view_all`.
- Single choke point so org logic is in one place, not sprinkled as `if role ==`.

### 4. Super-admin org switcher

When a `super_admin` loads the dashboard, show an org dropdown: **"All orgs"**
(default — every org merged, each card badged with its org) + each org by name.
Selecting an org sets the viewing-org for that session (filters Forge to exactly
what that org sees). Members/org_admins never see the dropdown — pinned to their
org.

### 5. Org-scoped posting

Replace the single hard-wired key with an **org → Postiz config** map:
```
mainstay  → POSTIZ_MAINSTAY_API_KEY
rucktalk  → POSTIZ_RUCKTALK_API_KEY   (Ground Rush / RuckTalk org)
```
- `postiz_client.py` functions (`list_integrations`, `create_post`, `upload_media`,
  JWT helpers) take the org (via Scope) and select the right key/base.
- A clip posts under the Postiz account of **its own org**. Channel selection at
  post time is unchanged (user picks the integration).
- Orgs with no connected Postiz (e.g. a fresh `groundrush`) get data-isolation
  only — posting disabled with a clear "no account connected" state.

### 6. Admin panel + migration

- Existing user-admin panel gains an **org field** on user creation. `org_admin`
  may only create users in their own org; `super_admin` may create in any org and
  create new orgs.
- **One-time migration script** (idempotent, reversible): create `orgs` table +
  seed; add `org_id` columns; backfill existing rows → `mainstay`; rewrite the
  four users' roles/orgs in `forge_users.json`.

### 7. Testing — the part that matters most

Isolation tests are the guarantee that Mainstay can't see RuckTalk:
- A `member` of org A gets empty/404 on org B's sources, clips, jobs, and search —
  **reads and writes** both.
- A write that tries to claim another org_id in its payload is forced back to the
  viewer's org.
- `super_admin` + "All orgs" sees both; "view as org A" shows only A.
- Posting routes a RuckTalk clip to `POSTIZ_RUCKTALK_API_KEY` and a Mainstay clip
  to `POSTIZ_MAINSTAY_API_KEY` (assert key selection, mock the HTTP call).
- Migration backfill lands every legacy row on `mainstay` and is a no-op on
  second run.

## Open items

- **RuckTalk TikTok:** not connected in the RuckTalk Postiz org (only FB/IG/YT
  returned). If TikTok posting is wanted, the channel must be connected in that
  org. v1 posts the three connected channels regardless.
- **`groundrush` org:** seeded but has no users or Postiz yet — placeholder until
  Mike populates it.

## Rollout / risk

- Multi-tenancy touches **every query in Forge**, so the isolation test suite is
  the gate, not a nicety — a missed read path is a data leak between companies.
- `forge-web.service` does **not** hot-reload — `sudo systemctl restart
  forge-web.service` after deploy.
- Migration runs once against `data/forge_live.db`; take a DB copy first.
- Branch: `feat/forge-studio` (current) or a dedicated `feat/forge-multi-tenant`.
