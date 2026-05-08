# Lucius — Hermes Agent on Server 111 (Test Deployment) — Design Spec

**Date:** 2026-05-08
**Author:** Alfred (with Mike Johnson)
**Status:** Approved for implementation planning

---

## Goal

Stand up a parallel agent on server 111 powered by Nous Research's **Hermes Agent** framework, fronted by Telegram bot **@Luciuslabsbot**, running side-by-side with Oracle on 117 for a 2-week evaluation. If Lucius performs comparably or better than Oracle for day-to-day butler/operator work, we plan a phased migration off OpenClaw. If not, Lucius is decommissioned and 117 keeps the job.

## Why

- OpenClaw on 117 has been showing reliability friction (cron double-fires, stuck escalations, model-chain fragility — see `claw_failsafe_model.md` and `claw_heartbeat.md`).
- Hermes Agent v0.13.0 (released 2026-05-07) ships with native Telegram/Discord/Slack/etc. gateway, MCP support, built-in skills system, cross-session memory, and provider-agnostic LLM backends — covering most of what we have hand-built around OpenClaw.
- 105's Ollama bridge already serves the model recommended by the Ollama-Hermes integration docs (`kimi-k2.6:cloud` ≥ 64K context). Zero new model infra required.
- 111 is otherwise underused (CasaOS dev/test box from the Paperclip era); putting Lucius there utilizes existing capacity and isolates the test.

## Architecture

```
                                       Mike (Telegram)
                                              │
            ┌──────────────────────────┬──────┴───────────────────┐
            ▼                          ▼                          ▼
   @groundrushlabsbot           @alfredblogbot              @Luciuslabsbot
        (Alfred / 105)              (Oracle / 117)            (Lucius / 111)
            │                          │                          │
            │                          │                          │
            ▼                          ▼                          ▼
     ┌─────────────┐           ┌──────────────┐          ┌──────────────────┐
     │   Alfred    │           │   OpenClaw   │          │   Hermes Agent   │
     │   Labs API  │           │   gateway    │          │   v0.13.0        │
     │   (105)     │           │   (117)      │          │   (111)          │
     └──────┬──────┘           └──────┬───────┘          └────────┬─────────┘
            │                         │                           │
            │ tools                   │ tools                     │ tools (MCP)
            ▼                         ▼                           ▼
       integrations            ~/.openclaw/scripts/        claw-tools MCP server
                                  integrations                    (111)
                                                                   │
                                                                   │ wraps 40+ scripts
                                                                   ▼
                                                            scripts copied from
                                                            117 → 111 verbatim
                                                            for full isolation

   ┌────────────────────────────────────────────────────────────────────────┐
   │                            MODEL BACKEND                               │
   │  Lucius → Ollama on 105 (http://75.43.156.105:11434/v1)                │
   │  Model: kimi-k2.6:cloud (matches Oracle's primary, no API key needed)  │
   │  Same bridge already used by Oracle for Kimi K2.6 today.               │
   └────────────────────────────────────────────────────────────────────────┘

   ┌────────────────────────────────────────────────────────────────────────┐
   │                        LONG-TERM MEMORY ARCHITECTURE                   │
   │                                                                        │
   │  Lucius reads → Grey Matter on 117:9621 (READ-ONLY, via                │
   │                 lightrag_client.py recall/query)                       │
   │                                                                        │
   │  Lucius writes → ~/.hermes/ on 111 (its native memory layer)           │
   │                                                                        │
   │  Lucius proposes → ~/.hermes/promote_queue.jsonl on 111                │
   │                  → 7 AM ET cron emails Telegram digest to Mike         │
   │                  → Mike approves N entries via reply                   │
   │                  → approved entries push to 117 Grey Matter            │
   │                  → namespaced with `lucius_` prefix for traceability   │
   └────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Hermes Agent on 111

- **Install:** `curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash` as `brucewayne9` on 111. Verifies to `~/.hermes/`.
- **Service:** systemd-user unit `hermes-gateway.service` enabled and started under `brucewayne9`. Mirrors how OpenClaw runs on 117 (also user-systemd) for operational parity.
- **Config files:**
  - `~/.hermes/.env` — secrets (Telegram token, etc.)
  - `~/.hermes/config.yaml` — non-secret settings
- **Version pin:** v0.13.0 ("Tenacity Release") at install. Note version in spec; auto-update OFF for the test window so Mike doesn't catch a regression mid-eval.

### 2. Model Backend

- **Primary:** `kimi-k2.6:cloud` via `http://75.43.156.105:11434/v1` (no API key — Ollama local endpoint, already firewalled to allow 111 → 105:11434 per `opencode_111_setup.md`).
- **No fallback chain at v1.** Hermes Agent doesn't ship the same multi-model failover OpenClaw has. If Kimi K2.6 hiccups during the test, that's a real data point. We can add a fallback later if it becomes a blocker.
- **Context:** ≥ 64K — Kimi K2.6 satisfies this.

### 3. Tool Layer — `claw-tools` MCP Server

**Full isolation, per Mike's directive.** Copy the 40+ integration scripts from 117 to 111 verbatim, then wrap them as a single MCP server.

- **Source:** `brucewayne9@75.43.156.117:~/.openclaw/workspace/scripts/integrations/`
- **Destination:** `brucewayne9@75.43.156.111:~/.lucius/workspace/scripts/integrations/`
  - Use `~/.lucius/` (not `~/.openclaw/`) to make it visually obvious which box you're on.
- **MCP server:** A new `mcp-claw-tools` Python package on 111 that exposes each script's CLI commands as MCP tools. One MCP server, one binary, simpler ops; can split by domain later if it grows past ~80 tools.
- **Tools migrated day one (25 total):**
  - **CRM / comms:** `crm.py`, `email_client.py`, `google_calendar.py`
  - **Social (primary channels):** `meta_social.py`, `linkedin.py`, `youtube.py`
  - **Web research:** `search.py`, `scraper.py`
  - **Image / design / media:** `comfyui_gen.py`, `flyer_designer.py`, `image_fx.py`, `image_tools.py`, `screenshot.py`, `stock_photos.py`, `design_memory.py`, `design_review.py`, `telegram_tts.py`
  - **Content pipeline:** `auto_blogger.py`, `video_render.py`, `remotion_render.py`, `weather.py`
  - **Workspace / project:** `google_workspace.py`, `website_designer.py`, `mission_control.py`
  - **Memory:** `lightrag_client.py` (READ ops only — `recall`/`query`; `insert` not exposed at v1, gated behind promote queue)
- **Tools deferred to "add as needed" (15 total):**
  - **Money / ads (T3 anyway):** `stripe_api.py`, `meta_ads.py`, `google_ads.py`
  - **Email marketing:** `brevo.py`
  - **Social secondary:** `tiktok.py`, `postiz.py`, `social_calendar/`
  - **Production / ops:** `wordpress.py`, `azuracast.py`, `homeassistant.py`, `n8n.py`, `twilio.py`, `mainstay.py`
  - **Analytics:** `google_analytics.py`
  - **API bridge:** `call_alfred_labs.py`
- **Disabled tools (not migrated):** `hunter.py`, `sponsorship.py` (you disabled 2026-03-21), `paperclip.py` (killed 2026-04-13).
- **Config / secrets:** `~/.lucius/config/.env` — copy of relevant keys from `/home/aialfred/alfred/config/.env`. Lock to `chmod 600`.

### 4. Telegram Gateway

- **Bot:** `@Luciuslabsbot` (ID 8750983299, display name "Lucius Fox") — already verified via `getMe` 2026-05-08, token in `config/.env` as `TELEGRAM_BOT_TOKEN_LUCIUS`.
- **Setup:** `hermes gateway setup` → choose Telegram → paste bot token → enable for chat ID `7582976864` (Mike's chat).
- **Identity guard:** A startup check that calls `getMe` and fails-fast if the username ≠ `Luciuslabsbot`. Prevents repeat of the 2026-04 Alfred/Oracle bot-ID swap incident.
- **No mailbox:** Lucius doesn't get its own email account at v1. If outbound mail is needed during the test, it routes through Oracle's `bgordon@groundrushlabs.com`.

### 5. Long-Term Memory — Read-Only + Promote Queue

- **Reads:** Lucius can call `lightrag_client.py recall <query>` and `lightrag_client.py query <query>` against `http://75.43.156.117:9621` for full historical context. Same auth as Oracle uses today.
- **Writes (forbidden direct):** Lucius **cannot** call `lightrag_client.py insert` directly. The MCP wrapper for `lightrag_client.py` exposes ONLY `recall` and `query` for v1.
- **Promote queue:** When Lucius decides "this fact deserves long-term memory," it appends to `/home/brucewayne9/.lucius/promote_queue.jsonl` on 111. Format:
  ```json
  {"ts": "2026-05-09T14:32:00Z", "session_id": "...", "content": "Mike said FaR campaign is paused indefinitely, no re-launch planned", "reasoning": "Project status update; Oracle's last memory was 2026-04-30 pause", "proposed_track_id": "lucius_pending"}
  ```
- **Daily digest:** Cron at 7 AM ET on 111 reads the queue, sends a Telegram message via Lucius bot to Mike with numbered entries: "Lucius wants to remember these — reply with comma-separated numbers to approve, or 'none' to skip all."
- **Approval handler:** Mike's reply triggers a script that pulls approved entries → ingests them into 117 Grey Matter via `lightrag_client.py insert`, namespaced with `lucius_` prefix in the track_id so they're traceable and prunable. Approved entries are deleted from the queue; rejected entries are moved to `promote_queue.rejected.jsonl` for audit.
- **Future graduation:** After 2 weeks of clean operation, Lucius can be promoted to "auto-write to Grey Matter under `lucius_*` namespace" without manual approval. Out of scope for v1.

### 6. Heartbeat / Monitoring

- **Extend `scripts/alfred_claw_monitor.py` on 105** to also probe 111's `hermes gateway` health endpoint (default port determined at install; spec assumes :18790 to avoid collision with 18789 on 117).
- **Same alert pattern:** 2 consecutive failures → Telegram + email to Mike with "FIX IT / LEAVE IT" choice.
- **State extension:** Add `lucius_status`, `lucius_failures`, `lucius_last_check` to `data/claw_monitor_state.json`. Don't overwrite existing fields — additive only.
- **No auto-failover for Lucius.** If Lucius dies, alert Mike; don't try to restart silently. The whole point of this test is to observe failure modes honestly, not paper over them.

### 7. Coexistence with 117 / OpenClaw / Oracle

- **Zero changes to 117.** OpenClaw stays running, Oracle stays answering on @alfredblogbot, all crons keep firing.
- **Zero changes to Grey Matter ingest paths.** Nightly braindump, Obsidian/Nextcloud sync, PreCompact hooks, manual inserts all keep going to 117 only. Lucius's promote queue is the only new write path, and only after Mike approves.
- **Zero changes to Alfred (105).** Alfred Labs API and the @groundrushlabsbot bot keep working as-is.

## Data Flow Walkthroughs

### Inbound user message
1. Mike sends Telegram message to @Luciuslabsbot
2. Hermes gateway on 111 receives via long-poll
3. Hermes runtime processes — calls model on 105 Ollama, calls tools via claw-tools MCP, looks up history via lightrag_client.py recall (read-only)
4. Gateway sends reply back to Mike via Telegram

### Lucius wants to remember a fact
1. Hermes' internal logic decides "this is worth long-term"
2. Hermes calls a custom skill `propose_memory(content, reasoning)` (we ship this skill at v1)
3. Skill appends entry to `~/.lucius/promote_queue.jsonl`
4. Hermes also keeps the entry in its native `~/.hermes/` memory so it's available short-term

### Daily promote digest
1. Cron at 7 AM ET on 111 invokes `~/.lucius/scripts/promote_digest.py`
2. Script reads queue, builds numbered Telegram message, sends to Mike via Lucius bot
3. Mike replies with approval list (or "none")
4. A webhook handler (or a follow-up cron job at 7:15 AM that reads recent Lucius bot messages) parses the approval, calls `lightrag_client.py insert` against 117 for each approved entry with `track_id=lucius_<timestamp>_<idx>`
5. Approved entries removed from queue; rejected entries moved to `.rejected.jsonl`

### Heartbeat alert
1. Cron on 105 (`*/10`) probes 111:18790/health
2. Two consecutive failures → email Mike + Telegram via Alfred
3. Mike replies "FIX IT" or "LEAVE IT" — same flow as Oracle today
4. On recovery → log to `~/.lucius/INCIDENTS.md` on 111

## Testing & Success Criteria

**Duration:** 2 weeks from go-live (target: 2026-05-09 → 2026-05-23, Mike's call on actual start date).

**Success bar (vibes-with-strikes):**

The test is judged by Mike's overall satisfaction with Lucius for day-to-day butler work. To make that less squishy, we track strikes:

- **A "strike"** = Mike has to fall back to Oracle (or Alfred) to complete a task that Lucius failed at, hallucinated through, or refused to do.
- **3 strikes in 14 days** = test fails, Lucius decommissioned, OpenClaw stays.
- **0–2 strikes in 14 days** = test passes, plan a phased migration off OpenClaw at Mike's pace.
- **Catastrophic strikes** (sent something wrong as Mike, leaked a secret, blew up a server) = instant fail, decommission immediately, post-mortem.

**Tasks the test should exercise across the 2 weeks:**
- Morning brief delivery (already a daily Oracle task)
- CRM lookups + simple updates
- Calendar checks
- Email triage / unread report
- Social-media post drafts (low-risk pillar accounts only — RuckTalk/AG, NOT FaR campaigns since FaR is paused)
- Image generation via ComfyUI
- Web research + scraping
- Recall queries against Grey Matter (history retrieval)
- Memory promotion proposals (the queue should fire at least a few times in 2 weeks; if it never fires, that's data too)

**Out-of-scope for the test:** Stripe payments, ad spend changes, production server commands (104/100/117/121), data deletion, vendor commitments, sending AS Mike. Lucius is butler-tier autonomy at most (T1 + drafts of T2). T3 still goes to Mike.

## Out of Scope

- Decommissioning OpenClaw / 117 — explicitly NOT happening in this spec; that's a follow-up only if the test passes.
- Migrating Oracle's Grey Matter to a clone on 111 — rejected during brainstorming; Lucius reads 117's Grey Matter directly.
- Auto-write to Grey Matter — gated behind promote queue at v1.
- Lucius's own email mailbox — defer.
- Multi-model failover for Lucius — defer.
- Lucius taking over Oracle's existing crons (daily social, daily blog, episode pipeline) — defer; Oracle keeps those during the test.
- Migrating skills/agents from OpenClaw's 14-agent setup — defer; let Hermes' single-agent + skills system do the work and see what we miss.

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Hermes 0.13.0 has bugs (it's brand new) | Pin to 0.13.0; auto-update OFF; test catches issues, that's the point |
| `claw-tools` MCP wrapper has parity drift from the originals on 117 | Copy scripts verbatim, don't modify; if a script needs a fix, fix on 117 first then re-copy |
| Promote queue grows unwatched and floods Mike's morning | Cap digest at 10 entries/day; older entries auto-roll to next day |
| Lucius hallucinates a "send email AS Mike" action | T2/T3 actions blocked at the tool layer; same enforcement as Oracle |
| Bot token leak | `config/.env` is `chmod 600`; never echoed in any logs; Lucius has no ability to publish env vars |
| 105 Ollama goes down during test | Same as Oracle today — Oracle also depends on 105 Ollama for kimi-k2.6 |
| Lucius writes to 117 Grey Matter accidentally | MCP wrapper for `lightrag_client.py` exposes ONLY `recall`/`query`; `insert` not exposed |
| Bot identity confusion (the 2026-04 swap) | Startup `getMe` check + canonical memory entry [`lucius_bot.md`](../../../.claude/projects/-home-aialfred-alfred/memory/lucius_bot.md) |

## Open Questions

None blocking spec writeup. Open execution-time items:
- Mike to confirm cutover date (default: as soon as plan is approved).
- Mike to OK any new APIs needed for the moved scripts (none expected; reusing existing tokens from `config/.env`).

## File Inventory

**New files this project will create:**

| Path | Purpose |
|---|---|
| `~/.lucius/workspace/scripts/integrations/*` (on 111) | Mirror of 117's integration scripts |
| `~/.lucius/config/.env` (on 111) | Subset of secrets needed by tools |
| `~/.lucius/scripts/promote_digest.py` (on 111) | Daily promote-queue digest sender |
| `~/.lucius/scripts/promote_apply.py` (on 111) | Approval-reply parser + GM ingester |
| `~/.lucius/promote_queue.jsonl` (on 111) | Hermes-proposed memories awaiting Mike approval |
| `~/.lucius/promote_queue.rejected.jsonl` (on 111) | Audit log of rejected proposals |
| `~/.lucius/INCIDENTS.md` (on 111) | Operational log |
| `~/.config/systemd/user/hermes-gateway.service` (on 111) | systemd-user unit for Hermes Agent |
| `mcp-claw-tools/` Python package (on 111) | MCP server wrapping the 40+ scripts |

**Files modified:**

| Path | Change |
|---|---|
| `/home/aialfred/alfred/config/.env` (on 105) | Added `TELEGRAM_BOT_TOKEN_LUCIUS` + `LUCIUS_TELEGRAM_BOT_ID` (already done 2026-05-08) |
| `/home/aialfred/alfred/scripts/alfred_claw_monitor.py` (on 105) | Add Lucius-on-111 health probe + state fields |
| `/home/aialfred/alfred/data/claw_monitor_state.json` (on 105) | New `lucius_*` fields appended (additive only) |

**Files explicitly NOT modified:**
- Anything under `/home/brucewayne9/.openclaw/` on 117
- 117's systemd units, OpenClaw config, sessions
- Grey Matter ingest pipelines, Obsidian/Nextcloud sync
- Alfred Labs API code on 105 (except `claw_monitor.py`)
- Existing Telegram bot configs for Alfred and Oracle

## Sources

- [Hermes Agent quickstart (Nous Research)](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart)
- [Ollama × Hermes integration](https://docs.ollama.com/integrations/hermes)
- [NousResearch/hermes-agent on GitHub](https://github.com/nousresearch/hermes-agent)
- [Hermes AI Agent + Ollama: FREE + 1 Click Setup! (YouTube)](https://www.youtube.com/watch?v=FEaXsMmuqeQ)
- Internal: `claw_failsafe_model.md`, `claw_heartbeat.md`, `claw_tools_inventory.md`, `claw_migration_117.md`, `opencode_111_setup.md`, `feedback_verify_bot_identity.md`, `feedback_oracle_readonly_servers.md`
