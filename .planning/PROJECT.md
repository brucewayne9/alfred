# Alfred Platform Stabilization & Ad Management

## What This Is

A stabilization and feature project for the Alfred dual-system AI assistant platform. Alfred Labs (Server 105, FastAPI + React, 353+ tools) and Alfred Claw (Server 101, OpenClaw/Node.js, Telegram bot) are infrastructure-stable with reliable integrations, full conversational ad campaign management (Meta + Google), automated multi-server backups to Google Drive, and disaster recovery capability across the entire 7-server fleet.

## Core Value

Alfred must be a reliable daily operations tool — every integration works correctly, no duplicate messages, no broken queues, and Mike can manage ad campaigns and CRM contacts conversationally without touching the ad platforms or CRM directly.

## Requirements

### Validated

- ✓ FastAPI backend with 353+ LLM-callable tools — existing
- ✓ React frontend with Vite + TypeScript + Tailwind — existing
- ✓ Multi-model LLM routing (local Ollama, cloud, Claude Code) — existing
- ✓ JWT + passkey + TOTP authentication — existing
- ✓ Telegram bot via OpenClaw on Server 101 — existing
- ✓ ChromaDB vector memory + SQLite conversations — existing
- ✓ WebSocket real-time notifications — existing
- ✓ Gmail, Stripe, Home Assistant integrations — existing
- ✓ Health monitor + escalation bridge (105→101) — existing
- ✓ LightRAG server restored with circuit breaker self-healing — v1.0
- ✓ GA4 property IDs synced to Labs — v1.0
- ✓ Telegram duplicate message bug resolved — v1.0
- ✓ Twenty CRM integration reliable (atomic rollback, 500-result search) — v1.0
- ✓ Meta Ads full campaign control (22 tools, read-after-write verification) — v1.0
- ✓ Google Ads budget/status control (mutations + audit logging) — v1.0
- ✓ Claw config fixed (USER.md, HEARTBEAT.md, grep, tool args, embeddings) — v1.0
- ✓ Log rotation and stale gateway cleanup on Claw — v1.0
- ✓ Labs git repo cleaned — v1.0
- ✓ SSH key access from 105 to all 7 servers — v1.1
- ✓ Server audit — catalog services, Docker, databases per server — v1.1
- ✓ Daily config backup script (configs, databases, env files, crontabs) — v1.1
- ✓ Weekly full backup script (Docker volumes, app data, media, package lists) — v1.1
- ✓ Google Drive upload via Workspace integration — v1.1
- ✓ 30-day retention with automatic cleanup — v1.1
- ✓ Telegram failure alerts on backup errors — v1.1
- ✓ Per-server restore documentation — v1.1
- ✓ Backup validation + status API endpoint — v1.1
- ✓ AI-generated performance suggestions for ad campaigns — v1.1
- ✓ Cross-platform ad performance summary — Meta + Google combined — v1.1
- ✓ Confirmation guardrail pattern for financial mutations — v1.1

### Active

(None — next milestone requirements to be defined via `/gsd:new-milestone`)

### Out of Scope

- Mobile app — web-first, PWA works well
- Migration off Twenty CRM — integration now reliable
- OpenClaw version upgrade — current version stable after fixes
- Google OAuth re-authorization flow — existing scopes working
- Bare metal OS snapshots — use hosting provider snapshots instead
- Cross-datacenter replication — over-engineered for current 7-server setup
- Incremental/differential backups — weekly full + daily config sufficient at this scale
- Backup encryption at rest — Drive is authenticated, complexity without proportional benefit

## Context

Shipped v1.0 Ops Ready (2026-02-21): 45 files changed (+7,136 lines), 40 commits, 24 days.
Shipped v1.1 Infrastructure Resilience (2026-02-26): 17 files changed (+4,709 lines), 41 commits, 1 day.
Tech stack: FastAPI + React (Labs/105), OpenClaw/Node.js (Claw/101), Twenty CRM (117).
Seven-server infrastructure: 101 (Claw, SSH:2222), 104 (Prod), 105 (Labs, orchestrator), 98 (Loovacast Dev), 100 (Loovacast Prod), 117 (Dokploy/CRM, SSH:22), 121 (Mailcow).
All servers accessible via SSH from 105 with dedicated per-server ed25519 keys.
Automated backups running: daily configs (Mon-Sat 2 AM), weekly full (Sunday 2 AM), Drive uploads, 30-day retention.
Backup validation at 5 AM daily with Telegram alerts on failure.
Ad campaigns (Rod Wave tours, One Music Festival) fully manageable conversationally with guardrails.

## Constraints

- **SSH access:** Claw fixes executed via SSH from 105 (port 2222)
- **TOOLS.md limit:** 20,000 chars on Claw — new tools must fit within budget
- **OpenClaw version:** Stay on 2026.2.14, no `doctor --fix`
- **Config safety:** Always backup `openclaw.json` before changes
- **Existing architecture:** Fix within current FastAPI + tool registry pattern on Labs

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Fix via SSH from 105 | Single Claude Code session manages both servers | ✓ Good — all Claw fixes deployed successfully |
| Full ads API integration | Mike needs conversational campaign control for active campaigns | ✓ Good — Google + Meta both operational |
| Fix CRM before building new | Reliability of existing tools before new features | ✓ Good — atomic rollback + expanded search |
| Targeted iptables edit (not iptables-save) | Preserve Docker dynamic rules on 117 | ✓ Good — LightRAG accessible without breaking Docker |
| Behavioral Telegram dedup via AGENTS.md | Instruct agent rather than modifying gateway config | ✓ Good — dedup resolved without OpenClaw changes |
| Ollama nomic-embed-text for embeddings | OpenAI 401 on Claw, local embeddings more reliable | ✓ Good — 355 embeddings cached, working |
| Immediate rollback (no retry) on CRM step-2 | HTTP errors are deterministic | ✓ Good — clean failure, no orphaned records |
| Shared budget warning in data layer | LLM reads warning naturally, no hardcoded logic | ✓ Good — conversational budget safety |
| Read-after-write on all Meta write ops | Trust but verify for financial mutations | ✓ Good — 19/22 tools validated with verification |
| 105 as backup orchestrator | Central point, SSH into all servers, upload to Drive | ✓ Good — all 7 servers backed up successfully |
| Google Workspace for Drive uploads | Reuse existing integration, no new auth setup | ✓ Good — Drive folder hierarchy working |
| Daily configs + weekly full schedule | Balance safety vs storage, 30-day retention | ✓ Good — clean separation of concerns |
| Per-server dedicated SSH keys | Fine-grained revocation without affecting other servers | ✓ Good — 6 key pairs deployed |
| Rule-based heuristics for ad suggestions | Deterministic, low-latency, LLM already provides NL layer | ✓ Good — 7 rules covering key scenarios |
| Programmatic guardrail enforcement | Code-level block, not LLM description hints | ✓ Good — confirmed=True required for all 12 mutation tools |

---
*Last updated: 2026-02-26 after v1.1 milestone*
