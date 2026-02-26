# Alfred Platform Stabilization & Ad Management

## What This Is

A stabilization and feature project for the Alfred dual-system AI assistant platform. Alfred Labs (Server 105, FastAPI + React, 353+ tools) and Alfred Claw (Server 101, OpenClaw/Node.js, Telegram bot) are now infrastructure-stable with reliable integrations. Full conversational ad campaign management for both Meta Ads and Google Ads is operational for active Rod Wave tour and One Music Festival campaigns.

## Current Milestone: v1.1 Infrastructure Resilience

**Goal:** Establish automated multi-server backups to Google Drive with full disaster recovery capability, set up SSH access from Labs (105) to all servers, and complete carried-forward ad management features.

**Target features:**
- SSH key distribution from 105 to all 7 servers
- Server auditing (what's running on each — Docker, bare metal, databases)
- Automated daily config backups (2 AM) + weekly full backups (Sunday 2 AM)
- Google Drive upload via existing Workspace integration
- 30-day retention with automatic cleanup
- Telegram failure alerts to Mike
- Full restore capability (package lists, Docker configs, systemd, crontabs)
- AI ad performance suggestions (carried from v1.0)
- Cross-platform ad summary (carried from v1.0)
- Confirmation guardrails for financial mutations (carried from v1.0)

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

### Active

- [ ] SSH key access from 105 to all 7 servers (INFRA-01)
- [ ] Server audit — catalog services, Docker, databases per server (INFRA-02)
- [ ] Daily config backup script (configs, databases, env files, crontabs) (BACKUP-01)
- [ ] Weekly full backup script (Docker volumes, app data, media, package lists) (BACKUP-02)
- [ ] Google Drive upload via Workspace integration (BACKUP-03)
- [ ] 30-day retention with automatic cleanup (BACKUP-04)
- [ ] Telegram failure alerts on backup errors (BACKUP-05)
- [ ] Cron scheduling on 105 (daily 2 AM, weekly Sunday 2 AM) (BACKUP-06)
- [ ] Full restore documentation per server (RECOVERY-01)
- [ ] AI-generated performance suggestions for ad campaigns (ADS-01)
- [ ] Cross-platform ad performance summary — Meta + Google combined (ADS-02)
- [ ] Confirmation guardrail pattern for financial mutations (ADS-03)

### Out of Scope

- Mobile app — web-first, PWA works well
- Migration off Twenty CRM — integration now reliable
- OpenClaw version upgrade — current version stable after fixes
- Google OAuth re-authorization flow — existing scopes working

## Context

Shipped v1.0 Ops Ready with 45 files changed (+7,136 lines) across 40 commits in 24 days.
Tech stack: FastAPI + React (Labs/105), OpenClaw/Node.js (Claw/101), Twenty CRM (117).
Seven-server infrastructure: 101 (Claw, SSH:2222), 104 (Prod), 105 (Labs, orchestrator), 98 (Loovacast Dev), 100 (Loovacast Prod), 117 (Dokploy/CRM, SSH:22), 121 (Mailcow). SSH access needs auditing — confirmed on 101 and 117, others unknown.
Ad campaigns are for Rod Wave tours and One Music Festival — real-time budget/performance decisions needed.
All ad management tools now include verification, alerting, and audit trails.
Google Workspace integration exists (`google_workspace.py`) with Drive API access — will use for backup uploads.
Backup orchestrator will run on 105 with cron, SSH into each server, collect backups, upload to Drive.

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

| 105 as backup orchestrator | Central point, SSH into all servers, upload to Drive | — Pending |
| Google Workspace for Drive uploads | Reuse existing integration, no new auth setup | — Pending |
| Daily configs + weekly full schedule | Balance safety vs storage, 30-day retention | — Pending |

---
*Last updated: 2026-02-26 after v1.1 milestone start*
