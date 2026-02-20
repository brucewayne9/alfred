# Alfred Platform Stabilization & Ad Management

## What This Is

A stabilization and feature project for the Alfred dual-system AI assistant platform. Alfred Labs (Server 105, FastAPI + React, 353 tools) and Alfred Claw (Server 101, OpenClaw/Node.js, Telegram bot) both need infrastructure repairs, bug fixes, and reliable integrations. Additionally, full conversational ad campaign management for Meta Ads and Google Ads is needed for active Rod Wave tour and One Music Festival campaigns.

## Core Value

Alfred must be a reliable daily operations tool — every integration works correctly, no duplicate messages, no broken queues, and Mike can manage ad campaigns and CRM contacts conversationally without touching the ad platforms or CRM directly.

## Requirements

### Validated

- ✓ FastAPI backend with 353 LLM-callable tools — existing
- ✓ React frontend with Vite + TypeScript + Tailwind — existing
- ✓ Multi-model LLM routing (local Ollama, cloud, Claude Code) — existing
- ✓ JWT + passkey + TOTP authentication — existing
- ✓ Telegram bot via OpenClaw on Server 101 — existing
- ✓ ChromaDB vector memory + SQLite conversations — existing
- ✓ WebSocket real-time notifications — existing
- ✓ Gmail, Stripe, Home Assistant integrations — existing
- ✓ Health monitor + escalation bridge (105→101) — existing

### Active

- [ ] LightRAG server restored and accessible from both systems
- [ ] GA4 property IDs synced to Labs (Claw already fixed)
- [ ] Telegram duplicate message bug resolved on Claw
- [ ] Twenty CRM integration reliable — search, workflows, notes, tasks all correct
- [ ] Meta Ads full campaign control — on/off, budgets, pause, performance, AI suggestions
- [ ] Google Ads full campaign control — on/off, budgets, pause, performance, AI suggestions
- [ ] QUEUE.md grep fixed (needs -E flag for alternation)
- [ ] USER.md trimmed to fit 3,955 char limit
- [ ] HEARTBEAT.md trimmed to fit 293 char limit
- [ ] OpenAI project unarchived or switched — 401 errors on memory embeddings
- [ ] Log rotation fixed — daily log files created correctly
- [ ] Stale openclaw-gateway.service cleaned up
- [ ] Labs git repo gc — unreachable loose objects cleaned
- [ ] Claw tool argument errors fixed (python33→python3, CRM commands, email args, HEARTBEAT_OK)

### Out of Scope

- New feature development beyond ads — stabilization focus
- Mobile app — web-first
- Migration off Twenty CRM — fix integration, don't replace
- OpenClaw version upgrade — fix within current 2026.2.14

## Current Milestone: v1.0 Ops Ready

**Goal:** Make Alfred a reliable daily operations tool — fix all infrastructure issues, stabilize integrations, and enable conversational ad campaign management.

**Target features:**
- Infrastructure repairs (LightRAG, log rotation, gateway cleanup, git gc)
- Claw config fixes (USER.md, HEARTBEAT.md, grep, tool args, OpenAI 401)
- Twenty CRM integration reliability
- GA4 property sync to Labs
- Telegram duplicate message fix
- Meta Ads full campaign control
- Google Ads full campaign control

## Context

- **Two-server architecture:** Labs on 105 (FastAPI+React), Claw on 101 (OpenClaw/Telegram). Claude Code runs on 105, can SSH to 101 via `ssh -p 2222 brucewayne9@75.43.156.101`.
- **LightRAG** runs on Lonewolf (117) at port 9621. Shared knowledge graph used by both systems. Currently down with circuit breakers active.
- **Twenty CRM** at crm.groundrushlabs.com on Lonewolf (117), Dokploy managed, v1.14.0.
- **Ad campaigns** are for Rod Wave tours and One Music Festival. Real-time budget/performance decisions needed. Both Meta and Google Ads API credentials already configured.
- **Claw config issues** (USER.md, HEARTBEAT.md, log rotation, grep, tool args) are all fixable via SSH from 105.
- **OpenAI 401** on Claw affects memory embeddings and RAG features.

## Constraints

- **SSH access:** Claw fixes executed via SSH from 105 (port 2222)
- **TOOLS.md limit:** 20,000 chars on Claw — new tools must fit within budget
- **OpenClaw version:** Stay on 2026.2.14, no `doctor --fix`
- **Config safety:** Always backup `openclaw.json` before changes
- **Existing architecture:** Fix within current FastAPI + tool registry pattern on Labs

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Fix via SSH from 105 | Single Claude Code session manages both servers | — Pending |
| Full ads API integration | Mike needs conversational campaign control for active campaigns | — Pending |
| Fix CRM before building new | Reliability of existing tools before new features | — Pending |

---
*Last updated: 2026-02-20 after milestone v1.0 started*
