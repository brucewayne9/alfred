# Phase 2: Alfred Claw Config Fixes - Context

**Gathered:** 2026-02-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Batch all SSH-only changes to Server 101: fix Telegram duplicate messages, correct tool argument errors, trim context files to size limits, fix escalation bridge grep, configure log rotation, and clean up stale gateway service. All work happens on Alfred Claw (101) via SSH.

</domain>

<decisions>
## Implementation Decisions

### USER.md trimming
- Keep both family details AND business context — trim equally from each
- Include personality/communication preferences (not just facts)
- Cut historical info first (past roles, backstory) — keep current state
- Cut prose and long descriptions — favor bullet-point facts
- Claude should read current USER.md, identify bloat, and propose cuts
- Must fit within 3,955 char limit

### Embeddings strategy
- Switch primary embeddings from OpenAI to Ollama nomic-embed-text (local on 101)
- Ollama is already running on 101 — check if nomic-embed-text is pulled, pull if needed
- Keep OpenAI as fallback if Ollama embedding fails (local-first, cloud-fallback)
- Research how OpenClaw configures embedding provider during planning

### HEARTBEAT.md content
- Include both service health AND last-activity timestamps — condensed to fit
- Local services only (Ollama, gateway, Telegram) — no external service checks
- Brief details format: e.g., "Ollama:OK(3d) Gateway:DOWN(since 14:30)"
- Research whether 293-char limit is hard OpenClaw constraint or configurable
- Must fit within whatever the actual limit is

### Claude's Discretion
- Telegram dedup fix approach (pure technical debugging)
- Tool argument corrections (python33→python3, CRM commands, email args, HEARTBEAT_OK)
- QUEUE.md grep -E flag fix (straightforward)
- Log rotation implementation details
- Stale openclaw-gateway.service removal approach

</decisions>

<specifics>
## Specific Ideas

- USER.md should preserve personality so Alfred matches Mike's communication style
- Heartbeat format inspired by compact status lines: "Service:STATUS(context)" pattern
- Embeddings: prefer local-first for cost savings and no external dependency

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-alfred-claw-config-fixes*
*Context gathered: 2026-02-20*
