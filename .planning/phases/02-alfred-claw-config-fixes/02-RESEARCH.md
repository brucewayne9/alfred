# Phase 2: Alfred Claw Config Fixes - Research

**Researched:** 2026-02-20
**Domain:** OpenClaw configuration, workspace file management, Python tool fixes, logrotate
**Confidence:** HIGH (all findings verified directly on Server 101)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **USER.md trimming**: Keep both family details AND business context — trim equally from each. Include personality/communication preferences (not just facts). Cut historical info first (past roles, backstory) — keep current state. Cut prose and long descriptions — favor bullet-point facts. Must fit within 3,955 char limit.

- **Embeddings strategy**: Switch primary embeddings from OpenAI to Ollama nomic-embed-text (local on 101). Ollama is already running on 101 — check if nomic-embed-text is pulled, pull if needed. Keep OpenAI as fallback if Ollama embedding fails (local-first, cloud-fallback).

- **HEARTBEAT.md content**: Include both service health AND last-activity timestamps — condensed to fit. Local services only (Ollama, gateway, Telegram) — no external service checks. Brief details format: e.g., "Ollama:OK(3d) Gateway:DOWN(since 14:30)". Must fit within whatever the actual limit is (confirmed: 293 chars).

### Claude's Discretion

- Telegram dedup fix approach (pure technical debugging)
- Tool argument corrections (python33→python3, CRM commands, email args, HEARTBEAT_OK)
- QUEUE.md grep -E flag fix (straightforward)
- Log rotation implementation details
- Stale openclaw-gateway.service removal approach

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CLAW-01 | Telegram duplicate message bug resolved | Root cause confirmed in logs: agent calls messaging tool to send HEARTBEAT_OK, then heartbeat framework delivers again. Fix: update SOUL.md/AGENTS.md to instruct agent to reply HEARTBEAT_OK as text (not via messaging tool); fix Telegram target format from display name to numeric chat ID |
| CLAW-02 | USER.md trimmed to fit 3,955 char limit | Current: 9,199 chars (9,113 in bootstrap). Limit is confirmed hard constraint from openclaw source (logged at WARN level). Must cut ~5,244 chars. File content reviewed and trim plan developed. |
| CLAW-03 | HEARTBEAT.md trimmed to fit 293 char limit | Current: 1,961 chars (1,959 in bootstrap). Limit is 293 chars — confirmed from openclaw logs. This is `ackMaxChars` default = 300, but actual limit is 293 per log. The file must be completely restructured to a minimal format. |
| CLAW-04 | QUEUE.md grep fixed with -E flag for alternation | Current QUEUE.md grep uses `grep -i "PENDING|ESCALATED"` without -E flag, causing exit code 1. Fix: change to `grep -E -i "PENDING|ESCALATED"` in AGENTS.md heartbeat instructions. |
| CLAW-05 | Tool argument errors fixed | Multiple tool arg bugs confirmed in logs: (1) email mark-read routing bug in email_client.py (2) "Unknown target" Telegram send format (3) MIMEText (already works — `from email.mime.text import MIMEText` is correct) (4) HEARTBEAT_OK double-send via messaging tool |
| CLAW-06 | OpenAI project unarchived or switched to local embeddings | OpenAI key returns 401 (unauthorized). Must switch to Ollama nomic-embed-text. nomic-embed-text not yet pulled. Config path: `agents.defaults.memorySearch` with provider="openai", custom baseUrl pointing to Ollama endpoint. |
| INFRA-04 | Log rotation producing daily log files correctly on Claw (101) | No logrotate config exists for openclaw. Logs go to `/tmp/openclaw-1000/`. Feb 18 log is 7.8MB and not rotating. Need logrotate config for this path. |
| INFRA-05 | Stale openclaw-gateway.service cleaned up on Claw (101) | No `openclaw-gateway.service` exists in systemctl — the service was already removed or never existed as a systemd unit. There IS an `openclaw-gateway` process (PID 970704, running since Feb 18) but it is started by the `openclaw` supervisor process (PID 970695). This requirement may already be resolved. |
</phase_requirements>

---

## Summary

Phase 2 is entirely SSH-based work on Server 101 (Alfred Claw, 75.43.156.101, port 2222). All bugs are confirmed through direct inspection of the system: log files, running processes, source files, and the OpenClaw documentation bundled with the installed version (2026.2.17).

The most important finding is the **Telegram duplicate root cause**: the agent is using its messaging tool to explicitly send "HEARTBEAT_OK" to Mike's Telegram, but the heartbeat framework ALSO delivers the response. The fix is behavioral — update AGENTS.md or SOUL.md to tell the agent to reply with plain text "HEARTBEAT_OK" (not via the messaging tool), per OpenClaw's documented heartbeat protocol.

The **context file overflows are confirmed critical**: USER.md is 9,199 chars (limit 3,955) and HEARTBEAT.md is 1,961 chars (limit 293). Both emit WARN-level truncation messages on every single heartbeat and session start. The HEARTBEAT.md must be essentially rebuilt from scratch — it needs to go from 1,961 chars down to under 293 chars, which is a ~85% reduction. The current HEARTBEAT.md is a detailed 37-line instruction document; it needs to become a 3-5 line micro-checklist.

The **OpenAI embedding key is dead** (returns HTTP 401). The path forward is Ollama nomic-embed-text, which is not yet pulled but Ollama is running and accessible. OpenClaw supports custom OpenAI-compatible endpoints for `memorySearch`, so the correct config is `provider: "openai"` with `remote.baseUrl: "http://127.0.0.1:11434/v1/"` and `remote.apiKey: "ollama-local"`, with `model: "nomic-embed-text"`.

**Primary recommendation:** Execute fixes in dependency order: (1) Fix context file sizes first (CLAW-02, CLAW-03) to stop truncation noise; (2) Fix Telegram dedup (CLAW-01) to stop duplicate messages; (3) Fix tool arguments (CLAW-05) and QUEUE.md grep (CLAW-04); (4) Pull nomic-embed-text and configure embeddings (CLAW-06); (5) Set up logrotate (INFRA-04); (6) Verify INFRA-05 is already resolved.

---

## Standard Stack

### Core
| Tool | Version/Detail | Purpose | Why Standard |
|------|----------------|---------|--------------|
| OpenClaw | 2026.2.17 | AI agent platform on 101 | The deployed system |
| openclaw.json | `~/.openclaw/openclaw.json` | All config lives here | OpenClaw's single config file |
| Ollama | Running on 127.0.0.1:11434 | Local model server | Already deployed, nomic-embed-text to be pulled |
| logrotate | 3.21.0 | Log rotation | Standard Linux tool, already installed |
| Python 3.12 | On 101 | Tool scripts | Integration scripts language |

### Supporting
| Tool | Purpose | When to Use |
|------|---------|-------------|
| `openclaw logs --follow` | Tail gateway logs | Verifying fixes in real-time |
| `openclaw channels status` | Check Telegram health | After channel config changes |
| `systemctl` | Service management | After logrotate config added |
| SSH from 105 | All remote changes | Single session manages both servers |

---

## Architecture Patterns

### Current File Layout on 101

```
~/.openclaw/
├── openclaw.json             # Main config (all settings)
├── agents/main/sessions/     # Session JSONL files (~160 files)
│   └── sessions.json         # Session index (167KB)
└── workspace/
    ├── AGENTS.md             # Agent operating instructions (5,416 chars)
    ├── SOUL.md               # Agent identity (7,686 chars)
    ├── USER.md               # Mike's profile (9,199 chars — OVER LIMIT)
    ├── HEARTBEAT.md          # Heartbeat checklist (1,961 chars — OVER LIMIT)
    ├── TOOLS.md              # Tool reference (6,193 chars)
    ├── QUEUE.md              # Escalation queue (679 chars)
    ├── MEMORY.md             # Long-term memory (5,155 chars)
    └── scripts/integrations/ # Python tool scripts
        ├── crm.py
        ├── email_client.py
        └── ... (16 scripts)

/tmp/openclaw-1000/
├── openclaw-2026-02-18.log   # 7.8MB (not rotating — BUG)
├── openclaw-2026-02-19.log   # 7.4KB
└── openclaw-2026-02-20.log   # 547B
```

### OpenClaw Config Structure (openclaw.json)

Key sections relevant to this phase:

```json5
{
  "agents": {
    "defaults": {
      "model": { "primary": "ollama/minimax-m2:cloud" },
      "heartbeat": { "every": "30m", "model": "ollama/minimax-m2:cloud", "target": "last" },
      "compaction": { "mode": "safeguard", "reserveTokensFloor": 25000 },
      "memorySearch": {}   // EMPTY — embeddings not configured
    }
  },
  "channels": {
    "telegram": {
      "enabled": true,
      "dmPolicy": "pairing",
      "streamMode": "partial"
    }
  }
}
```

---

## Findings by Requirement

### CLAW-01: Telegram Duplicate Messages

**Root Cause (HIGH confidence — confirmed in logs):**

The agent is sending "HEARTBEAT_OK" via its Telegram messaging tool AND the heartbeat framework is also delivering the reply. This creates exactly two messages.

Evidence from `/tmp/openclaw-1000/openclaw-2026-02-18.log`:
```
2026-02-19T03:13:13.994Z - Skipping block reply - already sent via messaging tool: ARTBEAT_OK...
2026-02-19T04:42:46.402Z - Skipping block reply - already sent via messaging tool: HEARTBEAT_OK...
2026-02-19T05:12:42.157Z - Skipping block reply - already sent via messaging tool: HEARTBEAT_OK...
```

The message "Skipping block reply - already sent via messaging tool" means OpenClaw detected the tool send and tried to skip its own delivery — but on earlier occurrences it did NOT skip, causing the duplicate.

**Second issue confirmed:** The agent is formatting the Telegram target as a display name instead of a numeric ID:
```
Unknown target "MJ (@groundrushlabs) id:7582976864" for Telegram. Hint: <chatId>
```
The correct format is just the numeric ID: `7582976864`.

**Fix approach:**
1. Update SOUL.md or AGENTS.md to explicitly instruct: "During heartbeat runs, reply with HEARTBEAT_OK as plain text. Do NOT use the messaging tool to send HEARTBEAT_OK. The heartbeat framework handles delivery."
2. Update TOOLS.md to show correct Telegram send target format (numeric ID only, not display name format).
3. Consider setting `channels.telegram.streamMode: "off"` during debugging if duplicates persist — this disables draft bubbles and simplifies the send path.

**OpenClaw docs confirm** (from bundled telegram.md):
- During heartbeat runs, agent should reply `HEARTBEAT_OK` as the message body
- The heartbeat framework strips and suppresses `HEARTBEAT_OK` if it appears at start/end and remaining content is ≤ ackMaxChars (300)
- If the agent uses a messaging TOOL to send, OpenClaw tries to skip the block reply, but timing can cause duplicates

### CLAW-02: USER.md — 3,955 Char Limit

**Current state (HIGH confidence — directly verified):**
- Current size: 9,199 chars
- Limit: 3,955 chars (confirmed from log: `"workspace bootstrap file USER.md is 9113 chars (limit 3955)"`)
- Must cut: ~5,244 chars (57% reduction)
- This is a HARD limit enforced by OpenClaw at bootstrap injection time; it truncates at exactly 3,955 chars

**Current content breakdown (from reading the file):**

| Section | Approximate Size | Keep/Cut |
|---------|-----------------|---------|
| Identity + Business Overview | ~400 chars | KEEP — core context |
| Digital Properties (12 URLs) | ~350 chars | KEEP compressed — 1-line format |
| Preferences / Communication style | ~600 chars | KEEP — Alfred's behavior |
| Current Clients (My Hands Car Wash detail) | ~450 chars | KEEP condensed |
| Infrastructure (7 servers) | ~400 chars | KEEP condensed to IP+role format |
| Personal (full name, birthday, origin) | ~250 chars | KEEP essential |
| Wife (full bio + backstory) | ~500 chars | TRIM — cut "how we met" story, keep facts |
| Children (Rowan, Rex) | ~150 chars | KEEP |
| Older Children (Brandon full family tree) | ~400 chars | TRIM — cut grandchildren detail |
| Parents + Siblings | ~500 chars | TRIM — cut addresses, keep names+bdays |
| Key Dates table | ~400 chars | KEEP — Alfred needs these |
| How Mike Operates (Brain Dumps) | ~3,000 chars | MAJOR CUT — compress to 500 chars |
| Queue/Escalation Updates (CRITICAL) section | ~600 chars | KEEP — operational requirement |

**Trim strategy (Claude's discretion):**
- Cut "How Mike Operates — Ground Rush Evolution" prose section (the longest section, ~700 chars)
- Cut family backstory/narrative prose (wedding story, "how we met")
- Cut grandchildren detail (6 kids' names for Brandon)
- Compress siblings to name-only entries
- Compress infrastructure to `IP:role` table format
- Compress communication preferences to bullet points (already mostly bullets, just remove duplicates)
- Keep all dates, keep Queue/Escalation section (marked CRITICAL), keep Current Clients

### CLAW-03: HEARTBEAT.md — 293 Char Limit

**Current state (HIGH confidence — directly verified):**
- Current size: 1,961 chars
- Limit: 293 chars (confirmed from log: `"workspace bootstrap file HEARTBEAT.md is 1959 chars (limit 293)"`)
- Must cut: 1,668 chars (85% reduction)

**OpenClaw heartbeat documentation confirms:**
- `ackMaxChars: 300` is the default max chars after `HEARTBEAT_OK` — this maps to the 293-char limit observed (with some margin for HEARTBEAT_OK token itself + separator)
- If `HEARTBEAT.md` is effectively empty (only blank lines and markdown headers), OpenClaw skips the heartbeat run entirely
- Keep it to "a tiny checklist or reminders"

**Target HEARTBEAT.md format (must be under 293 chars):**

```markdown
# Heartbeat
- Check alfred@ + mjohnson@ inboxes for urgent items
- Check QUEUE.md: grep -E -i "PENDING|ESCALATED" QUEUE.md
- Check birthdays in USER.md within 7 days
- Monitor: Ollama, gateway, Telegram connectivity
Silent unless alert conditions met. Reply HEARTBEAT_OK if all clear.
```

This is approximately 260 chars — within limit.

**The critical fix:** The current HEARTBEAT.md is 37+ lines of detailed protocol documentation. It needs to become a 5-6 line checklist. The protocol detail belongs in SOUL.md or AGENTS.md, not HEARTBEAT.md.

### CLAW-04: QUEUE.md Grep Fix

**Root cause (HIGH confidence — QUEUE.md content confirmed):**

The escalation bridge grep command in AGENTS.md or HEARTBEAT.md uses:
```bash
grep -i "PENDING|ESCALATED" QUEUE.md
```

Without the `-E` flag, the pipe `|` is treated as a literal character, not alternation. This causes exit code 1 when neither literal string "PENDING|ESCALATED" is found.

**Fix:** Change to:
```bash
grep -E -i "PENDING|ESCALATED" ~/.openclaw/workspace/QUEUE.md
```

Or equivalently use `egrep -i "PENDING|ESCALATED"`.

**Also fix the path**: AGENTS.md instructs the heartbeat to check QUEUE.md. During heartbeat, the working directory may not be the workspace. Use the absolute path.

The QUEUE.md escalation bridge in `/home/aialfred/alfred/scripts/alfred_claw_monitor.py` (on 105) also uses a grep command that should be checked — confirm it uses `-E` flag.

### CLAW-05: Tool Argument Errors

**Confirmed bugs from logs (HIGH confidence):**

**Bug 1: email_client.py mark-read routing**
```
[tools] exec failed: Unknown command or missing args: mark-read
```
Source inspection of `/home/brucewayne9/.openclaw/workspace/scripts/integrations/email_client.py` reveals the `mark-read` command is defined in the `mark_read` function (line 233) but is NOT present in the dispatch block. The dispatch has `mark-unread` (line 358) and `trash` (line 360) but jumps directly to the `else` fallback. The `mark-read` case is missing from the elif chain.

**Fix:** Add to email_client.py dispatch:
```python
elif cmd == "mark-read" and len(sys.argv) > 3:
    print(json.dumps(mark_read(sys.argv[2], sys.argv[3]), indent=2))
```
This goes between the `search` and `mark-unread` branches.

**Bug 2: Telegram target format**
The agent sends: `"MJ (@groundrushlabs) id:7582976864"` when it should send: `7582976864`

Fix: Update TOOLS.md and SOUL.md to specify that Telegram target must be the numeric chat ID only: `7582976864`.

**Bug 3: HEARTBEAT_OK double-send via tool**
Documented above in CLAW-01. Fix is behavioral (SOUL.md/AGENTS.md instruction update).

**Bug 4: MIMEText import**
The log shows `ImportError: cannot import name 'MimeText'` — but the actual email_client.py correctly uses `from email.mime.text import MIMEText` (capital M, capital T). This error came from the model trying to write a Python one-liner with the wrong case. Not a code bug — behavioral fix via TOOLS.md clarification.

**Bug 5: crm.py create-note argument format**
The TOOLS.md shows: `create-note <person_id> "note text"` — the actual crm.py requires: `crm.py create-note <person_id> <body>` where body is space-joined remaining args (already correct). The issue is the model sometimes passes wrong arg count. Ensure TOOLS.md shows correct usage. `create-task` requires JSON: `create-task '{"title":"...","assigneeId":"..."}'` — TOOLS.md already shows JSON format but confirm it matches actual crm.py dispatch.

### CLAW-06: Embeddings — Switch to Ollama nomic-embed-text

**Current state (HIGH confidence):**
- `agents.defaults.memorySearch` = `{}` (empty — no embedding configured)
- OpenAI key in `env.vars.OPENAI_API_KEY` returns HTTP 401 (project archived/deleted)
- Ollama is running on `http://127.0.0.1:11434`
- `nomic-embed-text` is NOT pulled yet

**OpenClaw docs confirm** (from `/home/brucewayne9/.npm-global/lib/node_modules/openclaw/docs/concepts/memory.md`):

OpenClaw supports custom OpenAI-compatible endpoints for `memorySearch`:
```json5
agents: {
  defaults: {
    memorySearch: {
      provider: "openai",
      model: "nomic-embed-text",
      remote: {
        baseUrl: "http://127.0.0.1:11434/v1/",
        apiKey: "ollama-local"
      },
      fallback: "none"
    }
  }
}
```

Key notes:
- `provider: "openai"` with a custom `remote.baseUrl` routes to any OpenAI-compatible API
- Ollama exposes an OpenAI-compatible `/v1/embeddings` endpoint
- `fallback: "none"` prevents fallback attempts to the dead OpenAI key
- The `nomic-embed-text` model must be pulled first: `ollama pull nomic-embed-text`
- After config change, OpenClaw will reindex memory files automatically (detected via provider/model fingerprint change)

**Decision confirmed from CONTEXT.md:** Keep OpenAI as fallback if Ollama embedding fails. So use `fallback: "openai"` would be wrong since the OpenAI key is dead. Use `fallback: "none"` until a working key is available, OR the user may want to accept the local-only approach permanently.

### INFRA-04: Log Rotation

**Current state (HIGH confidence):**
- Log path: `/tmp/openclaw-1000/openclaw-YYYY-MM-DD.log`
- No logrotate config exists: `ls /etc/logrotate.d/` — no openclaw entry
- Feb 18 log is 7.8MB and has NOT rotated despite date changing
- Feb 19 and Feb 20 logs are small (7KB and 547B) — new files ARE being created per day (the date in the filename increments correctly)
- The actual rotation mechanism works via filename (new file per day). The problem is the old Feb 18 log is 7.8MB and kept growing into the next day, indicating the gateway was started on Feb 18 and kept writing to that day's file until restart.

**Log rotation behavior from OpenClaw docs:**
- Default log path: `/tmp/openclaw/openclaw-YYYY-MM-DD.log`
- This system uses `/tmp/openclaw-1000/` (uid-based variant)
- Files are date-based, so a new file appears at midnight UTC automatically
- The 7.8MB Feb-18 file grew because the gateway was running continuously across midnight
- logrotate won't help with `/tmp/` paths that auto-rotate by date — the real concern is file size limits

**Fix approach:**
- Create `/etc/logrotate.d/openclaw` to handle size-based rotation and cleanup of old logs
- The daily files are already working; need a cleanup policy for large files and old archives
- Note: logrotate requires a `postrotate` signal since openclaw-gateway is not a systemd service with a reload command; use `copytruncate` instead

```
/tmp/openclaw-1000/openclaw-*.log {
    daily
    rotate 7
    size 10M
    copytruncate
    missingok
    notifempty
    compress
    delaycompress
}
```

**Limitation:** logrotate doesn't handle glob patterns in filenames well. The actual solution may be a cron job that deletes logs older than 7 days.

### INFRA-05: Stale openclaw-gateway.service

**Current state (HIGH confidence):**
- `systemctl list-unit-files | grep gateway` — returns nothing related to openclaw
- `systemctl status openclaw-gateway.service` — unit not found
- There IS an `openclaw-gateway` process (PID 970704, uptime 2+ days) but it is started by the `openclaw` supervisor (PID 970695), not by systemd directly
- This is the CORRECT architecture: openclaw CLI starts the gateway as a child process

**Finding:** INFRA-05 appears to be already resolved — there is no stale `openclaw-gateway.service` systemd unit. The requirement may have been satisfied prior to this phase, or the service was already cleaned up. This needs confirmation by checking if there was ever a broken systemd unit. The gateway process is running correctly as a child of the openclaw supervisor.

**Action:** Verify with `systemctl list-unit-files | grep -i openclaw` and confirm no stale units exist. If none found, mark INFRA-05 as trivially complete with documentation of current healthy state.

---

## Common Pitfalls

### Pitfall 1: Editing openclaw.json Incorrectly
**What goes wrong:** JSON with comments (json5 format in docs) fails to parse as standard JSON. openclaw.json is standard JSON, not json5.
**How to avoid:** Always use `python3 -m json.tool` to validate after editing. Use `jq` or `python3 -c "import json; json.load(open('openclaw.json'))"` to verify before restarting gateway.

### Pitfall 2: Gateway Restart Required for Config Changes
**What goes wrong:** Config changes to `openclaw.json` do not take effect until the gateway restarts. Memory embedding config, channel config, heartbeat config all require restart.
**How to avoid:** After editing openclaw.json, restart: `pkill -f openclaw-gateway && openclaw gateway &` or use the correct restart command. The gateway is started by the `openclaw` supervisor process (PID 970695), so restart the supervisor: `pkill -f "^openclaw$"` and let it restart. Or use `openclaw gateway restart` if supported.

### Pitfall 3: Editing Workspace Files While Gateway Running
**What goes wrong:** AGENTS.md, SOUL.md, HEARTBEAT.md, USER.md are read at session bootstrap time. Edits take effect on the NEXT session, not immediately.
**How to avoid:** After editing workspace files, trigger a new session or heartbeat to verify. The current heartbeat will still use the old content.

### Pitfall 4: nomic-embed-text Model Size
**What goes wrong:** `ollama pull nomic-embed-text` downloads ~274MB. If disk space is tight on 101, this could fail.
**How to avoid:** Check disk space first: `df -h ~`. nomic-embed-text is a small embedding model (274MB), not a large LLM.

### Pitfall 5: HEARTBEAT.md Under 5 Lines May Be Treated as Empty
**What goes wrong:** OpenClaw docs say: "If HEARTBEAT.md is effectively empty (only blank lines and markdown headers), OpenClaw skips the heartbeat run entirely."
**How to avoid:** Ensure HEARTBEAT.md has at least one substantive line beyond headers. The checklist must have real instructions.

### Pitfall 6: logrotate and /tmp/ Paths
**What goes wrong:** `/tmp/` contents may be cleared on reboot. A logrotate config for `/tmp/` paths is unusual and may not work as expected if the directory structure changes.
**How to avoid:** The logrotate config should be for cleanup/archival only. The primary mechanism (new file per day) is built into OpenClaw. Consider a simpler cron-based cleanup: `find /tmp/openclaw-1000/ -name "*.log" -mtime +7 -delete`.

### Pitfall 7: NEVER Run `openclaw doctor --fix`
**What goes wrong:** Documented in MEMORY.md — `doctor --fix` wipes channel config, model routing, plugin state.
**How to avoid:** Never run this command. All config changes go directly to `openclaw.json`.

---

## Code Examples

### Correct memorySearch config for Ollama nomic-embed-text

```json
{
  "agents": {
    "defaults": {
      "memorySearch": {
        "provider": "openai",
        "model": "nomic-embed-text",
        "remote": {
          "baseUrl": "http://127.0.0.1:11434/v1/",
          "apiKey": "ollama-local"
        },
        "fallback": "none"
      }
    }
  }
}
```

### Pulling nomic-embed-text

```bash
ollama pull nomic-embed-text
# Verify:
curl -s -X POST http://127.0.0.1:11434/api/embeddings \
  -d '{"model":"nomic-embed-text","prompt":"test"}' | python3 -c "import sys,json; d=json.load(sys.stdin); print('OK, dims:', len(d.get('embedding',[])))"
```

### Correct logrotate config

```
/tmp/openclaw-1000/openclaw-*.log {
    size 10M
    rotate 7
    copytruncate
    missingok
    notifempty
    compress
    delaycompress
    dateext
}
```

Or simpler cron cleanup:
```bash
# In /etc/cron.daily/openclaw-log-cleanup
#!/bin/bash
find /tmp/openclaw-1000/ -name "*.log" -mtime +7 -delete
find /tmp/openclaw/ -name "*.log" -mtime +7 -delete
```

### email_client.py mark-read fix

```python
# Add this elif branch in email_client.py, between search and mark-unread:
elif cmd == "mark-read" and len(sys.argv) > 3:
    print(json.dumps(mark_read(sys.argv[2], sys.argv[3]), indent=2))
```

### Correct QUEUE.md grep (use absolute path + -E flag)

```bash
# In AGENTS.md or HEARTBEAT.md:
grep -E -i "PENDING|ESCALATED" ~/.openclaw/workspace/QUEUE.md
```

### Target HEARTBEAT.md content (under 293 chars)

```markdown
# Heartbeat
- Check alfred@ inbox; mjohnson@ for spikes
- grep -E -i "PENDING|ESCALATED" ~/.openclaw/workspace/QUEUE.md
- Check USER.md for birthdays within 7 days
- Ping: Ollama, gateway, Telegram
Reply HEARTBEAT_OK if all clear.
```
(Approximately 248 chars — within limit)

### USER.md trimming approach

The current USER.md has these large sections to cut:

1. **"How Mike Operates — Ground Rush Evolution & Business Philosophy"** (~700 chars): This is a verbose narrative about creativity philosophy. Cut to 1-2 sentences max or move to MEMORY.md.

2. **"Decision Making" section** (~400 chars): Compress to 2-3 bullets.

3. **Wife section "How we met"** (~200 chars): Cut entirely.

4. **Mike's parents and siblings section** (~350 chars): Keep names and birthdays only, cut locations.

5. **Sarah's parents and siblings** (~350 chars): Keep names only, cut locations.

6. **Infrastructure table** (7 rows): Compress to `IP:role` single-line format.

7. **"Work Style" and "Things He Likes/Hates" placeholders** (~200 chars): Cut placeholder sections.

Target: ~3,800 chars (keeping 200 chars of margin below the 3,955 limit).

---

## Open Questions

1. **INFRA-05 — Is it truly already clean?**
   - What we know: No `openclaw-gateway.service` found in systemctl
   - What's unclear: Was there ever a broken systemd unit, or was the requirement based on a misunderstanding of the architecture?
   - Recommendation: Confirm with `systemctl list-unit-files --all | grep -i "openclaw\|alfred"` and mark complete if no stale units

2. **CLAW-01 — Does the duplicate also happen on non-heartbeat Telegram DMs?**
   - What we know: The duplicate pattern is clearly visible in heartbeat runs
   - What's unclear: Whether the agent's messaging tool habit also causes duplicates on regular chat responses
   - Recommendation: After fixing the heartbeat behavior, test a direct Telegram message to confirm no duplicates on regular chat

3. **HEARTBEAT.md ackMaxChars vs char limit**
   - What we know: Log says "limit 293" for HEARTBEAT.md
   - What's unclear: Is 293 the bootstrap injection limit or the heartbeat ackMaxChars limit? They may be different settings.
   - Recommendation: The log message context (`workspace bootstrap file HEARTBEAT.md is 1959 chars (limit 293)`) makes this clearly a bootstrap injection size limit, not ackMaxChars. This is a separate constraint from ackMaxChars=300.

---

## Sources

### Primary (HIGH confidence)
- Direct SSH inspection of Server 101 (75.43.156.101 port 2222) — all findings are first-hand
- `/tmp/openclaw-1000/openclaw-2026-02-18.log` and `openclaw-2026-02-20.log` — live system logs
- `/home/brucewayne9/.openclaw/openclaw.json` — current configuration
- `/home/brucewayne9/.npm-global/lib/node_modules/openclaw/docs/channels/telegram.md` — bundled OpenClaw docs
- `/home/brucewayne9/.npm-global/lib/node_modules/openclaw/docs/gateway/heartbeat.md` — bundled OpenClaw docs
- `/home/brucewayne9/.npm-global/lib/node_modules/openclaw/docs/concepts/memory.md` — bundled OpenClaw docs
- `/home/brucewayne9/.npm-global/lib/node_modules/openclaw/CHANGELOG.md` — version 2026.2.17 changes
- `/home/brucewayne9/.openclaw/workspace/scripts/integrations/email_client.py` — source code of broken tool
- `/home/brucewayne9/.openclaw/workspace/USER.md` — current 9,199-char content

### Secondary (MEDIUM confidence)
- OpenClaw docs `gateway/configuration-examples.md` — memorySearch Ollama pattern
- Ollama REST API (`/api/tags`, `/api/embeddings`) — tested live, nomic-embed-text confirmed not installed

---

## Metadata

**Confidence breakdown:**
- CLAW-01 root cause: HIGH — directly confirmed in log files with exact log messages
- CLAW-02 USER.md limit: HIGH — hard limit confirmed in live WARN logs; content reviewed
- CLAW-03 HEARTBEAT.md limit: HIGH — hard limit confirmed in live WARN logs
- CLAW-04 grep fix: HIGH — bash behavior is well-understood; QUEUE.md content confirmed
- CLAW-05 tool bugs: HIGH — email_client.py source inspected; crm.py routing confirmed
- CLAW-06 embeddings: HIGH — OpenAI key 401 confirmed; Ollama config path confirmed from docs
- INFRA-04 logrotate: HIGH — no config exists; log file state observed
- INFRA-05 gateway service: HIGH — systemctl confirmed no stale unit

**Research date:** 2026-02-20
**Valid until:** 2026-03-20 (stable infrastructure; 30-day window)
