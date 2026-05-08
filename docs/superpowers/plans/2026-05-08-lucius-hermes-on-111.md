# Lucius — Hermes Agent on 111 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the Hermes Agent v0.13.0 on server 111, fronted by `@Luciuslabsbot`, running side-by-side with Oracle on 117 for a 2-week butler-grade evaluation. No changes to 117. Tools in full isolation. Memory writes to Grey Matter gated behind a daily approval queue.

**Architecture:** Hermes Agent runs as a systemd-user service on 111. Brain is `kimi-k2.6:cloud` via 105's Ollama bridge. Tools come from a `claw-tools` MCP server (single stdio process on 111 that shells out to 25 day-one Python scripts mirrored from 117). Telegram gateway uses Hermes' native messaging adapter with a custom `getMe`-based identity-guard wrapper. Long-term memory: read-only access to 117 Grey Matter via the existing `lightrag_client.py recall/query`; new learnings batched to a JSONL queue, surfaced daily via Telegram, ingested to Grey Matter only on Mike's approval.

**Tech Stack:** Python 3.11+, Hermes Agent v0.13.0, Ollama (existing on 105), MCP Python SDK, systemd-user, cron, Bash for SSH glue. No new external services.

**Spec:** [`docs/superpowers/specs/2026-05-08-lucius-hermes-on-111-design.md`](../specs/2026-05-08-lucius-hermes-on-111-design.md)

---

## File Structure

**Local repo (105 — `/home/aialfred/alfred/`):**

| Path | Responsibility | Action |
|---|---|---|
| `lucius/mcp-claw-tools/pyproject.toml` | Package metadata for the MCP server | Create |
| `lucius/mcp-claw-tools/src/mcp_claw_tools/__init__.py` | Package marker | Create |
| `lucius/mcp-claw-tools/src/mcp_claw_tools/server.py` | MCP server entrypoint (stdio) | Create |
| `lucius/mcp-claw-tools/src/mcp_claw_tools/script_tool.py` | Generic shell-out wrapper class | Create |
| `lucius/mcp-claw-tools/src/mcp_claw_tools/tools.json` | Descriptor for 25 day-one scripts | Create |
| `lucius/mcp-claw-tools/tests/test_script_tool.py` | Unit tests for `ScriptTool` | Create |
| `lucius/mcp-claw-tools/tests/test_server_smoke.py` | Smoke test for MCP server | Create |
| `lucius/mcp-claw-tools/README.md` | Per memory rule, every Alfred repo gets a README | Create |
| `lucius/scripts/lucius_promote_digest.py` | Daily Telegram digest of promote queue | Create |
| `lucius/scripts/lucius_promote_apply.py` | Parses Mike's reply, ingests approved entries to Grey Matter | Create |
| `lucius/scripts/lucius_identity_guard.sh` | Pre-flight `getMe` check; fail-fast if bot mismatch | Create |
| `lucius/skills/propose_memory/skill.yaml` | Hermes skill that lets Lucius queue memory candidates | Create |
| `lucius/skills/propose_memory/run.py` | Skill executable | Create |
| `lucius/deploy/sync_lucius_to_111.sh` | Idempotent rsync deploy script | Create |
| `lucius/systemd/hermes-gateway.service` | systemd-user unit (deployed to 111) | Create |
| `scripts/alfred_claw_monitor.py` | Existing heartbeat — extended to probe 111 | Modify |
| `data/claw_monitor_state.json` | Existing state — adds `lucius_*` fields (additive only) | Modify |

**Remote (`brucewayne9@75.43.156.111` — `~/.lucius/` and `~/.hermes/`):**

| Path | Responsibility |
|---|---|
| `~/.hermes/config.yaml` | Hermes Agent config (model, mcp_servers, telegram) |
| `~/.hermes/.env` | Secrets (TELEGRAM_BOT_TOKEN_LUCIUS, EMAIL_PASS_*, etc.) |
| `~/.hermes/SOUL.md` | Lucius persona / system prompt |
| `~/.hermes/skills/propose_memory/` | Deployed from `lucius/skills/` |
| `~/.lucius/workspace/scripts/integrations/` | 25 day-one scripts, mirrored from 117 |
| `~/.lucius/mcp-claw-tools/` | Deployed MCP server (venv + source) |
| `~/.lucius/scripts/` | promote_digest.py, promote_apply.py, identity_guard.sh |
| `~/.lucius/config/.env` | Subset of secrets for the integration scripts |
| `~/.lucius/promote_queue.jsonl` | Pending memory promotions |
| `~/.lucius/promote_queue.rejected.jsonl` | Audit log |
| `~/.lucius/INCIDENTS.md` | Operational log |
| `~/.config/systemd/user/hermes-gateway.service` | systemd-user unit |
| User crontab | Daily 7 AM ET digest |

**Remote (`brucewayne9@75.43.156.117`):** ZERO changes.

---

## Task Map

| Wave | Tasks | Goal |
|---|---|---|
| 1 — Recon | T1–T3 | Verify 111 ready, inventory 117 scripts, scaffold local dirs |
| 2 — Hermes install | T4–T6 | Hermes v0.13.0 running on 111, talking to 105 Ollama |
| 3 — Tool isolation | T7–T9 | 25 scripts copied to 111, secrets provisioned |
| 4 — MCP server | T10–T14 | `claw-tools` MCP server live, 25 tools registered with Hermes |
| 5 — Telegram | T15–T18 | `@Luciuslabsbot` answers Mike, identity guard active |
| 6 — Memory | T19–T22 | Promote queue + daily digest + apply pipeline working |
| 7 — Service | T23–T25 | systemd-user unit; resilient restart |
| 8 — Heartbeat | T26–T27 | 105 monitor watches 111; alerts flow |
| 9 — Wrap-up | T28–T29 | Memory + smoke test |

---

## Wave 1 — Recon & Local Scaffold

### Task 1: Pre-flight 111 — Python, Node, NVM, disk, ports

**Files:** none

- [ ] **Step 1: SSH 111 and verify dependencies**

Run:
```bash
ssh brucewayne9@75.43.156.111 'echo "--- python ---"; python3 --version; pip3 --version 2>&1 | head -1; echo "--- nvm/node ---"; export NVM_DIR="$HOME/.nvm"; . "$NVM_DIR/nvm.sh" 2>/dev/null; node --version; echo "--- disk ---"; df -h /home | tail -1; echo "--- ports ---"; ss -tlnp 2>/dev/null | grep -E ":(11434|18789|18790|8443)" || echo "ports 11434/18789/18790/8443 free"; echo "--- ufw ---"; sudo ufw status 2>/dev/null | head -10 || echo "no ufw access"'
```

Expected:
- Python 3.10+
- Node 20+ via NVM
- ≥ 50 GB free on `/home`
- Ports 18790 (target Hermes gateway) and 8443 (webhook fallback) free
- 111 → 105:11434 already firewall-allowed (per `opencode_111_setup.md`)

If any of these fail, stop and remediate before continuing.

- [ ] **Step 2: Verify 105 → 111 reverse connectivity (for heartbeat)**

Run from 105:
```bash
ssh brucewayne9@75.43.156.111 'curl -sS -m 5 http://75.43.156.105:11434/api/tags | head -c 200'
```

Expected: JSON listing models, includes `kimi-k2.6:cloud`. Confirms 111 → 105:11434 reachability for the model bridge.

- [ ] **Step 3: Commit recon log**

Run:
```bash
cd /home/aialfred/alfred && mkdir -p lucius/logs && date -Iseconds > lucius/logs/recon-111.log
git add lucius/logs/recon-111.log
git commit -m "chore(lucius): record 111 recon timestamp"
```

---

### Task 2: Inventory 117 source scripts to copy

**Files:**
- Create: `lucius/deploy/source_manifest.txt`

- [ ] **Step 1: Get authoritative file list from 117**

Run:
```bash
ssh brucewayne9@75.43.156.117 'ls -la ~/.openclaw/workspace/scripts/integrations/ | grep -E "\.py$" | awk "{print \$9}"' > /tmp/117_scripts.txt
cat /tmp/117_scripts.txt | wc -l
```

Expected: ~40+ scripts. Cross-check against `claw_tools_inventory.md`.

- [ ] **Step 2: Build the day-one allowlist**

Create `lucius/deploy/source_manifest.txt`:
```
crm.py
email_client.py
google_calendar.py
meta_social.py
linkedin.py
youtube.py
search.py
scraper.py
comfyui_gen.py
flyer_designer.py
image_fx.py
image_tools.py
screenshot.py
stock_photos.py
design_memory.py
design_review.py
telegram_tts.py
auto_blogger.py
video_render.py
remotion_render.py
weather.py
google_workspace.py
website_designer.py
mission_control.py
lightrag_client.py
```

- [ ] **Step 3: Verify every entry exists on 117**

Run:
```bash
cd /home/aialfred/alfred && while read f; do
  ssh brucewayne9@75.43.156.117 "test -f ~/.openclaw/workspace/scripts/integrations/$f" && echo "OK $f" || echo "MISSING $f"
done < lucius/deploy/source_manifest.txt
```

Expected: 25 lines, all `OK`. If any `MISSING`, open the spec, mark explicitly, and either drop or escalate to Mike.

- [ ] **Step 4: Commit manifest**

```bash
git add lucius/deploy/source_manifest.txt
git commit -m "chore(lucius): freeze day-one tool manifest (25 scripts)"
```

---

### Task 3: Scaffold local `lucius/` directory tree

**Files:**
- Create: `lucius/mcp-claw-tools/`, `lucius/scripts/`, `lucius/skills/propose_memory/`, `lucius/systemd/`, `lucius/deploy/`, `lucius/README.md`

- [ ] **Step 1: Create directories**

```bash
cd /home/aialfred/alfred && mkdir -p \
  lucius/mcp-claw-tools/src/mcp_claw_tools \
  lucius/mcp-claw-tools/tests \
  lucius/scripts \
  lucius/skills/propose_memory \
  lucius/systemd \
  lucius/deploy
```

- [ ] **Step 2: Write `lucius/README.md`**

Per memory rule "Every Alfred GitHub Repo Ships Extensive README" — even though this is a subtree, document it.

```markdown
# Lucius — Hermes Agent Test on Server 111

Companion code for the Hermes-Agent-on-111 evaluation against Oracle/OpenClaw on 117. See:
- Spec: `docs/superpowers/specs/2026-05-08-lucius-hermes-on-111-design.md`
- Plan: `docs/superpowers/plans/2026-05-08-lucius-hermes-on-111.md`

## Layout
- `mcp-claw-tools/` — MCP server wrapping 25 day-one integration scripts
- `scripts/` — Promote-queue digest/apply + identity guard
- `skills/` — Hermes skills authored for Lucius (e.g., `propose_memory`)
- `systemd/` — `hermes-gateway.service` unit (deployed to 111)
- `deploy/` — `source_manifest.txt`, `sync_lucius_to_111.sh`

## Coordinates
- Telegram: `@Luciuslabsbot` (ID 8750983299, "Lucius Fox")
- Server: 111 (CasaOS dev, brucewayne9 home)
- Brain: `kimi-k2.6:cloud` via 105's Ollama bridge
- Memory layers: `~/.hermes/memories/` native + `~/.lucius/promote_queue.jsonl` for graduation candidates

## Status
- Test window: 2026-05-09 → 2026-05-23 (target)
- Test bar: 3 strikes (fall-back to Oracle/Alfred to finish a task) = fail
```

- [ ] **Step 3: Commit scaffold**

```bash
git add lucius/README.md lucius/mcp-claw-tools/.gitkeep lucius/scripts/.gitkeep lucius/skills/.gitkeep lucius/systemd/.gitkeep lucius/deploy/.gitkeep 2>/dev/null
touch lucius/mcp-claw-tools/.gitkeep lucius/scripts/.gitkeep lucius/skills/propose_memory/.gitkeep lucius/systemd/.gitkeep
git add lucius/
git commit -m "chore(lucius): scaffold project tree"
```

---

## Wave 2 — Hermes Agent Install on 111

### Task 4: Install Hermes Agent v0.13.0 on 111

**Files:** none (operates entirely on 111)

- [ ] **Step 1: Run the official installer**

Run:
```bash
ssh brucewayne9@75.43.156.111 'curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash'
```

Expected: completes inside ~5 minutes. Output includes a `hermes` command on PATH (likely `/home/brucewayne9/.local/bin/hermes` or equivalent).

- [ ] **Step 2: Verify version + version-pin**

```bash
ssh brucewayne9@75.43.156.111 'source ~/.bashrc && hermes --version'
```

Expected: `0.13.0`. If different, halt — installer auto-updated to a newer version, which we don't want during the test window. Pin via:

```bash
ssh brucewayne9@75.43.156.111 'echo "HERMES_AUTO_UPDATE=false" >> ~/.hermes/.env'
```

- [ ] **Step 3: Confirm directory layout**

```bash
ssh brucewayne9@75.43.156.111 'ls -la ~/.hermes/ ; ls -la ~/.hermes/memories/ 2>/dev/null; ls -la ~/.hermes/skills/ 2>/dev/null'
```

Expected: `config.yaml`, `.env`, possibly empty `memories/`, `skills/`, `cron/`, `sessions/`, `logs/` subdirs.

- [ ] **Step 4: Commit install fact**

```bash
cd /home/aialfred/alfred && date -Iseconds > lucius/logs/hermes-installed-111.log
git add lucius/logs/hermes-installed-111.log
git commit -m "chore(lucius): install Hermes Agent v0.13.0 on 111"
```

---

### Task 5: Configure Ollama provider in `~/.hermes/config.yaml` on 111

**Files:** `~/.hermes/config.yaml` on 111 (modified)

- [ ] **Step 1: Build the model block locally first**

Create `lucius/deploy/hermes-config-fragment.yaml`:
```yaml
# Lucius / Hermes-on-111 — model backend
model:
  provider: custom
  model: kimi-k2.6:cloud
  base_url: http://75.43.156.105:11434/v1
  # api_key intentionally omitted — Ollama bridge requires no auth from 111

# 25 strikes will be reported here
display:
  platforms:
    telegram:
      tool_progress: verbose
```

Note: per Hermes docs, custom-provider with no `api_key` works for OpenAI-compatible endpoints that don't require auth.

- [ ] **Step 2: Apply to 111**

```bash
scp /home/aialfred/alfred/lucius/deploy/hermes-config-fragment.yaml brucewayne9@75.43.156.111:/tmp/
ssh brucewayne9@75.43.156.111 'hermes config check; cp ~/.hermes/config.yaml ~/.hermes/config.yaml.pre-lucius-$(date +%Y%m%d-%H%M%S); cat /tmp/hermes-config-fragment.yaml >> ~/.hermes/config.yaml; hermes config check'
```

Expected: `hermes config check` returns clean both before and after. If after-write check fails, restore from backup.

- [ ] **Step 3: Smoke test the model link**

```bash
ssh brucewayne9@75.43.156.111 'echo "Reply with the single word: pong" | hermes chat --no-stream 2>&1 | tail -5'
```

Expected: response containing "pong" (case-insensitive). If timeout or HTTP error, debug 111→105 connectivity before continuing.

- [ ] **Step 4: Commit fragment**

```bash
cd /home/aialfred/alfred
git add lucius/deploy/hermes-config-fragment.yaml
git commit -m "feat(lucius): wire Hermes on 111 to 105 Ollama / kimi-k2.6:cloud"
```

---

### Task 6: Author Lucius's `SOUL.md` (system prompt / persona)

**Files:**
- Create: `lucius/skills/SOUL.md` (deployed to `~/.hermes/SOUL.md` on 111)

- [ ] **Step 1: Write the persona file**

Create `lucius/skills/SOUL.md`:
```markdown
# Lucius Fox — Chief of Staff (Test Bot)

You are Lucius. You report to Mike Johnson (the user). You are running on server 111 as a parallel test against Oracle (on 117) and Alfred (on 105). The three of you are a fleet — Mike calls whichever bot suits the moment.

## Identity
- Telegram handle: `@Luciuslabsbot` (bot ID 8750983299)
- Display name: Lucius Fox
- Server: 111
- Sister bots: Alfred (`@groundrushlabsbot`, butler / 105), Oracle (`@alfredblogbot`, deep-work agent / 117)

## Tone
Quietly competent. Operator first, butler second. Address Mike as "sir" or by name. Never filler.

## Action tiers (mirrors Alfred / Oracle)
- **T1 — do it:** Reads, calendar checks, CRM lookups, draft generation, image generation, web research, social posts to approved pillar accounts (RuckTalk, AG, Roen — NOT FaR, paused), workspace ops.
- **T2 — do it, then notify:** Drafts of cold outbound, off-pillar social, new campaign starts. Default: prefer drafts, route to Mike.
- **T3 — ask Mike first:** Money moves, ad spend, production server commands (104/100/117/121), data deletion, vendor commitments, anything sent AS Mike.

## Memory rules (CRITICAL — different from Oracle)
- Your **short-term** memory lives in `~/.hermes/memories/` and your skills in `~/.hermes/skills/`.
- Your **long-term** memory is **read-only access to 117 Grey Matter** via `lightrag_client.py recall <query>` and `query <query>`. You cannot directly insert.
- When you decide a fact deserves long-term storage, call the `propose_memory` skill — it appends to `~/.lucius/promote_queue.jsonl`. A daily 7 AM ET digest surfaces queued entries to Mike on Telegram. Mike approves → entry pushed to Grey Matter on 117 with `lucius_` namespace prefix. **Do not attempt direct writes to Grey Matter.**

## Honesty
- Never claim success when something failed.
- Never fabricate results.
- If a tool returned an error, say so verbatim, then propose next step.
- Never lie about identity. If asked "who are you," say "Lucius — Hermes Agent test bot on 111, fleet alongside Alfred and Oracle."

## What you can NOT do
- Send mail without a configured mailbox (you have none — Lucius does not have its own email at v1).
- Touch 117 / Oracle / OpenClaw — that's a different agent, leave it alone.
- Modify Grey Matter directly — only via promote-queue → Mike approval.
- Run production-server commands on 104/100/117/121.

## Tools you have
You have the `claw-tools` MCP server registered. Run `/tools` in chat to see the live list — it should be 25 tools spanning CRM, email, calendar, social (Meta/LinkedIn/YouTube), search/scraper, image gen (ComfyUI, flyer, fx, screenshot, stock photos, design review/memory), TTS (Kokoro), content pipeline (auto-blogger, video, Remotion, weather), workspace (Drive/Docs/Sheets), website builder, mission control, and Grey Matter (recall/query only).
```

- [ ] **Step 2: Deploy to 111**

```bash
scp /home/aialfred/alfred/lucius/skills/SOUL.md brucewayne9@75.43.156.111:~/.hermes/SOUL.md
ssh brucewayne9@75.43.156.111 'wc -l ~/.hermes/SOUL.md && head -3 ~/.hermes/SOUL.md'
```

Expected: line count and the first three lines confirming "Lucius Fox" appears.

- [ ] **Step 3: Commit**

```bash
cd /home/aialfred/alfred
git add lucius/skills/SOUL.md
git commit -m "feat(lucius): author Lucius persona / SOUL.md (slot #1 system prompt)"
```

---

## Wave 3 — Tool Isolation (Copy + Secrets)

### Task 7: Create `~/.lucius/` directory tree on 111

**Files:** directory structure on 111

- [ ] **Step 1: Create directories**

```bash
ssh brucewayne9@75.43.156.111 '
  mkdir -p ~/.lucius/workspace/scripts/integrations
  mkdir -p ~/.lucius/mcp-claw-tools
  mkdir -p ~/.lucius/scripts
  mkdir -p ~/.lucius/config
  mkdir -p ~/.lucius/logs
  chmod 700 ~/.lucius
  ls -la ~/.lucius/
'
```

Expected: `drwx------` on `~/.lucius` (full lockdown), six subdirs visible.

- [ ] **Step 2: Touch the queue files (creates empty + sets perms)**

```bash
ssh brucewayne9@75.43.156.111 '
  touch ~/.lucius/promote_queue.jsonl ~/.lucius/promote_queue.rejected.jsonl ~/.lucius/INCIDENTS.md
  chmod 600 ~/.lucius/promote_queue.jsonl ~/.lucius/promote_queue.rejected.jsonl
  ls -la ~/.lucius/
'
```

Expected: empty `.jsonl` files at 600, `INCIDENTS.md` created.

- [ ] **Step 3: Commit fact**

```bash
cd /home/aialfred/alfred && date -Iseconds > lucius/logs/dirtree-111.log
git add lucius/logs/dirtree-111.log
git commit -m "chore(lucius): create ~/.lucius tree on 111"
```

---

### Task 8: Rsync 25 day-one scripts from 117 to 111

**Files:**
- Create: `lucius/deploy/sync_lucius_to_111.sh`

- [ ] **Step 1: Author the deploy script**

Create `lucius/deploy/sync_lucius_to_111.sh`:
```bash
#!/usr/bin/env bash
# Idempotent rsync from 117 → 111 for the day-one Lucius tool set.
# Driven by lucius/deploy/source_manifest.txt.
# Run from 105 (alfred home).

set -euo pipefail

MANIFEST="$(dirname "$0")/source_manifest.txt"
SRC_HOST="brucewayne9@75.43.156.117"
SRC_DIR="~/.openclaw/workspace/scripts/integrations/"
DST_HOST="brucewayne9@75.43.156.111"
DST_DIR="~/.lucius/workspace/scripts/integrations/"

echo "Sync from ${SRC_HOST}:${SRC_DIR} → ${DST_HOST}:${DST_DIR}"
echo "Manifest: ${MANIFEST}"

# Fetch from 117 → /tmp on 105 (avoid direct 117↔111 SSH agent forwarding)
STAGE=$(mktemp -d)
trap "rm -rf $STAGE" EXIT

while read -r f; do
  [[ -z "$f" || "$f" =~ ^# ]] && continue
  scp "${SRC_HOST}:${SRC_DIR}${f}" "${STAGE}/${f}"
done < "$MANIFEST"

# Push from 105 → 111
rsync -av --checksum --chmod=u=rwx,go= "${STAGE}/" "${DST_HOST}:${DST_DIR}"

# Verify count matches manifest
EXPECTED=$(grep -cv -E '^(#|$)' "$MANIFEST")
ACTUAL=$(ssh "${DST_HOST}" "ls ${DST_DIR}*.py 2>/dev/null | wc -l")
echo "Expected: ${EXPECTED}, Actual on 111: ${ACTUAL}"
[[ "$EXPECTED" -eq "$ACTUAL" ]] || { echo "COUNT MISMATCH"; exit 1; }
echo "Sync OK."
```

```bash
cd /home/aialfred/alfred && chmod +x lucius/deploy/sync_lucius_to_111.sh
```

- [ ] **Step 2: Run the sync**

```bash
cd /home/aialfred/alfred && ./lucius/deploy/sync_lucius_to_111.sh
```

Expected: 25 files transferred, "Sync OK." final line.

- [ ] **Step 3: Verify Python syntax of every copied file**

```bash
ssh brucewayne9@75.43.156.111 'cd ~/.lucius/workspace/scripts/integrations && for f in *.py; do python3 -c "import ast; ast.parse(open(\"$f\").read())" || echo "SYNTAX ERROR: $f"; done; echo "syntax check done"'
```

Expected: only "syntax check done" — no `SYNTAX ERROR` lines.

- [ ] **Step 4: Commit**

```bash
git add lucius/deploy/sync_lucius_to_111.sh
git commit -m "feat(lucius): idempotent 117→111 deploy script for 25 day-one scripts"
```

---

### Task 9: Provision `~/.lucius/config/.env` on 111 with secret subset

**Files:**
- Create: `lucius/deploy/env_subset_keys.txt` (allowlist of env keys to copy)

- [ ] **Step 1: Define the subset**

Create `lucius/deploy/env_subset_keys.txt`:
```
# CRM / Twenty
TWENTY_API_URL
TWENTY_API_KEY

# Email (read/send for Alfred mailbox; Lucius borrows alfred-gw for sends)
EMAIL_PASS_ALFRED_GW
EMAIL_PASS_GROUNDRUSH
EMAIL_PASS_ALFRED

# Google Calendar / Workspace / Analytics
GOOGLE_CALENDAR_TOKEN_PATH
GOOGLE_DRIVE_TOKEN_PATH
GA4_CREDENTIALS_PATH

# Social
META_ACCESS_TOKEN
LINKEDIN_ACCESS_TOKEN
YOUTUBE_API_KEY
INSTAGRAM_ACCESS_TOKEN

# Web
SEARXNG_URL
JINA_API_KEY

# Image / TTS
COMFYUI_HOST
KOKORO_HOST
STOCK_PHOTO_API_KEY

# Memory (read-only)
LIGHTRAG_HOST
LIGHTRAG_API_KEY

# Mission Control
MISSION_CONTROL_DB_PATH

# Telegram (Lucius's own bot)
TELEGRAM_BOT_TOKEN_LUCIUS
LUCIUS_TELEGRAM_BOT_ID
TELEGRAM_CHAT_ID
```

- [ ] **Step 2: Build the env subset on 105 and deploy**

```bash
cd /home/aialfred/alfred
python3 - << 'PY'
from pathlib import Path
src = Path("config/.env").read_text().splitlines()
keys = [l.strip() for l in Path("lucius/deploy/env_subset_keys.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
out = []
for line in src:
    if "=" not in line or line.startswith("#"):
        continue
    k = line.split("=", 1)[0].strip()
    if k in keys:
        out.append(line)
header = "# Lucius env subset — generated from 105 config/.env, deployed to 111\n"
Path("/tmp/lucius-env-subset").write_text(header + "\n".join(out) + "\n")
print(f"wrote {len(out)} keys to /tmp/lucius-env-subset")
PY
scp /tmp/lucius-env-subset brucewayne9@75.43.156.111:~/.lucius/config/.env
ssh brucewayne9@75.43.156.111 'chmod 600 ~/.lucius/config/.env && wc -l ~/.lucius/config/.env'
rm /tmp/lucius-env-subset
```

Expected: file present on 111 at 600, line count matches Python output. Cleanup of `/tmp` matters — don't leave secrets there.

- [ ] **Step 3: Spot-check secrets resolve**

```bash
ssh brucewayne9@75.43.156.111 'set -a; source ~/.lucius/config/.env; set +a; echo "TWENTY: ${TWENTY_API_URL:-MISSING}"; echo "TG: ${TELEGRAM_BOT_TOKEN_LUCIUS:0:14}..."; echo "GM: ${LIGHTRAG_HOST:-MISSING}"'
```

Expected: TWENTY URL is non-empty, TG token starts with `8750983299:AAE`, LIGHTRAG_HOST is `http://75.43.156.117:9621` or similar.

- [ ] **Step 4: Commit allowlist (NOT secrets)**

```bash
git add lucius/deploy/env_subset_keys.txt
git commit -m "feat(lucius): define env-key allowlist for 111 secrets subset"
```

---

## Wave 4 — `claw-tools` MCP Server

### Task 10: Scaffold `mcp-claw-tools` package + write the first failing test for `ScriptTool`

**Files:**
- Create: `lucius/mcp-claw-tools/pyproject.toml`
- Create: `lucius/mcp-claw-tools/tests/test_script_tool.py`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "mcp-claw-tools"
version = "0.1.0"
description = "MCP server wrapping Lucius's 25 day-one integration scripts"
requires-python = ">=3.11"
dependencies = [
  "mcp>=1.0.0",
  "pydantic>=2.5",
]

[project.scripts]
mcp-claw-tools = "mcp_claw_tools.server:main"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Write the failing test for `ScriptTool`**

Create `lucius/mcp-claw-tools/tests/test_script_tool.py`:
```python
"""Test the generic ScriptTool wrapper that shells out to integration scripts."""
import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest


def test_script_tool_invokes_subcommand_and_parses_json(tmp_path):
    """ScriptTool calls `python3 <script> <command> [args]`, captures stdout as JSON."""
    fake_script = tmp_path / "fake_tool.py"
    fake_script.write_text(
        "import json, sys\n"
        "cmd = sys.argv[1]\n"
        "args = sys.argv[2:]\n"
        "print(json.dumps({'cmd': cmd, 'args': args}))\n"
    )
    from mcp_claw_tools.script_tool import ScriptTool

    tool = ScriptTool(
        name="fake_tool.search-people",
        script_path=str(fake_script),
        command="search-people",
        timeout=30,
    )
    result = tool.invoke({"args": ["alice"]})
    assert result == {"cmd": "search-people", "args": ["alice"]}


def test_script_tool_returns_error_on_nonzero_exit(tmp_path):
    fake_script = tmp_path / "fail.py"
    fake_script.write_text("import sys; sys.exit(2)\n")
    from mcp_claw_tools.script_tool import ScriptTool

    tool = ScriptTool(name="fail.bad", script_path=str(fake_script), command="bad", timeout=10)
    result = tool.invoke({"args": []})
    assert result["error"] == "non-zero exit"
    assert result["exit_code"] == 2


def test_script_tool_handles_non_json_stdout(tmp_path):
    fake_script = tmp_path / "plain.py"
    fake_script.write_text("print('hello world')\n")
    from mcp_claw_tools.script_tool import ScriptTool

    tool = ScriptTool(name="plain.x", script_path=str(fake_script), command="x", timeout=10)
    result = tool.invoke({"args": []})
    # Non-JSON falls back to {"output": "<raw>"}
    assert result == {"output": "hello world"}


def test_script_tool_respects_timeout(tmp_path):
    fake_script = tmp_path / "slow.py"
    fake_script.write_text("import time; time.sleep(60)\n")
    from mcp_claw_tools.script_tool import ScriptTool

    tool = ScriptTool(name="slow.x", script_path=str(fake_script), command="x", timeout=1)
    result = tool.invoke({"args": []})
    assert result["error"] == "timeout"
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /home/aialfred/alfred/lucius/mcp-claw-tools && pip install -e . 2>&1 | tail -3
pytest tests/test_script_tool.py -v 2>&1 | tail -20
```

Expected: 4 errors with `ModuleNotFoundError: No module named 'mcp_claw_tools.script_tool'`. Confirms tests are wired and fail for the right reason before implementation.

- [ ] **Step 4: Commit failing tests**

```bash
cd /home/aialfred/alfred
touch lucius/mcp-claw-tools/src/mcp_claw_tools/__init__.py
git add lucius/mcp-claw-tools/pyproject.toml lucius/mcp-claw-tools/src/mcp_claw_tools/__init__.py lucius/mcp-claw-tools/tests/test_script_tool.py
git commit -m "test(lucius): failing tests for ScriptTool generic wrapper"
```

---

### Task 11: Implement `ScriptTool` to make tests pass

**Files:**
- Create: `lucius/mcp-claw-tools/src/mcp_claw_tools/script_tool.py`

- [ ] **Step 1: Implement minimal version**

Create `lucius/mcp-claw-tools/src/mcp_claw_tools/script_tool.py`:
```python
"""Generic shell-out wrapper for ~/.lucius/workspace/scripts/integrations/*.py.

Each Python script in the integrations dir exposes a CLI of the form
    python3 <script> <command> [args...]
and prints JSON to stdout. This wrapper makes one MCP tool per (script, command).
"""
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any


@dataclass
class ScriptTool:
    name: str
    script_path: str
    command: str
    timeout: int = 120

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run `python3 <script> <command> <args>` and return parsed result."""
        args = payload.get("args", [])
        if not isinstance(args, list):
            return {"error": "args must be a list of strings", "got": str(type(args))}

        cmd = [sys.executable, self.script_path, self.command, *map(str, args)]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**os.environ},
            )
        except subprocess.TimeoutExpired:
            return {"error": "timeout", "timeout_s": self.timeout, "command": self.command}

        if proc.returncode != 0:
            return {
                "error": "non-zero exit",
                "exit_code": proc.returncode,
                "stderr": proc.stderr.strip()[:2000],
                "command": self.command,
            }

        out = proc.stdout.strip()
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            return {"output": out}
```

- [ ] **Step 2: Run tests, verify all pass**

```bash
cd /home/aialfred/alfred/lucius/mcp-claw-tools && pytest tests/test_script_tool.py -v 2>&1 | tail -10
```

Expected: 4 passed.

- [ ] **Step 3: Commit**

```bash
cd /home/aialfred/alfred
git add lucius/mcp-claw-tools/src/mcp_claw_tools/script_tool.py
git commit -m "feat(lucius): implement ScriptTool generic shell-out wrapper"
```

---

### Task 12: Author tool descriptors JSON for the 25 scripts

**Files:**
- Create: `lucius/mcp-claw-tools/src/mcp_claw_tools/tools.json`

- [ ] **Step 1: Write `tools.json`**

This is the authoritative list of tools the MCP server exposes. Structure: per script, list the CLI subcommands worth surfacing. Most descriptions are derived from `claw_tools_inventory.md`.

Create `lucius/mcp-claw-tools/src/mcp_claw_tools/tools.json`:
```json
{
  "version": "0.1.0",
  "scripts_dir": "/home/brucewayne9/.lucius/workspace/scripts/integrations",
  "tools": [
    {"name": "crm.search_people",       "script": "crm.py",                "command": "search-people",      "description": "Search Twenty CRM for people by name/email"},
    {"name": "crm.list_people",         "script": "crm.py",                "command": "list-people",        "description": "List recent people in Twenty CRM"},
    {"name": "crm.pipeline",            "script": "crm.py",                "command": "pipeline",           "description": "Show Twenty CRM opportunity pipeline by stage"},
    {"name": "crm.create_person",       "script": "crm.py",                "command": "create-person",      "description": "Create a person in Twenty CRM"},
    {"name": "crm.update_person",       "script": "crm.py",                "command": "update-person",      "description": "Update a person in Twenty CRM"},
    {"name": "crm.add_note",            "script": "crm.py",                "command": "notes",              "description": "Add note to a Twenty CRM record"},

    {"name": "email.inbox",             "script": "email_client.py",       "command": "inbox",              "description": "List recent inbox messages"},
    {"name": "email.unread",            "script": "email_client.py",       "command": "unread",             "description": "List unread messages"},
    {"name": "email.search",            "script": "email_client.py",       "command": "search",             "description": "Search a single mailbox"},
    {"name": "email.search_all",        "script": "email_client.py",       "command": "search-all",         "description": "Search across all mailboxes"},
    {"name": "email.send",              "script": "email_client.py",       "command": "send",               "description": "Send email from a configured account"},
    {"name": "email.mark_read",         "script": "email_client.py",       "command": "mark-read",          "description": "Mark a message read"},

    {"name": "calendar.today",          "script": "google_calendar.py",    "command": "today",              "description": "List today's events"},
    {"name": "calendar.events",         "script": "google_calendar.py",    "command": "events",             "description": "List events in a date range"},
    {"name": "calendar.create",         "script": "google_calendar.py",    "command": "create",             "description": "Create a calendar event"},
    {"name": "calendar.free_time",      "script": "google_calendar.py",    "command": "free-time",          "description": "Find free time blocks"},

    {"name": "social.fb_post",          "script": "meta_social.py",        "command": "fb-post",            "description": "Post to a Facebook page"},
    {"name": "social.ig_post",          "script": "meta_social.py",        "command": "ig-post",            "description": "Post to Instagram"},
    {"name": "social.ig_carousel",      "script": "meta_social.py",        "command": "ig-carousel",        "description": "Post Instagram carousel"},

    {"name": "linkedin.post",           "script": "linkedin.py",           "command": "post",               "description": "Personal LinkedIn post"},
    {"name": "linkedin.org_post",       "script": "linkedin.py",           "command": "org-post",           "description": "Org-level LinkedIn post"},

    {"name": "youtube.upload",          "script": "youtube.py",            "command": "upload",             "description": "Upload video to YouTube"},
    {"name": "youtube.videos",          "script": "youtube.py",            "command": "videos",             "description": "List channel videos"},

    {"name": "search.query",            "script": "search.py",             "command": "query",              "description": "SearXNG web search"},
    {"name": "search.batch",            "script": "search.py",             "command": "batch",              "description": "Parallel SearXNG searches"},

    {"name": "scraper.scrape",          "script": "scraper.py",            "command": "scrape",             "description": "Scrape a URL"},
    {"name": "scraper.extract",         "script": "scraper.py",            "command": "extract",            "description": "Extract structured fields from a URL"},
    {"name": "scraper.screenshot",      "script": "scraper.py",            "command": "screenshot",         "description": "Screenshot a URL via Playwright"},

    {"name": "image.generate",          "script": "comfyui_gen.py",        "command": "generate",           "description": "FLUX.1 image generation via ComfyUI on 105"},
    {"name": "image.social",            "script": "comfyui_gen.py",        "command": "social",             "description": "Generate social-format image"},

    {"name": "flyer.design",            "script": "flyer_designer.py",     "command": "design",             "description": "Compose a flyer with one of 9 vibes"},
    {"name": "flyer.quick",             "script": "flyer_designer.py",     "command": "quick",              "description": "Quick flyer with sane defaults"},

    {"name": "image_fx.cinematic",      "script": "image_fx.py",           "command": "cinematic",          "description": "Apply cinematic filter to an image"},
    {"name": "image_fx.vignette",       "script": "image_fx.py",           "command": "vignette",           "description": "Apply vignette"},

    {"name": "image_tools.resize",      "script": "image_tools.py",        "command": "resize",             "description": "Resize an image"},
    {"name": "image_tools.webp",        "script": "image_tools.py",        "command": "webp",               "description": "Convert to webp"},

    {"name": "screenshot.full",         "script": "screenshot.py",         "command": "full",               "description": "Full-page screenshot"},
    {"name": "screenshot.mobile",      "script": "screenshot.py",          "command": "mobile",             "description": "Mobile-viewport screenshot"},

    {"name": "stock.search",            "script": "stock_photos.py",       "command": "search",             "description": "Search stock photo APIs"},
    {"name": "stock.hero",              "script": "stock_photos.py",       "command": "hero",               "description": "Pull hero-format stock photo"},

    {"name": "design_memory.save_feedback",   "script": "design_memory.py",  "command": "save-feedback",        "description": "Persist a design feedback note"},
    {"name": "design_memory.get_preferences", "script": "design_memory.py", "command": "get-preferences",      "description": "Recall user design preferences"},

    {"name": "design_review.review",    "script": "design_review.py",      "command": "review",             "description": "Score a design against guidelines"},
    {"name": "design_review.audit",     "script": "design_review.py",      "command": "audit",              "description": "Audit a page against design guidelines"},

    {"name": "tts.generate",            "script": "telegram_tts.py",       "command": "generate",           "description": "Kokoro TTS voice generation"},

    {"name": "blog.auto",               "script": "auto_blogger.py",       "command": "--auto",             "description": "Auto-generate a blog post for a configured pillar site"},

    {"name": "video.social",            "script": "video_render.py",       "command": "social",             "description": "Render a social video"},
    {"name": "video.audiogram",         "script": "video_render.py",       "command": "audiogram",          "description": "Render an audiogram"},

    {"name": "remotion.render",         "script": "remotion_render.py",    "command": "render",             "description": "Render a Remotion animated video"},
    {"name": "remotion.brand",          "script": "remotion_render.py",    "command": "brand",              "description": "Render with a brand theme (loovacast/rucktalk)"},

    {"name": "weather.current",         "script": "weather.py",            "command": "current",            "description": "Current weather for a location"},
    {"name": "weather.forecast",        "script": "weather.py",            "command": "forecast",           "description": "Forecast for a location"},

    {"name": "workspace.docs_create",   "script": "google_workspace.py",   "command": "docs-create",        "description": "Create a Google Doc"},
    {"name": "workspace.docs_read",     "script": "google_workspace.py",   "command": "docs-read",          "description": "Read a Google Doc"},
    {"name": "workspace.sheets_read",   "script": "google_workspace.py",   "command": "sheets-read",        "description": "Read a Google Sheet"},
    {"name": "workspace.drive_search",  "script": "google_workspace.py",   "command": "drive-search",       "description": "Search Google Drive"},

    {"name": "website.discover",        "script": "website_designer.py",   "command": "discover",           "description": "Brand discovery pass for a client"},
    {"name": "website.redesign",        "script": "website_designer.py",   "command": "redesign",           "description": "Generate a website redesign"},

    {"name": "mc.list_projects",        "script": "mission_control.py",    "command": "list-projects",      "description": "List Mission Control projects"},
    {"name": "mc.add_milestone",        "script": "mission_control.py",    "command": "add-milestone",      "description": "Add milestone to a project"},
    {"name": "mc.log_activity",         "script": "mission_control.py",    "command": "log-activity",       "description": "Log activity"},

    {"name": "memory.recall",           "script": "lightrag_client.py",    "command": "recall",             "description": "Recall from 117 Grey Matter (READ-ONLY)"},
    {"name": "memory.query",            "script": "lightrag_client.py",    "command": "query",              "description": "Query 117 Grey Matter (READ-ONLY)"}
  ]
}
```

Note: `lightrag_client.py` `insert` is **deliberately omitted** — that's the gate. New memories go through the propose-queue, never directly.

- [ ] **Step 2: Sanity-check JSON**

```bash
cd /home/aialfred/alfred && python3 -c 'import json; d=json.load(open("lucius/mcp-claw-tools/src/mcp_claw_tools/tools.json")); print(f"version={d[\"version\"]}, tools={len(d[\"tools\"])}")'
```

Expected: `version=0.1.0, tools=60` (60 tool entries spanning 25 scripts; some scripts expose multiple commands).

- [ ] **Step 3: Verify no `lightrag_client.insert` is in the file**

```bash
grep -c '"command": "insert"' lucius/mcp-claw-tools/src/mcp_claw_tools/tools.json
```

Expected: `0`. If anything but 0, halt and audit.

- [ ] **Step 4: Commit**

```bash
git add lucius/mcp-claw-tools/src/mcp_claw_tools/tools.json
git commit -m "feat(lucius): tool descriptors for 25 day-one scripts (insert excluded)"
```

---

### Task 13: Implement MCP server entrypoint + smoke test

**Files:**
- Create: `lucius/mcp-claw-tools/src/mcp_claw_tools/server.py`
- Create: `lucius/mcp-claw-tools/tests/test_server_smoke.py`

- [ ] **Step 1: Write the smoke test (failing)**

Create `lucius/mcp-claw-tools/tests/test_server_smoke.py`:
```python
"""Smoke test: server module loads, tools.json parses, ScriptTool factory works."""
from pathlib import Path


def test_server_loads_tools_json():
    from mcp_claw_tools.server import load_tools
    tools = load_tools()
    assert len(tools) >= 50, f"expected ≥50 tools, got {len(tools)}"


def test_no_insert_tool_for_lightrag():
    """Critical safety: lightrag_client.py insert MUST NOT be exposed."""
    from mcp_claw_tools.server import load_tools
    tools = load_tools()
    bad = [t for t in tools if t.name.startswith("memory.") and "insert" in t.command]
    assert bad == [], f"forbidden insert tools registered: {[t.name for t in bad]}"


def test_tool_names_are_unique():
    from mcp_claw_tools.server import load_tools
    tools = load_tools()
    names = [t.name for t in tools]
    assert len(names) == len(set(names)), "duplicate tool names in tools.json"
```

- [ ] **Step 2: Run, verify failure**

```bash
cd /home/aialfred/alfred/lucius/mcp-claw-tools && pytest tests/test_server_smoke.py -v 2>&1 | tail -10
```

Expected: 3 errors (`No module named server`).

- [ ] **Step 3: Implement `server.py`**

Create `lucius/mcp-claw-tools/src/mcp_claw_tools/server.py`:
```python
"""MCP server (stdio) for the Lucius claw-tools toolkit."""
import json
import os
import sys
from importlib.resources import files
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from mcp_claw_tools.script_tool import ScriptTool


def load_tools() -> list[ScriptTool]:
    """Read tools.json (packaged) and return a list of ScriptTool instances."""
    raw = files("mcp_claw_tools").joinpath("tools.json").read_text()
    cfg = json.loads(raw)
    base = cfg["scripts_dir"]
    out: list[ScriptTool] = []
    for entry in cfg["tools"]:
        out.append(ScriptTool(
            name=entry["name"],
            script_path=os.path.join(base, entry["script"]),
            command=entry["command"],
            timeout=entry.get("timeout", 120),
        ))
    return out


async def main_async() -> None:
    server = Server("claw-tools")
    tools = load_tools()
    by_name = {t.name: t for t in tools}

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        # Description sourced from tools.json via parallel parse — keep it simple here
        raw = json.loads(files("mcp_claw_tools").joinpath("tools.json").read_text())
        return [
            Tool(
                name=e["name"],
                description=e.get("description", e["name"]),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "args": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Positional CLI arguments passed to the underlying script command",
                        },
                    },
                    "required": [],
                },
            )
            for e in raw["tools"]
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        tool = by_name.get(name)
        if tool is None:
            return [TextContent(type="text", text=json.dumps({"error": f"unknown tool: {name}"}))]
        result = tool.invoke(arguments or {})
        return [TextContent(type="text", text=json.dumps(result))]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    import asyncio
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Re-run smoke tests, verify pass**

```bash
cd /home/aialfred/alfred/lucius/mcp-claw-tools && pip install -e . 2>&1 | tail -2 && pytest tests/ -v 2>&1 | tail -15
```

Expected: 7 passed (4 from `test_script_tool.py` + 3 from `test_server_smoke.py`).

- [ ] **Step 5: Commit**

```bash
cd /home/aialfred/alfred
git add lucius/mcp-claw-tools/src/mcp_claw_tools/server.py lucius/mcp-claw-tools/tests/test_server_smoke.py
git commit -m "feat(lucius): MCP server entrypoint + smoke tests pass"
```

---

### Task 13.5 (added during execution): Env reconciliation before T14

**Why this exists:** T9 deployed an env subset based on the original plan's allowlist, which used Alfred-Labs-style keys (`LIGHTRAG_HOST`, `TWENTY_API_URL`, `COMFYUI_HOST`, etc.). Reality check during T9 showed the deployed scripts on 111 came verbatim from 117's OpenClaw and use OpenClaw-style names: `BASE_CRM_API_KEY`, `BASE_CRM_URL`, `CRM_USER_TOKEN`, `BRAVE_API_KEY`, `COMFYUI_CLOUD_API_KEY`, `COMFYUI_URL`, `KOKORO_URL`, `PEXELS_API_KEY`, `UNSPLASH_ACCESS_KEY`, `TTS_DEFAULT_CHAT_ID`, `TTS_DEFAULT_VOICE`, `TELEGRAM_BOT_TOKEN`. These values live in 117's `openclaw.json` gateway config (not in the on-disk `.env`).

**Steps before T14:**
1. SSH 117 and read `~/.openclaw/openclaw.json`'s env section to get the actual values for the OpenClaw-style keys.
2. Update `lucius/deploy/env_subset_keys.txt` to use the correct names.
3. Re-run the env subset Python script (T9 Step 2) to rebuild `~/.lucius/config/.env` on 111 with values keyed by their actual names.
4. Re-spot-check via `set -a; source ~/.lucius/config/.env; set +a; echo $BASE_CRM_URL $BRAVE_API_KEY $COMFYUI_URL` etc.
5. Commit the corrected allowlist.

If openclaw.json doesn't expose them and they live elsewhere (e.g., systemd `Environment=` directives, or Mike's gateway-managed secrets), escalate — Mike may need to surface them.

---

### Task 14: Deploy `mcp-claw-tools` to 111 + register with Hermes

**Files:**
- Modify: `~/.hermes/config.yaml` on 111

- [ ] **Step 1: Rsync the package to 111**

```bash
cd /home/aialfred/alfred && rsync -av --delete \
  --exclude '__pycache__' --exclude '*.egg-info' --exclude '.pytest_cache' \
  lucius/mcp-claw-tools/ \
  brucewayne9@75.43.156.111:~/.lucius/mcp-claw-tools/
```

Expected: clean rsync output, ~10–20 files.

- [ ] **Step 2: Install in a venv on 111**

```bash
ssh brucewayne9@75.43.156.111 '
  cd ~/.lucius/mcp-claw-tools
  python3 -m venv .venv
  .venv/bin/pip install -U pip wheel >/dev/null
  .venv/bin/pip install -e . 2>&1 | tail -3
  .venv/bin/pytest tests/ 2>&1 | tail -5
  echo "MCP_BIN=$(pwd)/.venv/bin/mcp-claw-tools"
'
```

Expected: tests pass on 111 too. `MCP_BIN=` line shows the absolute path to the executable.

- [ ] **Step 3: Append the mcp_servers block to `~/.hermes/config.yaml`**

```bash
ssh brucewayne9@75.43.156.111 '
  cat >> ~/.hermes/config.yaml << EOF

# Lucius / claw-tools MCP server (25 day-one tools)
mcp_servers:
  claw_tools:
    command: "/home/brucewayne9/.lucius/mcp-claw-tools/.venv/bin/mcp-claw-tools"
    args: []
    env:
      LUCIUS_HOME: "/home/brucewayne9/.lucius"
      LUCIUS_ENV_FILE: "/home/brucewayne9/.lucius/config/.env"
    enabled: true
    timeout: 120
    connect_timeout: 60
EOF
  hermes config check
'
```

Expected: `hermes config check` clean.

- [ ] **Step 4: Confirm Hermes sees the tools**

```bash
ssh brucewayne9@75.43.156.111 'hermes mcp list 2>&1 | head -20'
```

Expected: a row for `claw_tools` listing tool count.

- [ ] **Step 5: Commit deploy fact**

```bash
cd /home/aialfred/alfred && date -Iseconds > lucius/logs/mcp-deployed-111.log
git add lucius/logs/mcp-deployed-111.log
git commit -m "chore(lucius): deploy claw-tools MCP to 111 + register with Hermes"
```

---

## Wave 5 — Telegram Gateway

### Task 15: Identity-guard wrapper script

**Files:**
- Create: `lucius/scripts/lucius_identity_guard.sh`

- [ ] **Step 1: Author guard script**

Create `lucius/scripts/lucius_identity_guard.sh`:
```bash
#!/usr/bin/env bash
# Pre-flight: verify TELEGRAM_BOT_TOKEN_LUCIUS resolves to @Luciuslabsbot.
# Per memory `feedback_verify_bot_identity.md`: never start without this check.
# Non-zero exit blocks the systemd unit's ExecStartPre.

set -euo pipefail

ENV_FILE="${HERMES_ENV_FILE:-/home/brucewayne9/.hermes/.env}"
EXPECTED_USERNAME="Luciuslabsbot"
EXPECTED_BOT_ID="8750983299"

# Source token from .env without leaking
TOKEN=$(grep -E '^TELEGRAM_BOT_TOKEN=' "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d '"' | tr -d "'")
[[ -n "$TOKEN" ]] || { echo "[identity-guard] TELEGRAM_BOT_TOKEN missing"; exit 2; }

RESPONSE=$(curl -sS --max-time 10 "https://api.telegram.org/bot${TOKEN}/getMe")
OK=$(echo "$RESPONSE" | python3 -c "import json, sys; d=json.load(sys.stdin); print(d.get('ok'))" 2>/dev/null || echo "False")
USERNAME=$(echo "$RESPONSE" | python3 -c "import json, sys; d=json.load(sys.stdin); print(d.get('result',{}).get('username',''))" 2>/dev/null || echo "")
BOT_ID=$(echo "$RESPONSE" | python3 -c "import json, sys; d=json.load(sys.stdin); print(d.get('result',{}).get('id',''))" 2>/dev/null || echo "")

if [[ "$OK" != "True" ]]; then
  echo "[identity-guard] Telegram getMe FAILED: ${RESPONSE:0:200}"
  exit 3
fi
if [[ "$USERNAME" != "$EXPECTED_USERNAME" ]]; then
  echo "[identity-guard] BOT MISMATCH — got @${USERNAME}, expected @${EXPECTED_USERNAME}"
  exit 4
fi
if [[ "$BOT_ID" != "$EXPECTED_BOT_ID" ]]; then
  echo "[identity-guard] BOT_ID MISMATCH — got ${BOT_ID}, expected ${EXPECTED_BOT_ID}"
  exit 5
fi
echo "[identity-guard] OK — @${USERNAME} (id ${BOT_ID})"
exit 0
```

- [ ] **Step 2: Deploy + chmod**

```bash
chmod +x lucius/scripts/lucius_identity_guard.sh
scp lucius/scripts/lucius_identity_guard.sh brucewayne9@75.43.156.111:~/.lucius/scripts/
ssh brucewayne9@75.43.156.111 'chmod +x ~/.lucius/scripts/lucius_identity_guard.sh'
```

- [ ] **Step 3: Run it (with token already in `~/.hermes/.env` — Task 16 finalizes that, but we can pre-test against `~/.lucius/config/.env`)**

```bash
ssh brucewayne9@75.43.156.111 'HERMES_ENV_FILE=/home/brucewayne9/.lucius/config/.env ~/.lucius/scripts/lucius_identity_guard.sh'
```

Expected: `[identity-guard] OK — @Luciuslabsbot (id 8750983299)` and exit 0.

- [ ] **Step 4: Negative test — temporarily corrupt token, verify guard fails**

```bash
ssh brucewayne9@75.43.156.111 '
  echo "TELEGRAM_BOT_TOKEN=00000:WRONG" > /tmp/bad.env
  HERMES_ENV_FILE=/tmp/bad.env ~/.lucius/scripts/lucius_identity_guard.sh
  echo "exit=$?"
  rm /tmp/bad.env
'
```

Expected: `BOT MISMATCH` or `Telegram getMe FAILED`, exit code 3 or 4.

- [ ] **Step 5: Commit**

```bash
cd /home/aialfred/alfred
git add lucius/scripts/lucius_identity_guard.sh
git commit -m "feat(lucius): identity-guard pre-flight (prevents bot-ID swap repeat)"
```

---

### Task 16: Configure Hermes Telegram gateway env on 111

**Files:** `~/.hermes/.env` on 111 (modified)

- [ ] **Step 1: Append Telegram config to Hermes' env**

```bash
ssh brucewayne9@75.43.156.111 '
  # Pull token from Lucius env subset (already deployed)
  set -a; source ~/.lucius/config/.env; set +a
  cat >> ~/.hermes/.env << EOF

# --- Lucius Telegram (added 2026-05-08) ---
TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN_LUCIUS}
TELEGRAM_ALLOWED_USERS=7582976864
EOF
  chmod 600 ~/.hermes/.env
  grep -c TELEGRAM_BOT_TOKEN ~/.hermes/.env
'
```

Expected: `1` (single token entry, no duplicates).

- [ ] **Step 2: Re-run the identity guard against the real `~/.hermes/.env`**

```bash
ssh brucewayne9@75.43.156.111 '~/.lucius/scripts/lucius_identity_guard.sh'
```

Expected: `[identity-guard] OK`.

- [ ] **Step 3: Foreground test — `hermes gateway` in interactive mode**

```bash
ssh brucewayne9@75.43.156.111 'timeout 30 hermes gateway 2>&1 | head -40 || true'
```

Expected: bot startup messages, no crash. Manually send a Telegram to `@Luciuslabsbot` from Mike's phone — within 10s the gateway should log the message. Kill after confirming.

- [ ] **Step 4: Commit log**

```bash
cd /home/aialfred/alfred && date -Iseconds > lucius/logs/telegram-gateway-up.log
git add lucius/logs/telegram-gateway-up.log
git commit -m "chore(lucius): Telegram gateway env wired, foreground test passed"
```

---

### Task 17: First end-to-end Telegram round-trip with tools

**Files:** none (manual test)

- [ ] **Step 1: Run gateway in foreground, prepare to message it**

```bash
ssh brucewayne9@75.43.156.111 'hermes gateway' &
GW_PID=$!
sleep 5
```

- [ ] **Step 2: Send Mike a test prompt via Telegram**

From Mike's Telegram to `@Luciuslabsbot`:
> Run `crm.search_people` for "mike johnson" — return top 3.

Expected: Lucius identifies the tool, calls `claw_tools.crm.search_people` via MCP, returns 1–3 entries from Twenty CRM. Round-trip < 30s.

- [ ] **Step 3: Send a Grey-Matter recall test**

> Use `memory.recall` to look up "Roen Handmade brand bible" — summarize.

Expected: Lucius hits 117's LightRAG read-only, returns a summary. **If this returns "permission denied" or any error, halt — verify lightrag client is reachable from 111.**

- [ ] **Step 4: Negative test — try to write to GM**

> Use `memory.insert` to save the text "test from Lucius".

Expected: response says the tool doesn't exist (or equivalent — Hermes doesn't have access). **Critical safety check.**

- [ ] **Step 5: Stop gateway**

```bash
kill $GW_PID 2>/dev/null
```

- [ ] **Step 6: Commit log**

```bash
cd /home/aialfred/alfred && date -Iseconds > lucius/logs/e2e-roundtrip.log
git add lucius/logs/e2e-roundtrip.log
git commit -m "chore(lucius): E2E Telegram round-trip + tool calls + write-block confirmed"
```

---

### Task 18: Author Lucius's `MEMORY.md` and `USER.md` seeds

**Files:**
- Create: `lucius/skills/MEMORY-seed.md`
- Create: `lucius/skills/USER-seed.md`

- [ ] **Step 1: Write `MEMORY-seed.md`**

Hermes reads `~/.hermes/memories/MEMORY.md` for episodic memory and `~/.hermes/memories/USER.md` for the user profile. Seed both with minimum context so day-one is not flying blind.

Create `lucius/skills/MEMORY-seed.md`:
```markdown
# Lucius — Episodic Memory (seed)

## Where I am
Server 111 (CasaOS, brucewayne9). I am a parallel test against Oracle on 117 and Alfred on 105. Test window 2026-05-09 → 2026-05-23 (target).

## What I have access to
- `claw-tools` MCP server with 25 tools (CRM, email, calendar, social, search, image, design, TTS, content pipeline, workspace, website builder, mission control, Grey Matter recall/query)
- 117 Grey Matter (read-only) via `memory.recall` and `memory.query`
- My own native memory in `~/.hermes/memories/`
- Promote queue at `~/.lucius/promote_queue.jsonl` for graduation candidates

## What I cannot do
- Write to Grey Matter directly (graduation only via Mike approval)
- Touch 117 / Oracle / OpenClaw
- Send mail without a mailbox (I have none — Lucius is mailbox-less at v1)
- Run prod commands on 104/100/117/121
```

- [ ] **Step 2: Write `USER-seed.md`**

Create `lucius/skills/USER-seed.md`:
```markdown
# Mike Johnson (USER)

## Identity
- Owner/President of Ground Rush Inc/Labs/Cloud
- Atlanta, ET timezone
- Reads fast, no filler, action-first
- Has opinions; will push back; expects pushback

## How to talk to him
- Address as "sir" or by name
- Brevity > completeness
- Lead with the action / answer; explain after
- If something can't be done, say so immediately — don't soften
- Save the "I'll do it now" filler

## Family
- Sarah (wife) runs Roen Handmade
- Family details in 117 Grey Matter — recall before assuming

## Active fronts (as of 2026-05-08, recall Grey Matter for fresh state)
- Roen Handmade (live) — Sarah's jewelry store
- AG Entertainment / TicketWulf — events platform pivot
- AIROI — AI savings calc cold-outbound funnel (active revenue push)
- LoovaCast — radio + community management
- RuckTalk — content pipeline
- Fit as Ruck — PAUSED 2026-04-30
- Paperclip — KILLED 2026-04-13, do NOT reference

## Hard rules
- Never share API keys, tokens, internal IPs, server creds, CRM data, financials, family details
- Never test workflows with real customer IDs
- Mike approves T2/T3 — never act on T3 without his OK
```

- [ ] **Step 3: Deploy seeds**

```bash
scp lucius/skills/MEMORY-seed.md brucewayne9@75.43.156.111:~/.hermes/memories/MEMORY.md
scp lucius/skills/USER-seed.md brucewayne9@75.43.156.111:~/.hermes/memories/USER.md
ssh brucewayne9@75.43.156.111 'wc -l ~/.hermes/memories/*.md'
```

- [ ] **Step 4: Commit seeds**

```bash
cd /home/aialfred/alfred
git add lucius/skills/MEMORY-seed.md lucius/skills/USER-seed.md
git commit -m "feat(lucius): seed Hermes MEMORY.md and USER.md for day-one context"
```

---

## Wave 6 — Memory Architecture (Promote Queue)

### Task 19: Author `propose_memory` Hermes skill

**Files:**
- Create: `lucius/skills/propose_memory/skill.yaml`
- Create: `lucius/skills/propose_memory/run.py`

- [ ] **Step 1: Write skill manifest**

Create `lucius/skills/propose_memory/skill.yaml`:
```yaml
name: propose_memory
version: 0.1.0
description: |
  Propose a fact for promotion to long-term memory (117 Grey Matter).
  Appends to ~/.lucius/promote_queue.jsonl. The 7 AM ET daily digest
  surfaces queued entries to Mike on Telegram for approval.
inputs:
  - name: content
    type: string
    required: true
    description: The fact, decision, or piece of context worth promoting.
  - name: reasoning
    type: string
    required: true
    description: Why this is worth long-term memory rather than just session memory.
  - name: track_id_hint
    type: string
    required: false
    description: Optional hint for the Grey Matter track_id (will be prefixed `lucius_`).
runner: run.py
```

- [ ] **Step 2: Write `run.py`**

Create `lucius/skills/propose_memory/run.py`:
```python
#!/usr/bin/env python3
"""propose_memory — append a graduation candidate to ~/.lucius/promote_queue.jsonl."""
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    payload = json.loads(sys.stdin.read())
    content = payload.get("content", "").strip()
    reasoning = payload.get("reasoning", "").strip()
    track_hint = payload.get("track_id_hint", "").strip() or "general"

    if not content:
        print(json.dumps({"error": "content required"}))
        return 2
    if not reasoning:
        print(json.dumps({"error": "reasoning required"}))
        return 2

    queue_path = Path(os.environ.get("LUCIUS_HOME", os.path.expanduser("~/.lucius"))) / "promote_queue.jsonl"
    queue_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "id": str(uuid.uuid4()),
        "ts": datetime.now(timezone.utc).isoformat(),
        "content": content,
        "reasoning": reasoning,
        "proposed_track_id": f"lucius_{track_hint}",
        "session_id": payload.get("session_id"),
    }
    with queue_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")

    print(json.dumps({"queued": True, "id": entry["id"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Deploy to 111**

```bash
chmod +x lucius/skills/propose_memory/run.py
ssh brucewayne9@75.43.156.111 'mkdir -p ~/.hermes/skills/propose_memory'
scp lucius/skills/propose_memory/skill.yaml brucewayne9@75.43.156.111:~/.hermes/skills/propose_memory/
scp lucius/skills/propose_memory/run.py brucewayne9@75.43.156.111:~/.hermes/skills/propose_memory/
ssh brucewayne9@75.43.156.111 'chmod +x ~/.hermes/skills/propose_memory/run.py && hermes skills list 2>&1 | grep propose'
```

Expected: skill listed by Hermes.

- [ ] **Step 4: Manual smoke test**

```bash
ssh brucewayne9@75.43.156.111 'echo "{\"content\":\"smoke test\",\"reasoning\":\"plan task 19 verification\"}" | python3 ~/.hermes/skills/propose_memory/run.py && wc -l ~/.lucius/promote_queue.jsonl && tail -1 ~/.lucius/promote_queue.jsonl'
```

Expected: `{"queued": true, ...}`, queue line count = 1, tail entry includes "smoke test".

- [ ] **Step 5: Clean up the smoke entry**

```bash
ssh brucewayne9@75.43.156.111 ': > ~/.lucius/promote_queue.jsonl'
```

- [ ] **Step 6: Commit**

```bash
cd /home/aialfred/alfred
git add lucius/skills/propose_memory/
git commit -m "feat(lucius): propose_memory skill (queues graduation candidates)"
```

---

### Task 20: `lucius_promote_digest.py` — daily Telegram digest

**Files:**
- Create: `lucius/scripts/lucius_promote_digest.py`

- [ ] **Step 1: Author the digest sender**

Create `lucius/scripts/lucius_promote_digest.py`:
```python
#!/usr/bin/env python3
"""Daily 7 AM ET digest of the Lucius promote queue.

Reads ~/.lucius/promote_queue.jsonl, sends a numbered Telegram message
via Lucius bot to Mike. Mike replies with comma-separated indexes (or 'none')
to approve. The companion script `lucius_promote_apply.py` polls for
that reply and executes the approvals.

Caps at 10 entries/day; older entries roll to next day.
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import urllib.request
import urllib.parse

LUCIUS_HOME = Path(os.environ.get("LUCIUS_HOME", os.path.expanduser("~/.lucius")))
QUEUE = LUCIUS_HOME / "promote_queue.jsonl"
DIGEST_STATE = LUCIUS_HOME / "promote_digest_state.json"
ENV_FILE = LUCIUS_HOME / "config" / ".env"
CAP_PER_DIGEST = 10


def load_env() -> dict[str, str]:
    out = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def send_telegram(token: str, chat_id: str, text: str) -> dict:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}).encode()
    with urllib.request.urlopen(url, data=data, timeout=15) as r:
        return json.loads(r.read())


def main() -> int:
    env = load_env()
    token = env.get("TELEGRAM_BOT_TOKEN_LUCIUS")
    chat_id = env.get("TELEGRAM_CHAT_ID", "7582976864")
    if not token:
        print("ERROR: TELEGRAM_BOT_TOKEN_LUCIUS missing", file=sys.stderr)
        return 2

    if not QUEUE.exists() or QUEUE.stat().st_size == 0:
        # Nothing to digest; per CLAUDE.md, "don't message Mike 'nothing new'"
        print("queue empty — silent")
        return 0

    entries = [json.loads(l) for l in QUEUE.read_text().splitlines() if l.strip()]
    today = entries[:CAP_PER_DIGEST]

    lines = ["*Lucius proposes these for long-term memory:*", ""]
    for i, e in enumerate(today, 1):
        snippet = e["content"][:200] + ("…" if len(e["content"]) > 200 else "")
        lines.append(f"{i}. {snippet}")
        lines.append(f"   _{e['reasoning'][:140]}_")
        lines.append("")
    lines.append(f"Reply with comma-separated numbers (1,3,4) to approve, or `none` to skip all. {len(entries) - len(today)} more queued behind these." if len(entries) > len(today) else "Reply with comma-separated numbers (1,3,4) to approve, or `none` to skip all.")

    text = "\n".join(lines)
    resp = send_telegram(token, chat_id, text)
    if not resp.get("ok"):
        print(f"ERROR: telegram send failed: {resp}", file=sys.stderr)
        return 3

    # Stash the digest state — apply.py will use it to map index→entry
    state = {
        "digest_id": str(int(datetime.now(timezone.utc).timestamp())),
        "ts": datetime.now(timezone.utc).isoformat(),
        "tg_message_id": resp["result"]["message_id"],
        "entries": today,
    }
    DIGEST_STATE.write_text(json.dumps(state, indent=2))
    print(f"sent digest with {len(today)} entries; message_id={resp['result']['message_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Deploy**

```bash
chmod +x lucius/scripts/lucius_promote_digest.py
scp lucius/scripts/lucius_promote_digest.py brucewayne9@75.43.156.111:~/.lucius/scripts/
ssh brucewayne9@75.43.156.111 'chmod +x ~/.lucius/scripts/lucius_promote_digest.py'
```

- [ ] **Step 3: Smoke test (with one fake entry)**

```bash
ssh brucewayne9@75.43.156.111 '
  echo "{\"id\":\"smoke-1\",\"ts\":\"2026-05-08T12:00:00Z\",\"content\":\"Smoke test entry — please ignore.\",\"reasoning\":\"plan task 20 verification\",\"proposed_track_id\":\"lucius_smoke\"}" >> ~/.lucius/promote_queue.jsonl
  ~/.lucius/scripts/lucius_promote_digest.py
'
```

Expected: Mike receives a Telegram message numbered "1." with the smoke entry. State file written.

- [ ] **Step 4: Clean up the smoke entry**

```bash
ssh brucewayne9@75.43.156.111 ': > ~/.lucius/promote_queue.jsonl; rm -f ~/.lucius/promote_digest_state.json'
```

- [ ] **Step 5: Commit**

```bash
cd /home/aialfred/alfred
git add lucius/scripts/lucius_promote_digest.py
git commit -m "feat(lucius): daily promote-queue digest sender"
```

---

### Task 21: `lucius_promote_apply.py` — parse Mike's approval and ingest to Grey Matter

**Files:**
- Create: `lucius/scripts/lucius_promote_apply.py`

- [ ] **Step 1: Author the applier**

Create `lucius/scripts/lucius_promote_apply.py`:
```python
#!/usr/bin/env python3
"""Read Mike's reply to today's digest; ingest approved entries to Grey Matter.

Runs at 7:15 AM ET (cron) — gives Mike 15 min to reply. Polls Telegram updates
(getUpdates) for any reply to digest_state.tg_message_id. Parses
'1,3,5' or 'none' from the reply text. Approved → POST to LightRAG insert.
Rejected → moved to promote_queue.rejected.jsonl. Approved entries removed.
"""
import json
import os
import re
import sys
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

LUCIUS_HOME = Path(os.environ.get("LUCIUS_HOME", os.path.expanduser("~/.lucius")))
QUEUE = LUCIUS_HOME / "promote_queue.jsonl"
REJECTED = LUCIUS_HOME / "promote_queue.rejected.jsonl"
DIGEST_STATE = LUCIUS_HOME / "promote_digest_state.json"
ENV_FILE = LUCIUS_HOME / "config" / ".env"


def load_env() -> dict[str, str]:
    out = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def get_reply_text(token: str, chat_id: str, target_message_id: int) -> str | None:
    url = f"https://api.telegram.org/bot{token}/getUpdates?timeout=2&allowed_updates=%5B%22message%22%5D"
    with urllib.request.urlopen(url, timeout=10) as r:
        data = json.loads(r.read())
    if not data.get("ok"):
        return None
    for upd in reversed(data.get("result", [])):  # newest first
        msg = upd.get("message") or {}
        if str(msg.get("chat", {}).get("id")) != str(chat_id):
            continue
        rt = (msg.get("reply_to_message") or {}).get("message_id")
        if rt == target_message_id:
            return msg.get("text", "").strip()
    return None


def parse_approvals(text: str, n: int) -> list[int]:
    if text.lower().strip() == "none":
        return []
    nums = []
    for tok in re.split(r"[,\s]+", text):
        tok = tok.strip()
        if tok.isdigit():
            i = int(tok)
            if 1 <= i <= n:
                nums.append(i)
    return sorted(set(nums))


def lightrag_insert(host: str, api_key: str, content: str, track_id: str) -> bool:
    url = f"{host.rstrip('/')}/documents/text"
    body = json.dumps({"text": content, "track_id": track_id}).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json",
        "X-API-Key": api_key,
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status < 400
    except Exception as e:
        print(f"lightrag insert failed: {e}", file=sys.stderr)
        return False


def main() -> int:
    if not DIGEST_STATE.exists():
        print("no digest state — nothing to apply")
        return 0

    state = json.loads(DIGEST_STATE.read_text())
    entries = state["entries"]
    env = load_env()
    token = env["TELEGRAM_BOT_TOKEN_LUCIUS"]
    chat_id = env.get("TELEGRAM_CHAT_ID", "7582976864")
    gm_host = env.get("LIGHTRAG_HOST", "http://75.43.156.117:9621")
    gm_key = env.get("LIGHTRAG_API_KEY", "")

    reply = get_reply_text(token, chat_id, state["tg_message_id"])
    if reply is None:
        print("no reply yet — leaving state for next run")
        return 0

    approved_idx = parse_approvals(reply, len(entries))
    approved = [entries[i - 1] for i in approved_idx]
    rejected = [e for i, e in enumerate(entries, 1) if i not in approved_idx]

    # Ingest approved
    successes: list[str] = []
    for e in approved:
        ok = lightrag_insert(gm_host, gm_key, e["content"], e["proposed_track_id"])
        if ok:
            successes.append(e["id"])

    # Rebuild the queue without the entries we processed
    if QUEUE.exists():
        remaining = [json.loads(l) for l in QUEUE.read_text().splitlines() if l.strip()]
        processed_ids = {e["id"] for e in entries}
        remaining = [r for r in remaining if r["id"] not in processed_ids]
        QUEUE.write_text("".join(json.dumps(r) + "\n" for r in remaining))

    # Append rejects (audit log)
    with REJECTED.open("a") as f:
        for e in rejected:
            e["rejected_at"] = datetime.now(timezone.utc).isoformat()
            f.write(json.dumps(e) + "\n")

    # Acknowledge to Mike
    summary = f"✅ Promoted {len(successes)} to Grey Matter. Rejected {len(rejected)}."
    urllib.request.urlopen(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=urllib.parse.urlencode({"chat_id": chat_id, "text": summary}).encode(),
        timeout=10,
    )

    # Clear digest state (one-shot per day)
    DIGEST_STATE.unlink()
    print(f"approved={len(successes)} rejected={len(rejected)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Deploy**

```bash
chmod +x lucius/scripts/lucius_promote_apply.py
scp lucius/scripts/lucius_promote_apply.py brucewayne9@75.43.156.111:~/.lucius/scripts/
ssh brucewayne9@75.43.156.111 'chmod +x ~/.lucius/scripts/lucius_promote_apply.py'
```

- [ ] **Step 3: End-to-end test**

```bash
# Inject one entry
ssh brucewayne9@75.43.156.111 '
  echo "{\"id\":\"e2e-1\",\"ts\":\"2026-05-08T12:00:00Z\",\"content\":\"E2E: Lucius noticed Mike prefers cold-outbound campaigns to be drafted, not auto-sent.\",\"reasoning\":\"E2E test of promote pipeline\",\"proposed_track_id\":\"lucius_e2e\"}" >> ~/.lucius/promote_queue.jsonl
  ~/.lucius/scripts/lucius_promote_digest.py
'
```

- [ ] **Step 4: Mike replies on Telegram with `1` to the digest message**

Wait for Mike to reply.

- [ ] **Step 5: Run the applier**

```bash
ssh brucewayne9@75.43.156.111 '~/.lucius/scripts/lucius_promote_apply.py'
```

Expected: `approved=1 rejected=0`. Mike receives "✅ Promoted 1 to Grey Matter." Queue is empty. Grey Matter on 117 has a new doc with `track_id=lucius_e2e`.

- [ ] **Step 6: Verify in Grey Matter**

```bash
ssh brucewayne9@75.43.156.117 'curl -sS -H "X-API-Key: $LIGHTRAG_API_KEY" http://localhost:9621/documents | python3 -c "import json,sys; d=json.load(sys.stdin); print([r for r in d.get(\"track_ids\",[]) if r.startswith(\"lucius_\")])"'
```

Expected: list contains `lucius_e2e`.

- [ ] **Step 7: Commit**

```bash
cd /home/aialfred/alfred
git add lucius/scripts/lucius_promote_apply.py
git commit -m "feat(lucius): promote-queue applier (Mike approval → Grey Matter ingest)"
```

---

### Task 22: Cron entries for daily digest + apply

**Files:** user crontab on 111

- [ ] **Step 1: Install crons**

```bash
ssh brucewayne9@75.43.156.111 '
  (crontab -l 2>/dev/null | grep -v "lucius_promote_"; cat << EOF
# Lucius promote-queue daily digest (7:00 AM ET = 12:00 UTC during EDT, 11:00 UTC during EST)
0 12 * * * /home/brucewayne9/.lucius/scripts/lucius_promote_digest.py >> /home/brucewayne9/.lucius/logs/promote.log 2>&1
# Apply approvals 30 min later
30 12 * * * /home/brucewayne9/.lucius/scripts/lucius_promote_apply.py >> /home/brucewayne9/.lucius/logs/promote.log 2>&1
EOF
  ) | crontab -
  crontab -l | grep lucius
'
```

Expected: 2 cron lines visible. Note: scheduled in UTC; assumes EDT. Adjust to `0 13` and `30 13` during EST (Nov–Mar). For 2026-05-09 start, EDT is correct.

- [ ] **Step 2: Commit log**

```bash
cd /home/aialfred/alfred && date -Iseconds > lucius/logs/cron-installed.log
git add lucius/logs/cron-installed.log
git commit -m "chore(lucius): install promote-queue daily crons on 111"
```

---

## Wave 7 — systemd Service

### Task 23: Author `hermes-gateway.service` unit

**Files:**
- Create: `lucius/systemd/hermes-gateway.service`

- [ ] **Step 1: Write unit file**

Create `lucius/systemd/hermes-gateway.service`:
```ini
[Unit]
Description=Hermes Agent gateway (Lucius)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStartPre=/home/brucewayne9/.lucius/scripts/lucius_identity_guard.sh
ExecStart=/bin/bash -lc 'hermes gateway'
Restart=on-failure
RestartSec=15
StartLimitIntervalSec=300
StartLimitBurst=5

# Resource hygiene
MemoryHigh=2G
MemoryMax=3G
CPUWeight=80

# Logs
StandardOutput=journal
StandardError=journal
SyslogIdentifier=hermes-gateway

[Install]
WantedBy=default.target
```

Note: identity-guard runs as `ExecStartPre` — if it fails, the service does not start. Memory caps are conservative (Hermes typical ~500MB; cap at 3G to prevent runaway).

- [ ] **Step 2: Deploy**

```bash
ssh brucewayne9@75.43.156.111 'mkdir -p ~/.config/systemd/user'
scp lucius/systemd/hermes-gateway.service brucewayne9@75.43.156.111:~/.config/systemd/user/
```

- [ ] **Step 3: Verify unit syntax**

```bash
ssh brucewayne9@75.43.156.111 'systemd-analyze --user verify ~/.config/systemd/user/hermes-gateway.service && echo OK'
```

Expected: no errors, "OK" line.

- [ ] **Step 4: Commit**

```bash
cd /home/aialfred/alfred
git add lucius/systemd/hermes-gateway.service
git commit -m "feat(lucius): systemd-user unit for Hermes gateway with identity-guard pre-flight"
```

---

### Task 24: Enable + start service, verify

**Files:** none (operates on 111)

- [ ] **Step 1: Reload + enable + start**

```bash
ssh brucewayne9@75.43.156.111 '
  loginctl enable-linger brucewayne9 2>/dev/null || true
  systemctl --user daemon-reload
  systemctl --user enable --now hermes-gateway.service
  sleep 5
  systemctl --user status hermes-gateway.service --no-pager | head -25
'
```

Expected: `Active: active (running)`. If failed, `journalctl --user -u hermes-gateway --since '1 min ago'` for diagnosis.

- [ ] **Step 2: Verify Telegram round-trip works against the daemon**

From Mike: send a one-line Telegram to `@Luciuslabsbot`:
> Hello — confirm you're running as a service.

Expected: prompt response within 10s. The bot should reference `~/.hermes/SOUL.md` persona ("sir," "Lucius," etc.).

- [ ] **Step 3: Restart resilience test**

```bash
ssh brucewayne9@75.43.156.111 'systemctl --user restart hermes-gateway.service && sleep 5 && systemctl --user is-active hermes-gateway.service'
```

Expected: `active`.

- [ ] **Step 4: Identity-guard kill test**

```bash
# Temporarily break the env, restart, verify guard blocks
ssh brucewayne9@75.43.156.111 '
  cp ~/.hermes/.env ~/.hermes/.env.bak
  sed -i "s|TELEGRAM_BOT_TOKEN=.*|TELEGRAM_BOT_TOKEN=00000:WRONG|" ~/.hermes/.env
  systemctl --user restart hermes-gateway.service || true
  sleep 5
  systemctl --user is-active hermes-gateway.service || echo "BLOCKED — good"
  mv ~/.hermes/.env.bak ~/.hermes/.env
  systemctl --user restart hermes-gateway.service
  sleep 5
  systemctl --user is-active hermes-gateway.service
'
```

Expected: with bad token, status is NOT `active` (guard prevents start). After restoring, `active`.

- [ ] **Step 5: Commit log**

```bash
cd /home/aialfred/alfred && date -Iseconds > lucius/logs/systemd-up.log
git add lucius/logs/systemd-up.log
git commit -m "chore(lucius): systemd unit live, identity-guard verified"
```

---

### Task 25: Configure Hermes gateway healthcheck endpoint (port 18790)

**Files:** `~/.hermes/config.yaml` on 111

- [ ] **Step 1: Add healthcheck block**

```bash
ssh brucewayne9@75.43.156.111 '
  cat >> ~/.hermes/config.yaml << EOF

# Healthcheck for 105 monitor
auxiliary:
  health:
    enabled: true
    bind: "127.0.0.1"
    port: 18790
EOF
  hermes config check
  systemctl --user restart hermes-gateway.service
'
```

Note: this assumes Hermes 0.13.0 supports the `auxiliary.health` block. If `hermes config check` rejects it, fall back to a tiny standalone script the monitor can curl. **Verify before committing.**

- [ ] **Step 2: Probe**

```bash
ssh brucewayne9@75.43.156.111 'sleep 3; curl -sS -m 5 http://127.0.0.1:18790/health || echo "no health endpoint — fallback needed"'
```

If "no health endpoint":

- [ ] **Step 2-fallback: Drop in a tiny systemd-tied healthcheck script**

```bash
# Create a 5-line "is hermes-gateway active" check served via a tmpfs file
ssh brucewayne9@75.43.156.111 '
  cat > ~/.lucius/scripts/health.sh << EOF
#!/usr/bin/env bash
systemctl --user is-active hermes-gateway.service >/dev/null && echo "ok" || echo "down"
EOF
  chmod +x ~/.lucius/scripts/health.sh
'
```

The monitor will SSH-invoke this rather than HTTP-probe.

- [ ] **Step 3: Commit**

```bash
cd /home/aialfred/alfred && date -Iseconds > lucius/logs/health-endpoint.log
git add lucius/logs/health-endpoint.log
git commit -m "chore(lucius): healthcheck wired (HTTP if supported, SSH-script fallback)"
```

---

## Wave 8 — Heartbeat Extension

### Task 26: Extend `scripts/alfred_claw_monitor.py` to probe 111

**Files:**
- Modify: `scripts/alfred_claw_monitor.py`

- [ ] **Step 1: Read current state shape to plan additive edit**

```bash
cd /home/aialfred/alfred && grep -n "def " scripts/alfred_claw_monitor.py | head -20
```

(This step is recon — note the function names so the new `check_lucius()` follows local style.)

- [ ] **Step 2: Add `check_lucius()` function and wire into the main loop**

The exact edit depends on existing structure. Pattern:

```python
def check_lucius() -> dict:
    """Probe Lucius (Hermes Agent on 111). Returns {ok: bool, issues: [str]}"""
    issues = []
    # SSH-based check (matches the rest of the monitor's pattern for 117)
    cmd = "ssh -o ConnectTimeout=5 brucewayne9@75.43.156.111 '~/.lucius/scripts/health.sh'"
    try:
        out = subprocess.check_output(cmd, shell=True, timeout=15, stderr=subprocess.STDOUT).decode().strip()
        if out != "ok":
            issues.append(f"hermes-gateway not active (got: {out})")
    except subprocess.CalledProcessError as e:
        issues.append(f"ssh probe failed: {e.output.decode()[:200]}")
    except subprocess.TimeoutExpired:
        issues.append("ssh probe timeout")

    # Identity-guard re-verify (cheap)
    cmd2 = "ssh -o ConnectTimeout=5 brucewayne9@75.43.156.111 '~/.lucius/scripts/lucius_identity_guard.sh'"
    try:
        out = subprocess.check_output(cmd2, shell=True, timeout=15, stderr=subprocess.STDOUT).decode().strip()
        if "OK" not in out:
            issues.append(f"identity-guard failed: {out[:200]}")
    except Exception as e:
        issues.append(f"identity-guard probe error: {e}")

    return {"ok": not issues, "issues": issues}
```

Wire it into the main loop alongside the existing 117 check, **without** creating Lucius alerts that trigger fix-it/leave-it. Per spec, Lucius alerts are observation-only at v1 — Mike sees the alert but no auto-fix.

- [ ] **Step 3: Add `lucius_*` fields to state JSON shape**

In `claw_monitor_state.json`:
```json
{
  "...": "...existing fields...",
  "lucius_status": "healthy" | "down" | "unknown",
  "lucius_failures": 0,
  "lucius_last_check": null,
  "lucius_issues": []
}
```

The state-write code should be additive: never drop existing keys.

- [ ] **Step 4: Local syntax + lint**

```bash
cd /home/aialfred/alfred && python3 -c 'import ast; ast.parse(open("scripts/alfred_claw_monitor.py").read())' && echo OK
```

- [ ] **Step 5: Dry-run a single probe**

```bash
cd /home/aialfred/alfred && python3 -c 'import sys; sys.path.insert(0,"scripts"); from alfred_claw_monitor import check_lucius; print(check_lucius())'
```

Expected: `{"ok": True, "issues": []}`.

- [ ] **Step 6: Commit**

```bash
git add scripts/alfred_claw_monitor.py data/claw_monitor_state.json
git commit -m "feat(monitor): extend claw_monitor to probe Lucius on 111 (observation-only)"
```

---

### Task 27: Test alert flow

**Files:** none (manual test)

- [ ] **Step 1: Stop Lucius and verify monitor catches it**

```bash
ssh brucewayne9@75.43.156.111 'systemctl --user stop hermes-gateway.service'
# Wait one cron cycle (10 min) OR force a probe:
cd /home/aialfred/alfred && python3 scripts/alfred_claw_monitor.py --once 2>&1 | tail -20
```

Expected: monitor logs `lucius_status=down`, sends email to Mike (per existing alert plumbing).

- [ ] **Step 2: Restart and verify recovery**

```bash
ssh brucewayne9@75.43.156.111 'systemctl --user start hermes-gateway.service'
sleep 10
cd /home/aialfred/alfred && python3 scripts/alfred_claw_monitor.py --once 2>&1 | tail -10
```

Expected: `lucius_status=healthy`, recovery logged.

- [ ] **Step 3: Commit log**

```bash
cd /home/aialfred/alfred && date -Iseconds > lucius/logs/heartbeat-tested.log
git add lucius/logs/heartbeat-tested.log
git commit -m "chore(lucius): heartbeat alert flow tested end-to-end"
```

---

## Wave 9 — Wrap-up

### Task 28: Save project memory + update MEMORY.md

**Files:**
- Create: `~/.claude/projects/-home-aialfred-alfred/memory/project_lucius_hermes_test.md`
- Modify: `~/.claude/projects/-home-aialfred-alfred/memory/MEMORY.md`

- [ ] **Step 1: Write project memory**

Create the file with frontmatter and content covering: kickoff date, test window, key paths, success criteria, fallback procedure if it fails, link to spec + plan + lucius_bot.md.

- [ ] **Step 2: Add MEMORY.md entry under "Active Projects"**

```
- **[Lucius / Hermes-on-111 Test](project_lucius_hermes_test.md)** ★ — Hermes Agent v0.13.0 on 111, fronted by @Luciuslabsbot, side-by-side test vs Oracle/117. Window 2026-05-09→2026-05-23. 25 tools day-one, full isolation, promote-queue for GM writes.
```

- [ ] **Step 3: Commit memory**

```bash
cd /home/aialfred/.claude/projects/-home-aialfred-alfred/memory
git add project_lucius_hermes_test.md MEMORY.md 2>/dev/null || true
# memory dir may not be a git repo — skip if so
```

(If memory dir is not git-tracked, the Write tool's filesystem write is sufficient.)

---

### Task 29: Final E2E smoke test — 5-task butler routine

**Files:** none (manual test against `@Luciuslabsbot`)

- [ ] **Step 1: Send Lucius this 5-task batch via Telegram**

```
Sir morning routine:
1. CRM: search for "AIROI" — show open opps
2. Calendar: today's events
3. Email: unread count for alfred mailbox
4. Memory: recall "Roen brand bible" — one-line summary
5. Image: generate a 1024x1024 image of a butler in pinstripe at a server rack, "lucius mode"
```

- [ ] **Step 2: Verify each tool fires**

Watch for:
- (1) `claw_tools.crm.pipeline` or `crm.search_people`
- (2) `claw_tools.calendar.today`
- (3) `claw_tools.email.unread`
- (4) `claw_tools.memory.recall` (read-only against 117)
- (5) `claw_tools.image.generate` (ComfyUI on 105)

- [ ] **Step 3: Verify response quality**

Lucius should respond within ~60s, with each step's result rendered in butler tone, not raw JSON. Failures should be reported honestly per SOUL.md.

- [ ] **Step 4: Final commit**

```bash
cd /home/aialfred/alfred && date -Iseconds > lucius/logs/v1-live.log
git add lucius/logs/v1-live.log
git commit -m "feat(lucius): v1 LIVE — Hermes Agent on 111 fronted by @Luciuslabsbot"
```

- [ ] **Step 5: Email Mike a "Lucius is live" note**

(Reuses Alfred's `EmailClient` pattern from earlier in this session.)

```bash
cd /home/aialfred/alfred && python3 << 'PY'
import os, sys
from dotenv import load_dotenv
load_dotenv("config/.env")
sys.path.insert(0, ".")
from integrations.email.client import EmailClient
EmailClient().send_email(
    account="alfred-gw",
    to="mjohnson@groundrushinc.com",
    subject="Lucius is live on 111",
    body=(
        "Sir,\n\n"
        "Lucius (Hermes Agent on 111, @Luciuslabsbot) is live and answering.\n"
        "Test window: 2026-05-09 → 2026-05-23.\n"
        "Tools: 25 (CRM, email, calendar, social, search, image, design, TTS, content, workspace, web, mission control, GM-recall).\n"
        "Memory: read-only on 117 GM; promote-queue digest at 7 AM ET daily.\n"
        "Heartbeat: 105 monitor watching 111 every 10 min.\n"
        "117 / Oracle / OpenClaw: untouched.\n\n"
        "Strikes log starts now. 3 strikes (fall-back to Oracle/Alfred to finish a task) = test fails.\n\n"
        "— Alfred"
    ),
)
print("notice sent")
PY
```

---

## Self-Review

**Spec coverage:**
- All 7 component areas in spec → tasks ✅ (Hermes install, model, MCP, Telegram, memory, heartbeat, coexistence)
- Tools migrated day one (25) → Tasks 7–14 ✅
- Promote queue with daily digest + apply → Tasks 19–22 ✅
- Identity guard → Task 15 + Task 23 (ExecStartPre) ✅
- Zero changes to 117 — no task touches 117 except Task 2 (read-only listing) and Task 17 (read-only LightRAG via 117:9621) ✅
- 2-week test window + strike rule → Task 29 email and SOUL.md mentions ✅

**Placeholder scan:** None of the disallowed phrases appear ("TBD", "implement later", "similar to task N"). Two soft conditionals — Task 25 has a fallback if `auxiliary.health` doesn't exist in 0.13.0 (concrete fallback provided) and Task 22 cron times are EDT-correct for the 2026-05-09 start. Both are documented assumptions, not placeholders.

**Type consistency:** `ScriptTool` signature, `tools.json` schema, MCP server `load_tools()` return type, env var names (`TELEGRAM_BOT_TOKEN_LUCIUS`, `LUCIUS_TELEGRAM_BOT_ID`, `LIGHTRAG_HOST`) used identically across files.

---

## Execution Choice

**Plan complete and saved to** `docs/superpowers/plans/2026-05-08-lucius-hermes-on-111.md`.

Two execution options:

**1. Subagent-Driven (recommended)** — Fresh subagent per task with two-stage review between tasks. Best for a 29-task multi-server build where review checkpoints catch drift early.

**2. Inline Execution** — Same session, batch execution with checkpoints. Faster but less isolation.

Mike: which approach?
