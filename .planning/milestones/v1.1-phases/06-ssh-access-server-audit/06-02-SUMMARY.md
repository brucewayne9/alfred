---
phase: 06-ssh-access-server-audit
plan: 02
subsystem: infra
tags: [server-audit, inventory, ssh, docker, python, json, markdown]

# Dependency graph
requires:
  - phase: 06-01
    provides: SSH key pairs and named aliases (server-98, server-100, claw, server-104, lonewolf, server-121) for remote command execution
provides:
  - data/infrastructure/audit.py — reusable Python audit script for future re-runs
  - data/infrastructure/inventory.json — machine-readable inventory of all 7 servers (6 remote + 105 local)
  - data/infrastructure/inventory.md — human-readable server inventory with per-server sections
  - Complete catalog of Docker containers, systemd services, databases, disk usage, cron jobs per server
  - Cross-server connection map (labs-R820/LightRAG references 105 for embeddings)
affects:
  - 07-backup-automation (uses inventory to tailor per-server backup scripts)
  - 08-monitoring (uses inventory as baseline for health checks)

# Tech tracking
tech-stack:
  added: [subprocess SSH remote execution, json.dump inventory, Python audit script]
  patterns:
    - "SSH alias-based remote command execution via subprocess.run(['ssh', alias, cmd])"
    - "Dual-format inventory: JSON as source of truth, markdown as human-readable view"
    - "--markdown-only flag pattern for regenerating docs without re-running expensive SSH audit"
    - "is_active() exact string check for systemd status (avoids 'inactive' false positive)"

key-files:
  created:
    - data/infrastructure/audit.py — full audit script with audit_server(), generate_markdown(), --markdown-only flag
    - data/infrastructure/inventory.json — 7-server inventory (gitignored, regenerate with audit.py)
    - data/infrastructure/inventory.md — human-readable inventory (gitignored, regenerate with audit.py)
  modified:
    - .gitignore — changed from directory exclusion to file-level exclusion so audit.py is tracked

key-decisions:
  - "Gitignore changed from data/infrastructure/ (directory) to individual file patterns for inventory.json/md — git directory exclusions cannot be negated with !, so file-level patterns needed to track audit.py"
  - "Database detection uses exact 'active' match on first line of systemctl output — 'inactive' contains 'active' substring which caused false positives"
  - "audit.py skips df header row by checking parts[0] != 'Filesystem' — df --output flag still emits a header on some systems"

patterns-established:
  - "Audit script pattern: collect all data per server into dict, write JSON, generate markdown from JSON"
  - "Cross-server reference detection: check /etc/hosts, env files, Docker env vars, Traefik config"

requirements-completed: [INFRA-02]

# Metrics
duration: 6min
completed: 2026-02-26
---

# Phase 6 Plan 02: Server Audit Summary

**Complete infrastructure inventory of 7 servers collected via SSH — Docker containers, services, databases, disk, cron, and cross-server connections cataloged in dual JSON/markdown format for Phase 7 backup scripting**

## Performance

- **Duration:** 6 min (fully automated — SSH keys from Plan 01 used throughout)
- **Started:** 2026-02-26T15:36:24Z
- **Completed:** 2026-02-26T15:42:30Z
- **Tasks:** 2 (both automated)
- **Files modified:** 2 (audit.py created, .gitignore modified)

## Accomplishments

- Created `data/infrastructure/audit.py` — a 340-line Python script that SSHes into 6 remote servers and collects local data from 105, producing a complete inventory
- Collected 12 data categories per server: hostname/OS/kernel/uptime, disk usage, memory, Docker containers, Docker Compose projects, systemd services, listening ports, PostgreSQL/MySQL/Redis/MongoDB/SQLite detection, cron jobs, cross-server IP references, and notable findings
- Generated `inventory.json` (7 servers) and `inventory.md` (7 server sections with summary table and cross-server connections map)
- Notable discovery: labs-R820 (lonewolf/117) has a LightRAG installation that references Alfred Labs (105) for embedding generation (`http://75.43.156.105:11434`)

## Key Findings from Audit

| Server | IP | Docker | Services | Databases | Disk |
|--------|-----|--------|----------|-----------|------|
| alfred (local) | 75.43.156.105 | 2 containers | 49 | PostgreSQL, Redis | 32% |
| GroundRushRadio | 75.43.156.98 | 3 containers | 37 | none | 46% |
| labs-edge-server | 75.43.156.100 | 8 containers | 43 | MySQL | 29% |
| oracle (Alfred Claw) | 75.43.156.101 | not installed | 39 | none | 3% |
| labsliveserver | 75.43.156.104 | 55 containers | 40 | MySQL | 35% |
| labs-R820 (Lonewolf) | 75.43.156.117 | 24 containers | 40 | none | 30% |
| gloundrush-cloud-mail | 75.43.156.121 | 20 containers | 36 | none | 1% |

**Notable:** labsliveserver (104) has 55 Docker containers — highest container density. Alfred Claw (101) has no Docker installed (bare Python/systemd).

## Task Commits

Each task was committed atomically:

1. **Task 1: Create audit script and collect server inventory** — `6be3891` (feat)
2. **Task 2: Generate markdown, fix disk/db parsing** — `ecdd306` (feat)

**Plan metadata:** `(see final docs commit below)`

## Files Created/Modified

- `data/infrastructure/audit.py` — Full infrastructure audit script (tracked in git)
- `data/infrastructure/inventory.json` — Machine-readable 7-server inventory (gitignored)
- `data/infrastructure/inventory.md` — Human-readable inventory (gitignored)
- `.gitignore` — Changed from directory exclusion to file-level patterns for inventory.json/md

## Decisions Made

- Changed `.gitignore` from `data/infrastructure/` (full directory) to individual file patterns (`inventory.json`, `inventory.md`) so that `audit.py` can be tracked in git. Git cannot negate a directory-level ignore pattern with `!`, so file-level patterns were required.
- Database detection uses `is_active()` which checks the exact first line of `systemctl is-active` output equals `"active"` — the substring check `"active" in status` would match "inactive", causing false positives on all servers.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed false positive database detection**
- **Found during:** Task 1 (after running audit and reviewing results)
- **Issue:** `bool(pg_status and "active" in pg_status.lower())` matched "inactive" because "active" is a substring of "inactive". Every server showed all 4 databases as "detected" including mail server with no databases.
- **Fix:** Added `is_active()` function that checks `first_line == "active"` for exact match on systemctl output
- **Files modified:** data/infrastructure/audit.py
- **Verification:** Re-ran audit — mail server (121) correctly shows 0 databases, alfred (105) correctly shows PostgreSQL + Redis
- **Committed in:** 6be3891 (Task 1 commit, included the fix before final commit)

**2. [Rule 1 - Bug] Fixed duplicate header row in disk usage tables**
- **Found during:** Task 2 (markdown review)
- **Issue:** `df --output=source,...` emits a header line ("Filesystem Size Used...") which was being captured as a data row, resulting in a duplicate table header in the markdown
- **Fix:** Added `parts[0] != "Filesystem" and parts[0] != "Source"` guard in disk parsing loop
- **Files modified:** data/infrastructure/audit.py
- **Verification:** Markdown shows clean disk tables with no duplicate header rows
- **Committed in:** ecdd306 (Task 2 commit)

**3. [Rule 3 - Blocking] Changed gitignore to file-level patterns to enable audit.py commit**
- **Found during:** Task 1 commit (git refused to stage audit.py)
- **Issue:** `data/infrastructure/` directory-level gitignore prevented `git add data/infrastructure/audit.py`, even with `!` negation — git does not apply negation to directory-level patterns
- **Fix:** Changed gitignore from `data/infrastructure/` to `data/infrastructure/inventory.json` + `data/infrastructure/inventory.md`
- **Files modified:** .gitignore
- **Verification:** `git check-ignore` confirms inventory files are excluded; `git add audit.py` succeeds
- **Committed in:** 6be3891

---

**Total deviations:** 3 auto-fixed (2 bugs, 1 blocking)
**Impact on plan:** All fixes improved correctness and allowed the plan's git tracking requirement to be met. No scope creep.

## Issues Encountered

None beyond the auto-fixed deviations above.

## User Setup Required

None — all audit data was collected automatically via SSH keys set up in Plan 01.

## Next Phase Readiness

- Inventory established: all 7 servers cataloged with services, Docker containers, databases, disk usage
- Phase 7 (backup automation) can now tailor scripts per server based on actual services found:
  - labsliveserver (104): 55 containers — Docker volume backup priority
  - labs-R820 (117): 24 containers with LightRAG/embeddings
  - alfred (105): PostgreSQL + Redis need database dumps
  - labs-edge-server (100): MySQL running
- Cross-server connection noted: lonewolf (117) LightRAG connects to alfred (105) Ollama for embeddings

## Self-Check: PASSED
