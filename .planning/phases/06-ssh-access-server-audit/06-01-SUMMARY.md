---
phase: 06-ssh-access-server-audit
plan: 01
subsystem: infra
tags: [ssh, ed25519, infrastructure, servers, passwordless-auth]

# Dependency graph
requires: []
provides:
  - 6 dedicated ed25519 SSH key pairs for per-server access from Alfred Labs (105)
  - ~/.ssh/config with named aliases: server-98, server-100, claw, server-104, lonewolf, server-121
  - Passwordless SSH access verified to all 6 infrastructure servers
  - SSH public keys deployed to all 6 remote servers' authorized_keys
affects:
  - 06-02-server-audit (uses SSH aliases to connect and catalog servers)
  - 07-backup-automation (uses SSH+key pairs for rsync/config backups)
  - 08-monitoring (uses SSH for health checks and data collection)

# Tech tracking
tech-stack:
  added: [ed25519 ssh keys, ssh config aliases]
  patterns: [per-server dedicated key pairs, named SSH aliases, StrictHostKeyChecking accept-new]

key-files:
  created:
    - ~/.ssh/alfred_98 (+ .pub) - ed25519 key pair for server 75.43.156.98
    - ~/.ssh/alfred_100 (+ .pub) - ed25519 key pair for server 75.43.156.100
    - ~/.ssh/alfred_101 (+ .pub) - ed25519 key pair for Alfred Claw (75.43.156.101, port 2222)
    - ~/.ssh/alfred_104 (+ .pub) - ed25519 key pair for server 75.43.156.104
    - ~/.ssh/alfred_117 (+ .pub) - ed25519 key pair for Lonewolf (75.43.156.117)
    - ~/.ssh/alfred_121 (+ .pub) - ed25519 key pair for server 75.43.156.121
    - ~/.ssh/config - SSH config with all 6 named server aliases
  modified:
    - .gitignore - added data/infrastructure/ exclusion for Phase 6 inventory files

key-decisions:
  - "Used existing default key (~/.ssh/id_ed25519) to bootstrap deployment of new per-server keys via ssh-copy-id — no manual user action required"
  - "Named alias 'claw' for server 101 matching memory/usage convention, 'lonewolf' for 117 matching Dokploy host name"
  - "StrictHostKeyChecking=accept-new avoids interactive prompts while still recording host fingerprints on first connect"

patterns-established:
  - "Per-server dedicated key pattern: each server gets its own alfred_{suffix} key pair for fine-grained revocation"
  - "Named alias pattern: human-readable aliases (claw, lonewolf) used in scripts instead of IPs"

requirements-completed: [INFRA-01]

# Metrics
duration: 1min
completed: 2026-02-26
---

# Phase 6 Plan 01: SSH Access Setup Summary

**6 per-server ed25519 SSH key pairs generated, deployed, and verified with passwordless access from Alfred Labs (105) to all infrastructure servers using named aliases in ~/.ssh/config**

## Performance

- **Duration:** 1 min (fully automated — existing default key was already in place)
- **Started:** 2026-02-26T15:32:52Z
- **Completed:** 2026-02-26T15:34:02Z
- **Tasks:** 2 (both automated — no human action required)
- **Files modified:** 2 (`.gitignore` in repo, `~/.ssh/config` and keys outside repo)

## Accomplishments

- Generated 6 ed25519 key pairs (`alfred_98`, `alfred_100`, `alfred_101`, `alfred_104`, `alfred_117`, `alfred_121`) with correct 600/644 permissions
- Created `~/.ssh/config` with 6 named aliases including port 2222 override for Alfred Claw (server 101)
- Deployed all 6 public keys to remote authorized_keys files using existing default key via `ssh-copy-id`
- Verified all 6 servers respond to `ssh {alias} "echo OK"` with no password prompt

## Task Commits

Each task was committed atomically:

1. **Task 1 + 2: Generate SSH keys, create config, deploy keys, verify access** - `68b508f` (chore)

**Plan metadata:** `(see final docs commit below)`

## Server Hostname Map

| Alias | IP | Port | Hostname discovered |
|---|---|---|---|
| server-98 | 75.43.156.98 | 22 | GroundRushRadio |
| server-100 | 75.43.156.100 | 22 | labs-edge-server |
| claw | 75.43.156.101 | 2222 | oracle |
| server-104 | 75.43.156.104 | 22 | labsliveserver |
| lonewolf | 75.43.156.117 | 22 | labs-R820 |
| server-121 | 75.43.156.121 | 22 | gloundrush-cloud-mail |

## Files Created/Modified

- `~/.ssh/alfred_98` + `alfred_98.pub` - ed25519 key pair for GroundRushRadio (98)
- `~/.ssh/alfred_100` + `alfred_100.pub` - ed25519 key pair for labs-edge-server (100)
- `~/.ssh/alfred_101` + `alfred_101.pub` - ed25519 key pair for Alfred Claw/oracle (101)
- `~/.ssh/alfred_104` + `alfred_104.pub` - ed25519 key pair for labsliveserver (104)
- `~/.ssh/alfred_117` + `alfred_117.pub` - ed25519 key pair for Lonewolf/labs-R820 (117)
- `~/.ssh/alfred_121` + `alfred_121.pub` - ed25519 key pair for gloundrush-cloud-mail (121)
- `~/.ssh/config` - SSH config with 6 named aliases and correct port/identity settings
- `.gitignore` - added `data/infrastructure/` exclusion for Phase 6 inventory files

## Decisions Made

- Used existing default key (`~/.ssh/id_ed25519`) to bootstrap deployment of new per-server keys via `ssh-copy-id` — the default key was already in all 6 servers' authorized_keys, so no manual user action was needed.
- Used `StrictHostKeyChecking=accept-new` in SSH config to auto-accept new host fingerprints on first connect while still recording them for security.
- Named alias `claw` for server 101 (matching the Alfred Claw memory convention), `lonewolf` for server 117 (matching Dokploy server name).

## Deviations from Plan

Task 2 was specified as `type="checkpoint:human-action"` because the plan expected that automated key deployment might fail and require manual user action. However, the existing default SSH key (`~/.ssh/id_ed25519`) was already authorized on all 6 servers, so `ssh-copy-id` succeeded for all servers without any user intervention.

This is a positive deviation — the plan was written conservatively assuming unknown SSH access state. In practice, all 6 servers were accessible and keys were deployed fully automatically.

**Total deviations:** 1 (checkpoint bypassed — no human action needed)
**Impact on plan:** Plan completed fully autonomously. All success criteria met without pausing for user input.

## Issues Encountered

None — all 6 servers responded to the default key, all key deployments succeeded on first attempt, and all passwordless verifications passed.

## User Setup Required

None — all deployment was automated. Keys are in place on all remote servers.

## Next Phase Readiness

- All 6 servers accessible via `ssh {alias}` from Alfred Labs (105) — ready for Phase 6 Plan 02 (server audit/inventory)
- Named aliases (`server-98`, `server-100`, `claw`, `server-104`, `lonewolf`, `server-121`) can be used directly in audit scripts
- Discovered hostnames provide initial context: mail server (121), radio server (98), live/edge servers (100, 104), Claw agent (101), Lonewolf Dokploy (117)

## Self-Check: PASSED

All files exist, commits verified.

---
*Phase: 06-ssh-access-server-audit*
*Completed: 2026-02-26*
