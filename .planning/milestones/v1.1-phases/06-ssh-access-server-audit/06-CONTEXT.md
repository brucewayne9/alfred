# Phase 6: SSH Access & Server Audit - Context

**Gathered:** 2026-02-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Set up passwordless SSH from 105 (Alfred Labs) to all 7 servers (98, 100, 101, 104, 117, 121) and catalog running services, containers, databases, and disk usage on each server. Monitoring, alerting, and automated remediation are separate phases.

</domain>

<decisions>
## Implementation Decisions

### Inventory depth & detail
- Map cross-server connections (which servers talk to which — e.g., 105→101 for monitoring, 117 runs Traefik)
- Let the audit discover unusual setups rather than pre-flagging — surface quirks and document them as found

### Inventory format & location
- Dual format: JSON as source of truth + auto-generated markdown for readability
- Inventory files NOT committed to git — add to .gitignore (contains sensitive IPs and service info)

### SSH key strategy
- Per-server SSH keys (separate ed25519 key for each server)
- Username: `brucewayne9` on all servers
- Port 2222 for server 101 (Alfred Claw), standard port 22 for all others

### Claude's Discretion
- Inventory detail depth (services, versions, resource usage, databases — pick what's useful for later phases)
- Docker container catalog depth (names + images vs full compose context)
- Inventory file location (data/infrastructure/, .planning/infrastructure/, or config/ — pick most logical)
- Whether to include summary table at top of markdown inventory
- SSH config aliases (whether to set up named aliases like `ssh claw`)
- SSH key naming convention (by server role, IP suffix, or other scheme)

</decisions>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches for infrastructure auditing. The inventory should be practical for future phases (health checks, backup automation, etc.).

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 06-ssh-access-server-audit*
*Context gathered: 2026-02-26*
