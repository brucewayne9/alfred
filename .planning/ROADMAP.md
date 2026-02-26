# Roadmap: Alfred Platform Stabilization & Ad Management

## Milestones

- [x] **v1.0 Ops Ready** — Phases 1-5 (shipped 2026-02-21)
- [ ] **v1.1 Infrastructure Resilience** — Phases 6-9 (in progress)

## Phases

<details>
<summary>✅ v1.0 Ops Ready (Phases 1-5) — SHIPPED 2026-02-21</summary>

- [x] Phase 1: Infrastructure Repairs (2/2 plans) — completed 2026-02-20
- [x] Phase 2: Alfred Claw Config Fixes (5/5 plans) — completed 2026-02-20
- [x] Phase 3: CRM Reliability (1/1 plan) — completed 2026-02-21
- [x] Phase 4: Google Ads Budget Control (2/2 plans) — completed 2026-02-21
- [x] Phase 5: Ad Workflow Validation & Hardening (3/3 plans) — completed 2026-02-21

Full details: `.planning/milestones/v1.0-ROADMAP.md`

</details>

### v1.1 Infrastructure Resilience

- [ ] **Phase 6: SSH Access & Server Audit** — 105 can reach all 7 servers; each server's services cataloged
- [ ] **Phase 7: Backup System** — Daily + weekly backups running on cron, uploaded to Google Drive with retention
- [ ] **Phase 8: Recovery & Alerting** — Failure alerts active, restore docs written, backup integrity verified
- [ ] **Phase 9: Ad Intelligence** — AI suggestions, cross-platform summary, and confirmation guardrails for ad mutations

## Phase Details

### Phase 6: SSH Access & Server Audit
**Goal**: 105 can SSH into all 7 servers and each server's running services are cataloged
**Depends on**: Nothing (first phase of v1.1)
**Requirements**: INFRA-01, INFRA-02
**Success Criteria** (what must be TRUE):
  1. Running `ssh` from 105 to each of 98, 100, 101, 104, 117, 121 succeeds without a password prompt
  2. A server inventory document exists listing services, Docker containers, databases, and disk usage per server
  3. Alfred Claw on 101 is reachable via SSH from 105 using port 2222 (existing confirmed access validated in new key setup)
**Plans**: TBD

### Phase 7: Backup System
**Goal**: Automated daily and weekly backups run on cron, collect artifacts from all servers, and land in Google Drive with 30-day retention
**Depends on**: Phase 6 (SSH access required to collect backups from remote servers)
**Requirements**: BACKUP-01, BACKUP-02, BACKUP-03, BACKUP-04
**Success Criteria** (what must be TRUE):
  1. At 2 AM each day, a script runs on 105, SSHes into each server, and collects configs, env files, crontabs, and systemd units
  2. At 2 AM each Sunday, a second script runs and additionally collects Docker volumes, app data, and package lists
  3. Both scripts upload collected artifacts to an organized Google Drive folder structure via the existing Workspace integration
  4. Drive folders older than 30 days are automatically pruned so storage does not grow unbounded
  5. Running either script manually produces a backup that appears in Drive within 5 minutes
**Plans**: TBD

### Phase 8: Recovery & Alerting
**Goal**: Mike is alerted on any backup failure, and every server has documented + validated restore procedures
**Depends on**: Phase 7 (backup system must exist before recovery procedures and alerting can be built against it)
**Requirements**: RECOV-01, RECOV-02, RECOV-03
**Success Criteria** (what must be TRUE):
  1. When any server is unreachable or any backup script errors, Mike receives a Telegram message identifying which server failed
  2. A per-server restore document exists that describes how to rebuild that server from scratch using the Drive backups
  3. A validation script runs against the latest backup set and reports whether each server's backup is present and non-empty
  4. Mike can ask Alfred "did last night's backup succeed?" and get a meaningful answer
**Plans**: TBD

### Phase 9: Ad Intelligence
**Goal**: Alfred can suggest ad optimizations, show a combined Meta + Google view, and require confirmation before any financial mutation
**Depends on**: Nothing (independent of infrastructure work — can be built in parallel with Phases 6-8 or after)
**Requirements**: ADS-01, ADS-02, ADS-03
**Success Criteria** (what must be TRUE):
  1. Asking Alfred "how are my campaigns performing?" returns a single response covering both Meta and Google campaigns with key metrics
  2. Asking Alfred "what should I do with my ad budget?" returns specific, actionable suggestions based on current campaign data
  3. Any tool call that would change a budget, bid, or campaign status presents a confirmation step — Alfred does not execute the mutation until Mike explicitly approves
**Plans**: TBD

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Infrastructure Repairs | v1.0 | 2/2 | Complete | 2026-02-20 |
| 2. Alfred Claw Config Fixes | v1.0 | 5/5 | Complete | 2026-02-20 |
| 3. CRM Reliability | v1.0 | 1/1 | Complete | 2026-02-21 |
| 4. Google Ads Budget Control | v1.0 | 2/2 | Complete | 2026-02-21 |
| 5. Ad Workflow Validation & Hardening | v1.0 | 3/3 | Complete | 2026-02-21 |
| 6. SSH Access & Server Audit | v1.1 | 0/? | Not started | — |
| 7. Backup System | v1.1 | 0/? | Not started | — |
| 8. Recovery & Alerting | v1.1 | 0/? | Not started | — |
| 9. Ad Intelligence | v1.1 | 0/? | Not started | — |
