# Requirements: Alfred Platform v1.1

**Defined:** 2026-02-26
**Core Value:** Alfred must be a reliable daily operations tool — every integration works correctly, no duplicate messages, no broken queues, and Mike can manage ad campaigns and CRM contacts conversationally without touching the ad platforms or CRM directly.

## v1.1 Requirements

Requirements for Infrastructure Resilience milestone. Each maps to roadmap phases.

### Infrastructure

- [ ] **INFRA-01**: SSH key access established from 105 to all 7 servers (98, 100, 101, 104, 117, 121)
- [ ] **INFRA-02**: Server audit completed — services, Docker containers, databases, disk usage cataloged per server

### Backup

- [ ] **BACKUP-01**: Daily config backup script runs at 2 AM — captures configs, databases, env files, crontabs, systemd units
- [ ] **BACKUP-02**: Weekly full backup script runs Sunday 2 AM — captures Docker volumes, app data, media, package lists
- [ ] **BACKUP-03**: Backups uploaded to organized Google Drive folder structure via Workspace integration
- [ ] **BACKUP-04**: 30-day retention with automatic cleanup of old backups from Drive

### Recovery & Alerting

- [ ] **RECOV-01**: Telegram failure alert sent to Mike when any backup fails or server is unreachable
- [ ] **RECOV-02**: Per-server restore documentation — how to rebuild from backup on a fresh server
- [ ] **RECOV-03**: Restore validation script — verifies backup integrity and restorability

### Ad Management (carried from v1.0)

- [ ] **ADS-01**: AI-generated performance suggestions for ad campaigns
- [ ] **ADS-02**: Cross-platform ad performance summary — Meta + Google combined
- [ ] **ADS-03**: Confirmation guardrail pattern for financial mutations

## v1.2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Monitoring

- **MON-01**: Real-time server health dashboard in Alfred Labs UI
- **MON-02**: Automated server resource alerts (disk space, CPU, memory thresholds)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Bare metal OS snapshots | Too large for Drive, use hosting provider snapshots instead |
| Cross-datacenter replication | Over-engineered for current 7-server setup |
| Incremental/differential backups | Weekly full + daily config is sufficient for this scale |
| Backup encryption at rest | Drive is authenticated, encryption adds complexity without proportional benefit for now |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| INFRA-01 | — | Pending |
| INFRA-02 | — | Pending |
| BACKUP-01 | — | Pending |
| BACKUP-02 | — | Pending |
| BACKUP-03 | — | Pending |
| BACKUP-04 | — | Pending |
| RECOV-01 | — | Pending |
| RECOV-02 | — | Pending |
| RECOV-03 | — | Pending |
| ADS-01 | — | Pending |
| ADS-02 | — | Pending |
| ADS-03 | — | Pending |

**Coverage:**
- v1.1 requirements: 12 total
- Mapped to phases: 0
- Unmapped: 12

---
*Requirements defined: 2026-02-26*
*Last updated: 2026-02-26 after initial definition*
