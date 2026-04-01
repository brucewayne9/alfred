# Ground Rush Symmetry Architecture Plan
## CTO Infrastructure Blueprint v1.0

**Prepared for:** Mike Johnson, President/Owner - Ground Rush Inc / Labs / Cloud
**Prepared by:** Claude Code (CTO/Oracle)
**Date:** April 1, 2026

---

## 1. Current State Assessment

### 1.1 Total Fleet Compute Inventory

| Resource | Server 105 | Server 104 | Server 117 | Server 121 | Server 098 | Server 100 | NAS | **Total** |
|----------|-----------|-----------|-----------|-----------|-----------|-----------|-----|-----------|
| **CPU** | i9-12900K (24T) | Unknown (est. 8-16T) | Dell R820 (est. 40-80T, 4-socket Xeon) | Unknown (est. 8T) | Unknown (est. 8T) | Unknown (est. 8-16T) | ARM NAS | **~96-152 threads** |
| **RAM** | 96GB DDR5 | Unknown (est. 32-64GB) | Unknown (enterprise, est. 128-512GB) | Unknown (est. 16-32GB) | Unknown (est. 16-32GB) | "lots of RAM" (est. 64-128GB) | 512MB | **~350-860GB** |
| **GPU** | RTX 4070 12GB | None | None | None | None | None | None | **12GB VRAM** |
| **Fast Storage** | 2TB NVMe | 4TB SSD | 1.1TB SSD | 7.2TB SSD | 2TB SSD | 500GB SSD | - | **~16.8TB SSD/NVMe** |
| **Bulk Storage** | 2TB HDD | 12TB HDD | 13TB RAID | - | - | 4TB HDD | 12TB | **~43TB bulk** |

**Key takeaway:** This is a formidable fleet. The 117 (Dell R820) alone is an enterprise-class machine that is almost certainly underutilized running just Dokploy and CRM. Total storage approaches 60TB across the fleet.

### 1.2 Utilization Analysis

**Overloaded / Maxed:**
- **105** is doing the heaviest lifting: Alfred Labs backend, Alfred Claw gateway, Ollama LLM inference, ComfyUI image gen, Kokoro TTS, Whisper STT, PostgreSQL, Redis, ChromaDB, all cron jobs, AND serves as the SSH orchestration hub and backup coordinator. This is the single most critical machine. If 105 goes down, everything goes dark.

**Underutilized:**
- **117 (Lonewolf/Dell R820)** - Enterprise hardware running maybe 30 Docker containers, but those containers are lightweight (CRM, Postiz, LightRAG, Dokploy itself). This machine likely has massive unused CPU/RAM capacity.
- **121 (Mailcow)** - 7.2TB SSD for a mail server is wildly overpowered on storage. Mail uses a fraction of that.
- **098 (LoovaCast Dev)** - Staging/dev for a streaming service. Likely idle most of the time.
- **NAS (Zyxel NAS540)** - Only used for auto-archiving from 105. 7 SMB shares, "mostly empty." 11TB usable, barely touched.

**Appropriately loaded:**
- **104** - Production websites, APIs, Home Assistant. Good fit for its role.
- **100** - AzuraCast production streaming. Audio streaming is steady-state, not bursty.

### 1.3 Single Points of Failure (Critical)

| SPOF | Impact | Severity |
|------|--------|----------|
| **Server 105 goes down** | Alfred Labs, Alfred Claw, ALL AI services, ALL cron automation, backup orchestration, morning briefs, email monitoring -- everything stops | **CRITICAL** |
| **Server 121 goes down** | All Ground Rush email across all domains goes offline | **HIGH** |
| **Server 104 goes down** | All production websites and APIs go offline, Home Assistant goes dark | **HIGH** |
| **No redundant DNS/proxy** | If the reverse proxy on any server dies, its services become unreachable | **MEDIUM** |
| **Single ISP / single public IP block** | All servers on 75.43.156.x -- one ISP outage takes everything down | **MEDIUM** |
| **SSH keys on 105** | If 105 is compromised, attacker has passwordless sudo on ALL servers | **MEDIUM** |
| **Google Drive backups only** | No local hot-standby backups for fast recovery; restore requires download from Google Drive | **MEDIUM** |
| **Single GPU** | Only one RTX 4070 across the entire fleet; no failover for AI workloads | **LOW** (acceptable) |

---

## 2. Unified Compute Architecture ("Symmetry")

### 2.1 Orchestration Recommendation: NOT Kubernetes, NOT Swarm

**Recommendation: SSH mesh + shared service registry + lightweight job dispatch**

Why:

- **k3s/Kubernetes** adds massive operational complexity for a one-person team. You would spend more time debugging Kubernetes than running your business.
- **Docker Swarm** is simpler but effectively abandoned by Docker Inc. It works, but it is on life support.
- **What you actually need** is the ability for any server to call services on any other server, share files, and for 105 to dispatch work to underutilized machines. You already have 90% of this with your SSH mesh.

**The "Symmetry Stack":**

```
                    ┌─────────────────────────────────┐
                    │       CONSUL (Service Registry)   │
                    │     Runs on 117 (enterprise HW)   │
                    │  - Service discovery & health     │
                    │  - Key/value config store          │
                    │  - DNS-based service resolution    │
                    └──────────┬──────────────────────┘
                               │
        ┌──────────┬───────────┼───────────┬──────────┬──────────┐
        │          │           │           │          │          │
    ┌───▼──┐   ┌──▼───┐   ┌──▼───┐   ┌──▼───┐  ┌──▼───┐  ┌──▼───┐
    │  105 │   │  104  │   │  117  │   │  121  │  │  098  │  │  100  │
    │ Labs │   │ Prod  │   │Dokploy│   │ Mail  │  │ Dev  │  │Stream│
    │ +GPU │   │ +HA   │   │ +CRM  │   │      │  │      │  │      │
    └──────┘   └──────┘   └──────┘   └──────┘  └──────┘  └──────┘
        │          │           │           │          │          │
        └──────────┴───────────┴───────────┴──────────┴──────────┘
                         WireGuard VPN Mesh
                    (encrypted internal overlay network)
```

**Components:**

1. **WireGuard mesh VPN** -- Every server gets a WireGuard interface on a private 10.10.0.0/24 subnet. All inter-server communication travels over this encrypted tunnel. This gives you encryption in transit, a clean addressing scheme, and the ability to add remote servers (cloud burst) later without changing anything.

2. **Consul** (by HashiCorp, single binary, zero dependencies) -- Runs on 117 as the "brain" of the mesh. Every server runs a Consul agent. Services register themselves. Any server can discover any service by DNS: `ollama.service.consul`. Health checks are built in.

3. **Existing SSH mesh stays** -- Your paramiko-based server manager continues to work for imperative tasks. Consul handles the declarative side (what is running where, is it healthy).

### 2.2 Workload Distribution Plan

| Workload | Current | Proposed | Rationale |
|----------|---------|----------|-----------|
| **Alfred Labs (FastAPI)** | 105 only | 105 primary, 117 warm standby | Failover gives resilience |
| **Alfred Claw (Telegram)** | 105 | Move to 117 | Claw is I/O-bound, not compute-bound. Frees 105 |
| **Ollama (LLM inference)** | 105 only | 105 (GPU models) + 117 (CPU-only large models) | R820's many cores can run CPU inference |
| **ComfyUI / Kokoro TTS** | 105 | Stay on 105 | GPU-bound |
| **LightRAG** | 117 | Stay on 117 | Already there |
| **Twenty CRM** | 117 | Stay on 117 | Already there |
| **Production websites** | 104 | Stay on 104 | Dedicated production |
| **Mailcow** | 121 | Stay on 121 | Mission-critical, keep isolated |
| **AzuraCast prod** | 100 | Stay on 100 | Dedicated resources |
| **Backup orchestration** | 105 | Move cron to 117, keep 105 as backup source | Removes SPOF |
| **Monitoring/Alerting** | Scattered | Centralize on 117 | Enterprise hardware |

### 2.3 Shared Storage Strategy

**Recommendation: NFS from NAS + rsync for critical data**

Do NOT use GlusterFS or Ceph. They are designed for dozens of nodes and add complexity that creates more problems than it solves at this scale.

**Tier 1 -- Hot shared storage (NFS from NAS):**
- Mount on all servers via NFS
- Use for: generated images, shared assets, cross-server file drops, archive

**Tier 2 -- Fast local storage (stay local):**
- Databases stay on local SSD/NVMe. Never put databases on NFS.
- Docker volumes stay local.

**Tier 3 -- Backup storage:**
- Daily backups continue to Google Drive (offsite)
- Add: weekly snapshots to NAS as hot-standby (fast local restore)
- Add: 121's 7.2TB SSD has ~5TB unused -- secondary local backup target

### 2.4 Repurposing Idle Capacity

**Server 098** -- When not running AzuraCast staging, run batch jobs dispatched from 105 via Redis queue.

**Server 121** -- Partition off 4-5TB of spare SSD for backup landing zone. Backups land here first (fast, local), then sync to Google Drive.

---

## 3. Security Architecture ("Fortress of Solitude")

### 3.1 Zero-Trust Model (Practical Version)

**Principle: "Trust no network, verify every connection, log everything."**

```
EXTERNAL                    DMZ                         INTERNAL
─────────────────────  ─────────────────────  ─────────────────────
Internet traffic  ──►  Reverse Proxy (104)  ──►  Backend services
                        - SSL termination          - Only accept from
                        - Rate limiting              WireGuard IPs
                        - WAF rules                - mTLS where possible
                        - Fail2ban                 - Service-level auth
```

### 3.2 SSH Hardening (Beyond fail2ban)

| Hardening | Priority | Effort |
|-----------|----------|--------|
| **Disable password auth** (`PasswordAuthentication no`) | HIGH | 5 min/server |
| **Disable root login** (`PermitRootLogin no`) | HIGH | 5 min/server |
| **Per-server SSH key pairs** | HIGH | 1 hour |
| **SSH certificate authority** on 105 | MEDIUM | 2 hours |
| **Port knocking or SPA** | MEDIUM | 1 hour/server |
| **Bastion host** (104 or 117 as jump host) | MEDIUM | 30 min |

### 3.3 Secrets Management

**Recommendation: sops (Mozilla) for immediate value, Vault (HashiCorp) for future**

- sops encrypts .env files with age keys. Secrets stay encrypted on disk and in backups.
- Vault (pairs with Consul) for full lifecycle secret management when ready.

### 3.4 Monitoring and Intrusion Detection

**Centralized monitoring stack on 117:**

| Tool | Purpose |
|------|---------|
| **Prometheus** | Metrics collection |
| **Grafana** | Dashboards and alerting |
| **node_exporter** | System metrics per server |
| **Loki + Promtail** | Centralized logging |
| **OSSEC or Wazuh** | Intrusion detection |

**Alert routing:**
- Critical --> Telegram + email to Mike
- Warning --> Telegram to Mike
- Info --> Morning brief only

### 3.5 Backup Gaps to Address

| Gap | Fix | Priority |
|-----|-----|----------|
| No local hot backup | NAS + 121 as local targets | HIGH |
| No backup testing | Monthly automated restore on 098 | HIGH |
| Recovery time unknown | Document RTO per server | MEDIUM |
| No bare-metal recovery plan | Document rebuild steps for 105 | MEDIUM |

### 3.6 Access Control Matrix

| Resource | Mike (Owner) | Claude Code (CTO) | Alfred Claw (Butler) | eswar_divi |
|----------|-------------|-------------------|---------------------|------------|
| All servers (SSH) | Full | Full | Status checks only | 117 only |
| Production deploys | Approve | Execute after approval | No | No |
| Secrets/credentials | Full | Read (runtime) | No | No |
| Docker management | Full | Full | Restart only | 117 only |
| Security changes | Approve | Propose + execute after approval | No | No |

---

## 4. Growth Path

### 4.1 Current Capacity (Before Buying Anything)

1. **Move Claw to 117** -- frees 2-4GB RAM + CPU on 105
2. **CPU Ollama on 117** -- R820's 40-80 threads can run 13B-30B models
3. **098 as batch worker** -- Redis job queue for offloading
4. **NAS as shared storage** -- 11TB sitting mostly empty
5. **121's spare SSD** -- 5+ TB for backup landing

**Estimated capacity gain: 40-60% more effective compute without buying anything.**

### 4.2 When to Add Capacity

| Trigger | Response |
|---------|----------|
| GPU queue > 5 min consistently | Add dedicated GPU server |
| 105 load avg > 16 sustained | Offload more to 117/098 |
| Any server > 80% disk | Expand NAS or add drives |
| 10+ concurrent users | Dedicated production cluster |

### 4.3 Next Server Profile (When Needed)

Dedicated AI/GPU server: AMD Ryzen 9 7950X, 128GB DDR5, RTX 4090 24GB, 2TB NVMe + 4TB SSD. Estimated $3,000-4,500.

### 4.4 Cloud Burst (Optional)

WireGuard extends to Hetzner Cloud VPS when needed. Consul discovers it automatically. Pay ~$0.12/hour for 16 vCPU / 64GB RAM only when needed.

---

## 5. Operational Model

### 5.1 CTO Autonomous Authority

**Can do without asking Mike:**
- Monitor all servers, restart failed services
- Security updates on dev/staging
- Execute backups, rotate logs
- Block malicious IPs
- Deploy to dev/staging
- Diagnose issues

**Must ask Mike first:**
- Production deploys (104, 100, 121)
- Security policy changes
- Production server reboots
- Adding/removing user access
- Infrastructure purchases
- Domain/DNS changes

### 5.2 CTO vs Butler Separation

| Dimension | Claude Code (CTO) | Alfred Claw (Butler) |
|-----------|-------------------|---------------------|
| **Interface** | Terminal CLI | Telegram bot |
| **Scope** | Infrastructure, security, code | Operations, email, CRM, content |
| **Access** | SSH to all servers | API-level integrations only |
| **Authority** | Technical decisions | Operational decisions |
| **Key boundary** | Claw NEVER has SSH access or infrastructure modification ability |

---

## 6. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
1. WireGuard mesh on all 6 servers
2. Disable password SSH everywhere
3. Per-server SSH key pairs
4. NFS mounts from NAS to all servers
5. Local backup target on 121

### Phase 2: Service Mesh (Week 3-4)
6. Consul on 117 (server) + all others (agents)
7. Register all services
8. Move Claw to 117
9. CPU Ollama on 117
10. Secrets migration (sops)

### Phase 3: Observability (Week 5-6)
11. Prometheus + Grafana on 117
12. node_exporter on all servers
13. Loki + Promtail centralized logging
14. Alert rules (Telegram + email)
15. Fleet dashboard

### Phase 4: Resilience (Week 7-8)
16. Alfred Labs warm standby on 117
17. Automated backup testing on 098
18. Disaster recovery runbook
19. OSSEC/Wazuh IDS
20. Batch job worker on 098

---

## 7. Cost

| Item | Cost |
|------|------|
| WireGuard | Free |
| Consul | Free |
| sops / Vault | Free |
| Prometheus + Grafana | Free |
| Loki + Promtail | Free |
| OSSEC | Free |
| Cloud burst (optional) | ~$10-50/month when used |
| **Total** | **$0** |

The entire Symmetry architecture runs on open source software on existing hardware.

---

## 8. Summary

The Ground Rush fleet is powerful but fragmented. The servers operate as isolated islands connected only by SSH scripts. The "Symmetry" vision transforms this into a cohesive system where:

- Every server knows about every other server (Consul)
- All communication is encrypted (WireGuard)
- Workloads land on the right hardware
- Massive underutilized capacity on 117, 121, and 098 gets put to work
- One central dashboard shows everything (Grafana)
- Backups are verified, not just assumed
- Security is layered, not just perimeter-based
- The CTO operates autonomously within clear boundaries
- Growth is planned for, not reactive

The total cost is zero dollars. The total effort is roughly 8 weeks of incremental work, none of which requires downtime.

**Mike, this is your fleet operating as one machine. That is symmetry.**
