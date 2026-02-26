# Server Restore Procedures

*Generated: 2026-02-26*
*Maintained by: Alfred Labs backup system*

## Overview

The Alfred backup system runs on two schedules from alfred-labs (75.43.156.105):

- **Daily (Mon-Sat, 2 AM):** Config files, database dumps, crontabs, systemd service listings
- **Weekly (Sunday, 2 AM):** Everything in daily PLUS package list, Docker volume exports, Docker image listings

All artifacts are uploaded to Google Drive under:

```
Alfred Backups/{server_name}/{backup_type}/YYYY-MM-DD/
```

Where `{backup_type}` is `daily` or `weekly` and `YYYY-MM-DD` is the backup run date.

Each backup is a single `.tar.gz` file containing all collected files and command outputs for that server.

## Prerequisites for Any Restore

Before starting a restore, ensure you have:

1. **Google Drive access** — Log into the Google Workspace account that holds the Alfred Backups folder
2. **SSH key pair** — The `~/.ssh/alfred_{server}` private key (backed up in alfred-labs config)
3. **Fresh server** — Ubuntu 22.04 or 24.04 LTS (match the OS version from inventory)
4. **Python 3.11+** — For running Alfred scripts post-restore
5. **Network access** — New server must be reachable on its designated IP (update DNS/firewall as needed)

### How to Extract a Backup

```bash
# Download the .tar.gz from Google Drive
# Then extract:
tar -xzf {server}-{date}.tar.gz
cd {server}-{date}/
ls -la  # review all captured files and command outputs
```

---

## Server 1: alfred-labs (75.43.156.105)

### Overview

- **IP:** 75.43.156.105
- **OS:** Ubuntu 22.04.5 LTS
- **Role:** Alfred Labs web application — FastAPI backend (port 8400), React frontend (port 80/443 via Caddy), backup orchestrator for all 7 servers
- **Key services:** alfred.service (FastAPI), caddy.service (reverse proxy), postgresql@14-main.service, redis-server.service, telegram-bot.service, ollama.service, docker.service
- **SSH:** Port 22, user `aialfred`

### What is Backed Up

**Daily files:**
- `/etc/crontab` — system crontab
- `/home/aialfred/alfred/config/.env` — all API keys, database URLs, secrets
- `/home/aialfred/alfred/config/settings.py` — Pydantic settings model
- `/home/aialfred/alfred/config/users.json` — user accounts (bruce, etc.)
- `/home/aialfred/alfred/config/google_token.json` — Google Workspace OAuth token
- `/var/lib/redis/dump.rdb` — Redis snapshot

**Daily commands:**
- `crontab-user.txt` — user crontab (`crontab -l`)
- `systemd-custom-services.txt` — custom systemd service file listings
- `postgresql-dump.sql` — full PostgreSQL dump (`pg_dumpall -U postgres`)

**Weekly extras:**
- `packages-dpkg.txt` — full installed package list (`dpkg --list`)
- `docker-volumes.txt` — Docker volume listing
- `docker-images.txt` — Docker image listing
- `data-files.txt` — listing of all files under `/home/aialfred/alfred/data/`
- `docker-volume-export.sh` — Docker volume mountpoints

### Restore Steps

**Step 1: Base OS setup**

```bash
# Install Ubuntu 22.04 LTS on fresh machine
# After first boot, update and install packages from weekly backup

sudo apt-get update && sudo apt-get upgrade -y

# From the weekly backup, packages-dpkg.txt has the full list
# Restore key packages:
sudo apt-get install -y python3.11 python3.11-venv python3-pip \
  postgresql-14 redis-server caddy docker.io docker-compose-plugin \
  nodejs npm curl git fail2ban

# For ollama (LLM service):
curl -fsSL https://ollama.ai/install.sh | sh
```

**Step 2: User setup**

```bash
# Create aialfred user
sudo adduser aialfred
sudo usermod -aG sudo,docker aialfred

# Restore SSH authorized_keys
# From the backup, look for authorized_keys in the alfred config or manually copy
sudo -u aialfred mkdir -p /home/aialfred/.ssh
# Copy the SSH public key for the new server and any personal keys
sudo -u aialfred nano /home/aialfred/.ssh/authorized_keys
sudo chmod 700 /home/aialfred/.ssh && sudo chmod 600 /home/aialfred/.ssh/authorized_keys
```

**Step 3: Clone and restore Alfred repository**

```bash
# Clone the Alfred codebase (or restore from git)
sudo -u aialfred git clone <alfred-git-repo> /home/aialfred/alfred

# Restore config files from backup tarball
cp backup/config/.env /home/aialfred/alfred/config/.env
cp backup/config/settings.py /home/aialfred/alfred/config/settings.py
cp backup/config/users.json /home/aialfred/alfred/config/users.json
cp backup/config/google_token.json /home/aialfred/alfred/config/google_token.json

# Install Python dependencies
cd /home/aialfred/alfred
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Step 4: PostgreSQL restoration**

```bash
# PostgreSQL is already installed in Step 1
sudo systemctl start postgresql

# Restore from the postgresql-dump.sql in backup:
sudo -u postgres psql < backup/postgresql-dump.sql

# Verify databases restored:
sudo -u postgres psql -c "\l"
```

**Step 5: Redis restoration**

```bash
sudo systemctl stop redis-server
# Restore the Redis snapshot:
sudo cp backup/dump.rdb /var/lib/redis/dump.rdb
sudo chown redis:redis /var/lib/redis/dump.rdb
sudo systemctl start redis-server

# Verify Redis data loaded:
redis-cli DBSIZE
```

**Step 6: Alfred data directory restoration**

```bash
# conversations.db and learning.db are SQLite files
# The weekly backup's data-files.txt shows what was in /home/aialfred/alfred/data/
# These SQLite files must be restored separately if they were exported

# If you have the SQLite files directly:
mkdir -p /home/aialfred/alfred/data
cp backup/conversations.db /home/aialfred/alfred/data/conversations.db
cp backup/learning.db /home/aialfred/alfred/data/learning.db

# ChromaDB data — restore the full chroma directory:
cp -r backup/chromadb/ /home/aialfred/alfred/data/chromadb/
```

**Step 7: Docker containers**

```bash
# portainer
docker run -d \
  --name portainer \
  --restart always \
  -p 8000:8000 -p 9000:9000 -p 9443:9443 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v portainer_data:/data \
  portainer/portainer-ce:lts

# audio-news-api (rebuild from source in Alfred repo)
cd /home/aialfred/alfred
docker build -t audio-news-api -f docker/audio-news-api/Dockerfile .
docker run -d --name audio-news-api-container \
  --restart always \
  -p 8765:8765 audio-news-api
```

**Step 8: Systemd services**

```bash
# Alfred FastAPI service
sudo cp /home/aialfred/alfred/config/alfred.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable alfred
sudo systemctl start alfred

# Caddy (web server / reverse proxy)
# Caddy config lives at /etc/caddy/Caddyfile — restore from backup if present
sudo systemctl enable caddy
sudo systemctl start caddy
```

**Step 9: Crontab restoration**

```bash
# From backup crontab-user.txt, restore the user crontab:
crontab -e
# Or pipe directly:
cat backup/crontab-user.txt | crontab -

# Also restore /etc/crontab:
sudo cp backup/crontab /etc/crontab
```

### Post-Restore Validation

```bash
# Alfred API responding:
curl -s http://localhost:8400/health

# PostgreSQL running:
sudo systemctl status postgresql

# Redis running:
redis-cli ping  # should return PONG

# Alfred service running:
sudo systemctl status alfred

# Docker containers up:
docker ps

# Check all backup cron jobs registered:
crontab -l
```

---

## Server 2: groundrush-radio (75.43.156.98)

### Overview

- **IP:** 75.43.156.98
- **OS:** Ubuntu 22.04.5 LTS
- **Role:** GroundRush Radio — AzuraCast radio streaming platform, News Mews Radio (station ID 22)
- **Key services:** AzuraCast (Docker-based), web radio streaming, Liquidsoap, Icecast
- **SSH:** Port 22, alias `server-98`
- **Docker:** 3 containers

### What is Backed Up

**Daily files:**
- `/etc/crontab` — system crontab

**Daily commands:**
- `crontab-user.txt` — user crontab
- `systemd-custom-services.txt` — custom systemd service listings
- `dotenv-files.txt` — paths to `.env` files under `/opt`, `/var/www`, `/home`
- `azuracast-config.txt` — paths to AzuraCast config files (`.conf`, `.env`)

**Weekly extras:**
- `packages-dpkg.txt` — installed package list
- `docker-volumes.txt` — Docker volume listing
- `docker-images.txt` — Docker image listing

**Note:** AzuraCast database (MySQL) is NOT directly dumped by the Alfred backup system. AzuraCast has its own internal backup mechanism. The `azuracast-config.txt` provides paths to locate the configs.

### Restore Steps

**Step 1: Base OS setup**

```bash
sudo apt-get update && sudo apt-get upgrade -y
# Restore packages from packages-dpkg.txt (weekly backup)
sudo apt-get install -y docker.io docker-compose curl git
```

**Step 2: SSH setup**

```bash
# Copy SSH authorized_keys for the alfred backup key
mkdir -p ~/.ssh
echo "<alfred_radio_public_key>" >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys
```

**Step 3: AzuraCast installation**

```bash
# AzuraCast is Docker-based and managed via its official installer
mkdir -p /var/azuracast
cd /var/azuracast
curl -fsSL https://raw.githubusercontent.com/AzuraCast/AzuraCast/stable/docker.sh > docker.sh
chmod a+x docker.sh
./docker.sh install

# AzuraCast will start its own MySQL container internally
# After installation, restore AzuraCast data:
# 1. Use AzuraCast's own restore command with an AzuraCast backup archive if available
./docker.sh restore /path/to/azuracast-backup.tar.gz
```

**Step 4: Restore AzuraCast config files**

```bash
# From dotenv-files.txt and azuracast-config.txt in the backup,
# identify the .env paths, then restore:
# Common location: /var/azuracast/.env
cp backup-extracted/azuracast-config/.env /var/azuracast/.env

# Restart AzuraCast after config restore:
cd /var/azuracast && ./docker.sh restart
```

**Step 5: Crontab restoration**

```bash
cat backup/crontab-user.txt | crontab -
sudo cp backup/crontab /etc/crontab
```

### Post-Restore Validation

```bash
# AzuraCast web UI accessible:
curl -I http://localhost/
# Should return 200 or 302

# Check radio stream:
curl -s http://localhost:8000/listen  # Icecast stream endpoint

# Check AzuraCast containers running:
cd /var/azuracast && docker-compose ps
```

**Key gotcha:** AzuraCast station ID 22 (News Mews Radio) configuration is stored inside AzuraCast's MySQL database. Without an AzuraCast-native backup, you will need to manually reconfigure the station via the AzuraCast web UI.

---

## Server 3: labs-edge (75.43.156.100)

### Overview

- **IP:** 75.43.156.100
- **OS:** Ubuntu 22.04.5 LTS
- **Role:** Edge server for labs — various Docker services, MySQL databases
- **Key services:** Docker containers (8 total), MySQL
- **SSH:** Port 22, alias `server-100`

### What is Backed Up

**Daily files:**
- `/etc/crontab` — system crontab

**Daily commands:**
- `crontab-user.txt` — user crontab
- `systemd-custom-services.txt` — custom systemd service listings
- `mysql-dump.sql` — full MySQL dump (`mysqldump --all-databases`)
- `dotenv-files.txt` — paths to `.env` files under `/opt`

**Weekly extras:**
- `packages-dpkg.txt` — installed package list
- `docker-volumes.txt` — Docker volume listing
- `docker-images.txt` — Docker image listing

### Restore Steps

**Step 1: Base OS setup**

```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y docker.io docker-compose-plugin mysql-server curl git
```

**Step 2: SSH setup**

```bash
mkdir -p ~/.ssh
echo "<alfred_edge_public_key>" >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys
```

**Step 3: MySQL restoration**

```bash
# Start MySQL
sudo systemctl start mysql

# Restore from the mysql-dump.sql in backup:
sudo mysql < backup/mysql-dump.sql

# Verify databases restored:
sudo mysql -e "SHOW DATABASES;"
```

**Step 4: Docker services restoration**

```bash
# From dotenv-files.txt, find the .env file paths (they'll be under /opt)
# These .env files contain the docker-compose configuration paths

# For each Docker service found in the backup:
# 1. Locate the docker-compose.yml (usually in /opt/<service>/)
# 2. Restore the .env file
# 3. Run docker-compose up

# Example pattern:
cd /opt/<service>/
cp backup-extracted/<service>.env .env
docker compose up -d
```

**Step 5: Crontab restoration**

```bash
cat backup/crontab-user.txt | crontab -
sudo cp backup/crontab /etc/crontab
```

### Post-Restore Validation

```bash
# MySQL running:
sudo systemctl status mysql
sudo mysql -e "SHOW DATABASES;"

# Docker containers up:
docker ps

# Check container count matches pre-failure (8 containers expected):
docker ps | wc -l
```

---

## Server 4: alfred-claw (75.43.156.101)

### Overview

- **IP:** 75.43.156.101
- **OS:** Ubuntu 24.04.4 LTS
- **Role:** Alfred Claw — OpenClaw AI agent, Telegram bot (`@alfredblogbot`), RSS API service, no Docker installed
- **Key services:** openclaw (main AI agent), alfred-rss-api.service (port 8401), telegram bot via openclaw
- **SSH:** Port 2222 (non-standard!), user `brucewayne9`, alias `claw`
- **Docker:** Not installed

### What is Backed Up

**Daily files:**
- `/etc/crontab` — system crontab
- `/root/.openclaw/openclaw.json` — OpenClaw main configuration (models, channels, compaction settings)
- `/root/.openclaw/workspace/USER.md` — user profile and preferences
- `/root/.openclaw/workspace/SOUL.md` — personality configuration
- `/root/.openclaw/workspace/AGENTS.md` — agent instructions (TOOLS.md, AGENTS.md)
- `/root/.openclaw/workspace/TOOLS.md` — tool definitions
- `/root/.openclaw/workspace/HEARTBEAT.md` — heartbeat configuration
- `/root/.openclaw/workspace/QUEUE.md` — current task queue

**Daily commands:**
- `crontab-user.txt` — user crontab
- `systemd-custom-services.txt` — custom systemd service listings

**Weekly extras:**
- `packages-dpkg.txt` — installed package list
- `docker-volumes.txt` — not applicable (no Docker)
- `docker-images.txt` — not applicable (no Docker)

### Restore Steps

**Step 1: Base OS setup**

```bash
# Ubuntu 24.04 LTS on fresh machine
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y python3 python3-pip nodejs npm curl git
```

**Step 2: SSH setup (CRITICAL — non-standard port)**

```bash
# SSH runs on port 2222, NOT 22
# First, change SSH port:
sudo nano /etc/ssh/sshd_config
# Set: Port 2222

sudo systemctl restart ssh

# Add authorized keys for backup user (brucewayne9):
sudo adduser brucewayne9
sudo usermod -aG sudo brucewayne9
sudo -u brucewayne9 mkdir -p /home/brucewayne9/.ssh
echo "<alfred_claw_public_key>" >> /home/brucewayne9/.ssh/authorized_keys
sudo chmod 700 /home/brucewayne9/.ssh
sudo chmod 600 /home/brucewayne9/.ssh/authorized_keys
```

**Step 3: OpenClaw installation**

```bash
# Install OpenClaw via npm (global):
sudo npm install -g openclaw

# Or from the known version:
sudo npm install -g openclaw@2026.2.17
```

**Step 4: Restore OpenClaw workspace**

```bash
# Create workspace directories
sudo mkdir -p /root/.openclaw/workspace

# Restore config files from backup:
sudo cp backup/openclaw.json /root/.openclaw/openclaw.json
sudo cp backup/USER.md /root/.openclaw/workspace/USER.md
sudo cp backup/SOUL.md /root/.openclaw/workspace/SOUL.md
sudo cp backup/AGENTS.md /root/.openclaw/workspace/AGENTS.md
sudo cp backup/TOOLS.md /root/.openclaw/workspace/TOOLS.md
sudo cp backup/HEARTBEAT.md /root/.openclaw/workspace/HEARTBEAT.md
sudo cp backup/QUEUE.md /root/.openclaw/workspace/QUEUE.md

# Set proper permissions:
sudo chmod 600 /root/.openclaw/openclaw.json
sudo chown -R root:root /root/.openclaw/
```

**Step 5: Restore systemd services**

```bash
# From systemd-custom-services.txt in backup, identify service files
# Restore the alfred-rss-api service:
sudo cp backup/alfred-rss-api.service /etc/systemd/system/

sudo systemctl daemon-reload
sudo systemctl enable alfred-rss-api
sudo systemctl start alfred-rss-api
```

**Step 6: Start OpenClaw**

```bash
# Start OpenClaw (runs as root for system access):
sudo openclaw start

# Verify gateway is running:
sudo openclaw status
```

**Step 7: Crontab restoration**

```bash
# Root crontab (OpenClaw runs as root):
sudo crontab -e
# Or:
sudo bash -c 'cat backup/crontab-user.txt | crontab -'

sudo cp backup/crontab /etc/crontab
```

### Post-Restore Validation

```bash
# SSH accessible on port 2222:
ssh -p 2222 brucewayne9@75.43.156.101 "echo OK"

# OpenClaw status:
sudo openclaw status

# RSS API running:
curl http://localhost:8401/health

# Telegram bot responding:
# Send a message to @alfredblogbot on Telegram

# Log check:
tail -f /tmp/openclaw-1000/openclaw-$(date +%Y-%m-%d).log
```

**Key gotcha:** Sessions are at `/root/.openclaw/agents/main/sessions/sessions.json` (NOT `/root/.openclaw/sessions.json`). After restore, OpenClaw will start fresh sessions — this is expected. Also, NEVER run `openclaw doctor --fix` as it wipes channel config, model routing, and plugin state.

---

## Server 5: labsliveserver (75.43.156.104)

### Overview

- **IP:** 75.43.156.104
- **OS:** Ubuntu 22.04.5 LTS
- **Role:** Live services host — 55 Docker containers, primary MySQL/PostgreSQL workloads
- **Key services:** 55 Docker containers (databases, web apps, APIs), MySQL, PostgreSQL
- **SSH:** Port 22, alias `server-104`
- **Note:** HA SSL cert has a server-side TLS issue — access via HTTP or direct IP

### What is Backed Up

**Daily files:**
- `/etc/crontab` — system crontab

**Daily commands:**
- `crontab-user.txt` — user crontab
- `systemd-custom-services.txt` — custom systemd service listings
- `mysql-dump.sql` — full MySQL dump (`mysqldump --all-databases`)
- `dotenv-files.txt` — paths to `.env` files under `/opt`

**Weekly extras:**
- `packages-dpkg.txt` — installed package list
- `docker-volumes.txt` — Docker volume listing
- `docker-images.txt` — Docker image listing
- `docker-volume-export.sh` — volume mountpoint map (allowlist: `_db_`, `_data_`, `postgres`, `mysql`, `redis`, `mongo` patterns)

**Note:** Docker volume exports use an allowlist of 6 patterns to select database and data volumes from the 55-container set. Not all volumes are exported weekly — only critical data volumes.

### Restore Steps

**Step 1: Base OS setup**

```bash
sudo apt-get update && sudo apt-get upgrade -y
# Restore packages from packages-dpkg.txt
sudo apt-get install -y docker.io docker-compose-plugin mysql-server postgresql \
  redis-server curl git python3 python3-pip
```

**Step 2: SSH setup**

```bash
mkdir -p ~/.ssh
echo "<alfred_104_public_key>" >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys
```

**Step 3: MySQL restoration**

```bash
sudo systemctl start mysql

# Restore from the mysql-dump.sql in daily backup:
sudo mysql < backup/mysql-dump.sql

# Verify:
sudo mysql -e "SHOW DATABASES;"
```

**Step 4: Docker volumes restoration (from weekly backup)**

```bash
# The docker-volume-export.sh in weekly backup lists volume names and mountpoints
# For each critical volume (matching allowlist patterns):

# Import a Docker volume from a tar export:
docker volume create <volume_name>
docker run --rm \
  -v <volume_name>:/volume \
  -v $(pwd)/backup:/backup \
  alpine \
  sh -c "tar -xzf /backup/<volume_name>.tar.gz -C /volume --strip-components=1"
```

**Step 5: Docker services restoration**

```bash
# With 55 containers, restoration priority:
# 1. Database containers first (MySQL, PostgreSQL, Redis, MongoDB)
# 2. Application containers second
# 3. Reverse proxy / Nginx last

# From dotenv-files.txt, find all service directories under /opt
# Restore each service:
for service_dir in /opt/*/; do
  if [ -f "$service_dir/docker-compose.yml" ]; then
    cd "$service_dir"
    docker compose up -d
  fi
done
```

**Step 6: Crontab restoration**

```bash
cat backup/crontab-user.txt | crontab -
sudo cp backup/crontab /etc/crontab
```

### Post-Restore Validation

```bash
# MySQL running and databases present:
sudo mysql -e "SHOW DATABASES;" | wc -l

# PostgreSQL running:
sudo systemctl status postgresql

# Docker container count (expect ~55):
docker ps | wc -l

# Check key containers:
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "Up|Exited"
```

**Key gotcha:** With 55 containers, bring them up in dependency order (databases first). Volume allowlist patterns are: `_db_`, `_data_`, `postgres`, `mysql`, `redis`, `mongo`. Containers NOT matching the allowlist have their volumes excluded from weekly backup — reconfiguration required for those.

---

## Server 6: lonewolf / labs-R820 (75.43.156.117)

### Overview

- **IP:** 75.43.156.117
- **OS:** Ubuntu 22.04.5 LTS
- **Role:** Dokploy-managed infrastructure — 24 Docker containers, Traefik reverse proxy, Twenty CRM (v1.14.0), LightRAG knowledge graph
- **Key services:** Dokploy, Traefik, Twenty CRM (PostgreSQL), LightRAG, various managed apps at crm.groundrushlabs.com
- **SSH:** Port 22, user `brucewayne9`, alias `lonewolf`
- **Docker:** 24 containers

### What is Backed Up

**Daily files:**
- `/etc/crontab` — system crontab

**Daily commands:**
- `crontab-user.txt` — user crontab
- `systemd-custom-services.txt` — custom systemd service listings
- `dokploy-config.txt` — paths to Dokploy JSON/YAML config files
- `traefik-config.txt` — paths to Traefik configuration files

**Weekly extras:**
- `packages-dpkg.txt` — installed package list
- `docker-volumes.txt` — Docker volume listing
- `docker-images.txt` — Docker image listing
- `dokploy-app-data.txt` — Dokploy app configuration JSON files
- `docker-volume-export.sh` — volume mountpoint map

### Restore Steps

**Step 1: Base OS setup**

```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y docker.io docker-compose-plugin curl git
```

**Step 2: SSH setup**

```bash
# User is brucewayne9 (same pattern as alfred-claw)
sudo adduser brucewayne9
sudo usermod -aG sudo,docker brucewayne9
sudo -u brucewayne9 mkdir -p /home/brucewayne9/.ssh
echo "<alfred_lonewolf_public_key>" >> /home/brucewayne9/.ssh/authorized_keys
sudo chmod 700 /home/brucewayne9/.ssh
sudo chmod 600 /home/brucewayne9/.ssh/authorized_keys
```

**Step 3: Dokploy installation**

```bash
# Dokploy is the primary management layer — install it first
curl -sSL https://dokploy.com/install.sh | sh

# Dokploy installs Traefik and its own infrastructure automatically
# After install, Dokploy is accessible at http://<server-ip>:3000
```

**Step 4: Restore Dokploy configuration**

```bash
# From dokploy-config.txt in backup, identify the config file paths
# Common Dokploy config locations:
#   /opt/dokploy/
#   /var/lib/dokploy/

# Restore Dokploy app definitions (from dokploy-app-data.txt in weekly backup):
# Each app has a JSON config — restore these to the Dokploy data directory
sudo cp backup-extracted/*.json /opt/dokploy/

# Restart Dokploy to pick up restored configs:
docker compose -f /opt/dokploy/docker-compose.yml restart
```

**Step 5: Restore Traefik configuration**

```bash
# From traefik-config.txt in backup, identify Traefik config file paths
# Common locations: /etc/traefik/, /opt/traefik/

# Restore Traefik config files:
sudo cp backup-extracted/traefik.yml /etc/traefik/traefik.yml
sudo cp backup-extracted/dynamic/ /etc/traefik/dynamic/

# Traefik restarts automatically via Dokploy
```

**Step 6: Twenty CRM (PostgreSQL) restoration**

```bash
# Twenty CRM uses PostgreSQL (managed by Dokploy)
# Restore from docker-volume-export.sh output — find the Twenty CRM volume

# If you have a volume backup tar:
docker volume create twenty_crm_db
docker run --rm \
  -v twenty_crm_db:/volume \
  -v $(pwd)/backup:/backup \
  alpine \
  sh -c "tar -xzf /backup/twenty_crm_db.tar.gz -C /volume --strip-components=1"

# Then restart Twenty via Dokploy UI or:
docker compose -f /opt/dokploy/docker-compose.yml up -d
```

**Step 7: Crontab restoration**

```bash
cat backup/crontab-user.txt | crontab -
sudo cp backup/crontab /etc/crontab
```

### Post-Restore Validation

```bash
# Dokploy UI accessible:
curl -I http://localhost:3000/

# Traefik dashboard (if enabled):
curl -I http://localhost:8080/dashboard/

# Twenty CRM accessible:
curl -I https://crm.groundrushlabs.com/

# Docker containers (expect ~24):
docker ps | wc -l

# LightRAG embedding service:
curl http://localhost:11434/api/tags  # Ollama API
```

**Key gotcha:** Twenty CRM v1.14.0 auto-registers cron jobs via its worker container. After restore, let the worker run for a few minutes before verifying CRM functionality. The `aix.groundrushlabs.com` hostname resolves to this server's IP — update DNS if IP changes.

---

## Server 7: cloud-mail (75.43.156.121)

### Overview

- **IP:** 75.43.156.121
- **OS:** Ubuntu 24.04.3 LTS
- **Role:** Mail server — Mailu mail stack, Postfix, Dovecot, 20 Docker containers
- **Key services:** Mailu (Docker-based mail suite), Postfix (SMTP), Dovecot (IMAP), Nginx, Redis
- **SSH:** Port 22, alias `server-121`
- **Docker:** 20 containers
- **Handles mail for:** groundrushlabs.com, doowoprnb.com (SMTP auth: lumabot@groundrushlabs.com)

### What is Backed Up

**Daily files:**
- `/etc/crontab` — system crontab

**Daily commands:**
- `crontab-user.txt` — user crontab
- `systemd-custom-services.txt` — custom systemd service listings
- `mail-config.txt` — paths to Postfix/Dovecot/Exim/Mailu config files
- `dotenv-files.txt` — paths to `.env` files under `/opt`, `/var/www`, `/home`

**Weekly extras:**
- `packages-dpkg.txt` — installed package list
- `docker-volumes.txt` — Docker volume listing
- `docker-images.txt` — Docker image listing
- `docker-volume-export.sh` — volume mountpoint map

### Restore Steps

**Step 1: Base OS setup**

```bash
# Ubuntu 24.04 LTS
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y docker.io docker-compose-plugin curl git \
  postfix dovecot-core dovecot-imapd
```

**Step 2: SSH setup**

```bash
mkdir -p ~/.ssh
echo "<alfred_mail_public_key>" >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys
```

**Step 3: Mailu installation**

```bash
# Mailu is Docker-based — install via official Mailu setup
mkdir -p /opt/mailu
cd /opt/mailu

# Generate docker-compose.yml from Mailu setup wizard or restore from backup:
# From the mail-config.txt and dotenv-files.txt in backup, find the .env path
# Common Mailu location: /opt/mailu/mailu.env

# Restore the Mailu environment file:
cp backup-extracted/<mailu-env-file> /opt/mailu/mailu.env

# Restore docker-compose.yml for Mailu:
cp backup-extracted/docker-compose.yml /opt/mailu/docker-compose.yml
```

**Step 4: Restore mail data volumes**

```bash
# Mailu stores mail data in Docker volumes
# From docker-volume-export.sh output (weekly backup), find Mailu volumes

# Restore mail storage volume:
docker volume create mailu_mail_data
docker run --rm \
  -v mailu_mail_data:/volume \
  -v $(pwd)/backup:/backup \
  alpine \
  sh -c "tar -xzf /backup/mailu_mail_data.tar.gz -C /volume --strip-components=1"

# Restore other Mailu volumes (certs, dkim, data):
for vol in mailu_certs mailu_data mailu_dkim mailu_filter mailu_overrides; do
  docker volume create $vol
  docker run --rm -v $vol:/volume -v $(pwd)/backup:/backup alpine \
    sh -c "tar -xzf /backup/${vol}.tar.gz -C /volume --strip-components=1 2>/dev/null || echo 'No backup for $vol'"
done
```

**Step 5: Start Mailu**

```bash
cd /opt/mailu
docker compose up -d

# Verify all Mailu containers started:
docker compose ps
```

**Step 6: Postfix/Dovecot config restoration (if standalone)**

```bash
# If using standalone Postfix/Dovecot (not Docker):
# From mail-config.txt in backup, identify config file paths:
# /etc/postfix/main.cf, /etc/postfix/master.cf
# /etc/dovecot/dovecot.conf

sudo cp backup-extracted/main.cf /etc/postfix/main.cf
sudo cp backup-extracted/master.cf /etc/postfix/master.cf
sudo cp backup-extracted/dovecot.conf /etc/dovecot/dovecot.conf

sudo systemctl restart postfix dovecot
```

**Step 7: DNS verification**

```bash
# Ensure DNS MX records still point to this IP
# DKIM keys should be in the restored mailu_dkim volume

# Test email delivery:
echo "Test restore" | mail -s "Restore Test" mike@groundrushlabs.com

# Check Postfix logs:
sudo tail -f /var/log/mail.log
```

**Step 8: Crontab restoration**

```bash
cat backup/crontab-user.txt | crontab -
sudo cp backup/crontab /etc/crontab
```

### Post-Restore Validation

```bash
# Mailu web UI accessible:
curl -I https://mail.groundrushlabs.com/

# SMTP port open:
nc -zv localhost 25
nc -zv localhost 465
nc -zv localhost 587

# IMAP port open:
nc -zv localhost 143
nc -zv localhost 993

# Docker containers (expect ~20):
docker ps | wc -l

# Send a test email through the system:
swaks --to mike@groundrushlabs.com \
  --from lumabot@groundrushlabs.com \
  --server localhost:587 \
  --auth LOGIN \
  --auth-user lumabot@groundrushlabs.com \
  --auth-password <EMAIL_PASS_LUMABOT>
```

**Key gotcha:** SMTP auth uses `lumabot@groundrushlabs.com` with `EMAIL_PASS_LUMABOT` from the `.env` file. The `alfred@groundrushlabs.com` account has no password set — never use it as SMTP auth sender (it's display-only). SMTP server: `mail.doowoprnb.com` port 465 (SSL). DKIM keys are critical for deliverability — ensure the `mailu_dkim` volume is correctly restored before sending.

---

## Quick Reference

| Server | IP | Critical Restore Order | Est. Time | Key Gotcha |
|--------|-----|------------------------|-----------|------------|
| alfred-labs | 75.43.156.105 | OS → PostgreSQL → Redis → Alfred app → Docker containers → Cron | 60-90 min | Backup orchestrator — restore this first; SQLite DBs (conversations.db, learning.db) are separate from PostgreSQL |
| groundrush-radio | 75.43.156.98 | OS → AzuraCast Docker install → AzuraCast restore | 45-60 min | AzuraCast has its own backup system; Alfred only backs up config paths, not MySQL data directly |
| labs-edge | 75.43.156.100 | OS → MySQL → Docker containers | 30-45 min | 8 containers; restore DB before app containers to avoid connection errors |
| alfred-claw | 75.43.156.101 | OS → SSH on port 2222 → OpenClaw install → Workspace restore → Services | 30-45 min | SSH is port 2222, NOT 22; NEVER run `openclaw doctor --fix`; no Docker on this server |
| labsliveserver | 75.43.156.104 | OS → MySQL → DB containers → App containers → Reverse proxy | 90-120 min | 55 containers — use volume allowlist for restoration priority; bring DB containers up first |
| lonewolf | 75.43.156.117 | OS → Dokploy install → Traefik → Twenty CRM PostgreSQL → Apps | 60-90 min | Dokploy manages Traefik; let Twenty CRM worker run after restore to re-register cron jobs |
| cloud-mail | 75.43.156.121 | OS → Mailu volumes restore → Mailu Docker stack → DNS verify | 45-60 min | DKIM keys must be restored before sending; SMTP auth is lumabot@, not alfred@ |

---

## Backup Folder Structure Reference

```
Google Drive/
└── Alfred Backups/
    ├── alfred-labs/
    │   ├── daily/
    │   │   └── YYYY-MM-DD/
    │   │       └── alfred-labs-YYYY-MM-DD.tar.gz
    │   └── weekly/
    │       └── YYYY-MM-DD/
    │           └── alfred-labs-YYYY-MM-DD.tar.gz
    ├── groundrush-radio/
    │   ├── daily/ ...
    │   └── weekly/ ...
    ├── labs-edge/
    │   ├── daily/ ...
    │   └── weekly/ ...
    ├── alfred-claw/
    │   ├── daily/ ...
    │   └── weekly/ ...
    ├── labsliveserver/
    │   ├── daily/ ...
    │   └── weekly/ ...
    ├── lonewolf/
    │   ├── daily/ ...
    │   └── weekly/ ...
    └── cloud-mail/
        ├── daily/ ...
        └── weekly/ ...
```

**Backup retention:** 30 days — the most recent 30 backup run folders are kept per server per type. Older folders are automatically pruned by the weekly backup script.

---

## Common Restore Commands Reference

```bash
# PostgreSQL — restore from pg_dumpall output:
sudo -u postgres psql < postgresql-dump.sql

# MySQL — restore from mysqldump output:
sudo mysql < mysql-dump.sql

# SQLite — direct file copy:
cp backup/conversations.db /target/path/conversations.db

# Docker volume — import from tar:
docker volume create <volume_name>
docker run --rm \
  -v <volume_name>:/volume \
  -v $(pwd)/backup:/backup \
  alpine \
  tar -xzf /backup/<volume_name>.tar.gz -C /volume --strip-components=1

# Crontab — restore user crontab:
cat crontab-user.txt | crontab -

# Crontab — restore system crontab:
sudo cp crontab /etc/crontab

# SSH authorized_keys:
mkdir -p ~/.ssh && cat backup-key.pub >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys
```
