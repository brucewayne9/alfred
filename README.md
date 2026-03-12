# Alfred — Personal AI Command Center

Alfred is a self-hosted, multi-surface AI assistant built for personal and business operations management. It combines a FastAPI backend, a React web interface, a Telegram bot (Alfred Claw), and 30+ tool integrations into a unified command center that handles email, CRM, calendar, image generation, voice interaction, social media scheduling, server management, and automated workflows — all running on a single GPU workstation.

---

## Surfaces

Alfred operates across three surfaces that share the same backend infrastructure:

| Surface | Interface | Primary Use |
|---------|-----------|-------------|
| **Alfred Labs** | React web app at `aialfred.groundrushcloud.com` | Web chat, voice, document analysis, dashboard |
| **Alfred Claw** | Telegram bot (`@alfredblogbot`) via OpenClaw | Day-to-day operations, mobile-first, 36 integration scripts |
| **Claude Code CLI** | Terminal on 105 | Development, infrastructure, deep analysis |

All three run on the same machine (105: i9-12900K, 96GB RAM, RTX 4070 12GB, 1.8TB NVMe) and share GPU services (ComfyUI, Kokoro TTS, Ollama).

---

## Architecture

```
                              ┌──────────────────────┐
                              │    Reverse Proxy      │
                              │  (aialfred.ground     │
                              │   rushcloud.com)       │
                              └──────────┬───────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              │                          │                          │
    ┌─────────▼─────────┐    ┌──────────▼──────────┐    ┌─────────▼─────────┐
    │   React Frontend   │    │   Telegram Gateway   │    │  Claude Code CLI  │
    │  (Vite + Tailwind)  │    │     (OpenClaw)        │    │   (Terminal)      │
    │    Port 5173 dev    │    │    Port 18789         │    │                   │
    └─────────┬─────────┘    └──────────┬──────────┘    └─────────┬─────────┘
              │                          │                          │
    ┌─────────▼──────────────────────────▼──────────────────────────▼─────────┐
    │                          FastAPI Backend                                │
    │                       core/api/main.py (Port 8400)                     │
    ├─────────────────┬──────────────────┬──────────────────┬────────────────┤
    │  Authentication  │   Brain/Router    │   Tool Registry   │ Notifications  │
    │  JWT + Passkeys  │  LLM Tier System  │  30+ Tools        │ WebSocket+Push │
    │  TOTP 2FA        │  local→cloud→cli  │  Auto-routing     │ VAPID Push     │
    ├─────────────────┼──────────────────┼──────────────────┼────────────────┤
    │    Memory        │   Voice Engine    │   Integrations    │  Scheduling    │
    │  ChromaDB        │  Whisper STT      │  Gmail, Calendar  │  Cron scripts  │
    │  SQLite History  │  Kokoro TTS       │  CRM, Stripe      │  Email monitor │
    │  LightRAG (117)  │  Wake Word        │  ComfyUI, Postiz  │  Morning brief │
    └─────────────────┴──────────────────┴──────────────────┴────────────────┘
              │                          │                          │
    ┌─────────▼─────────┐    ┌──────────▼──────────┐    ┌─────────▼─────────┐
    │     Ollama         │    │      ComfyUI         │    │    Kokoro TTS     │
    │  LLM inference     │    │  Image generation    │    │   Voice synthesis  │
    │  Port 11434        │    │  Port 8188           │    │   Port 8880       │
    └───────────────────┘    └─────────────────────┘    └───────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | FastAPI (Python 3.11), Uvicorn, async |
| **Frontend** | React 18, Vite, TypeScript, Tailwind CSS, Zustand |
| **Primary LLM** | gpt-5.3-codex via OpenAI Codex OAuth (Claw) |
| **Fallback LLMs** | cogito-2.1:671b, nemotron-3-nano, deepseek-v3.2, qwen3-coder, minimax-m2, kimi-k2.5, glm-5, gpt-oss:120b (all via Ollama cloud) |
| **Local LLM** | Ollama (port 11434) |
| **Vector Memory** | ChromaDB with nomic-embed-text embeddings |
| **Knowledge Graph** | LightRAG on 117 (minimax-m2 via Ollama) |
| **Database** | SQLite (conversations), ChromaDB (memory) |
| **Speech-to-Text** | Faster-Whisper (CPU, tiny model) |
| **Text-to-Speech** | Kokoro TTS (GPU, port 8880, ~912MB VRAM) |
| **Image Generation** | ComfyUI (GPU, port 8188, --lowvram) |
| **Image Models** | Juggernaut XL, FLUX.1 dev FP8, SVD-XT video, ControlNet, IP-Adapter |
| **Authentication** | JWT cookies, Passkeys (WebAuthn), TOTP 2FA |
| **Push Notifications** | VAPID WebPush + WebSocket |
| **Telegram** | OpenClaw gateway (Node 22, systemd) |
| **Social Media** | Postiz API (self-hosted on 117) |
| **CRM** | Twenty CRM v1.14.0 (self-hosted on 117) |
| **Email** | IMAP/SMTP (8 accounts) + Gmail API (Google Workspace) |

---

## Features

### AI & LLM
- **Multi-tier LLM routing** — local (Ollama) → cloud models → Claude Code CLI for complex tasks
- **Tool calling** — 30+ tools the LLM can invoke autonomously (email, CRM, calendar, image gen, etc.)
- **Vector memory** — ChromaDB-powered recall across all conversations
- **Knowledge graph** — LightRAG (Grey Matter) for deep domain knowledge retrieval
- **Conversation history** — persistent SQLite storage with search
- **Multi-agent orchestration** — sub-agents for parallel task execution

### Voice
- **"Hey Alfred" wake word** — custom OpenWakeWord ONNX model
- **Speech-to-text** — Faster-Whisper (runs on CPU, i9 handles it)
- **Text-to-speech** — Kokoro TTS (British male voice, GPU) or Qwen3-TTS (voice cloning)
- **Hands-free mode** — Silero VAD for continuous voice conversation
- **Smart acknowledgments** — contextual responses ("Checking now..." for tasks, silence for greetings)
- **iOS/Mac Shortcuts** — voice selection API for mobile

### Image & Video Generation
- **Juggernaut XL** — photorealistic SDXL image generation
- **FLUX.1 dev FP8** — superior prompt following and text rendering
- **SVD-XT** — product showcase video generation from photos
- **ControlNet** — edge/depth-guided image generation from reference images
- **IP-Adapter** — style transfer from reference photos
- **4x upscaling** — ultra-high resolution output
- **GPU management** — automatic VRAM budgeting, service start/stop, idle cleanup

### Email (8 Accounts)
- **Multi-account IMAP/SMTP** — groundrushinc, groundrush, rucktalk, loovacast, lumabot, support, alfred, oracle
- **Gmail API** — Google Workspace integration for mjohnson@groundrushinc.com
- **Auto HTML detection** — emails with HTML tags automatically render as styled HTML
- **Email monitoring** — automated checking across all accounts every 10 minutes
- **Auto-reply/forward** — rule-based email automation

### CRM (Twenty)
- People, companies, opportunities, tasks, notes — full CRUD
- Pipeline management (NEW → SCREENING → MEETING → PROPOSAL → CUSTOMER)
- Cross-object search and linked records
- GraphQL API integration

### Social Media (Postiz)
- 11 connected integrations (Facebook, Instagram, YouTube, WordPress)
- Scheduled posting with platform-specific settings
- Brands: LoovaCast, Ruck Talk, My Hands Car Wash, Luma
- Auto-blog pipeline with SEO articles and social scheduling

### Calendar
- Google Calendar integration (view, create, modify, delete events)
- Free time finder
- Attendee management

### Server Management
- SSH-based monitoring across 7+ servers
- Health checks, disk usage, service status
- Remote command execution

### Web Interface (Alfred Labs)
- **Progressive Web App** — installable on mobile/desktop
- **React SPA** — Vite + TypeScript + Tailwind with Zustand stores
- **Real-time chat** — WebSocket streaming with thinking indicators
- **Conversation sidebar** — browse and switch chat histories
- **Document analysis** — upload PDF, Word, Excel, CSV, images
- **Voice controls** — push-to-talk, hands-free, auto-speak

### Automated Workflows
- **Morning brief** — 6:30 AM daily HTML newsletter (stocks, news, servers, tasks, CRM)
- **Email monitor** — checks 8 accounts every 10 minutes, auto-replies/forwards
- **Evening task ping** — 6 PM Telegram prompt for tomorrow's tasks
- **Hunter.io CRM sync** — lead stage sync every 15 minutes
- **Claw health monitor** — OpenClaw gateway monitoring every 15 minutes with auto-repair

---

## Project Structure

```
alfred/
├── config/
│   ├── settings.py              # Pydantic settings (90+ env vars)
│   ├── .env                     # Runtime environment variables
│   ├── .env.example             # Template with all keys
│   ├── servers.json             # SSH server definitions
│   └── users.json               # User credentials (bcrypt hashed)
│
├── core/
│   ├── api/
│   │   └── main.py              # FastAPI app (~7300 lines) — all endpoints + embedded fallback UI
│   ├── brain/
│   │   ├── router.py            # Multi-tier LLM routing (local → cloud → CLI)
│   │   └── models.py            # Model definitions and capabilities
│   ├── memory/
│   │   ├── store.py             # ChromaDB vector memory
│   │   └── conversations.py     # SQLite conversation persistence
│   ├── security/
│   │   ├── auth.py              # JWT, passkeys (WebAuthn), TOTP 2FA
│   │   └── google_oauth.py      # Google OAuth flows
│   ├── tools/
│   │   ├── registry.py          # Tool registration, routing, execution
│   │   ├── definitions.py       # 30+ tool implementations
│   │   └── files.py             # Document parsing (PDF, Word, Excel, CSV)
│   ├── notifications/
│   │   ├── manager.py           # WebSocket + VAPID push notifications
│   │   └── watcher.py           # Notification event watcher
│   ├── learning/
│   │   ├── patterns.py          # Usage pattern detection
│   │   ├── feedback.py          # User feedback processing
│   │   └── preferences.py       # Learned user preferences
│   ├── orchestration/
│   │   └── agents.py            # Multi-agent task orchestration
│   └── briefing/
│       └── daily.py             # Daily briefing generation
│
├── integrations/                 # 27 service integrations
│   ├── gmail/                    # Gmail API (OAuth, read/send/search)
│   ├── email/                    # Multi-account IMAP/SMTP (8 accounts)
│   ├── calendar/                 # Google Calendar API
│   ├── base_crm/                 # Twenty CRM (GraphQL)
│   ├── comfyui/                  # Image/video generation (ComfyUI HTTP API)
│   ├── gpu_manager.py            # VRAM budgeting, service lifecycle
│   ├── stripe/                   # Payments and invoicing
│   ├── meta_ads/                 # Facebook/Instagram ad management
│   ├── google_ads/               # Google Ads campaigns
│   ├── google_analytics/         # GA4 reporting
│   ├── azuracast/                # Radio station management (LoovaCast)
│   ├── homeassistant/            # Smart home control
│   ├── nextcloud/                # File storage
│   ├── firecrawl/                # Web scraping
│   ├── lightrag/                 # Knowledge graph (Grey Matter)
│   ├── twilio/                   # Voice calls and SMS
│   ├── wordpress/                # Blog publishing
│   ├── servers/                  # SSH server management
│   ├── n8n/                      # Workflow automation
│   ├── google_docs/              # Document creation
│   ├── google_sheets/            # Spreadsheet operations
│   ├── google_slides/            # Presentation generation
│   ├── google_drive/             # File management
│   ├── ad_intelligence/          # Cross-platform ad analytics
│   └── finance/                  # Banking and transactions
│
├── interfaces/
│   ├── voice/
│   │   ├── stt.py                # Faster-Whisper speech-to-text
│   │   ├── tts.py                # Kokoro/Qwen3 text-to-speech
│   │   └── wakeword.py           # "Hey Alfred" OpenWakeWord detection
│   ├── telegram/                  # Telegram bot interface
│   ├── whatsapp/                  # WhatsApp Web.js integration
│   └── mcp/                       # Model Context Protocol server
│
├── frontend/                      # React SPA
│   ├── src/
│   │   ├── components/
│   │   │   ├── chat/              # Chat messages, markdown rendering
│   │   │   ├── input/             # ChatInput with voice controls
│   │   │   ├── layout/            # Header, AppLayout, sidebar
│   │   │   ├── voice/             # Voice mode UI
│   │   │   ├── auth/              # Login, passkey, 2FA flows
│   │   │   ├── sidebar/           # Conversation list
│   │   │   ├── notifications/     # Push notification UI
│   │   │   └── knowledge/         # Knowledge base browser
│   │   ├── stores/                # Zustand state (auth, chat, voice, sidebar, notification, knowledge)
│   │   ├── api/                   # API client functions
│   │   ├── hooks/                 # Custom React hooks
│   │   └── lib/                   # Utilities
│   ├── package.json               # React 18, Vite, Tailwind, Zustand
│   ├── vite.config.ts             # Dev server proxy + PWA config
│   └── tsconfig.json
│
├── scripts/                       # Automated cron tasks
│   ├── morning_brief.py           # 6:30 AM daily newsletter
│   ├── alfred_email_monitor.py    # Email checking (every 10 min)
│   ├── evening_task_ping.py       # Task prompt (6 PM daily)
│   ├── hunter_crm_sync.py         # CRM lead sync (every 15 min)
│   └── alfred_claw_monitor.py     # OpenClaw health + escalation bridge
│
├── data/
│   ├── conversations.db           # SQLite chat history
│   ├── chromadb/                   # Vector memory store
│   ├── learning.db                # Usage pattern database
│   ├── uploads/                    # User-uploaded files
│   ├── generated/                  # Generated documents and images
│   ├── audio/                      # Voice recordings
│   └── static/                     # Static served files
│
├── models/
│   └── hey_alfred.onnx            # Custom wake word model
│
├── requirements.txt               # Python dependencies (~235 packages)
├── run.sh                         # Build frontend + start server
└── README.md
```

---

## Server Infrastructure

| ID | IP | Role | Specs |
|----|-----|------|-------|
| **105** | 75.43.156.105 | Alfred Labs + Claw — main dev/ops machine | i9-12900K, 96GB RAM, RTX 4070 12GB, 1.8TB NVMe |
| **117** | 75.43.156.117 | Dokploy, Twenty CRM, LightRAG, Postiz | Docker managed |
| **104** | 75.43.156.104 | Production websites, Home Assistant | |
| **098** | 75.43.156.98 | LoovaCast dev/staging | |
| **100** | 75.43.156.100 | LoovaCast production | |
| **121** | 75.43.156.121 | Mailcow — all email services | |
| **101** | 75.43.156.101 | Cold backup (decommissioned) | |

---

## GPU Services (RTX 4070 12GB)

| Service | Systemd Unit | Port | VRAM | Purpose |
|---------|-------------|------|------|---------|
| ComfyUI | `comfyui` | 8188 | ~154MB idle, 5-10GB active | Image/video generation |
| Kokoro TTS | `kokoro` | 8880 | ~912MB | Text-to-speech |
| Ollama | `ollama` | 11434 | ~786MB per model | LLM inference |
| Qwen3-TTS | `qwen3-tts-ui` | 7860 | CPU only | Voice cloning TTS |

GPU Manager (`integrations/gpu_manager.py`) handles VRAM budgeting, automatic start/stop based on demand, idle timeouts, and service eviction when VRAM is scarce.

---

## Requirements

### Hardware
- **GPU**: NVIDIA with 8GB+ VRAM (12GB recommended for ComfyUI + TTS + Ollama)
- **RAM**: 32GB+ (96GB recommended for large model offloading)
- **CPU**: Modern multi-core (i9/Ryzen 9 for CPU-only Whisper)
- **Storage**: 50GB+ for models, 20GB+ for data

### Software
- Python 3.11+
- Node.js 20+ (for frontend build and OpenClaw)
- CUDA 12.1+ (for GPU services)
- Ollama
- ComfyUI (optional, for image generation)

---

## Installation

### 1. Clone and Setup
```bash
git clone https://github.com/brucewayne9/alfred.git
cd alfred
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp config/.env.example config/.env
# Edit config/.env with your API keys and settings
```

Key environment variables:
```bash
# LLM
OLLAMA_MODEL=llama3.1:8b
OLLAMA_URL=http://127.0.0.1:11434
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-20250514

# Security
SECRET_KEY=your-secret-key
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# Voice
TTS_MODEL=kokoro
TTS_VOICE=bm_daniel
WHISPER_MODEL=tiny

# Google APIs
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...

# CRM
TWENTY_API_URL=https://crm.yourdomain.com
TWENTY_API_KEY=...

# Email (per account)
EMAIL_PASS_GROUNDRUSH=...
EMAIL_PASS_ALFRED=...
```

### 3. Build Frontend
```bash
cd frontend
npm install
npm run build
cd ..
```

### 4. Setup Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b
```

### 5. Setup ComfyUI (Optional)
```bash
cd ~
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
python -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
# Download models (Juggernaut XL, FLUX.1 dev FP8, etc.) into models/checkpoints/
```

### 6. Create Systemd Services

**Alfred** (`/etc/systemd/system/alfred.service`):
```ini
[Unit]
Description=Alfred AI Assistant
After=network.target

[Service]
Type=simple
User=aialfred
WorkingDirectory=/home/aialfred/alfred
ExecStart=/home/aialfred/alfred/venv/bin/python -m uvicorn core.api.main:app --host 0.0.0.0 --port 8400
Restart=on-failure
RestartSec=10
Environment=PYTHONUNBUFFERED=1
Environment=CUDA_VISIBLE_DEVICES=0

[Install]
WantedBy=multi-user.target
```

**ComfyUI** (`/etc/systemd/system/comfyui.service`):
```ini
[Unit]
Description=ComfyUI Image Generation Server
After=network.target

[Service]
Type=simple
User=aialfred
WorkingDirectory=/home/aialfred/ComfyUI
ExecStart=/home/aialfred/ComfyUI/venv/bin/python main.py --listen 127.0.0.1 --port 8188 --lowvram --preview-method none --disable-auto-launch
Restart=on-failure
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

**Kokoro TTS** (`/etc/systemd/system/kokoro.service`):
```ini
[Unit]
Description=Kokoro TTS Server
After=network.target

[Service]
Type=simple
User=aialfred
WorkingDirectory=/home/aialfred/kokoro
ExecStart=/home/aialfred/kokoro/venv/bin/python server.py
Restart=on-failure
Environment=USE_GPU=true

[Install]
WantedBy=multi-user.target
```

### 7. Start Everything
```bash
sudo systemctl daemon-reload
sudo systemctl enable alfred comfyui kokoro ollama
sudo systemctl start alfred comfyui kokoro ollama
```

---

## Usage

### Web Interface
Access Alfred Labs at `http://your-server:8400` (or via reverse proxy).

### Telegram (Alfred Claw)
Message `@alfredblogbot` on Telegram. Claw has 36 standalone integration scripts and handles day-to-day operations including email, CRM, calendar, image generation, social media scheduling, and server management.

### Voice Modes

| Mode | How | Description |
|------|-----|-------------|
| **Wake Word** | Say "Hey Alfred" | Always-on listening, activates hands-free |
| **Hands-Free** | Toggle in UI | Continuous conversation via Silero VAD |
| **Push-to-Talk** | Click mic button | Manual voice recording |
| **Auto-Speak** | Toggle in UI | Alfred reads all responses aloud |

Exit hands-free: say "That will be all, Alfred" or "Goodbye Alfred"

### Example Queries

```
"Check my email"
"Send an email to john@example.com about the project update"
"What's on my calendar today?"
"Schedule a meeting with Sarah tomorrow at 2pm"
"Show me the CRM pipeline"
"Create a contact for Jane Smith at Acme Corp"
"Generate a photorealistic image of a sunset over mountains"
"Check the status of the production server"
"How much disk space is left on lonewolf?"
"Schedule an Instagram post for LoovaCast tomorrow at 4pm"
"What are my unread emails across all accounts?"
```

---

## API Endpoints

### Chat
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat` | Send message (with optional voice/tts params) |
| POST | `/chat/stream` | Stream response via SSE |
| WS | `/ws/chat` | WebSocket real-time chat |

### Conversations
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/conversations` | List all conversations |
| POST | `/conversations` | Create new conversation |
| GET | `/conversations/{id}` | Get conversation history |
| DELETE | `/conversations/{id}` | Delete conversation |

### Voice
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/voice/transcribe` | Transcribe audio file |
| POST | `/voice/speak` | Generate speech from text |
| GET | `/voice/voices` | List available TTS voices |
| POST | `/voice/chat/ack` | Get contextual acknowledgment audio |
| WS | `/ws/wakeword` | Wake word detection stream |

### Files
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload/document` | Upload document for analysis |
| POST | `/upload/image` | Upload image for vision |
| GET | `/download/{filename}` | Download generated file |
| GET | `/media/{filename}` | Serve generated images |

### Auth
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/login` | Username/password login |
| POST | `/auth/auto` | IP-based auto-login (local network) |
| POST | `/auth/passkey/register` | Register passkey (WebAuthn) |
| POST | `/auth/passkey/login` | Login with passkey |
| POST | `/auth/totp/setup` | Setup TOTP 2FA |
| POST | `/auth/totp/verify` | Verify TOTP code |

### Memory
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/memory/store` | Store a memory |
| POST | `/memory/recall` | Recall relevant memories |

### System
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/status` | System status (GPU, services) |

---

## Security

- **Authentication**: JWT tokens in HTTP-only cookies (no localStorage)
- **Passkeys**: WebAuthn/FIDO2 passwordless login
- **2FA**: TOTP-based two-factor authentication
- **Rate Limiting**: SlowAPI (30 req/min per endpoint)
- **CORS**: Restricted to configured domains
- **Passwords**: bcrypt hashed
- **File Validation**: Strict filename sanitization, size limits
- **Ollama**: Bound to localhost only
- **Email**: HTML auto-detection prevents raw tag rendering

---

## Cron Schedule

| Script | Schedule | Purpose |
|--------|----------|---------|
| `morning_brief.py` | 6:30 AM daily | HTML newsletter (stocks, news, servers, CRM, tasks) |
| `alfred_email_monitor.py` | Every 10 min | Check 8 email accounts, auto-reply/forward |
| `evening_task_ping.py ping` | 6 PM daily | Ask Mike for tomorrow's tasks via Telegram |
| `evening_task_ping.py check` | Every 5 min 6-11 PM | Check for Mike's task reply |
| `hunter_crm_sync.py` | Every 15 min | Sync Hunter.io leads to CRM |
| `alfred_claw_monitor.py` | Every 15 min | OpenClaw health check + escalation bridge |

---

## Troubleshooting

### Alfred won't start
```bash
sudo journalctl -u alfred -f          # Check logs
sudo lsof -i :8400                     # Check port conflicts
source venv/bin/activate && python -c "from core.api.main import app"  # Test imports
```

### Image generation fails
```bash
sudo systemctl status comfyui         # Check ComfyUI service
nvidia-smi                             # Check GPU/VRAM
curl http://127.0.0.1:8188/system_stats  # Check ComfyUI health
```

### Voice not working
```bash
python -c "import torch; print(torch.cuda.is_available())"  # CUDA check
curl http://localhost:8880/v1/audio/speech -X POST            # Kokoro health
sudo systemctl status kokoro                                   # TTS service
```

### Emails sending as raw HTML
This was fixed with auto-detection. If it recurs, check that `email_client.py` has the HTML auto-detect block in `send()` and `send_with_cc()`.

### ComfyUI connection refused
```bash
sudo systemctl start comfyui           # Start if stopped
sudo systemctl enable comfyui          # Enable auto-start
```

### ChromaDB issues
```bash
ls data/chromadb/                      # Check DB exists
# If corrupted, backup and reinitialize:
mv data/chromadb data/chromadb.bak
sudo systemctl restart alfred
```

---

## License

Private — All rights reserved.

---

## Acknowledgments

- [Anthropic](https://anthropic.com) — Claude API
- [Ollama](https://ollama.com) — Local LLM runtime
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — Image/video generation
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) — Speech recognition
- [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M) — Text-to-speech
- [Qwen3-TTS](https://github.com/QwenLM/Qwen2.5-Coder) — Voice cloning TTS
- [OpenWakeWord](https://github.com/dscripka/openWakeWord) — Wake word detection
- [Silero VAD](https://github.com/snakers4/silero-vad) — Voice activity detection
- [Twenty CRM](https://twenty.com) — Open-source CRM
- [OpenClaw](https://openclaw.dev) — Telegram AI gateway
- [LightRAG](https://github.com/HKUDS/LightRAG) — Knowledge graph RAG
- [Postiz](https://postiz.com) — Social media scheduling
