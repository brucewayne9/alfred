# Alfred - Personal AI Assistant

Alfred is a sophisticated, self-hosted AI assistant designed for personal and business management. Built with a multi-tier AI architecture, voice interaction capabilities, and deep integration with business tools.

## Features

### Core AI Capabilities
- **Multi-Tier LLM Routing** - Automatically routes queries between local Ollama models (fast, private) and Claude API (complex tasks)
- **Tool Calling** - LLM can invoke 30+ integrated tools for email, calendar, CRM, servers, and more
- **Memory System** - ChromaDB-powered vector memory for context recall across conversations
- **Conversation History** - SQLite-backed persistent conversation storage

### Voice Interaction
- **"Hey Alfred" Wake Word** - Custom OpenWakeWord model for hands-free activation
- **Speech-to-Text** - Whisper-based transcription (runs on GPU)
- **Text-to-Speech** - Kokoro TTS (British male voice) or Qwen3-TTS (voice cloning)
- **Hands-Free Mode** - Silero VAD for continuous voice conversation
- **Auto-Speak** - Automatically reads Alfred's responses aloud
- **Smart Acknowledgments** - Contextual responses ("Checking now..." for tasks, silence for greetings)
- **iOS/Mac Shortcuts** - Voice selection API for mobile integration

### Image & Document Handling
- **Image Vision** - Upload images and ask questions (Claude Vision API)
- **Image Generation** - SDXL Turbo via local ComfyUI (RTX 4070)
- **Document Analysis** - Parse PDF, Word, Excel, CSV, TXT, Markdown, JSON
- **Document Creation** - Generate documents in multiple formats for download

### Integrations
- **Gmail** - Read, send, search emails
- **Google Calendar** - View, create, modify events
- **Twenty CRM** - Full CRM access with Alfred role permissions:
  - People/Contacts (CRUD, search, link to companies)
  - Companies (CRUD, search)
  - Opportunities/Deals (CRUD, pipeline summary)
  - Tasks (CRUD, link to people/companies/opportunities)
  - Notes (CRUD, link to records)
  - Activities/Timeline
  - Metadata/Schema discovery
  - Cross-object search
- **Server Management** - SSH-based server monitoring and control
- **Finance** - Bank account and transaction tracking
- **AzuraCast Radio** - Full radio station control:
  - Now playing info (current song, listeners, DJ status)
  - Song history and upcoming queue
  - Playlist management (list, toggle, reshuffle)
  - Media library search
  - Listener statistics
  - Station restart

### Web Interface
- **Progressive Web App (PWA)** - Installable on mobile/desktop
- **Mobile Responsive** - Optimized for all screen sizes
- **Real-time Thinking Indicator** - Morphing shape animation while processing
- **Conversation Sidebar** - Browse and switch between chat histories

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Web Interface                            │
│                    (FastAPI + Inline HTML/JS)                    │
├─────────────────────────────────────────────────────────────────┤
│                         Core API Layer                           │
│              Authentication │ Rate Limiting │ CORS               │
├──────────────────┬──────────────────┬───────────────────────────┤
│   Brain/Router   │   Voice Engine   │      Tool Registry        │
│  (LLM Routing)   │  (STT/TTS/VAD)   │   (30+ Integrations)      │
├──────────────────┼──────────────────┼───────────────────────────┤
│     Ollama       │     Whisper      │        Gmail              │
│  (Local LLM)     │   (Local STT)    │       Calendar            │
├──────────────────┼──────────────────┼───────────────────────────┤
│   Claude API     │     Kokoro       │      Twenty CRM           │
│  (Cloud LLM)     │   (Local TTS)    │       Servers             │
├──────────────────┴──────────────────┼───────────────────────────┤
│              Memory Layer           │       ComfyUI             │
│    (ChromaDB + SQLite History)      │   (Image Generation)      │
└─────────────────────────────────────┴───────────────────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI (Python 3.11) |
| Local LLM | Ollama |
| Cloud LLM | Claude API (Anthropic) |
| Speech-to-Text | Faster-Whisper (CUDA) |
| Text-to-Speech | Kokoro TTS / Qwen3-TTS |
| Wake Word | OpenWakeWord (custom "Hey Alfred" model) |
| Voice Activity | Silero VAD (browser-based) |
| Vector Memory | ChromaDB |
| Database | SQLite |
| Image Generation | ComfyUI + SDXL Turbo |
| Authentication | JWT + bcrypt |
| Rate Limiting | SlowAPI |

## Requirements

### Hardware
- **GPU**: NVIDIA GPU with 8GB+ VRAM (for Whisper, TTS, and image generation)
- **RAM**: 16GB+ recommended
- **Storage**: 20GB+ for models

### Software
- Python 3.11+
- CUDA 12.1+
- Node.js 20+ (for Claude Code CLI, optional)
- Ollama

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/brucewayne9/alfred.git
cd alfred
```

### 2. Create Virtual Environment
```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
cp config/.env.example config/.env
# Edit config/.env with your API keys
```

### 5. Set Up Ollama
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3.1:8b
```

### 6. Set Up ComfyUI (Optional - for image generation)
```bash
cd ~
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
python -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Download SDXL Turbo model
wget -P models/checkpoints/ "https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sd_xl_turbo_1.0_fp16.safetensors"
```

### 7. Create Systemd Services

**Alfred Service** (`/etc/systemd/system/alfred.service`):
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
Environment="CUDA_VISIBLE_DEVICES=0"

[Install]
WantedBy=multi-user.target
```

**ComfyUI Service** (`/etc/systemd/system/comfyui.service`):
```ini
[Unit]
Description=ComfyUI Image Generation
After=network.target

[Service]
Type=simple
User=aialfred
WorkingDirectory=/home/aialfred/ComfyUI
ExecStart=/home/aialfred/ComfyUI/venv/bin/python main.py --listen 127.0.0.1 --port 8188
Restart=on-failure
Environment="CUDA_VISIBLE_DEVICES=0"

[Install]
WantedBy=multi-user.target
```

### 8. Start Services
```bash
sudo systemctl daemon-reload
sudo systemctl enable alfred comfyui
sudo systemctl start alfred comfyui
```

## Configuration

### Environment Variables (`config/.env`)

```bash
# LLM Configuration
OLLAMA_MODEL=llama3.1:8b
OLLAMA_URL=http://127.0.0.1:11434
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-20250514

# Security
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# Voice/TTS Configuration
TTS_MODEL=kokoro              # Options: kokoro, qwen3
TTS_VOICE=bm_daniel           # Kokoro: bm_daniel, af_sarah, etc.
WHISPER_MODEL=tiny            # Options: tiny, base, small, medium, large

# Qwen3-TTS (optional - for voice cloning)
QWEN3_TTS_URL=http://localhost:7860

# Google APIs (Gmail, Calendar)
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...

# Twenty CRM
TWENTY_API_URL=https://api.twenty.com
TWENTY_API_KEY=...
```

### Server Configuration (`config/servers.json`)

```json
{
  "server-name": {
    "host": "192.168.1.100",
    "username": "admin",
    "port": 22,
    "key_path": "/home/aialfred/.ssh/id_ed25519",
    "description": "Production Server"
  }
}
```

## Usage

### Web Interface
Access Alfred at `http://your-server:8400`

### Voice Modes

**"Hey Alfred" Wake Word** (Recommended)
- Always listening for the wake phrase "Hey Alfred"
- When detected, automatically activates hands-free mode
- Low resource usage - runs via WebSocket streaming
- Custom trained OpenWakeWord model

**Hands-Free Mode**
- Continuous voice conversation without button presses
- Uses Silero VAD to detect when you start/stop speaking
- Say "That will be all, Alfred" or "Goodbye Alfred" to exit
- Smart acknowledgments play for task queries, not greetings

**Push-to-Talk**
- Click the microphone button to speak
- Click again or wait for auto-stop

**Auto-Speak**
- When enabled, Alfred reads all responses aloud
- Uses configured TTS backend (Kokoro or Qwen3-TTS)

### Example Queries

**Email:**
- "Check my email"
- "Send an email to john@example.com about the meeting"
- "How many unread emails do I have?"

**Calendar:**
- "What's on my calendar today?"
- "Schedule a meeting with Mike tomorrow at 2pm"
- "Show me next week's schedule"

**CRM:**
- "Show me recent opportunities"
- "Create a new contact for Jane Smith at Acme Corp"
- "What tasks are due this week?"

**Servers:**
- "Check the status of production server"
- "What's the disk usage on lonewolf?"
- "Restart the web service on groundrush"

**Documents:**
- Upload a PDF and ask "What is this document about?"
- "Create a summary report as a PDF"
- "Analyze this spreadsheet"

**Images:**
- Upload an image and ask "What's in this picture?"
- "Generate an image of a sunset over mountains"
- "Create a picture of a golden retriever in snow"

### iOS/Mac Shortcuts Integration

Alfred's API supports voice selection for Shortcuts:

```
POST /chat
{
  "message": "What's on my calendar today?",
  "voice": "Gwen_Stacy",        // Optional: specific voice
  "tts_backend": "qwen3"        // Optional: kokoro or qwen3
}
```

**Available Voices:**
- **Kokoro:** bm_daniel (British male), af_sarah, bf_emma, am_adam, etc.
- **Qwen3:** Cloned voices (Gwen_Stacy, etc.) or designed voices (design:Natalie)

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat` | Send a message to Alfred |
| POST | `/chat/stream` | Stream a response |
| GET | `/conversations` | List conversations |
| POST | `/conversations` | Create new conversation |
| GET | `/conversations/{id}` | Get conversation history |
| DELETE | `/conversations/{id}` | Delete conversation |
| POST | `/voice/transcribe` | Transcribe audio |
| POST | `/voice/speak` | Generate speech |
| GET | `/voice/voices` | List available TTS voices |
| POST | `/voice/chat/ack` | Get contextual acknowledgment audio |
| WS | `/ws/wakeword` | Wake word detection stream |
| POST | `/upload/document` | Upload document for analysis |
| POST | `/upload/image` | Upload image for vision |
| GET | `/download/{filename}` | Download generated file |
| POST | `/memory/store` | Store a memory |
| POST | `/memory/recall` | Recall memories |
| POST | `/auth/login` | Login |
| GET | `/health` | Health check |

## Security

- **Authentication**: JWT-based with HTTP-only cookies
- **Rate Limiting**: 30 requests/minute per endpoint
- **CORS**: Restricted to configured domain
- **Ollama**: Bound to localhost only
- **Passwords**: bcrypt hashed
- **File Validation**: Strict filename sanitization

## Project Structure

```
alfred/
├── config/
│   ├── .env                 # Environment variables
│   ├── settings.py          # Configuration loader
│   ├── servers.json         # Server definitions
│   └── users.json           # User credentials
├── core/
│   ├── api/
│   │   └── main.py          # FastAPI app + UI
│   ├── brain/
│   │   └── router.py        # LLM routing logic
│   ├── memory/
│   │   ├── store.py         # ChromaDB vector store
│   │   └── conversations.py # SQLite history
│   ├── security/
│   │   └── auth.py          # Authentication
│   └── tools/
│       ├── registry.py      # Tool registration
│       ├── definitions.py   # Tool implementations
│       └── files.py         # Document handling
├── integrations/
│   ├── gmail/               # Email integration
│   ├── calendar/            # Calendar integration
│   ├── base_crm/            # Twenty CRM
│   ├── servers/             # SSH management
│   ├── finance/             # Banking
│   └── comfyui/             # Image generation
├── interfaces/
│   └── voice/
│       ├── stt.py           # Speech-to-text (Whisper)
│       ├── tts.py           # Text-to-speech (Kokoro/Qwen3)
│       └── wakeword.py      # Wake word detection (OpenWakeWord)
├── models/
│   └── hey_alfred.onnx      # Custom wake word model
├── data/
│   ├── uploads/             # Uploaded files
│   ├── generated/           # Created documents
│   └── conversations.db     # Chat history
├── static/
│   └── gr-mic.jpeg          # Mic button image
├── requirements.txt
└── README.md
```

## Troubleshooting

### Alfred won't start
```bash
# Check logs
sudo journalctl -u alfred -f

# Check port availability
sudo lsof -i :8400
```

### Image generation fails
```bash
# Check ComfyUI status
sudo systemctl status comfyui

# Verify GPU access
nvidia-smi
```

### Voice not working
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check Whisper model
ls ~/.cache/huggingface/hub/
```

### Memory issues
```bash
# Check ChromaDB
ls data/chroma/

# Reinitialize if corrupted
rm -rf data/chroma/
sudo systemctl restart alfred
```

### Wake word not detecting
```bash
# Check if model exists
ls models/hey_alfred.onnx

# Check WebSocket connection in browser console (F12)
# Look for "Wake word WebSocket connected"

# Check server logs
sudo journalctl -u alfred | grep -i wake
```

### Training a custom wake word
```bash
# Install openWakeWord training tools
pip install openwakeword

# Collect audio samples (5-10 recordings of "Hey Alfred")
# Use the openWakeWord training notebook or CLI
# Place the resulting .onnx file in models/hey_alfred.onnx
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

Private - All rights reserved.

## Acknowledgments

- [Anthropic](https://anthropic.com) - Claude API
- [Ollama](https://ollama.com) - Local LLM runtime
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - Image generation
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) - Speech recognition
- [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M) - Text-to-speech
- [Qwen3-TTS](https://github.com/QwenLM/Qwen2.5-Coder) - Voice cloning TTS
- [OpenWakeWord](https://github.com/dscripka/openWakeWord) - Wake word detection
- [Silero VAD](https://github.com/snakers4/silero-vad) - Voice activity detection
- [Twenty](https://twenty.com) - Open-source CRM
