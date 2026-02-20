# Technology Stack

**Analysis Date:** 2026-02-20

## Languages

**Primary:**
- Python 3.12 - Backend API, integrations, LLM routing, tools, and ML models
- TypeScript 5.6.3 - React frontend (Vite + React 18)
- JavaScript - Vanilla JS embedded frontend (fallback), service worker

**Secondary:**
- SQL (SQLite + PostgreSQL) - Database queries and schema management
- YAML - Configuration files

## Runtime

**Environment:**
- Python 3.12 (running on 75.43.156.105 - Alfred Labs machine)
- Node.js v20.12.2 (via nvm) - Frontend build and dev server

**Package Manager:**
- pip - Python package management
- npm - Node.js package management (frontend)
- Lockfile: `requirements.txt` (Python), `package-lock.json` (Node.js)

## Frameworks

**Core:**
- FastAPI 0.128.0 - REST API and WebSocket server
- React 18.3.1 - Frontend UI (Vite + TypeScript + Tailwind)
- Vite 6.0.3 - Frontend build tool and dev server

**LLM & AI:**
- Ollama 0.6.1 - Local LLM inference (Qwen3-coder-next:cloud)
- Anthropic 0.76.0 - Claude API integration
- OpenAI 2.15.0 - ChatGPT integration
- LangChain 1.2.7 - LLM chain and agent orchestration
- LangGraph 1.0.7 - State machine workflows for AI agents
- LangSmith 0.6.6 - LLM monitoring and evaluation

**Memory & RAG:**
- ChromaDB 1.4.1 - Vector database for memory storage (persistent)
- Spacy 3.8.11 - NLP and entity extraction
- Faster-Whisper 1.2.1 - Speech-to-text (local)
- Kokoro 0.9.4 - Text-to-speech (local)

**Testing:**
- Not detected (no test files found)

**Build/Dev:**
- Tailwind CSS 3.4.15 - Frontend styling with CSS-in-JS
- PostCSS 8.4.49 - CSS post-processing
- AutoPrefixer 10.4.20 - Browser CSS vendor prefixing
- vite-plugin-pwa 0.21.1 - Progressive Web App support
- TypeScript compiler - Type checking and transpilation

## Key Dependencies

**Critical:**
- SQLAlchemy 2.0.46 - SQL ORM and database toolkit
- Pydantic 2.12.5 - Data validation and settings management
- Uvicorn 0.40.0 - ASGI server for FastAPI
- Starlette 0.50.0 - Async web framework (FastAPI depends on it)
- PyJWT 2.10.1 - JWT token creation and verification
- Passlib 1.7.4 - Password hashing (bcrypt)
- PyOAuth (python-jose 3.5.0) - OAuth/OIDC support

**API & Web:**
- httpx 0.28.1 - Async HTTP client
- requests 2.32.5 - Synchronous HTTP client
- aiohttp 3.13.3 - Async HTTP (fallback)
- websockets 16.0 - WebSocket protocol support

**Infrastructure:**
- Redis 7.1.0 - Caching and session management
- psycopg2-binary 2.9.11 - PostgreSQL adapter
- SQLAlchemy-based ORM - Relationships and migrations
- asyncssh 2.22.0 - SSH for remote command execution
- paramiko 4.0.0 - SSH protocol implementation

**Google Services:**
- google-api-python-client 2.188.0 - Gmail, Sheets, Calendar, Drive APIs
- google-analytics-data 0.18.18 - GA4 analytics API
- google-ads-api (via google-ads-googleads) - Google Ads API
- google-auth 2.48.0 - OAuth 2.0 authentication
- google-auth-oauthlib 1.2.4 - OAuth flow handling

**ML & NLP:**
- PyTorch 2.10.0+cu128 - Deep learning with CUDA 12.8
- TorchAudio 2.10.0+cu128 - Audio processing
- Transformers 5.0.0 - HuggingFace models (BERT, T5, etc.)
- ONNX Runtime 1.23.2 - Cross-platform model inference
- CUDA Toolkit 12.8 - GPU acceleration (NVIDIA)
- cuDNN 9.10.2.21 - GPU acceleration for neural networks

**External API Clients:**
- twilio 2.x - SMS and voice communications
- stripe (built-in to requests) - Payment processing

**Audio & Media:**
- PyAV 16.1.0 - Audio/video codec support
- SoundFile 0.13.1 - WAV file I/O
- Pillow 12.1.0 - Image processing

**Security & Cryptography:**
- cryptography 46.0.3 - Encryption and SSL/TLS
- PyNaCl 1.6.2 - Public key cryptography
- bcrypt 4.3.0 - Password hashing
- ECDSA 0.19.1 - Elliptic curve signatures
- py-vapid 1.9.4 - VAPID for Web Push
- pywebpush 2.2.0 - Web Push notifications

**Utilities:**
- python-dotenv 1.2.1 - Environment variable loading
- Pydantic-Settings 2.12.0 - Configuration management
- Loguru 0.7.3 - Advanced logging
- Click 8.3.1 - CLI creation
- Typer 0.21.1 - Modern CLI with type hints
- Invoke 2.2.1 - Task execution

## Configuration

**Environment:**
- Configuration loaded from `config/.env` via Pydantic Settings
- Settings class: `config/settings.py`
- Required env vars:
  - Database: `db_host`, `db_port`, `db_name`, `db_user`, `db_password`
  - LLM: `anthropic_api_key`, `openai_api_key`
  - Security: `secret_key`
  - OAuth: `google_client_id`, `google_client_secret`
  - Integrations: Various service API keys (Stripe, Twilio, etc.)
  - WebPush: `vapid_private_key`, `vapid_public_key`, `vapid_contact_email`

**Build:**
- `frontend/vite.config.ts` - Vite configuration with React plugin
- `frontend/tsconfig.json` - TypeScript compiler settings
- `frontend/tailwind.config.js` - Tailwind CSS configuration
- `core/api/main.py` - FastAPI app definition

## Platform Requirements

**Development:**
- Python 3.12 or higher
- Node.js v20.12.2 (via nvm)
- PostgreSQL 13+ (for production database)
- Redis 6+ (for caching/sessions)
- CUDA 12.8 (optional - for GPU acceleration)
- Ollama running locally (for local LLM inference)

**Production:**
- Linux server (75.43.156.105)
- PostgreSQL 13+ database
- Redis server
- FastAPI server (Uvicorn on port 8400)
- Node.js for frontend static asset building
- NVIDIA GPU with CUDA support (for Ollama inference)

---

*Stack analysis: 2026-02-20*
