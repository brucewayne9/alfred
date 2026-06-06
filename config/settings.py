from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # Paths
    base_dir: Path = Path("/home/aialfred/alfred")
    data_dir: Path = Path("/home/aialfred/alfred/data")

    # Database
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "alfred_main"
    db_vault: str = "alfred_vault"
    db_user: str = "alfred"
    db_password: str = ""

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Ollama (local LLM)
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3-coder-next:cloud"

    # Anthropic (cloud LLM)
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"

    # OpenAI (ChatGPT)
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Google OAuth
    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = "https://aialfred.groundrushcloud.com/auth/google/callback"

    # Voice
    whisper_model: str = "small"
    tts_model: str = "kokoro"
    tts_voice: str = "bm_daniel"  # Default voice (Kokoro: bm_daniel, Qwen3: demo_speaker0)
    voice_sample_path: str = ""

    # Server
    host: str = "0.0.0.0"
    port: int = 8400
    debug: bool = False

    # Security
    secret_key: str = ""
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 480

    # CRM (Twenty CRM)
    base_crm_api_key: str = ""
    base_crm_url: str = "https://crm.groundrushlabs.com"

    # n8n Workflow Automation
    n8n_url: str = ""
    n8n_api_key: str = ""

    # Nextcloud
    nextcloud_url: str = ""
    nextcloud_username: str = ""
    nextcloud_password: str = ""

    # Stripe
    stripe_api_key: str = ""
    stripe_fitasruck_webhook_secret: str = ""

    # Fit as Ruck
    brevo_fitasruck_rest_key: str = ""
    fitasruck_pdf_url: str = "https://fitasruck.com/wp-content/uploads/fitasruck-130859f216d7fbb0/fitasruck-8week.pdf"
    fitasruck_buyers_list_id: int = 3

    # AzuraCast Radio
    azuracast_url: str = ""
    azuracast_api_key: str = ""

    # --- Central Casting ---
    casting_db_path: str = "/home/aialfred/alfred/data/casting.db"
    casting_voices_dir: str = "/home/aialfred/alfred/data/casting/voices"
    casting_previews_dir: str = "/home/aialfred/alfred/data/casting/previews"
    qwen_tts_url: str = "http://75.43.156.105:7860"
    casting_ollama_url: str = "http://75.43.156.105:11434"
    casting_model: str = "kimi-k2.6:cloud"
    # AzuraCast deploy target (server 98)
    casting_ssh_host: str = "server-98"
    casting_az_db_pass: str = "Yc2tNakqcne2"
    casting_engine_voices_dir: str = "/var/azuracast"          # where <Base>_<mood>.wav live on 98
    casting_station_id: int = 22

    # Meta Ads
    meta_app_id: str = ""
    meta_app_secret: str = ""
    meta_access_token: str = ""
    meta_ad_account_id: str = ""

    # Telegram (Alfred bot)
    telegram_bot_token: str = ""
    telegram_chat_id: str = "7582976864"

    # Long Processing Notifications
    long_processing_threshold_seconds: int = 60
    long_processing_email_to: str = ""
    long_processing_email_from_account: str = "lumabot"

    # Web Push (VAPID)
    vapid_private_key: str = ""
    vapid_public_key: str = ""
    vapid_contact_email: str = ""

    # Roen Handmade — WooCommerce REST API
    # Generate at WP Admin → WooCommerce → Settings → Advanced → REST API
    # Permission: Read/Write. Add to config/.env as WC_ROEN_KEY / WC_ROEN_SECRET.
    wc_roen_key: str = ""
    wc_roen_secret: str = ""

    # Roen Handmade — WC outbound webhook → Alfred (sale notifications to Sarah).
    # Shared secret used to HMAC-verify incoming WC webhooks at /webhooks/roen/wc-order.
    # Generate once and register the same value on WP Admin → WooCommerce → Settings → Advanced → Webhooks.
    roen_wc_webhook_secret: str = ""
    # Telegram bot used to DM Sarah + Mike on new sales. Same token the Roen
    # intake bot uses (TELEGRAM_BOT_ROENHANDMADE_TOKEN).
    telegram_bot_roenhandmade_token: str = ""
    # Comma-separated chat IDs to ping on new sale. First entry is Sarah.
    roen_intake_allowed_chat_ids: str = ""

    # Roen Handmade — Meta (Facebook + Instagram) dedicated app
    # System User token is long-lived (never expires). Discovered IDs filled by setup.
    roen_meta_app_id: str = ""
    roen_meta_app_secret: str = ""
    roen_meta_system_user_token: str = ""
    roen_meta_system_user_id: str = ""
    roen_meta_page_id: str = ""
    roen_meta_ig_user_id: str = ""
    roen_meta_catalog_id: str = ""
    roen_site_base_url: str = "https://www.roenhandmade.com"

    # SEO — Google API access (Search Console + GA4 + PageSpeed Insights)
    # OAuth client for GSC + GA4 (user-consent flow). Create at console.cloud.google.com → APIs & Services → Credentials.
    seo_google_oauth_client_id: str = ""
    seo_google_oauth_client_secret: str = ""
    # PageSpeed Insights uses a simple API key (no OAuth). Free tier 25k/day.
    seo_psi_api_key: str = ""
    # Where the OAuth refresh token gets persisted on disk after Mike consents once.
    seo_oauth_token_path: str = "/home/aialfred/alfred/data/seo/google_oauth_token.json"
    # DataForSEO — paid SEO data API (keyword volumes, SERP, on-page audit, backlinks).
    # Pay-per-call from prepaid balance. Account: alfred@groundrushlabs.com.
    dataforseo_login: str = ""
    dataforseo_password: str = ""

    # AI Savings Audit funnel
    # cal_booking_link — Mike's Google Calendar appointment-schedule URL (15-min slot)
    # brevo_ai_audit_list_id — int ID of the Brevo list new audit leads land in
    #   (create the list in Brevo dashboard once, then paste the ID into config/.env)
    cal_booking_link: str = "https://calendar.app.google/4b6G2vqTxmEGWgGo9"
    brevo_ai_audit_list_id: int = 0

    # RuckTalk WordPress (server-100 / rt-wordpress container) — REST API access
    # Generate at WP Admin → Users → alfred → Application Passwords. Add to
    # config/.env as RUCKTALK_WP_APP_USER and RUCKTALK_WP_APP_PASSWORD.
    rucktalk_wp_app_user: str = "alfred"
    rucktalk_wp_app_password: str = ""

    @property
    def database_url(self) -> str:
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    @property
    def vault_database_url(self) -> str:
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_vault}"

    model_config = {"env_file": "/home/aialfred/alfred/config/.env", "extra": "ignore"}


settings = Settings()
