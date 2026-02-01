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
    ollama_model: str = "gpt-oss:120b-cloud"

    # Anthropic (cloud LLM)
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"

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

    # AzuraCast Radio
    azuracast_url: str = ""
    azuracast_api_key: str = ""

    # Meta Ads
    meta_app_id: str = ""
    meta_app_secret: str = ""
    meta_access_token: str = ""
    meta_ad_account_id: str = ""

    @property
    def database_url(self) -> str:
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    @property
    def vault_database_url(self) -> str:
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_vault}"

    model_config = {"env_file": "/home/aialfred/alfred/config/.env", "extra": "ignore"}


settings = Settings()
