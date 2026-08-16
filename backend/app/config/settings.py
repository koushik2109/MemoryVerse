from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "MemoryVerse API"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"

    SUPABASE_URL: str = ""
    SUPABASE_ANON_KEY: str = ""
    SUPABASE_SERVICE_ROLE_KEY: str = ""

    JWT_SECRET: str = "memory_verse_super_secret_jwt_key_2026"
    ALGORITHM: str = "HS256"

    DATABASE_URL: Optional[str] = None

    # AI / LLM configuration
    LLM_PROVIDER: str = "openai"          # openai | gemini | anthropic | none
    LLM_API_KEY: Optional[str] = None
    LLM_MODEL: str = "gpt-4o-mini"        # provider-specific model name
    AI_ENABLED: bool = True               # set False to disable AI chat

    # Email & OTP
    SMTP_SERVER: str = "smtp.gmail.com"
    SMTP_PORT: int = 465
    SMTP_USERNAME: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    OTP_HASH_SECRET: str = "development_otp_secret_key_change_in_production"
    OTP_EXPIRE_MINUTES: int = 10
    OTP_MAX_ATTEMPTS: int = 5

    # Redis Rate Limiting & Storage
    UPSTASH_REDIS_REST_URL: Optional[str] = None
    UPSTASH_REDIS_REST_TOKEN: Optional[str] = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()

