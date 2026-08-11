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

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()

