"""
RAG Audio Pipeline - Config
Loads settings from ai_engine/rag/.env
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the rag folder itself
_env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_env_path)

RAG_DIR = Path(__file__).parent

# LLM / Ollama
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")

# Supabase
SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY: str = os.getenv("SUPABASE_ANON_KEY", "")
SUPABASE_SERVICE_ROLE_KEY: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

# Models (all run locally)
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base")

# ESC-50
ESC50_META_PATH: Path = RAG_DIR / os.getenv("ESC50_META_PATH", "ESC-50-master/meta/esc50.csv")

# Retrieval
TOP_K: int = int(os.getenv("TOP_K", "10"))

# Supabase table name for audio memories
AUDIO_TABLE: str = "audio_memories"
