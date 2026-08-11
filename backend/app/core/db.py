from supabase import create_client, Client
from app.config.settings import settings
import logging

logger = logging.getLogger(__name__)

# Supabase Client Singleton
_supabase_client: Client | None = None

def get_supabase_client() -> Client:
    global _supabase_client
    if _supabase_client is None:
        key = settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_ANON_KEY
        _supabase_client = create_client(settings.SUPABASE_URL, key)
    return _supabase_client
