"""
Supabase client configuration.
Provides both anon client (respects RLS) and admin client (bypasses RLS).
"""
from functools import lru_cache

from supabase import Client, create_client

from config import get_settings

settings = get_settings()


@lru_cache
def get_supabase_client() -> Client:
    """
    Get Supabase client with anon key.
    This client respects Row Level Security policies.
    Use for user-facing operations.
    """
    return create_client(settings.supabase_url, settings.supabase_anon_key)


@lru_cache
def get_supabase_admin_client() -> Client:
    """
    Get Supabase client with service role key.
    This client bypasses Row Level Security.
    Use for server-side admin operations only.
    """
    return create_client(settings.supabase_url, settings.supabase_service_role_key)
