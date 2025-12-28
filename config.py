from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Keys
    gemini_api_key: str

    # Supabase Configuration
    supabase_url: str
    supabase_anon_key: str
    supabase_service_role_key: str

    # Legacy Admin Credentials (kept for backwards compatibility)
    admin_username: str = "admin"
    admin_password: str = ""

    # Security (legacy - Supabase handles JWT now)
    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440

    # File Upload
    upload_directory: str = "./uploads"
    max_file_size_mb: int = 50

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

@lru_cache
def get_settings():
    return Settings()

