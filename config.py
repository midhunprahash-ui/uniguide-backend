from functools import lru_cache

from pydantic_settings import BaseSettings


def _looks_like_placeholder_secret(value: str | None) -> bool:
    if not value:
        return True

    normalized = value.strip().lower()
    if not normalized:
        return True

    placeholder_tokens = (
        "your_",
        "placeholder",
        "changeme",
        "example",
        "test",
    )
    return any(token in normalized for token in placeholder_tokens)


class Settings(BaseSettings):
    # API Keys
    gemini_api_key: str
    tavily_api_key: str | None = None
    event_discovery_gemini_budget: int = 3
    event_discovery_confidence_threshold: float = 0.45
    event_discovery_extractor_mode: str = "hybrid"
    event_discovery_crawl4ai_max_urls: int = 6
    event_discovery_crawl4ai_timeout_ms: int = 12000
    event_discovery_crawl4ai_check_robots_txt: bool = True

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

    @property
    def has_valid_gemini_api_key(self) -> bool:
        return not _looks_like_placeholder_secret(self.gemini_api_key)

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

@lru_cache
def get_settings():
    return Settings()
