from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""
    tavily_api_key: str = ""
    pinecone_api_key: str = ""
    pinecone_index: str = "agent-memory"
    database_url: str = "postgresql://agent:agent@postgres:5432/agent"
    redis_url: str = "redis://redis:6379"
    external_api_token: str = ""
    api_token: str = ""
    langchain_tracing_v2: str | None = None
    langchain_api_key: str | None = None
    langchain_project: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
