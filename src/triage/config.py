from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str = "postgresql+asyncpg://triage:triage@localhost:5433/triage"
    REDIS_URL: str = "redis://localhost:6380/0"
    JEV_API_KEY: str = ""
    JEV_BASE_URL: str = "https://api.typesafe.ai"
    JEV_MODEL: str = "jev-latest"
    OLLAMA_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "qwen2.5:7b"
    HI: float = 0.7
    LO: float = 0.2
    DEEP_READ_MIN_CONFIDENCE: float = 0.7   # a 'not fit' verdict from the local LLM below this goes to the human, not the bin
    # Where an on-site or hybrid role is still acceptable. The user is Pakistan-based and won't relocate.
    TARGET_LOCATIONS: str = "pakistan,karachi,lahore,islamabad,rawalpindi"
    TRIAGE_ENABLED: bool = True    # fetch queues triage_post for new/changed posts; False for fetch-only runs

    @property
    def psycopg_url(self) -> str:
        """libpq-style URL for the LangGraph Postgres checkpointer."""
        return self.DATABASE_URL.replace("+asyncpg", "")


settings = Settings()
