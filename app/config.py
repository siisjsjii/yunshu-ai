from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置。三个 OPENAI_* 字段必填,其余有默认值。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 必填:无默认值
    openai_base_url: str
    openai_api_key: str
    openai_model: str

    # 可选:有默认值
    chat_temperature: float = 0.7
    extract_temperature: float = 0.0
    context_budget_tokens: int = 8192
    reserved_output_tokens: int = 1024
    safety_margin_tokens: int = 512
    session_ttl_seconds: int = 1800
    max_sessions: int = 1000
    session_lock_timeout_seconds: float = 60.0
    brand_name: str = "本店"


@lru_cache
def get_settings() -> Settings:
    return Settings()
