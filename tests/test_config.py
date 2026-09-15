import pytest
from pydantic import ValidationError

from app.config import Settings

REQUIRED = {
    "openai_base_url": "https://api.deepseek.com/v1",
    "openai_api_key": "sk-test",
    "openai_model": "deepseek-chat",
}


def test_reads_required_fields():
    settings = Settings(_env_file=None, **REQUIRED)
    assert settings.openai_base_url == "https://api.deepseek.com/v1"
    assert settings.openai_model == "deepseek-chat"


def test_optional_fields_have_defaults():
    settings = Settings(_env_file=None, **REQUIRED)
    assert settings.chat_temperature == 0.7
    assert settings.extract_temperature == 0.0
    assert settings.context_budget_tokens == 8192
    assert settings.reserved_output_tokens == 1024
    assert settings.safety_margin_tokens == 512
    assert settings.session_ttl_seconds == 1800
    assert settings.max_sessions == 1000
    assert settings.session_lock_timeout_seconds == 60.0


def test_missing_model_is_rejected():
    """OPENAI_MODEL 必填:不给默认值,避免换模型时静默用错模型名。"""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, openai_base_url="x", openai_api_key="y")


def test_missing_base_url_is_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, openai_api_key="y", openai_model="z")


def test_missing_api_key_is_rejected():
    """OPENAI_API_KEY 必填:不给默认值,避免无密钥时静默启动、首请求才炸。"""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, openai_base_url="x", openai_model="z")


def test_reads_api_key():
    assert Settings(_env_file=None, **REQUIRED).openai_api_key == "sk-test"
