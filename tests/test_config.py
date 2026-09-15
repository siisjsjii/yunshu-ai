import pytest
from pydantic import ValidationError

from app.config import Settings

REQUIRED = {
    "openai_base_url": "https://api.deepseek.com/v1",
    "openai_api_key": "sk-test",
    "openai_model": "deepseek-chat",
    "database_url": "mysql+asyncmy://u:p@h:3306/db",
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
    with pytest.raises(ValidationError) as exc:
        Settings(
            _env_file=None,
            openai_base_url="x",
            openai_api_key="y",
            database_url="mysql+asyncmy://u:p@h:3306/db",
        )
    assert "openai_model" in str(exc.value)


def test_missing_base_url_is_rejected():
    with pytest.raises(ValidationError) as exc:
        Settings(
            _env_file=None,
            openai_api_key="y",
            openai_model="z",
            database_url="mysql+asyncmy://u:p@h:3306/db",
        )
    assert "openai_base_url" in str(exc.value)


def test_missing_api_key_is_rejected():
    """OPENAI_API_KEY 必填:不给默认值,避免无密钥时静默启动、首请求才炸。"""
    with pytest.raises(ValidationError) as exc:
        Settings(
            _env_file=None,
            openai_base_url="x",
            openai_model="z",
            database_url="mysql+asyncmy://u:p@h:3306/db",
        )
    assert "openai_api_key" in str(exc.value)


def test_reads_api_key():
    assert Settings(_env_file=None, **REQUIRED).openai_api_key == "sk-test"


def test_database_url_is_required():
    """DATABASE_URL 必填,不给默认值 —— 默认值会拿一个可能不对的连接串去连。"""
    missing = {k: v for k, v in REQUIRED.items() if k != "database_url"}
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **missing)


def test_database_url_is_read_from_settings():
    settings = Settings(_env_file=None, **REQUIRED)
    assert settings.database_url == "mysql+asyncmy://u:p@h:3306/db"


def test_tool_defaults():
    settings = Settings(_env_file=None, **REQUIRED)
    assert settings.tool_timeout_seconds == 10.0
    assert settings.tool_retry_attempts == 1
    assert settings.tool_retry_delay_seconds == 0.3


def test_negative_retry_attempts_is_rejected():
    """重试次数为负 → 执行器的 attempts 算成 0,循环一次都不跑、
    last_message 停在空串,最终把**空错误文案**交给模型。启动即拒。"""
    with pytest.raises(ValidationError) as exc:
        Settings(_env_file=None, **REQUIRED, tool_retry_attempts=-1)
    assert "tool_retry_attempts" in str(exc.value)


def test_non_positive_timeout_is_rejected():
    """超时 <= 0 会让每一次工具调用立即超时。"""
    with pytest.raises(ValidationError) as exc:
        Settings(_env_file=None, **REQUIRED, tool_timeout_seconds=0)
    assert "tool_timeout_seconds" in str(exc.value)


def test_negative_retry_delay_is_rejected():
    with pytest.raises(ValidationError) as exc:
        Settings(_env_file=None, **REQUIRED, tool_retry_delay_seconds=-0.1)
    assert "tool_retry_delay_seconds" in str(exc.value)


def test_zero_bounds_are_allowed():
    """下界只在负数上收:0 次重试(等价于只试一次)与零延迟都是合法配置。"""
    settings = Settings(
        _env_file=None, **REQUIRED, tool_retry_attempts=0, tool_retry_delay_seconds=0
    )
    assert settings.tool_retry_attempts == 0
    assert settings.tool_retry_delay_seconds == 0
