from app.config import Settings
from app.llm import create_chat_model, create_extract_model

REQUIRED = {
    "openai_base_url": "https://api.deepseek.com/v1",
    "openai_api_key": "sk-test",
    "openai_model": "deepseek-chat",
}


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **REQUIRED, **overrides)


def test_chat_model_disables_responses_api():
    """硬约束:LangChain 1.x 默认走 Responses API,DeepSeek 不支持。"""
    model = create_chat_model(_settings())
    assert model.use_responses_api is False


def test_extract_model_disables_responses_api():
    model = create_extract_model(_settings())
    assert model.use_responses_api is False


def test_chat_model_uses_configured_base_url_and_model():
    model = create_chat_model(_settings())
    assert model.model_name == "deepseek-chat"
    assert str(model.openai_api_base) == "https://api.deepseek.com/v1"


def test_chat_model_uses_chat_temperature():
    model = create_chat_model(_settings(chat_temperature=0.3))
    assert model.temperature == 0.3


def test_extract_model_uses_zero_temperature_by_default():
    model = create_extract_model(_settings())
    assert model.temperature == 0.0


def test_extract_model_uses_configured_temperature():
    model = create_extract_model(_settings(extract_temperature=0.2))
    assert model.temperature == 0.2


def test_chat_model_streams_usage():
    """done 事件要带 usage,需要开启流式用量统计。"""
    model = create_chat_model(_settings())
    assert model.stream_usage is True
