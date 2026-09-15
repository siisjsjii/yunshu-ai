from langchain_openai import ChatOpenAI

from app.config import Settings


def _build(settings: Settings, *, temperature: float) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        temperature=temperature,
        # 硬约束:LangChain 1.x 的 OpenAI provider 默认走 Responses API,
        # DeepSeek 等 OpenAI 兼容网关只实现 Chat Completions。
        # 不关掉会调用失败,且报错指向"模型不存在",极难定位。
        use_responses_api=False,
        # 让最后一帧带上 usage_metadata,done 事件需要它。
        stream_usage=True,
    )


def create_chat_model(settings: Settings) -> ChatOpenAI:
    """对话用模型。temperature 偏高,客服回复需要亲和力。"""
    return _build(settings, temperature=settings.chat_temperature)


def create_extract_model(settings: Settings) -> ChatOpenAI:
    """抽取用模型。temperature 为 0,结构化输出需要稳定。"""
    return _build(settings, temperature=settings.extract_temperature)
