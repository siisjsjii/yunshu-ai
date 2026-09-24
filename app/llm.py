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
        # ⚠️ **这个参数不能省**(T16b,2026-09-24 实测)。不传它的时候
        # langchain-openai(1.6.2)把 `request_timeout=None` **原样**交给 openai SDK,
        # 而 SDK 对「显式给的 None」的处理是**不设超时** —— 不是它自己的
        # `DEFAULT_TIMEOUT = Timeout(connect=5.0, read=600, write=600, pool=600)`。
        # 实测:`model.root_async_client._client.timeout` 是 `Timeout(timeout=None)`,
        # 四相全 None;下一层(httpcore2 2.13.0 —— 这条路径**不是**遗留的
        # `httpcore` 1.0.9,那是 httpx 0.28.1 的依赖)同样把 `timeout=None` 交给
        # `connect_tcp`/`start_tls` ⇒ 连 DNS 与握手都没有上界。
        # ⚠️ 一个**标量**会把 SDK 原本的 `connect=5` 一并换成 60 ⇒ 连接阶段是
        # **放松**了 12 倍(修前它是 ∞,所以不是回归)。要保留 SDK 的 connect 值,
        # 就把它改成四元组 `(5, 60, 60, 60)`(实测可用:
        # `_client.timeout` 变成 `Timeout(connect=5.0, read=60.0, write=60.0, pool=60.0)`;
        # 注意**二**元组会让 write/pool 落回 None = 又不设上界了)。
        # 后果不是「慢」而是**永不返回**:对端静默一个字节都不回时,那次 await
        # 谁也等不回来 —— 飞轮任务因此卡在 running 占着单槽(见 config 里那两个
        # 上界的注释),而每次请求各建一个模型 ⇒ **每个**入口都被同一条命门覆盖。
        # 上界是**每一次往返**(connect/read/write/pool),不是整段流;
        # 数值与最坏耗时见 `settings.llm_timeout_seconds` 那一段。
        timeout=settings.llm_timeout_seconds,
    )


def create_chat_model(settings: Settings) -> ChatOpenAI:
    """对话用模型。temperature 偏高,客服回复需要亲和力。"""
    return _build(settings, temperature=settings.chat_temperature)


def create_extract_model(settings: Settings) -> ChatOpenAI:
    """抽取用模型。temperature 为 0,结构化输出需要稳定。"""
    return _build(settings, temperature=settings.extract_temperature)
