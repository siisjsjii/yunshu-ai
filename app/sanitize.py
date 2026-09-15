"""对外错误文本的脱敏。

设计文档 §6 要求错误响应**不回显 key 内容**:上游 SDK 的异常文本里
可能带着请求头或响应体中的密钥(例如 openai 的
"Error code: 401 - {...'Incorrect API key provided: sk-...'...}"),
直接把它拼进 SSE error 帧或 HTTPException.detail 就会泄漏。
"""

REDACTED = "***"


def redact_api_key(text: str, api_key: str) -> str:
    """把配置中的密钥从将要发往客户端的文本里抹掉。

    只依赖配置里的字面值,不做启发式匹配 —— 宁可漏掉未知形态的密钥,
    也不误伤正常回复内容。
    """
    if not api_key:
        return text
    return text.replace(api_key, REDACTED)
