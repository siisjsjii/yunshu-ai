"""ch09 探针:`bind_tools` 与 `response_format=json_object` 能否同时用。

全部打真实网关。**每个变体都把它实际挂上了什么打出来** —— 首版探针因为
`kwargs={}` 分支压根没挂 `response_format`,把「裸 bind_tools」当成了
「两者同时用」,是典型的假绿。

变体:
  1  tools(非 strict) + response_format,要求**作答**
  2  tools(非 strict) + response_format,要求**调工具**
  3  tools(strict=True) + response_format,要求**作答**
  4  tools(strict=True) + response_format,要求**调工具**
  5  裸 response_format 作对照(基线)
"""

import asyncio
import sys

from langchain_core.tools import tool

from app.config import get_settings
from app.llm import create_chat_model
from app.sanitize import redact_api_key


@tool
def get_weather(city: str) -> str:
    """查某个城市的天气。"""
    return f"{city} 晴,25 度"


JSON_PROMPT = """你是电商客服。**只输出一个 JSON 对象**,不要输出任何其它内容。

字段与顺序**必须**是:
1. useful:布尔值。证据足以回答为 true,不足为 false。
2. confidence:0 到 1 的小数。
3. answer:字符串。useful 为 false 时必须是空字符串。

用户问题:退货政策是什么?
证据:7 天无理由退货,需不影响二次销售。"""

TOOL_PROMPT = "杭州今天天气怎么样?用工具查。"

RF = {"response_format": {"type": "json_object"}}


def _head(s: str, n: int = 170) -> str:
    return s[:n].replace("\n", "\\n")


def _fail(exc: Exception, api_key: str) -> None:
    print(f"  FAIL {type(exc).__name__}: {redact_api_key(str(exc), api_key)[:200]}")


async def run(*, model, key, label, tools_kw, prompt, rf_via):
    """rf_via: 'bind' | 'kwargs' | None —— 明确告诉我 response_format 挂在哪。"""
    print(f"=== {label} ===")
    print(f"  挂法: tools={tools_kw or '<无>'}  response_format={rf_via}")
    bound = model.bind_tools([get_weather], **(tools_kw or {})) if tools_kw is not None else model
    if rf_via == "bind":
        bound = bound.bind(**RF)
        call_kw = {}
    elif rf_via == "kwargs":
        call_kw = dict(RF)
    else:
        call_kw = {}
    parts: list[str] = []
    acc = None
    try:
        async for chunk in bound.astream(prompt, **call_kw):
            acc = chunk if acc is None else acc + chunk
            if chunk.text:
                parts.append(chunk.text)
    except Exception as exc:  # noqa: BLE001
        _fail(exc, key)
        return
    calls = list(getattr(acc, "tool_calls", None) or [])
    print(f"  → tool_calls={[c['name'] for c in calls]} 碎片数={len(parts)}")
    if parts:
        print(f"  → 全文={_head(''.join(parts))}")


async def main() -> None:
    settings = get_settings()
    model = create_chat_model(settings)
    key = settings.openai_api_key
    print(f"model={settings.openai_model} base={settings.openai_base_url}\n")

    await run(model=model, key=key, label="5. 基线:裸 response_format,作答",
              tools_kw=None, prompt=JSON_PROMPT, rf_via="bind")
    print()
    await run(model=model, key=key, label="1. 非strict tools + rf(bind),作答",
              tools_kw={}, prompt=JSON_PROMPT, rf_via="bind")
    print()
    await run(model=model, key=key, label="2. 非strict tools + rf(bind),调工具",
              tools_kw={}, prompt=TOOL_PROMPT, rf_via="bind")
    print()
    await run(model=model, key=key, label="3. strict tools + rf(bind),作答",
              tools_kw={"strict": True}, prompt=JSON_PROMPT, rf_via="bind")
    print()
    await run(model=model, key=key, label="4. strict tools + rf(bind),调工具",
              tools_kw={"strict": True}, prompt=TOOL_PROMPT, rf_via="bind")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
