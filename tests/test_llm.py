import asyncio
import socket
import time

import pytest
from langchain_core.messages import HumanMessage
from openai import APITimeoutError

from app.config import Settings
from app.llm import create_chat_model, create_extract_model

REQUIRED = {
    "openai_base_url": "https://api.deepseek.com/v1",
    "openai_api_key": "sk-test",
    "openai_model": "deepseek-chat",
    "database_url": "mysql+asyncmy://u:p@h:3306/db",
}


def _settings(**overrides) -> Settings:
    # 合并再展开:`**REQUIRED, **overrides` 在两边同名时是 `TypeError:
    # got multiple values for keyword argument`(`openai_base_url` 会被下面那条
    # 黑洞用例覆盖 —— 没合并的话红的是用例自己,不是被测对象)。
    return Settings(_env_file=None, **{**REQUIRED, **overrides})


def _silent_gateway() -> tuple[socket.socket, int]:
    """一个「连接成功、响应永远不来」的回环端口:T16b 复现探针里的那个黑洞。

    `listen()` 之后**永不 accept** ⇒ 三次握手指令由内核完成(不像已关闭端口那样
    要等 ~2s 才拿到拒绝),而请求发出去之后一个字节都不会回来。这不是打桩:
    被测对象面对的是**真的 socket、真的 TCP、真的 read**。
    """
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    return srv, srv.getsockname()[1]


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


# --------------------------------------------------------------------------
# 模型往返的**墙钟上界**(T16b 的根因,2026-09-24)
# --------------------------------------------------------------------------
#
# 现场:飞轮后台任务三次卡在 `running` 不放(666s / 245s / 382s),占着**单槽**,
# 此后端点永远 409 —— 连手动那个「跑一轮飞轮」都拿不到槽,唯一出路是重启服务。
# 复盘出的根因有两半,这两条用例各钉一半:
#
# ① **无上界的等待**(下面两条):`app/llm.py` 不传 `timeout` 时,
#    langchain-openai(1.6.2)把 `request_timeout=None` **原样**交给 openai SDK,
#    而 openai SDK 对「显式给的 None」的处理是**不设超时**(不是它自己的
#    `DEFAULT_TIMEOUT = Timeout(connect=5.0, read=600, write=600, pool=600)`)——
#    实测 `model.root_async_client._client.timeout` 是 `Timeout(timeout=None)`,
#    四相全 None。对端静默时那次 await **永不返回**。
# ② **任务没有寿命上界**:不抛异常的那条路走不到 `finally`
#    (`tests/test_flywheel_task.py` 里那条 deadline 用例守它)。
#
# ⚠️ 这里的 `_settings(llm_timeout_seconds=…)` **显式传一个非默认值**:传默认值的话,
# 「读的是配置」与「写死一个常数」在断言上分不开(本仓那条老规矩)。


def test_the_configured_timeout_reaches_the_http_client():
    """`llm_timeout_seconds` 必须真的落到 httpx 客户端上,**不能**是 None。

    ⚠️ 断言打在 `root_async_client._client.timeout`(私有的那一层)是**刻意**的:
    就是这一层曾经是 `Timeout(timeout=None)`,而 `model.request_timeout is None`
    在任何一层都看不出来(它本来就是 None ⇒ 客户端「不设超时」)。
    只断 `model.request_timeout == 12.5` 的话,一个**收了值却丢掉**的实现照样绿。
    """
    for build in (create_chat_model, create_extract_model):
        model = build(_settings(llm_timeout_seconds=12.5))
        assert model.request_timeout == 12.5
        client = model.root_async_client
        assert client.timeout is not None, "客户端又回到「不设超时」那一档"
        for phase in ("connect", "read", "write", "pool"):
            assert getattr(client._client.timeout, phase) is not None, (
                f"{phase} 相没有上界 ⇒ 对端静默时那次 await 永不返回"
            )


@pytest.mark.anyio
async def test_a_silent_gateway_surfaces_as_a_timeout_instead_of_hanging():
    """对端**一个字节都不回**时,调用必须在有界时间内**报错**,而不是挂着。

    这条是行为层的那一半(结构层是上面那条):真的打一个黑洞端口,真的等。
    `wait_for(15)` 是**用例自己的**保险 —— 没有它,修好之前这条用例会**挂住**
    (挂住的测试比红的测试更坏:它会把整个 `not db` 那一轮拖死)。
    修好之后:0.5s 的读超时 × (1 + max_retries=2) 次尝试 + 两次退避 ≈ 3s 内抛。
    """
    srv, port = _silent_gateway()
    try:
        model = create_extract_model(_settings(
            openai_base_url=f"http://127.0.0.1:{port}/v1", llm_timeout_seconds=0.5))
        t0 = time.monotonic()
        with pytest.raises(APITimeoutError):
            await asyncio.wait_for(model.ainvoke([HumanMessage("hi")]), timeout=15)
        elapsed = time.monotonic() - t0
        # 10s 是**有判别力**的界:上面那条 wait_for(15) 只保证「用例不挂住」,
        # 而「是不是等待自己结束的」靠这一行 —— 修好之后实测 ≈3s。
        assert elapsed < 10, f"等待没有被有界地终止,实际 {elapsed:.1f}s"
    finally:
        srv.close()
