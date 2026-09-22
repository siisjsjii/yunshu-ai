"""ch07 T9:后台摘要执行体 —— 同会话去重 / 重读锚点 / 失败不冒泡 / engine 必 dispose。

**不联网、不碰 db、不依赖 LangChain。** 四组性质各有用例钉住,每一组都对应一个
「错了也不报错」的失效:

1. **同会话同一时刻只跑一个**,而**别的会话不受影响** —— 两个任务并发会把同一段
   原文压两遍(重复梗概里每一段单独看都正常);全局一把锁则会把所有会话串起来
   (每个会话都要等别人的模型往返);
2. **任务开头重读两个锚点,而且只压 `(summary_upto, layer1_from)` 这一段** ——
   拿触发时的快照会让一段历史**既不在层 2、也不在梗概里**;压过头(把层 1 也在
   注入的那几条一起压了)则让同一段内容在上下文里出现两遍。两者都不报错;
3. **失败不冒泡,也不把会话永久锁死** —— 线程体抛异常只在日志里留一行
   `summary fail`;而在跑标记必须在**失败路径上也摘掉**,否则这个会话**再也压不了**,
   用户侧看起来完全正常,只是梗概永远不更新;
4. **engine 自建、每条退出路径都 dispose** —— 漏掉异常路径就是每压一次泄漏一个
   连接池,而摘要任务每轮都可能起。

**为什么走 `_thread_target` / `_run_body` 这两个接缝**:起真线程再等它跑完会变成
时序断言,而时序断言是不稳定的;不稳定的断言最后会被人删掉,而不是修好。
`_thread_target` 是**同步**的线程体,所以「标记有没有被摘掉」可以确定性地验。

`_reload_state` 的替身是 `async def`:生产的那个要在**同一个** `asyncio.run` 里
读库(照抄 ch04 的形状),而同步函数没法 await 异步 engine —— 详见模块名的
那段说明与 T9 报告。
"""

import asyncio
import json
import logging
import threading
import types

import pytest

from app.config import Settings
from app.memory import tasks
from app.schemas import Message


#: `_env_file=None` 是本仓硬规矩(仓库根有真实 `.env`)。
REQUIRED = dict(
    _env_file=None,
    openai_base_url="https://example.invalid/v1",
    openai_api_key="sk-test",
    openai_model="m",
    database_url="mysql+asyncmy://u:p@127.0.0.1:3306/x",
)


def _settings(**over):
    return Settings(**{**REQUIRED, **over})


def _HISTORY() -> list[Message]:
    """id 1..9 的一小段历史。

    **故意比区间长**:重读回来的锚点是 `(3, 7)`,而历史是 1..9 —— 于是
    「只压区间内那几条」与「把整段历史都压了」在 `seen["turns"]` 上**输出不同**。
    若这里只给 4/5/6,那条断言就恒真了(本仓第 (e) 种假绿形态:输入小到
    触发不了被测行为)。
    """
    return [
        Message(id=i, role="user" if i % 2 else "assistant", content=f"第 {i} 条")
        for i in range(1, 10)
    ]


def _payloads(caplog, prefix: str) -> list[dict]:
    """取出本模块打的那几行(`"summary <event> <json>"`)的 payload。"""
    got = []
    for record in caplog.records:
        message = record.getMessage()
        if message.startswith(prefix):
            got.append(json.loads(message.split(" ", 2)[2]))
    return got


# --------------------------------------------------------------- 同会话去重


def test_only_one_summary_task_per_conversation(monkeypatch):
    """同一会话同一时刻只允许一个任务。

    两个任务并发会把同一段原文压两遍(或撞 `(conversation_id, seq)` 唯一键),
    而**重复梗概里每一段单独看都正常** —— 它与「压对了」在数据上只差一段,
    没有任何东西会报错。
    """
    release = threading.Event()

    def _blocking(*, conversation_id, settings, model_factory):
        release.wait(timeout=5)

    monkeypatch.setattr(tasks, "_run_body", _blocking)
    tasks._INFLIGHT.clear()

    first = tasks.run_summary_in_background(
        conversation_id="c1", settings=_settings(), model_factory=lambda s: None
    )
    second = tasks.run_summary_in_background(
        conversation_id="c1", settings=_settings(), model_factory=lambda s: None
    )
    assert first is True
    assert second is False            # ← 第二个必须被挡下

    # 另一个会话**不受影响** —— 全局一把锁会把所有会话串起来
    third = tasks.run_summary_in_background(
        conversation_id="c2", settings=_settings(), model_factory=lambda s: None
    )
    assert third is True

    release.set()
    tasks._INFLIGHT.clear()


def test_refused_task_is_logged_as_a_skip_with_its_reason(monkeypatch, caplog):
    """被挡下的那次不是「什么都没发生」—— 它要留一行 `summary skip`。

    没有这一行的话,「这一轮压根没触发」与「触发了但被上一个任务挡下」在日志里
    完全一样,而两者的处置完全不同(后者说明该会话的模型往返比轮次间隔还慢)。
    """
    release = threading.Event()

    def _blocking(*, conversation_id, settings, model_factory):
        release.wait(timeout=5)

    monkeypatch.setattr(tasks, "_run_body", _blocking)
    tasks._INFLIGHT.clear()
    try:
        with caplog.at_level(logging.INFO):
            tasks.run_summary_in_background(
                conversation_id="c1", settings=_settings(), model_factory=lambda s: None
            )
            tasks.run_summary_in_background(
                conversation_id="c1", settings=_settings(), model_factory=lambda s: None
            )
    finally:
        release.set()
        tasks._INFLIGHT.clear()

    skips = _payloads(caplog, "summary skip ")
    assert [s["reason"] for s in skips] == [tasks.SKIP_ALREADY_RUNNING]
    assert skips[0]["conversation_id"] == "c1"


def test_a_thread_that_will_not_start_does_not_wedge_the_conversation(monkeypatch, caplog):
    """线程**起不来**时也要撤销登记 —— 否则同样是永久锁死,只是发生在更早一步。

    这条比「跑完摘标记」更隐蔽:起线程失败会抛在登记**之后**,
    若没有这一处的撤销,那个会话从此每次都被当成「已有任务在跑」。

    `False` 在这里表示「没起任务」,与「被上一个任务挡下」是同一个返回值的两种
    成因 —— 日志里的 `reason` 分得清(见 `SKIP_THREAD_NOT_STARTED`)。
    """

    class _BadThread:
        def __init__(self, **kwargs):
            pass

        def start(self):
            raise RuntimeError("can't start new thread")

    # 只换 tasks 看到的那个 `threading`(不动全局的 threading 模块)。
    monkeypatch.setattr(tasks, "threading", types.SimpleNamespace(Thread=_BadThread))
    tasks._INFLIGHT.clear()

    with caplog.at_level(logging.INFO):
        started = tasks.run_summary_in_background(
            conversation_id="c1", settings=_settings(), model_factory=lambda s: None
        )

    assert started is False
    assert "c1" not in tasks._INFLIGHT          # ← 登记撤掉了
    assert [s["reason"] for s in _payloads(caplog, "summary skip ")] == [
        tasks.SKIP_THREAD_NOT_STARTED
    ]


def test_the_guard_is_released_after_a_successful_body(monkeypatch):
    """跑完就放行 —— 否则这个会话**一辈子只能压一次**。"""
    monkeypatch.setattr(tasks, "_run_body", lambda **kwargs: None)
    tasks._INFLIGHT.add("c1")

    tasks._thread_target(
        conversation_id="c1", settings=_settings(), model_factory=lambda s: None
    )

    assert "c1" not in tasks._INFLIGHT


def test_the_guard_is_released_even_when_the_body_raises(monkeypatch):
    """**失败路径也要摘标记。**

    漏了 `finally` 的话,一次失败就把这个会话**永久锁死**:之后每一次
    `run_summary_in_background` 都被当成「已有任务在跑」拒掉,而用户侧每一轮
    看起来都完全正常 —— 只是梗概永远不再更新,层 2 无限涨下去。

    这里注入的是**比生产更坏**的形态:`_run_body` 的契约是「绝不抛」,生产那条路
    由它内部的 `except Exception` 接住(上一条用例钉的就是它)。所以本用例问的是
    **真漏出来了会怎样** —— 答案必须是「标记照样摘掉」,`finally` 才是那条
    `BaseException`(KeyboardInterrupt / CancelledError)路径的保证。
    异常本身从 `_thread_target` 出来是**对的**(那条路没人处理,线程噪音如实反映);
    会要命的是标记跟着漏掉。
    """

    def _boom(**kwargs):
        raise RuntimeError("线程体炸了")

    monkeypatch.setattr(tasks, "_run_body", _boom)
    tasks._INFLIGHT.add("c1")

    with pytest.raises(RuntimeError):
        tasks._thread_target(
            conversation_id="c1", settings=_settings(), model_factory=lambda s: None
        )

    assert "c1" not in tasks._INFLIGHT


# --------------------------------------------------------------- 重读锚点


def test_task_rereads_anchors_instead_of_trusting_the_trigger_snapshot(monkeypatch):
    """任务开头必须**重读**两个锚点 —— 起任务到真跑之间边界可能已经变了。

    用触发时的快照去压,压出来的区间可能与当前层 2 对不上,
    结果是**一段历史被跳过**:既不在层 2、也不在梗概里。
    """
    seen = {}

    async def _fake_summarize(*, model, session, conversation_id, turns, upto_msg_id):
        seen["turns"] = [m.id for m in turns]
        seen["upto"] = upto_msg_id
        return "梗概"

    async def _reload(engine, conversation_id):
        return (3, 7, _HISTORY())

    monkeypatch.setattr(tasks, "summarize_range", _fake_summarize)
    monkeypatch.setattr(tasks, "_reload_state", _reload)
    # ↑ 重读回来的是 (summary_upto=3, layer1_from=7),与「触发时的快照」不同

    tasks._run_body(conversation_id="c1", settings=_settings(),
                    model_factory=lambda s: None)

    assert seen["turns"] == [4, 5, 6]        # ← 用的是**重读**到的 (3, 7)
    assert seen["upto"] == 7


def test_empty_interval_never_reaches_the_model(monkeypatch, caplog):
    """区间为空 ⇒ 一次模型构造、一次模型调用都不发生。

    `layer1_from == 0` 在本章是**有含义的值**:层 1 起于最早 ⇒ **层 2 为空**
    (spec §3.1 的表,`layers._middle` 的实现)。把它当成「到末尾」的实现会在
    这个形态下把**整段历史**压进去 —— 而它压的正是层 1 里还在逐轮注入的那几条,
    于是同一段内容在上下文里出现两遍,不报错、也不丢消息。

    **断言落在替身记下的调用上,不落在「抛没抛」上**:`_run_body` 会把任何
    异常接住记成 `summary fail`,所以用「替身直接抛」来判「没被调用」是假的
    —— 抛出来的那个异常会被吞掉,用例照样绿。
    """
    built = []

    def _factory(settings):
        built.append(settings)
        return object()

    calls = []

    async def _fake_summarize(**kwargs):
        calls.append(kwargs)
        return "梗概"

    async def _reload(engine, conversation_id):
        return (0, 0, _HISTORY())        # 两个锚点都是 0 ⇒ 层 2 为空

    monkeypatch.setattr(tasks, "summarize_range", _fake_summarize)
    monkeypatch.setattr(tasks, "_reload_state", _reload)

    with caplog.at_level(logging.INFO):
        tasks._run_body(conversation_id="c1", settings=_settings(), model_factory=_factory)

    assert calls == []
    assert built == []
    assert [s["reason"] for s in _payloads(caplog, "summary skip ")] == [
        tasks.SKIP_EMPTY_RANGE
    ]


def test_skip_reasons_separate_empty_range_from_blank_model_output(monkeypatch, caplog):
    """`summarize_range` 返回 `None` 的**两种成因必须分开记**。

    「区间为空」是**正常**(没有可压的东西);「模型吐了空」是**故障** ——
    而且它**没有退避**:下一轮触发会再打一次模型,一直打。两条混成一句 skip,
    运维看到的就是「一直在跳过」,而真实原因是模型坏了(spec §12.2 的连带)。
    """
    calls = []

    async def _blank_summarize(**kwargs):
        calls.append(kwargs)
        return None

    monkeypatch.setattr(tasks, "summarize_range", _blank_summarize)

    async def _empty_reload(engine, conversation_id):
        return (0, 0, _HISTORY())

    async def _nonempty_reload(engine, conversation_id):
        return (0, 5, _HISTORY())        # 区间是 1..4,有东西可压

    with caplog.at_level(logging.INFO):
        monkeypatch.setattr(tasks, "_reload_state", _empty_reload)
        tasks._run_body(conversation_id="c1", settings=_settings(),
                        model_factory=lambda s: None)
        monkeypatch.setattr(tasks, "_reload_state", _nonempty_reload)
        tasks._run_body(conversation_id="c1", settings=_settings(),
                        model_factory=lambda s: None)

    assert len(calls) == 1               # 第二次**真的**调了模型,否则这条用例恒真
    skips = _payloads(caplog, "summary skip ")
    assert [s["reason"] for s in skips] == [
        tasks.SKIP_EMPTY_RANGE,
        tasks.SKIP_BLANK_MODEL_OUTPUT,
    ]
    assert all(s["conversation_id"] == "c1" for s in skips)


# --------------------------------------------------------------- 失败不冒泡


def test_task_never_raises_into_the_caller(monkeypatch):
    """后台任务失败**不冒泡到请求路径** —— 它在**自己的线程**里。

    失败只留日志(spec §7.6 的 `summary fail`),边界不动。
    这条形状本身就保证了不冒泡(线程里抛不会传到请求),所以真正的断言是
    **`_run_body` 内部把它接住了** —— 否则线程会打一条
    `Exception in thread` 的噪音,而那是「没人处理」的样子。
    """
    async def _boom(**kwargs):
        raise RuntimeError("上游炸了")

    async def _reload(engine, conversation_id):
        return (0, 5, _HISTORY())

    monkeypatch.setattr(tasks, "summarize_range", _boom)
    monkeypatch.setattr(tasks, "_reload_state", _reload)

    tasks._run_body(conversation_id="c1", settings=_settings(),
                    model_factory=lambda s: None)   # ← **不抛**才算过

    # 并且锚点**没有被推进**(它只在成功后推)
    assert "c1" not in tasks._INFLIGHT


def test_anchor_query_failure_is_caught_too(monkeypatch):
    """**读锚点那一步失败**同样接住。

    与上一条不是同一件事:上一条炸在模型调用里(区间已读到),这条炸在**最开头**。
    没接住的话任务在 `asyncio.run` 里抛出去,调用方是**线程**,于是只有一条
    `Exception in thread` 噪音 —— 没有 `summary fail`、没有任何边界信息。
    """
    async def _boom(engine, conversation_id):
        raise RuntimeError("库连不上")

    monkeypatch.setattr(tasks, "_reload_state", _boom)

    tasks._run_body(conversation_id="c1", settings=_settings(),
                    model_factory=lambda s: None)   # ← **不抛**才算过


def test_fail_log_redacts_the_api_key(monkeypatch, caplog):
    """`summary fail` 记的是**异常文本**,而它常常就是上游响应体原文。

    openai 的 401 把 key 原样写在里面(本仓 ch01 实测过),所以记之前必须过
    `app.sanitize.redact_api_key`。替身**真的把密钥写进异常** —— 否则
    「日志里没有密钥」是恒真的(本仓的既有教训)。
    """
    key = "sk-live-KEY-0123456789"

    async def _boom(engine, conversation_id):
        raise RuntimeError(f"Error code: 401 - Incorrect API key provided: {key}")

    monkeypatch.setattr(tasks, "_reload_state", _boom)

    with caplog.at_level(logging.WARNING):
        tasks._run_body(conversation_id="c1", settings=_settings(openai_api_key=key),
                        model_factory=lambda s: None)

    fails = _payloads(caplog, "summary fail ")
    assert len(fails) == 1
    assert key not in json.dumps(fails, ensure_ascii=False)
    assert "401" in fails[0]["error"]        # ← 确实记了异常本身,不是「什么都不记」
    assert fails[0]["conversation_id"] == "c1"


# --------------------------------------------------------------- engine 释放


class _FakeEngine:
    """只记 `dispose` —— 「engine 有没有被释放」在本模块的观测面就这一个。"""

    def __init__(self):
        self.dispose_calls = 0

    async def dispose(self):
        self.dispose_calls += 1


class _FakeSession:
    """`async with` 的最小替身。

    **为什么连 `async_sessionmaker` 也要换掉**:新版 SQLAlchemy 的
    `async_sessionmaker(bind)` 会校验 `isinstance(bind, AsyncEngine)`,拿假 engine
    去建它**当场就抛** —— 于是 `_run` 根本走不到成功路径,而
    `assert engine.dispose_calls == 1` **照样绿**(异常路径也 dispose)。
    一个「注入的值在被测对象里还会被处理一次」的假绿,正是本仓第 (b) 种形态。
    换了替身之后,下一条用例的 `summary done` 断言才说明它真的走完了。
    """

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _patch_engine(monkeypatch, engine):
    monkeypatch.setattr(tasks, "create_async_engine", lambda *a, **kw: engine)
    monkeypatch.setattr(tasks, "async_sessionmaker", lambda *a, **kw: _FakeSession)


def test_engine_is_disposed_on_the_success_path(monkeypatch, caplog):
    """成功路径也要释放。**这条与下一条合起来才叫「每条退出路径」。**

    光断 `dispose_calls == 1` 不够 —— 异常路径同样会 dispose 一次,所以这里**还**
    断「真的走完了」(`summary done` 有、`summary fail` 没有)。少了后半句,
    成功路径上任何一处提前抛都会被这条断言悄悄放过。
    """
    engine = _FakeEngine()
    _patch_engine(monkeypatch, engine)

    async def _reload(engine_, conversation_id):
        return (0, 5, _HISTORY())

    async def _ok(**kwargs):
        return "梗概"

    monkeypatch.setattr(tasks, "_reload_state", _reload)
    monkeypatch.setattr(tasks, "summarize_range", _ok)

    with caplog.at_level(logging.INFO):
        tasks._run_body(conversation_id="c1", settings=_settings(),
                        model_factory=lambda s: None)

    assert [d["upto_msg_id"] for d in _payloads(caplog, "summary done ")] == [5]
    assert _payloads(caplog, "summary fail ") == []
    assert engine.dispose_calls == 1


def test_engine_is_disposed_when_the_body_fails(monkeypatch):
    """**异常路径最容易漏 `dispose`。**

    漏掉的话每压一次泄漏一个连接池,而摘要任务每轮都可能起 —— 症状是 MySQL
    侧连接数缓慢爬到上限,很久以后才表现为「随机连不上」,而那时没人会想到
    摘要任务。`finally` 少写一处就是这个后果。
    """
    engine = _FakeEngine()
    _patch_engine(monkeypatch, engine)

    async def _boom(engine_, conversation_id):
        raise RuntimeError("读库炸了")

    monkeypatch.setattr(tasks, "_reload_state", _boom)

    tasks._run_body(conversation_id="c1", settings=_settings(),
                    model_factory=lambda s: None)

    assert engine.dispose_calls == 1


# --------------------------------------------------------------- 生命周期日志


def test_start_and_done_carry_the_conversation_id(monkeypatch, caplog):
    """`summary start` / `summary done` 各一行、都带会话 id(spec §7.6)。

    顺带钉住两件事:
    - **不记模型返回的原文** —— 日志是密钥泄漏面也是日志膨胀源(spec §8);
    - `start` 的两端就是**重读到的**两个锚点(不是触发时的)。
    """
    marker = "梗概正文-MARKER-不该进日志"

    async def _ok(**kwargs):
        return marker

    async def _reload(engine, conversation_id):
        return (0, 5, _HISTORY())

    monkeypatch.setattr(tasks, "summarize_range", _ok)
    monkeypatch.setattr(tasks, "_reload_state", _reload)

    with caplog.at_level(logging.INFO):
        tasks._run_body(conversation_id="c1", settings=_settings(),
                        model_factory=lambda s: None)

    starts = _payloads(caplog, "summary start ")
    dones = _payloads(caplog, "summary done ")
    assert len(starts) == 1, "没有 start 行"
    assert len(dones) == 1, "没有 done 行"
    assert starts[0]["conversation_id"] == "c1"
    assert dones[0]["conversation_id"] == "c1"
    assert starts[0]["summary_upto"] == 0
    assert starts[0]["layer1_from"] == 5
    assert dones[0]["upto_msg_id"] == 5      # 覆盖到重读到的上界为止
    assert marker not in caplog.text


def test_trigger_line_carries_the_layer2_usage_and_budget(caplog):
    """`summary trigger` 由**触发方**打 —— 它才是唯一知道层 2 用量与预算的地方。

    本模块只把它收在这里,让五个生命周期节点在同一个模块里说同一套话。
    触发那条断言(验收 2)看的就是这一行里的两个数:只打一个「触发了」的
    半行日志,等于把「层 2 按截短后计数」这件事的观测面又抹掉了。
    """
    with caplog.at_level(logging.INFO):
        tasks.log_trigger(conversation_id="c1", layer2_tokens=120, layer2_budget=100)

    lines = _payloads(caplog, "summary trigger ")
    assert len(lines) == 1
    assert lines[0]["conversation_id"] == "c1"
    assert lines[0]["layer2_tokens"] == 120
    assert lines[0]["layer2_budget"] == 100
