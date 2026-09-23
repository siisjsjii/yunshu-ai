"""观测边界:关掉时全 no-op,且**不 import langfuse**。

这是"单测全程不联网"这条硬约束的守卫。测试的 Settings 不传三个 LANGFUSE_*
⇒ 任何一条用例都不该让 langfuse 被 import 进来。
"""

import os
import sys

import pytest

import app.observability as observability
from app.config import Settings
from app.observability import enabled, intent_scope, make_handler, span, trace_scope


def _settings(**over):
    base = {
        "openai_base_url": "http://x", "openai_api_key": "k",
        "openai_model": "m", "database_url": "mysql://x",
    }
    return Settings(**{**base, **over}, _env_file=None)


def _enabled_settings(**over):
    """三个 LANGFUSE_* 齐了 ⇒ `enabled()` 为真 ⇒ 走**开启态**路径。"""
    return _settings(
        langfuse_public_key="pk", langfuse_secret_key="sk", **over
    )


def test_disabled_when_any_key_missing():
    assert enabled(_settings()) is False
    assert enabled(_settings(langfuse_public_key="pk")) is False
    assert enabled(_settings(langfuse_public_key="pk", langfuse_secret_key="sk")) is True


def test_disabled_means_no_handler_and_no_langfuse_import():
    s = _settings()
    assert make_handler(s) is None

    with trace_scope(conversation_id="c1", settings=s):
        scope = intent_scope("商品咨询", settings=s)
        scope.enter()
        with span("tool:x", as_type="tool", input={"a": 1}, settings=s) as sp:
            assert sp is None
        scope.exit()

    assert "langfuse" not in sys.modules, "关掉时不许把 langfuse import 进来"


def test_span_never_raises_even_if_body_raises():
    """观测挂掉绝不许影响业务 —— 与 app/tools/audit.py 的 record_audit 同族。"""
    s = _settings()
    with pytest.raises(ValueError):
        with span("tool:x", settings=s):
            raise ValueError("业务异常必须原样穿出去")


def test_intent_scope_is_idempotent_and_safe_without_enter():
    s = _settings()
    scope = intent_scope("物流", settings=s)
    scope.exit()          # 没 enter 就 exit,不许炸
    scope.enter()
    scope.enter()         # 重复 enter,不许炸
    scope.exit()


# ── 开启态(enabled 路径)────────────────────────────────────────────────
#
# 上面几条只覆盖关掉态。而 `trace_scope` 真正的**控制流**只在开启态才跑得到,
# 偏偏那儿有过一个静默缺陷:被抓的异常后面**又 yield 一次** ⇒ contextlib 抛
# `RuntimeError: generator didn't stop after throw()`,**把原始业务异常替换掉**。
# 让它可测的唯一办法是把「建外层上下文」那一步抽成 `_outer_cm` 接缝,测试换成假的
# ⇒ 不联网、不 import langfuse,也能验到真实的控制流。


class _FakeCM:
    """假的外层上下文。**不 import langfuse,不发任何东西。**"""

    def __init__(self):
        self.entered = 0
        self.exited = 0

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, *exc):
        self.exited += 1
        return False


class _BusinessError(Exception):
    """只在本测试里存在的业务异常 —— 用它才能分清「原样穿出」与「被换掉」。"""


def _patch_seams(monkeypatch, cm):
    """把 enabled 路径的两条接缝都换成假的,返回 `_setup_env` 的调用记录。

    两条都要换:`_setup_env`(否则会往 os.environ 里灌配置)、`_outer_cm`
    (否则会真的 import langfuse 并建客户端)。
    """
    seen = []
    monkeypatch.setattr(
        observability, "_setup_env", lambda settings: seen.append("setup_env")
    )
    monkeypatch.setattr(
        observability, "_outer_cm", lambda settings, conversation_id: cm
    )
    return seen


def test_trace_scope_enabled_path_propagates_business_exception(monkeypatch):
    """**原缺陷的守卫**:业务异常必须原样穿出。

    原写法(异常被抓后再 yield 一次)下,这里拿到的是
    `RuntimeError: generator didn't stop after throw()`;若改成"只吞不抛",
    异常会被 contextlib 整个吞掉 ⇒ `pytest.raises` 落空。两种错法这条都红。
    """
    s = _enabled_settings()
    cm = _FakeCM()
    _patch_seams(monkeypatch, cm)

    with pytest.raises(_BusinessError):
        with trace_scope(conversation_id="c1", settings=s):
            raise _BusinessError("业务异常必须原样穿出去")

    assert cm.entered == 1
    assert cm.exited == 1


def test_trace_scope_enabled_path_enters_and_exits_exactly_once(monkeypatch):
    s = _enabled_settings()
    cm = _FakeCM()
    _patch_seams(monkeypatch, cm)

    with trace_scope(conversation_id="c1", settings=s):
        pass

    assert cm.entered == 1
    assert cm.exited == 1
    assert "langfuse" not in sys.modules


def test_trace_scope_enabled_path_does_not_touch_os_environ(monkeypatch):
    """`_setup_env` 被换成假的 ⇒ 真 `_setup_env` 一次都没跑 ⇒ `os.environ` 未被污染。

    `seen` 那条断言顺带钉住「enabled 路径**确实**会推配置」—— 少了它,
    这两条断言在"根本没走到那条路径"的实现下也会通过(恒真的假绿)。
    """
    s = _enabled_settings()
    cm = _FakeCM()
    seen = _patch_seams(monkeypatch, cm)
    before = dict(os.environ)

    with trace_scope(conversation_id="c1", settings=s):
        pass

    assert seen == ["setup_env"], "enabled 路径必须先推配置,再建外层上下文"
    assert dict(os.environ) == before, "不许把配置灌进单测进程的 os.environ"
    assert "langfuse" not in sys.modules, "这条用例也不许把 langfuse 引进单测进程"
