"""观测边界:关掉时全 no-op,且**不 import langfuse**。

这是"单测全程不联网"这条硬约束的守卫。测试的 Settings 不传三个 LANGFUSE_*
⇒ 任何一条用例都不该让 langfuse 被 import 进来。
"""

import sys

import pytest

from app.config import Settings
from app.observability import enabled, intent_scope, make_handler, span, trace_scope


def _settings(**over):
    base = {
        "openai_base_url": "http://x", "openai_api_key": "k",
        "openai_model": "m", "database_url": "mysql://x",
    }
    return Settings(**{**base, **over}, _env_file=None)


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
