"""观测边界:关掉时全 no-op,且**不 import langfuse**。

这是"单测全程不联网"这条硬约束的守卫。测试的 Settings 不传三个 LANGFUSE_*
⇒ 任何一条用例都不该让 langfuse 被 import 进来。
"""

import os
import re
import sys
from pathlib import Path

import pytest

import app.observability as observability
from app.config import Settings
from app.observability import (
    TagScope,
    enabled,
    intent_scope,
    make_handler,
    span,
    trace_scope,
)


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


class _FakeCM:
    """假的上下文管理器。**不 import langfuse,不发任何东西。**

    `exit_exc` 记下 `__exit__` 收到的异常三元组 —— 这一项是好几条断言的判别力所在:
    单看"异常有没有穿出去"分不清 `except BaseException` 与 `except Exception`
    (两者都不吞),**真正不同的是 `__exit__` 有没有拿到那个异常**。
    """

    def __init__(self):
        self.entered = 0
        self.exited = 0
        self.exit_exc = None

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, *exc):
        self.exited += 1
        self.exit_exc = exc
        return False


class _EnterBoomCM:
    """`__enter__` 就炸 —— 用来验"进入失败要降级,且不许再碰那个 CM"。"""

    def __init__(self):
        self.entered = 0
        self.exited = 0

    def __enter__(self):
        self.entered += 1
        raise RuntimeError("进入失败(模拟 propagate_attributes / 观测客户端炸了)")

    def __exit__(self, *exc):
        self.exited += 1
        return False


class _ExitBoomCM:
    """进得去、出不来 —— 用来验"清理失败不许盖掉业务异常"。"""

    def __init__(self):
        self.entered = 0
        self.exited = 0

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, *exc):
        self.exited += 1
        raise RuntimeError("清理失败")


def test_disabled_when_any_key_missing():
    assert enabled(_settings()) is False
    assert enabled(_settings(langfuse_public_key="pk")) is False
    assert enabled(_settings(langfuse_public_key="pk", langfuse_secret_key="sk")) is True
    # ⚠️ 这一条必须显式传 `langfuse_base_url=""`:它在 config 里有个**非空默认值**,
    # 而本文件每个 Settings 都带 `_env_file=None` ⇒ 不显式清空的话它**永远是满的**,
    # `enabled()` 里 base_url 那一项就**没有任何测试在守**(删掉它套件仍全绿)。
    assert (
        enabled(
            _settings(
                langfuse_public_key="pk",
                langfuse_secret_key="sk",
                langfuse_base_url="",
            )
        )
        is False
    )


def test_disabled_means_no_handler_and_no_langfuse_import():
    s = _settings()
    assert make_handler(s) is None

    with trace_scope(conversation_id="c1", settings=s):
        scope = intent_scope("商品咨询", settings=s)
        scope.enter()
        with span("tool:x", as_type="tool", input={"a": 1}, settings=s) as sp:
            assert sp is None
        scope.exit()

    # ⚠️ **进程级**断言:`sys.modules` 是整个 pytest 进程共享的。
    # T3 起若有别的测试文件 import 了 langfuse,这条会**假红** ——
    # 看到它翻红先查是不是别的文件引进来的,别当成 T2 的回归。
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


# ⚠️ 上面这条**判别力很弱**:它拿的是 disabled 的 settings,被测对象恒为
# `TagScope(None)`,`enter`/`exit` 双双在 `self._cm is None` 上短路 ⇒
# `_entered` 那个标志**一行都没跑到**(把它删掉这条照样绿)。
# 真正守着幂等语义的是下面两条 —— 它们**直接拿假 CM 构造 TagScope**。


def test_tag_scope_enter_and_exit_are_idempotent():
    """重复 `enter()` 不许重复进入,重复 `exit()` 不许重复退出。

    守卫 `_entered` 标志本身:去掉它 ⇒ `cm.entered`/`cm.exited` 都会变成 2。
    """
    cm = _FakeCM()
    scope = TagScope(cm)

    scope.enter()
    scope.enter()         # 重复 enter:不许再 __enter__ 一次
    scope.exit()
    scope.exit()          # 重复 exit:不许再 __exit__ 一次

    assert cm.entered == 1
    assert cm.exited == 1


def test_tag_scope_exit_without_enter_never_touches_cm():
    """没 `enter()` 就 `exit()`:不许抛,且**一次都不许碰那个 CM**。"""
    cm = _FakeCM()
    scope = TagScope(cm)

    scope.exit()

    assert cm.entered == 0
    assert cm.exited == 0


# ── 开启态(enabled 路径)────────────────────────────────────────────────
#
# 上面几条只覆盖关掉态。而 `trace_scope` 真正的**控制流**只在开启态才跑得到,
# 偏偏那儿有过一个静默缺陷:被抓的异常后面**又 yield 一次** ⇒ contextlib 抛
# `RuntimeError: generator didn't stop after throw()`,**把原始业务异常替换掉**。
# 让它可测的唯一办法是把「建外层上下文」那一步抽成 `_outer_cm` 接缝,测试换成假的
# ⇒ 不联网、不 import langfuse,也能验到真实的控制流。


class _BusinessError(Exception):
    """只在本测试里存在的业务异常 —— 用它才能分清「原样穿出」与「被换掉」。"""


class _CancelledError(BaseException):
    """`asyncio.CancelledError` 那一族:**只继承 `BaseException`,不是 `Exception`**。

    实测用的替身(真 `CancelledError` 要跑在事件循环里才拿得到,而本文件不联网、
    不起循环;继承关系一致就足以验 `except BaseException` 这条口径)。
    """


def _patch_seams(monkeypatch, cm):
    """把 enabled 路径的**三条**接缝全换成假的,返回 `_setup_env` 的调用记录。

    三条都要换,少一条就会真的联网 / 真的 import langfuse:

    - `_setup_env` —— 否则会往 `os.environ` 里灌配置;
    - `_outer_cm` —— `trace_scope` 的外层上下文;
    - `_observation_cm` —— `span` 的观测上下文。
    """
    seen = []
    monkeypatch.setattr(
        observability, "_setup_env", lambda settings: seen.append("setup_env")
    )
    monkeypatch.setattr(
        observability, "_outer_cm", lambda settings, conversation_id: cm
    )
    monkeypatch.setattr(
        observability,
        "_observation_cm",
        lambda settings, name, as_type, input: cm,
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
    # 异常**也要原样交给 __exit__** —— 只断言"穿出去了"是不够的(见下面 M-3 那条)。
    assert cm.exit_exc[0] is _BusinessError


def test_trace_scope_enabled_path_propagates_cancelled_error(monkeypatch):
    """`CancelledError` 那一族必须原样穿出 ⇒ 钉住 `except BaseException`。

    ⚠️ **只写 `pytest.raises(_CancelledError)` 是分辨不出来的**:改成
    `except Exception` 时它没被抓住,会顺顺当当地穿出去 —— 两种写法都绿。
    真正分得开的是 **`__exit__` 拿到的参数**:抓得到 ⇒ `exit_exc[0]` 是
    `_CancelledError`;抓不到 ⇒ 是 `None`(因为 `exc_info` 没被赋值)。
    """
    s = _enabled_settings()
    cm = _FakeCM()
    _patch_seams(monkeypatch, cm)

    with pytest.raises(_CancelledError):
        with trace_scope(conversation_id="c1", settings=s):
            raise _CancelledError()

    assert cm.entered == 1
    assert cm.exited == 1
    assert cm.exit_exc[0] is _CancelledError


def test_trace_scope_enabled_path_enters_and_exits_exactly_once(monkeypatch):
    s = _enabled_settings()
    cm = _FakeCM()
    _patch_seams(monkeypatch, cm)

    with trace_scope(conversation_id="c1", settings=s):
        pass

    assert cm.entered == 1
    assert cm.exited == 1
    assert cm.exit_exc == (None, None, None)
    # ⚠️ 进程级断言:别的测试文件一旦 import langfuse,这条会假红(详见文件上方)。
    assert "langfuse" not in sys.modules


def test_trace_scope_enter_failure_really_degrades(monkeypatch):
    """进入失败 ⇒ 真降级:**不许对那个从未进入过的 CM 调 `__exit__`**。

    修之前 `cm` 是**先赋值、后 `__enter__`** 的,`__enter__` 抛了 `cm` 也已经是
    非 None ⇒ 落到正常分支、对一个没进过的 CM 调 `__exit__`。日志嘴上说"降级为
    无观测",行为上却还在用它 —— 降级是假的、日志在说谎。
    (实测 `_AgnosticContextManager.__exit__` 在未进入时会抛
    `RuntimeError: generator didn't stop`,被 finally 吞掉 ⇒ 无业务影响,所以
    这个缺陷**只能靠 `exited == 0` 这种断言抓**。)
    """
    s = _enabled_settings()
    cm = _EnterBoomCM()
    _patch_seams(monkeypatch, cm)

    with trace_scope(conversation_id="c1", settings=s):
        pass          # 降级之后 body 必须照常执行

    assert cm.entered == 1
    assert cm.exited == 0, "进入失败之后不许再对这个 CM 调 __exit__"


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
    # ⚠️ 进程级断言(本文件第三处):T3 起若别的测试 import 了 langfuse 会假红。
    assert "langfuse" not in sys.modules, "这条用例也不许把 langfuse 引进单测进程"


# ── `span` 的 enabled 路径 ───────────────────────────────────────────────
#
# `span` 将来是**工具执行**(`app/tools/executor.py`)与**知识检索**
# (`app/retrieval/search.py`)唯一能用的观测入口 —— 它自己的吞异常/降级分支
# 必须有测试。在此之前本文件只在 disabled 下调过它,那两条分支一行都没跑到。


def test_span_enabled_path_enters_and_exits_exactly_once(monkeypatch):
    s = _enabled_settings()
    cm = _FakeCM()
    _patch_seams(monkeypatch, cm)

    with span("tool:x", as_type="tool", input={"a": 1}, settings=s) as sp:
        assert sp is cm            # 开启态 yield 的是真 handle,不是 None

    assert cm.entered == 1
    assert cm.exited == 1
    assert cm.exit_exc == (None, None, None)


def test_span_enter_failure_yields_none_and_body_still_runs(monkeypatch):
    """进入失败 ⇒ yield None、**不抛**,body 照常执行,且不许再碰那个 CM。"""
    s = _enabled_settings()
    cm = _EnterBoomCM()
    _patch_seams(monkeypatch, cm)
    ran = []

    with span("tool:x", settings=s) as sp:
        assert sp is None
        ran.append(True)

    assert ran == [True], "观测进不去绝不许拦住业务"
    assert cm.entered == 1
    assert cm.exited == 0


def test_span_business_exception_survives_cleanup_failure(monkeypatch):
    """**业务异常 + 清理也失败** ⇒ 出去的必须是业务异常本身。

    两个都错的时候最容易出的事就是清理的异常把业务的盖掉 —— 那正是本仓反复记的
    "报错指向别处"。这里同时钉住:清理失败被吞成 warning,而 `_BusinessError`
    原样穿出。
    """
    s = _enabled_settings()
    cm = _ExitBoomCM()
    _patch_seams(monkeypatch, cm)

    with pytest.raises(_BusinessError):
        with span("tool:x", settings=s):
            raise _BusinessError("业务异常必须原样穿出去")

    assert cm.entered == 1
    assert cm.exited == 1


def test_span_cleanup_failure_does_not_mask_normal_exit(monkeypatch):
    """清理失败但 body 正常 ⇒ 不许抛(`__exit__` 的异常一律吞成 warning)。"""
    s = _enabled_settings()
    cm = _ExitBoomCM()
    _patch_seams(monkeypatch, cm)

    with span("tool:x", settings=s) as sp:
        assert sp is cm

    assert cm.exited == 1


def test_no_module_outside_observability_imports_langfuse():
    """纪律 #1:`app/` 下除本模块外**零处** import langfuse。

    这是本章最关键的一条接缝 —— 别的模块只认那四个名字,内置工具与 MCP 工具
    在注册表里长得一模一样这件事,靠的就是这条边界。
    用**源码扫描**而不是运行时断言:要抓的是"有没有人写下去",不是"跑到了没有"。
    """
    app_dir = Path(observability.__file__).parent
    pattern = re.compile(r"^\s*(?:import|from)\s+langfuse\b", re.M)
    offenders = sorted(
        p.relative_to(app_dir).as_posix()
        for p in app_dir.rglob("*.py")
        if p.name != "observability.py" and pattern.search(p.read_text(encoding="utf-8"))
    )
    assert offenders == [], f"只有 app/observability.py 可以 import langfuse,但还发现:{offenders}"
