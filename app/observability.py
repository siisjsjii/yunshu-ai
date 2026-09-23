"""Langfuse 观测 —— **全章唯一**的边界。

三条纪律,**每条都有测试守着**(`tests/test_observability.py`;括号里是守着它的用例):

1. **`app/` 下除本模块外,零处 import langfuse。** 别的模块只认
   `trace_scope` / `intent_scope` / `span` / `make_handler` 四个名字。
   (`test_no_module_outside_observability_imports_langfuse` —— 源码扫描)
2. **关掉时全 no-op,且不 import langfuse。** 三个 `LANGFUSE_*` 任一为空
   ⇒ `enabled()` 为假 ⇒ 所有函数走空壳分支。**langfuse 的 import 一律放在
   函数体内**,顶层 import 会让"没配"的环境在导入期就炸。
   (`test_disabled_*` / `test_disabled_means_no_handler_and_no_langfuse_import`)
3. **本模块不抛异常**(`span` 的 `__exit__` 吞掉一切并 `logger.warning`)——
   与 `app/tools/audit.py:record_audit` 的"永不抛"同族。观测不许影响业务。
   (`test_span_enabled_path_*` / `test_trace_scope_enabled_path_*` ——
   开启态的降级与吞异常分支都有用例,不只是关掉态的空壳分支)

> ⚠️ 上面第 1 条是**源码**扫描,不是运行时断言:`__pycache__` 之外只看 `*.py` 的正文。
> 它抓的是"有没有人**写下**这行 import",而不是"这次跑到没跑到" ——
> 这正是那条纪律的字面意思。

⚠️ **两条实测出来的坑,改动前先读 spec §3.4:**

- `LangfuseSpan.update(**kwargs)` 的 kwargs 被**静默丢弃**
  (源码 docstring 逐字:`**kwargs: Additional keyword arguments (ignored)`)。
  ⇒ **不要**用 `span.update(**{"langfuse.trace.tags": [...]})` 设 trace 属性。
- `langfuse.propagate_attributes(...)` 返回的 `_AgnosticContextManager`
  **没有 `__aenter__`** ⇒ 中途进入只能用**同步** `__enter__`。
"""

import logging
from contextlib import contextmanager
from typing import Any, Iterator

from app.config import Settings

logger = logging.getLogger(__name__)


def enabled(settings: Settings) -> bool:
    """三个 LANGFUSE_* 齐了才算开着。缺一个 ⇒ 整套观测 no-op。"""
    return bool(
        settings.langfuse_public_key
        and settings.langfuse_secret_key
        and settings.langfuse_base_url
    )


def _setup_env(settings: Settings) -> None:
    """把配置灌进环境变量 —— langfuse 客户端只认 env。"""
    import os

    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
    os.environ.setdefault("LANGFUSE_BASE_URL", settings.langfuse_base_url)


def make_handler(settings: Settings) -> Any | None:
    """建一个 LangChain 回调。关掉时返回 None。

    ⚠️ 实测 4.15.4 的签名只有 `(*, public_key=None, trace_context=None)` ——
    文档里那套 `session_id=` / `user_id=` / `tags=` 构造参数是 **JS SDK** 的。
    会话与标签一律走 `trace_scope` / `intent_scope`,不要往这里塞。
    """
    if not enabled(settings):
        return None
    try:
        _setup_env(settings)
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()
    except Exception:  # noqa: BLE001 —— 观测建不起来不许拦住业务
        logger.warning("langfuse handler 构造失败,观测本次关闭", exc_info=True)
        return None


def _outer_cm(settings: Settings, conversation_id: str) -> Any:
    """建外层 trace 上下文。**单独抽出来是为了能测** —— 测试把它换成假的,
    于是 enabled 路径可以在不联网、不 import langfuse 的前提下被验到。
    """
    from langfuse import propagate_attributes

    return propagate_attributes(trace_name="cs-chat", session_id=conversation_id)


@contextmanager
def trace_scope(*, conversation_id: str, settings: Settings) -> Iterator[None]:
    """包住整段流,把会话 id 挂到 trace 上。

    实测:`propagate_attributes(session_id=...)` 在**这一层**是有效的,
    而且它覆盖**全部**观测(包括进入之前创建的 —— 因为它是外层)。

    ⚠️ **每条路径只允许 `yield` 一次** —— 这是 `@contextmanager` 的硬约束:
    业务异常是被 `throw()` 进生成器的,若被 `except` 抓住后再 `yield` 一次,
    contextlib 会抛 `RuntimeError: generator didn't stop after throw()`,
    **把原始业务异常替换掉**(本仓记过的"报错指向别处"那一类)。
    所以正常路径与降级路径**各自 yield 一次**,异常一律原样穿出 ——
    形状与下面的 `span` 对齐。
    """
    if not enabled(settings):
        yield
        return

    cm = None
    entered = False
    try:
        _setup_env(settings)
        cm = _outer_cm(settings, conversation_id)
        cm.__enter__()
        entered = True             # ← **只有 __enter__ 成功之后才算"进了"**
    except Exception:  # noqa: BLE001
        logger.warning("langfuse trace_scope 进入失败,降级为无观测", exc_info=True)

    if not entered:
        # ⚠️ 判据是 `entered` 而不是 `cm is not None`:`__enter__` 抛了的话 `cm`
        # 早已被赋值,拿它当"进了"会落到正常分支,**对一个从未进入过的 CM 调
        # __exit__**(实测 `_AgnosticContextManager.__exit__` 在未进入时会抛
        # `RuntimeError: generator didn't stop`)—— 虽被 finally 吞掉、无业务影响,
        # 但"降级"是假的、日志在说谎。
        yield                      # ← 降级路径:**只 yield 这一次**
        return

    exc_info = (None, None, None)
    try:
        yield                      # ← 正常路径:**只 yield 这一次**
    except BaseException:          # noqa: BLE001 —— CancelledError 是 BaseException
        exc_info = __import__("sys").exc_info()
        raise                      # 业务异常原样穿出去
    finally:
        try:
            cm.__exit__(*exc_info)
        except Exception:  # noqa: BLE001
            logger.warning("langfuse trace_scope 退出失败", exc_info=True)


class TagScope:
    """**可以中途进入**的标签作用域。

    `propagate_attributes` 是个上下文管理器,而意图要 `classify_intent` 跑完
    才知道(那时 `astream` 已经在跑了)⇒ 必须能手动 enter/exit。

    实测语义:**进入之后新建的观测带 tag,之前的没有**(spec §3.4-2)。
    开销的大头是意图分类**之后**的 agent 轮次,所以这个语义够用。
    """

    def __init__(self, cm: Any | None = None) -> None:
        self._cm = cm
        self._entered = False

    def enter(self) -> None:
        if self._cm is None or self._entered:
            return
        try:
            self._cm.__enter__()          # ← 同步,没有 __aenter__
            self._entered = True
        except Exception:  # noqa: BLE001
            logger.warning("langfuse intent tag 进入失败", exc_info=True)

    def exit(self) -> None:
        if self._cm is None or not self._entered:
            return
        try:
            self._cm.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            logger.warning("langfuse intent tag 退出失败", exc_info=True)
        finally:
            self._entered = False


def intent_scope(intent: str, *, settings: Settings) -> TagScope:
    """给 trace 打 `intent:<x>` 标签 —— 按意图统计花销就靠它(spec §3.5)。"""
    if not enabled(settings) or not intent:
        return TagScope(None)
    try:
        _setup_env(settings)
        from langfuse import propagate_attributes

        return TagScope(propagate_attributes(tags=["ch09", f"intent:{intent}"]))
    except Exception:  # noqa: BLE001
        logger.warning("langfuse intent_scope 构造失败", exc_info=True)
        return TagScope(None)


def _observation_cm(settings: Settings, name: str, as_type: str, input: Any) -> Any:
    """建一个手工观测上下文。**单独抽出来是为了能测** —— 理由与 `_outer_cm` 相同:
    测试把它换成假的,enabled 路径就能在不联网、不 import langfuse 的前提下被验到。
    `span` 将来要承担工具执行与知识检索的观测,它自己的吞异常/降级分支必须有测试。
    """
    from langfuse import get_client

    return get_client().start_as_current_observation(
        name=name, as_type=as_type, input=input
    )


@contextmanager
def span(
    name: str, *, as_type: str = "span", input: Any = None, settings: Settings
) -> Iterator[Any | None]:
    """手工开一个观测。关掉时 yield None。

    **它存在的理由**:Langfuse 的 LangChain 回调只覆盖 LangChain 的 run。
    本项目的**工具执行**(`app/tools/executor.py:execute_tool`)与**知识检索**
    (`app/retrieval/search.py:KnowledgeRetriever`)都不是 LangChain run,
    一个 span 都不会自动出现 —— 而"每个节点的工具调用、检索结果都能铺开看"
    这条需求,只有这里能落地。

    实测可用的 `as_type`:`span` / `generation` / `agent` / `tool` / `chain` /
    `retriever` / `evaluator` / `guardrail` / `embedding`。
    """
    if not enabled(settings):
        yield None
        return
    cm = None
    try:
        _setup_env(settings)
        cm = _observation_cm(settings, name, as_type, input)
        handle = cm.__enter__()    # ← 抛了就直接落进下面的 except,不会再碰这个 cm
    except Exception:  # noqa: BLE001
        logger.warning("langfuse span 进入失败", exc_info=True)
        yield None
        return

    exc_info = (None, None, None)
    try:
        yield handle
    except BaseException:  # noqa: BLE001 —— 业务异常原样穿出去
        exc_info = __import__("sys").exc_info()
        raise
    finally:
        try:
            cm.__exit__(*exc_info)
        except Exception:  # noqa: BLE001
            logger.warning("langfuse span 退出失败", exc_info=True)
