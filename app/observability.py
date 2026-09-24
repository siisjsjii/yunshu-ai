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

⚠️ **三条实测出来的坑,改动前先读 spec §3.4 / §15.4:**

- `LangfuseSpan.update(**kwargs)` 的 kwargs 被**静默丢弃**
  (源码 docstring 逐字:`**kwargs: Additional keyword arguments (ignored)`)。
  ⇒ **不要**用 `span.update(**{"langfuse.trace.tags": [...]})` 设 trace 属性。
- `langfuse.propagate_attributes(...)` 返回的 `_AgnosticContextManager`
  **没有 `__aenter__`** ⇒ 中途进入只能用**同步** `__enter__`。
- **只调 `propagate_attributes` 而不开"当前 span"的话,每个观测各自成一条 trace。**
  `start_as_current_observation` 的父级取自 **OTel 当前 span**,而 Langfuse 的
  LangChain 回调**不把观测挂成 current**(它靠 LangChain 的 run tree 定父子)⇒
  没有当前 span 时,回调建的观测与我们的手工 span **全都各自开新 trace**
  (`traceId` 互不相同、`parentObservationId` 全是 `null`)。
  ⇒ **`trace_scope` 里的根观测(`_root_cm`)是必需品,不是装饰**(T3 真机实测)。
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


def _root_cm(settings: Settings, conversation_id: str) -> Any:
    """建**根观测**(`chat`)。**单独抽出来是为了能测**,理由同 `_outer_cm`。

    ⚠️ **它不是一个装饰**:T3 的第一次真机冒烟实测到,只调
    `propagate_attributes` 而**不开当前 span** 时,**所有观测都各自成一条 trace**
    —— 请求那条 trace 上只有 LangChain 回调建的那些,手工 span 各自开新 trace,
    `traceId` 互不相同、`parentObservationId` 全是 `null`。成因:
    `start_as_current_observation` 的父级取自 **OTel 当前 span**,而 Langfuse 的
    LangChain 回调**不把观测挂成 current**(它靠 LangChain 的 run tree 定父子)⇒
    没有"当前 span"就没有东西可挂。

    ⚠️ **这句话一度写成「`retrieval` / `tool:*` 这两个手工 span」—— `tool:*` 已按
    spec §15.5 删除**(内置工具本来就有 LangChain 回调建的 `TOOL` 观测,手工那条是
    **同一事件表示两遍**)。**今天全仓手工 span 只剩 `retrieval` 一种形状**
    (`app/agent/nodes.py` 与 `app/agent/refund_nodes.py` 各一处),
    `tests/test_agent_refund.py` 有一条用例钉住「整轮里只有这一条手工 span」。

    ⇒ 有了它,回调建的观测与手工 span 才会落进**同一条** trace、`chat` 之下。

    `input` 只放 `conversation_id`:**用户原话由端点决定要不要放**(谁手里有谁放),
    这一层保持最小 —— 与 `journal` 那几行日志同一个口径:只记这一层真的知道的。
    """
    from langfuse import get_client

    return get_client().start_as_current_observation(
        name="chat", as_type="span", input={"conversation_id": conversation_id}
    )


@contextmanager
def trace_scope(*, conversation_id: str, settings: Settings) -> Iterator[None]:
    """包住整段流:开**根观测**(`chat`)+ 把会话 id 挂到 trace 上。

    两层,**顺序是语义的一部分**:

    ```
    propagate_attributes(trace_name, session_id)   ← 外层:trace 级属性
    └── start_as_current_observation("chat")       ← 内层:根观测 + **当前 span**
        └── 整段业务流(图、回调建的观测、手工 span)
    ```

    外层在里层**外侧**是刻意的:trace 级属性要能传播给根观测**及其全部孩子**。
    反过来(trace 属性套在根观测里面)的话,属性会在根观测之后才生效 ——
    根观测自己就不带 session。

    内层那个根观测是 T3 冒烟实测出来的**必需品**,不是装饰:没有"当前 span"时,
    Langfuse 的 LangChain 回调与我们的手工 span **各自成一条 trace**(详见 `_root_cm`)。

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

    outer = None
    root = None
    outer_in = False
    entered = False
    try:
        _setup_env(settings)
        outer = _outer_cm(settings, conversation_id)
        outer.__enter__()
        outer_in = True            # ← 只有 `__enter__` **返回了**才算进了
        root = _root_cm(settings, conversation_id)
        root.__enter__()
        entered = True             # ← **两层都进成功之后才算"进了"**
    except Exception:  # noqa: BLE001
        logger.warning("langfuse trace_scope 进入失败,降级为无观测", exc_info=True)
        # ⚠️ **进了一半必须退回去**:外层已经 `__enter__` 过、而建根观测那步抛了
        # ⇒ 不 unwind 的话 `propagate_attributes` 的 token **一直挂在这个任务上**,
        # 后续观测会继续被当成它的孩子(跨请求串味)。
        # (根只可能在外层**之后**进入,所以这里只需要回收外层 —— 根没进过就没东西要收。)
        if outer_in:
            _safe_exit(outer, "trace_scope 降级时回收外层上下文失败")

    if not entered:
        # ⚠️ 判据是 `entered` 而不是 `outer is not None`:`__enter__` 抛了的话
        # `outer` 早已被赋值,拿它当"进了"会落到正常分支,**对一个从未进入过的 CM
        # 调 __exit__**(实测 `_AgnosticContextManager.__exit__` 在未进入时会抛
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
        # 退出顺序与进入**相反**(根在外层里面)。
        _safe_exit(root, "langfuse trace_scope 退出根观测失败", exc_info)
        _safe_exit(outer, "langfuse trace_scope 退出失败", exc_info)


def _safe_exit(cm: Any, message: str, exc_info=(None, None, None)) -> None:
    """`__exit__` 失败一律吞成 warning —— 观测不许盖掉业务异常。"""
    try:
        cm.__exit__(*exc_info)
    except Exception:  # noqa: BLE001
        logger.warning(message, exc_info=True)


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
    `span` 今天的唯一使用方是**知识检索**(见 `span` 的 docstring),它自己的
    吞异常/降级分支因此必须有测试。
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

    **它存在的理由**:Langfuse 的 LangChain 回调只覆盖 **LangChain 的 run**。
    本项目的**知识检索**(`app/retrieval/search.py:KnowledgeRetriever`)是一个自写的
    普通类、**不是** LangChain run ⇒ 没有这个手工 span,知识检索在界面上**是空的**
    (`retrieval` 因此是今天唯一的使用方)。

    ⚠️ **不要按「工具执行也要靠它」来理解** —— 这句话一度是这么写的,**与实测相反**:
    `app/tools/executor.py:execute_tool` 里的 `spec.tool.ainvoke(...)` **是**一个
    LangChain run ⇒ 回调**已经**给了它一条嵌套正确的 `TOOL 'query_order'`
    (在 `agent` 之下)。手工再开一条是**同一个事件表示两遍**,已按 spec §15.5 删除
    (Langfuse 自己的最佳实践原话:`Don't emit duplicate dispatch + execution nodes`)。

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
