"""执行引擎的三道闸:权限、参数校验、重试规则。

⚠️ 本文件的**关键写法**:注入的是**未加工的输入**(裸 args、裸异常),
不是「已经被处理好的值」—— 本仓栽过三次的那类假绿就是
「测试把处理之后的形态喂给被测对象,于是处理那一步永远不被验」。
"""

import asyncio
import dataclasses
import json

import pytest
from langchain_core.tools import tool

from app.tools import executor
from app.tools.errors import ToolInfrastructureError, TransientToolError
from app.tools.executor import (
    ERROR_CONFIRMATION_REQUIRED,
    ERROR_INVALID_ARGS,
    ERROR_PERMISSION_DENIED,
    ERROR_TIMEOUT,
    APPROVED,
    DENIED,
    PENDING,
    execute_tool,
)
from app.tools.spec import READ, WRITE, ToolSpec


class _Settings:
    # ⚠️ **超时用真实的小数字,绝不要 monkeypatch `asyncio.sleep`。**
    # `executor.asyncio` **就是** `asyncio` 模块本身 —— `monkeypatch.setattr(
    # executor.asyncio, "sleep", ...)` 会把它**全局**换掉,于是被测工具里那句
    # `await asyncio.sleep(10)` **立刻返回**、`wait_for` 根本不超时,测试会以
    # 一个看不懂的方式红。这是「替身把被测行为整个取消掉了」——
    # 本仓第 (g) 类假绿的反面:**不是假绿,是假红**。
    # (初稿正是这么写的,已订正。)
    tool_timeout_seconds = 0.01
    tool_retry_attempts = 2
    tool_retry_delay_seconds = 0.0
    tool_result_max_tokens = 1200


#: 没传 `tool` 时的兜底 schema(只有「闸在前、根本走不到校验」的那几条用到)。
_FALLBACK_SCHEMA = {
    "type": "object",
    "properties": {"x": {"type": "string"}},
    "required": ["x"],
}


def _spec(*, name="demo", kind=READ, schema=None, tool=None) -> ToolSpec:
    """造一条登记项。

    ⚠️ **`schema` 不给时从 `tool` 派生**(与 `registry._spec_from_tool` 同款:
    `args_schema.model_json_schema()`)。初稿这里是一个固定要求字段 `x` 的
    默认 schema —— 于是「工具的入参」与「校验用的 schema」说的是**两件事**:
    `test_invalid_args_*` 要断的「文案点名到字段」会点在 `x` 上(而不是
    `order_id`),而 `test_approved_write_is_executed_once`、
    `test_first_try_success_reports_zero_retries` 这类会**先**被校验闸拦下,
    以「工具一次没被调用 / 种类是 invalid_args」的样子红 —— 那种红与它们
    真正要验的闸(权限、重试)**毫无关系**,是本仓记过的「报错指向别处」。
    """
    if schema is None:
        args_schema = getattr(tool, "args_schema", None) if tool is not None else None
        schema = args_schema.model_json_schema() if args_schema else _FALLBACK_SCHEMA
    return ToolSpec(
        name=name,
        description="测试用",
        input_schema=schema,
        kind=kind,
        source="builtin",
        tool=tool,
    )


def _tc(name: str, args: dict) -> dict:
    """造一条 `tool_call`。

    ⚠️ **`"type": "tool_call"` 这个键必须在**(CLAUDE.md 的硬约束,不是风格):
    `BaseTool.ainvoke` 判「这是不是一次工具调用」**只看**它 —— 缺键时它把
    整个 dict 当成**参数**去校验工具 schema,于是每次调用都返回一条
    「参数不合法」的**可恢复**失败。症状是「闸全对、工具一次没跑起来」,
    而报错指向 pydantic 的 `Field required` —— 与真正的原因毫无相似之处。
    初稿的内联 dict 全都缺这个键,六条用例因此红在上面那句 pydantic 报错上。
    """
    return {"name": name, "args": args, "id": "c1", "type": "tool_call"}


async def _collect(sink, kwargs):
    sink.append(kwargs)


class _CountingTool:
    """计次壳:把「执行器**尝试**了几次」记在 **`ainvoke` 边界**上。

    ⚠️ **计数绝不能写在工具体里**(CLAUDE.md 的硬约束,本仓栽过):
    `@tool` 包装后的 pydantic 校验发生在函数体**之前** —— 参数不合法时函数体
    根本不进入,按函数体计数恒为 0。于是「有前置校验闸」与「没有前置闸、
    靠循环里 `except ValidationError` 兜底」这两种实现给出**同一个观测值**,
    断言零判别力(初稿就是这么写的,审查者指出后订正)。

    计在 `ainvoke` 上才问得出那个真问题:**校验闸到底在 `ainvoke` 之前还是之后?**
    """

    def __init__(self, inner, counter: dict):
        self._inner = inner
        self._counter = counter

    async def ainvoke(self, tool_call):
        self._counter["n"] += 1
        return await self._inner.ainvoke(tool_call)


def _with_counter(spec, counter: dict) -> ToolSpec:
    """照 `spec` 造一份、只把 `tool` 换成计次壳。

    **先建 spec 再替换** —— `_spec` 要从真工具派生 `input_schema`
    (`args_schema`),而计次壳没有那个属性。
    """
    return dataclasses.replace(spec, tool=_CountingTool(spec.tool, counter))


# ---- 闸 1:权限 ---------------------------------------------------------


@pytest.mark.anyio
async def test_pending_write_is_not_executed():
    """未确认的写调用:**不执行**。用真工具来证 —— 计数器在函数体里,
    它必须**一次都没被调用**。"""
    calls = []

    @tool
    async def create_ticket(description: str) -> str:
        """建单。"""
        calls.append(description)
        return "ok"

    spec = _spec(name="create_ticket", kind=WRITE, tool=create_ticket)
    outcome = await execute_tool(
        tool_call=_tc("create_ticket", {"description": "坏了"}),
        registry={"create_ticket": spec},
        settings=_Settings(),
        conversation_id="c1",
        write_decision=PENDING,      # 显式写出默认值:Agent 那条路从不传别的
    )
    assert calls == []
    assert outcome.ok is False
    assert outcome.error_kind == ERROR_CONFIRMATION_REQUIRED


@pytest.mark.anyio
async def test_pending_write_preview_carries_the_args():
    """预览载荷 = 写操作的入参(前端要拿它渲染「工单类型 + 问题描述」)。"""
    spec = _spec(name="create_ticket", kind=WRITE)
    outcome = await execute_tool(
        tool_call=_tc("create_ticket", {"description": "耳机坏了", "ticket_type": "售后"}),
        registry={"create_ticket": spec},
        settings=_Settings(),
        conversation_id="c1",
        write_decision=PENDING,
    )
    assert outcome.preview == {"description": "耳机坏了", "ticket_type": "售后"}


@pytest.mark.anyio
async def test_pending_write_is_not_audited(monkeypatch):
    """拦截**不是**拒绝 —— 那一刻没有任何人拒绝任何事。

    落了审计的话,验收 5 要的「这条 create_ticket 状态是权限拒绝」
    会变成「其中一行是」,断言从唯一事实退化成含糊。
    """
    seen: list[dict] = []
    spec = _spec(name="create_ticket", kind=WRITE)
    monkeypatch.setattr(executor, "record_audit", lambda **kw: _collect(seen, kw))
    await execute_tool(
        tool_call=_tc("create_ticket", {"x": "1"}),
        registry={"create_ticket": spec},
        settings=_Settings(),
        conversation_id="c1",
        write_decision=PENDING,
    )
    await asyncio.sleep(0)          # 给可能存在的 fire-and-forget 一点机会
    assert seen == []


@pytest.mark.anyio
@pytest.mark.parametrize("raw", ["坏了", ["a"], 42])
async def test_pending_write_with_malformed_args_still_returns_recoverably(raw):
    """**畸形 `args` 不许把 `execute_tool` 炸穿。**

    写操作 + 待确认是唯一一条**根本走不到 `validate_args`** 的路径(校验闸在
    权限闸**之后**),于是它是这套分类学唯一的裸露面:`preview=dict(args)` 在
    `args` 不是 dict 时当场抛 `ValueError`/`TypeError`,而那个位置没有任何
    handler 罩着 ⇒ 异常逃出 `execute_tool` ⇒ 500,而不是一条可恢复结果。

    ⚠️ 断言的是「**正常返回**一条 `confirmation_required`」,**不是**「抛不抛」
    —— 后者写成 `pytest.raises(Exception)` 会把一个真 bug 也算过。
    """
    spec = _spec(name="create_ticket", kind=WRITE)
    outcome = await execute_tool(
        tool_call={"name": "create_ticket", "id": "c1",
                   "args": raw, "type": "tool_call"},
        registry={"create_ticket": spec},
        settings=_Settings(),
        conversation_id="c1",
        write_decision=PENDING,
    )
    assert outcome.ok is False
    assert outcome.error_kind == ERROR_CONFIRMATION_REQUIRED
    # 预览载荷必须是**能 JSON 序列化的 dict**(前端要拿它渲染卡片):
    # 原样塞一个 list/str 进去,坏在客户端而不是这里。
    assert isinstance(outcome.preview, dict)
    json.dumps(outcome.preview, ensure_ascii=False)


@pytest.mark.anyio
async def test_denied_write_is_not_executed_but_is_audited(monkeypatch):
    """取消:不执行 + **落 `permission_denied`**(验收 5 断的就是它)。"""
    calls = []
    seen: list[dict] = []

    @tool
    async def create_ticket(description: str) -> str:
        """建单。"""
        calls.append(description)
        return "ok"

    spec = _spec(name="create_ticket", kind=WRITE, tool=create_ticket)
    monkeypatch.setattr(executor, "record_audit", lambda **kw: _collect(seen, kw))
    outcome = await execute_tool(
        tool_call=_tc("create_ticket", {"description": "坏了"}),
        registry={"create_ticket": spec},
        settings=_Settings(),
        conversation_id="c1",
        write_decision=DENIED,
    )
    assert calls == []
    assert outcome.error_kind == ERROR_PERMISSION_DENIED
    assert [s["status"] for s in seen] == ["permission_denied"]


@pytest.mark.parametrize("bad", ["Approved", "approve", "yes", "ON", None, 1])
@pytest.mark.anyio
async def test_unknown_write_decision_raises_instead_of_executing(monkeypatch, bad):
    """**认不出的决议 ⇒ 响亮地抛,不当成「拒绝」也不执行。**

    闸 1 若写成「`== PENDING` 拦、`== DENIED` 拦、**其余一律放行**」,那么
    任何既不是 `"pending"` 也不是 `"denied"` 的值(拼错的 `"Approved"`、
    半接线调用方传的 `None`)都会**无确认、无审计地执行一次不可逆的写**。
    写操作是本章唯一不该猜的地方。

    ⚠️ 也**不许**改成「`!= APPROVED` 就按 `permission_denied` 处理」——
    那是把一个**编程错误**写成一条「**用户**点了取消」的审计行,而
    `tool_audit_logs` 正是验收 5 读的表。所以下面两条断言缺一不可:
    **工具没被调用**(`calls == []`)**且审计是空的**(`seen == []`)。

    ⚠️ **参数化的值必须真的是认不出来的那些。** 别拿 `"approved"` 当反例 ——
    它**就是** `executor.APPROVED` 的取值(`APPROVED = "approved"`,spec §5.3
    的小写口径),拿它当「拼错的决议」会让这条测试恒绿。真正的风险形态是
    **大小写**搞错(`"Approved"`)、**半接线调用方**传 `None`,以及一小撮
    真值(`1`)。下面是这一族,不是随手凑的数。
    """
    calls = []
    seen: list[dict] = []

    @tool
    async def create_ticket(description: str) -> str:
        """建单。"""
        calls.append(description)
        return "ok"

    spec = _spec(name="create_ticket", kind=WRITE, tool=create_ticket)
    monkeypatch.setattr(executor, "record_audit", lambda **kw: _collect(seen, kw))
    with pytest.raises(ToolInfrastructureError):
        await execute_tool(
            tool_call=_tc("create_ticket", {"description": "坏了"}),
            registry={"create_ticket": spec},
            settings=_Settings(),
            conversation_id="c1",
            write_decision=bad,
        )
    assert calls == [], "决议认不出来却把不可逆的写执行了"
    assert seen == [], "接线 bug 被写成了「用户点了取消」——污染验收 5 读的那张表"


@pytest.mark.anyio
async def test_approved_write_is_executed_once():
    calls = []

    @tool
    async def create_ticket(description: str) -> str:
        """建单。"""
        calls.append(description)
        return '{"ticket_no": "T-1"}'

    spec = _spec(name="create_ticket", kind=WRITE, tool=create_ticket)
    outcome = await execute_tool(
        tool_call=_tc("create_ticket", {"description": "坏了"}),
        registry={"create_ticket": spec},
        settings=_Settings(),
        conversation_id="c1",
        write_decision=APPROVED,
    )
    assert calls == ["坏了"]
    assert outcome.ok is True


# ---- 闸 2:参数校验 -----------------------------------------------------


@pytest.mark.anyio
async def test_invalid_args_are_caught_before_the_tool_runs(monkeypatch):
    """**注入的是裸 args,不是 ValidationError** —— 校验那一步必须真的发生。

    本仓栽过三次的假绿形态就是「注入已经被处理好的值」:那样
    「把校验错误翻成给模型看的话」这一步永远不被验。
    """
    calls = {"n": 0}
    seen: list[dict] = []

    @tool
    async def query_order(order_id: str) -> str:
        """查订单。"""
        return "{}"

    spec = _with_counter(_spec(name="query_order", tool=query_order), calls)
    monkeypatch.setattr(executor, "record_audit", lambda **kw: _collect(seen, kw))
    outcome = await execute_tool(
        tool_call=_tc("query_order", {}),                # 缺必填
        registry={"query_order": spec},
        settings=_Settings(),
        conversation_id="c1",
    )
    # 判别式在 `ainvoke` 计次上(见 `_CountingTool`):**没有前置闸**的实现会
    # 先调进 `ainvoke`、再由循环里的 `ValidationError` 兜成 invalid_args ——
    # 那一步会让这条变红,而按工具体计数的话它恒为 0、两种实现分不开。
    assert calls["n"] == 0, "参数不合法却把工具跑起来了(校验闸在 ainvoke 之后)"
    assert outcome.error_kind == ERROR_INVALID_ARGS
    assert "order_id" in outcome.content, "回灌文案必须点名到字段"
    assert [s["status"] for s in seen] == ["invalid_args"]


@pytest.mark.anyio
async def test_invalid_args_is_not_retried():
    """重放同样的参数只会同样失败 —— 重试纯属浪费。**断的是 `retry_count`。**"""
    spec = _spec(name="query_order")
    outcome = await execute_tool(
        tool_call=_tc("query_order", {}),
        registry={"query_order": spec},
        settings=_Settings(),
        conversation_id="c1",
    )
    assert outcome.error_kind == ERROR_INVALID_ARGS
    assert outcome.retry_count == 0


# ---- 闸 3:重试规则 -----------------------------------------------------


@pytest.mark.anyio
async def test_read_tool_is_retried_on_timeout():
    calls = []

    @tool
    async def query_order(order_id: str) -> str:
        """查订单。"""
        calls.append(order_id)
        # 真睡 10 秒,由 `_Settings.tool_timeout_seconds = 0.01` 掐掉 ——
        # **不要**去 patch `asyncio.sleep`(见 `_Settings` 的说明)。
        await asyncio.sleep(10)
        return "{}"

    spec = _spec(
        name="query_order", kind=READ,
        schema={"type": "object", "properties": {"order_id": {"type": "string"}}},
        tool=query_order,
    )
    outcome = await execute_tool(
        tool_call=_tc("query_order", {"order_id": "1002"}),
        registry={"query_order": spec},
        settings=_Settings(),
        conversation_id="c1",
    )
    assert outcome.ok is False
    assert outcome.error_kind == ERROR_TIMEOUT
    # 1 次原始 + 2 次重试 = 3 次尝试(TOOL_RETRY_ATTEMPTS=2,spec §6.2)
    assert outcome.retry_count == 2


@pytest.mark.anyio
async def test_write_tool_never_retries_even_when_config_says_two():
    """**结构保证,不是配置恰好为 0。**

    `_Settings.tool_retry_attempts` 在上面就是 **2** —— 这条测试要是绿不了,
    就说明「不重试」是靠配置凑出来的,而不是靠 `kind == write` 推出来的
    (验收 6 后半条断的就是这个)。
    """
    calls = []

    @tool
    async def create_ticket(description: str) -> str:
        """建单。"""
        calls.append(description)
        await asyncio.sleep(10)
        return "{}"

    spec = _spec(name="create_ticket", kind=WRITE, tool=create_ticket)
    outcome = await execute_tool(
        tool_call=_tc("create_ticket", {"description": "x"}),
        registry={"create_ticket": spec},
        settings=_Settings(),
        conversation_id="c1",
        write_decision=APPROVED,
    )
    assert outcome.error_kind == ERROR_TIMEOUT
    assert outcome.retry_count == 0
    assert len(calls) == 1


@pytest.mark.anyio
async def test_first_try_success_reports_zero_retries(monkeypatch):
    """⚠️ **本章最容易写错的字段**(spec §7.4)。

    写成 `settings.tool_retry_attempts` 的实现会报 **2** ——
    而它看起来完全正常,没有任何别的断言会红。
    """
    seen: list[dict] = []

    @tool
    async def query_order(order_id: str) -> str:
        """查订单。"""
        return '{"ok": true}'

    spec = _spec(name="query_order", tool=query_order)
    monkeypatch.setattr(executor, "record_audit", lambda **kw: _collect(seen, kw))
    outcome = await execute_tool(
        tool_call=_tc("query_order", {"order_id": "1002"}),
        registry={"query_order": spec},
        settings=_Settings(),
        conversation_id="c1",
    )
    assert outcome.ok is True
    assert outcome.retry_count == 0
    assert seen[0]["retry_count"] == 0


@pytest.mark.anyio
async def test_transient_failure_succeeds_on_retry():
    """**暂时性故障是本任务新增的那一类可重试故障**(spec §6.3)。

    这一条双向都钉住:① 它**真的重试了**(`retry_count == 1`);
    ② 第二次成功就是成功,不会因为「它抖过」而把结果也丢掉。
    """
    calls = []

    @tool
    async def query_order(order_id: str) -> str:
        """查订单。"""
        calls.append(order_id)
        if len(calls) < 2:
            raise TransientToolError("connection refused")
        return '{"ok": true}'

    spec = _spec(name="query_order", tool=query_order)
    outcome = await execute_tool(
        tool_call=_tc("query_order", {"order_id": "1002"}),
        registry={"query_order": spec},
        settings=_Settings(),
        conversation_id="c1",
    )
    assert outcome.ok is True
    assert outcome.retry_count == 1
    assert len(calls) == 2


@pytest.mark.anyio
async def test_transient_exhausted_raises_infrastructure_error():
    """三次都不成 ⇒ **上抛 502**,不是回灌一句「工具暂时不可用」。

    重试用尽之后那个故障就不叫暂时性了;推一句软话给模型等于把基础设施故障
    伪装成一次普通的工具失败 —— 与「数据库挂了不许伪装成你的订单查不到」同一条规矩。
    """
    calls = []

    @tool
    async def query_order(order_id: str) -> str:
        """查订单。"""
        calls.append(order_id)
        raise TransientToolError("connection refused")

    spec = _spec(name="query_order", tool=query_order)
    with pytest.raises(ToolInfrastructureError):
        await execute_tool(
            tool_call=_tc("query_order", {"order_id": "1002"}),
            registry={"query_order": spec},
            settings=_Settings(),
            conversation_id="c1",
        )
    assert len(calls) == 3, "暂时性故障必须真的重试到用尽(1 次原始 + 2 次重试)"


@pytest.mark.anyio
async def test_unknown_tool_is_not_audited(monkeypatch):
    """接线 bug 不是一次调用 —— 它该响亮地暴露,不该混进审计流水。"""
    seen: list[dict] = []
    monkeypatch.setattr(executor, "record_audit", lambda **kw: _collect(seen, kw))
    outcome = await execute_tool(
        tool_call=_tc("nope", {}),
        registry={},
        settings=_Settings(),
        conversation_id="c1",
    )
    assert outcome.error_kind == "tool_missing"
    assert seen == []
