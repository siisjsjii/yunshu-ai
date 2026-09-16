"""对话编排测试。全部用替身,不联网、不碰 DB。"""

import asyncio

import pytest
from langchain.tools import tool
from langchain_core.messages import SystemMessage

from app.config import Settings
from app.memory.trim import ContextOverflowError
from app.schemas import Message
from app.services.chat import prepare_turn, stream_turn
from app.tools.errors import ToolInfrastructureError

REQUIRED = {
    "openai_base_url": "https://example.invalid/v1",
    "openai_api_key": "sk-test",
    "openai_model": "test-model",
    "database_url": "mysql+asyncmy://u:p@h:3306/db",
}


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **REQUIRED, **overrides)


class FakeChunk:
    """模拟 AIMessageChunk:支持 + 累加,累加后携带 tool_calls。

    替身**必须补上 `"type": "tool_call"`**:真实链路上模型流出的
    tool_call chunk 经 `langchain_core.messages.tool.default_tool_parser`
    解析后,每个 tool_call 都带这个键(已在 langchain 1.4.0 / core 1.6.3
    上实测确认)。而 `BaseTool.ainvoke` 判"这是不是工具调用"靠的正是
    `_is_tool_call(x) = x.get("type") == "tool_call"` 这一个条件 ——
    缺键时它会把这个 dict **整个当成参数**去校验,于是每次调用都返回
    "参数不合法"的可恢复失败。那样的替身会让整组编排测试看似全绿,
    实际却一条都没走到真实路径上。
    """

    def __init__(self, text="", tool_calls=None, usage=None):
        self.text = text
        self.tool_calls = [{"type": "tool_call", **tc} for tc in (tool_calls or [])]
        self.usage_metadata = usage

    def __add__(self, other):
        return FakeChunk(
            text=self.text + other.text,
            tool_calls=self.tool_calls + other.tool_calls,
            usage=other.usage_metadata or self.usage_metadata,
        )


class _BoundModel:
    def __init__(self, inner):
        self._inner = inner

    async def astream(self, messages):
        self._inner.calls.append(("bound", list(messages)))
        for chunk in self._inner.batches.pop(0):
            yield chunk


class ScriptedModel:
    """按顺序回放预置 chunk 批次的替身。

    记录每次 astream 走的是**绑了工具**还是**未绑工具**的入口 ——
    这正是"只做单轮"的结构保证所在,必须有断言钉住。
    """

    def __init__(self, batches):
        self.batches = list(batches)
        self.calls = []
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return _BoundModel(self)

    async def astream(self, messages):
        self.calls.append(("unbound", list(messages)))
        for chunk in self.batches.pop(0):
            yield chunk


@tool
async def query_logistics(order_id: str) -> str:
    """替身:查物流。"""
    return '{"status": "已揽件"}'


@tool
async def query_order(order_id: str) -> str:
    """替身:查订单。

    与 query_logistics 成对,供「一轮两个 tool_call」的用例使用(见
    test_two_tool_calls_in_one_round_are_both_executed_and_paired)。
    本文件里另有两个函数内局部定义的 `query_order`(总是失败 / DB 挂了
    那两条),它们是各自独立的替身对象,通过显式注册表传入,与此处无冲突。
    """
    return '{"status": "已发货"}'


class RecordingSession:
    """替身 DB 会话:`add` 记下落库行,`commit` 默认成功。

    落库是要断言的产出(见 test_successful_turn_appends_user_tool_and_answer
    与 test_tool_free_turn_appends_user_and_answer),所以 `add` 一律记账 ——
    一个不记账的变体等于让"这一轮到底写没写库"无从观测。
    """

    def __init__(self):
        self.added = []

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        pass


def _collect(model, session, registry, tools=None):
    async def run():
        return [
            event
            async for event in stream_turn(
                settings=_settings(),
                model=model,
                session=session,
                conversation_id="s1",
                user_input="订单 1001 的物流到哪了",
                messages=[Message(role="user", content="订单 1001 的物流到哪了")],
                tools=tools if tools is not None else list(registry.values()),
                registry=registry,
            )
        ]

    return asyncio.run(run())


# ---------- 单轮结构保证 ----------

def test_second_round_uses_unbound_model():
    """**本章最关键的一条结构断言。**

    「只做单轮」不能靠提示词求模型自觉,必须靠第二轮不绑 tools。
    谁把第二轮改成 model_with_tools,这条就挂。
    """
    model = ScriptedModel(
        [
            [FakeChunk(tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}])],
            [FakeChunk("已揽件。")],
        ]
    )
    _collect(model, RecordingSession(), {"query_logistics": query_logistics})

    assert [kind for kind, _ in model.calls] == ["bound", "unbound"]
    # 绑上去的必须真的是注册表里的工具。本文件的 tool_call 全由替身**喂**出来,
    # 不是模型自己决定要调的 —— 所以 bind_tools([]) 让模型一个工具都看不到时,
    # 上面那条结构断言和其余全部编排测试照样全绿。这一行钉死绑定内容本身。
    assert model.bound_tools == [query_logistics]


def test_no_tool_call_means_single_api_call():
    """不调工具时只发一次请求,且文本已经流式推出。"""
    model = ScriptedModel([[FakeChunk("您"), FakeChunk("好")]])
    events = _collect(model, RecordingSession(), {})

    assert len(model.calls) == 1
    assert events[0] == ("token", {"text": "您"})
    assert events[-1][0] == "done"


def test_tool_event_order_is_call_then_result_then_answer():
    model = ScriptedModel(
        [
            [FakeChunk(tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}])],
            [FakeChunk("包裹"), FakeChunk("已揽件。")],
        ]
    )
    events = _collect(model, RecordingSession(), {"query_logistics": query_logistics})
    kinds = [kind for kind, _ in events]

    assert kinds == ["tool_call", "tool_result", "token", "token", "done"]
    assert events[0][1]["name"] == "query_logistics"
    assert events[0][1]["tool_call_id"] == "c1"
    assert events[1][1]["ok"] is True


def test_tool_result_is_fed_back_as_tool_message():
    model = ScriptedModel(
        [
            [FakeChunk(tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}])],
            [FakeChunk("已揽件。")],
        ]
    )
    _collect(model, RecordingSession(), {"query_logistics": query_logistics})

    second_round_messages = model.calls[1][1]
    from langchain.messages import ToolMessage

    tool_messages = [m for m in second_round_messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "c1"
    assert "已揽件" in tool_messages[0].content


# ---------- 错误分类 ----------

def test_recoverable_tool_failure_still_converges():
    """可恢复失败(查无此单)要回灌给模型,流正常 done,不是 error 帧。"""

    @tool
    async def query_order(order_id: str) -> str:
        """替身:总是找不到订单。"""
        from app.tools.errors import ToolNotFound
        raise ToolNotFound("未找到订单 9999")

    model = ScriptedModel(
        [
            [FakeChunk(tool_calls=[{"name": "query_order", "args": {"order_id": "9999"}, "id": "c1"}])],
            [FakeChunk("没查到该订单,请核对单号。")],
        ]
    )
    events = _collect(model, RecordingSession(), {"query_order": query_order})
    kinds = [kind for kind, _ in events]

    assert "tool_result" in kinds
    assert kinds[-1] == "done"
    assert "error" not in kinds
    result_payload = next(payload for kind, payload in events if kind == "tool_result")
    assert result_payload["ok"] is False

    # 落空的结果**必须回灌给模型**,否则第二轮消息里是一条带 tool_calls 的
    # assistant 后面没有配对的 tool 消息 —— OpenAI 兼容端点直接 400,
    # 而这正是 spec §6.3 要防的那个形态。事件序列本身看不出这一点。
    from langchain.messages import ToolMessage

    tool_messages = [m for m in model.calls[1][1] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "c1"
    assert "未找到订单 9999" in tool_messages[0].content


def test_infrastructure_failure_propagates():
    """基础设施故障必须向上抛,由 API 层推 error 帧 —— 不能伪装成"查不到"。"""

    @tool
    async def query_order(order_id: str) -> str:
        """替身:DB 挂了。"""
        from sqlalchemy.exc import OperationalError
        raise OperationalError("SELECT 1", {}, Exception("连接断开"))

    model = ScriptedModel(
        [[FakeChunk(tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}])]]
    )
    with pytest.raises(ToolInfrastructureError):
        _collect(model, RecordingSession(), {"query_order": query_order})


def test_tool_free_turn_appends_user_and_answer():
    """**无工具的一轮也要落库 —— 这是绝大多数流量的路径。**

    其余编排测试传的是不记账的替身会话,所以把无工具分支里的
    `append_turn(...)` 整段删掉,全部编排测试依然全绿,而普通闲聊
    从此一条历史都不存,且没有任何测试会喊。ch01 的
    `test_stream_turn_persists_both_messages_on_success` 钉的就是这条,
    改写时丢了 —— 这里补回,并让 Ruling 5 的两条完成路径都有覆盖。
    """
    session = RecordingSession()
    model = ScriptedModel([[FakeChunk("您"), FakeChunk("好")]])
    _collect(model, session, {})

    assert [obj.role for obj in session.added] == ["user", "assistant"]
    assert session.added[0].content == "订单 1001 的物流到哪了"
    assert session.added[1].content == "您好"


def test_history_is_not_written_when_second_round_breaks():
    """第二轮炸了 → 整轮不落库,不留孤儿行(沿用 ch01 语义)。"""

    class ExplodingSession(RecordingSession):
        async def commit(self):
            raise AssertionError("不应提交")

    class ExplodingModel(ScriptedModel):
        async def astream(self, messages):
            self.calls.append(("unbound", list(messages)))
            yield FakeChunk("前半句")
            raise RuntimeError("上游炸了")

    session = ExplodingSession()
    model = ExplodingModel(
        [[FakeChunk(tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}])]]
    )
    with pytest.raises(RuntimeError):
        _collect(model, session, {"query_logistics": query_logistics})

    assert session.added == []


def test_successful_turn_appends_user_tool_and_answer():
    """成功一轮落库四条:user / assistant(带 tool_calls) / tool / assistant。"""
    session = RecordingSession()
    model = ScriptedModel(
        [
            [FakeChunk(tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}])],
            [FakeChunk("已揽件。")],
        ]
    )
    _collect(model, session, {"query_logistics": query_logistics})

    roles = [obj.role for obj in session.added]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert session.added[1].tool_calls[0]["id"] == "c1"
    assert session.added[2].tool_call_id == "c1"
    assert session.added[3].content == "已揽件。"


def test_two_tool_calls_in_one_round_are_both_executed_and_paired():
    """一轮里模型发**两个** tool_call 时:两个都执行、都配对回灌、都落库。

    这不是假想 —— dev-notes/ch02.md 阶段 6 ④ 记着真实模型在物流提问上同轮发了
    两个 tool_call。而在此之前,**整个套件从没执行过 `for tool_call in tool_calls`
    的第二次迭代**:两个变异都能让既有 45 条编排测试全绿 ——
    ①循环体末尾 `break`(第二个工具根本不执行,事件序列少一对、回灌少一条);
    ②落库时写 `tool_calls[:1]`(assistant 行带两个申请、history 里只有一条 tool
    回复 —— 正是 spec §6.3 要防的、上游直接 400 的形态)。

    所以这里三层都断:事件层(顺序与配对)、发给模型的第二轮消息层、落库层。
    只断其中一层时,上面两个变异各有一个能从缝里溜过去。
    """
    from langchain.messages import AIMessage, ToolMessage

    session = RecordingSession()
    model = ScriptedModel(
        [
            [
                FakeChunk(
                    tool_calls=[
                        {
                            "name": "query_logistics",
                            "args": {"order_id": "1001"},
                            "id": "c1",
                        },
                        {
                            "name": "query_order",
                            "args": {"order_id": "1001"},
                            "id": "c2",
                        },
                    ]
                )
            ],
            [FakeChunk("包裹已揽件,订单正常。")],
        ]
    )
    events = _collect(
        model,
        session,
        {"query_logistics": query_logistics, "query_order": query_order},
    )

    # ---- 事件层:每个申请各自紧跟自己的结果,不是"先两个申请再两个结果"
    assert [kind for kind, _ in events] == [
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "token",
        "done",
    ]
    assert [p["tool_call_id"] for k, p in events if k == "tool_call"] == ["c1", "c2"]
    assert [p["tool_call_id"] for k, p in events if k == "tool_result"] == ["c1", "c2"]
    assert [p["name"] for k, p in events if k == "tool_call"] == [
        "query_logistics",
        "query_order",
    ]
    assert all(p["ok"] for k, p in events if k == "tool_result")

    # ---- 发给模型的第二轮消息层:一条 assistant(带两个申请)+ 紧邻的两条 tool
    second_round = model.calls[1][1]
    parent_index = next(
        i
        for i, m in enumerate(second_round)
        if isinstance(m, AIMessage) and m.tool_calls
    )
    assert [tc["id"] for tc in second_round[parent_index].tool_calls] == ["c1", "c2"]
    tool_messages = [m for m in second_round if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in tool_messages] == ["c1", "c2"]
    assert [second_round.index(m) for m in tool_messages] == [
        parent_index + 1,
        parent_index + 2,
    ]
    # 每条回灌的是**它自己那次调用**的结果,不是同一个工具跑两遍
    assert "已揽件" in tool_messages[0].content
    assert "已发货" in tool_messages[1].content

    # ---- 落库层:五条,且 assistant 行的 tool_calls 不能只剩头一个
    assert [obj.role for obj in session.added] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]
    assert [obj.tool_call_id for obj in session.added if obj.role == "tool"] == [
        "c1",
        "c2",
    ]
    assert [tc["id"] for tc in session.added[1].tool_calls] == ["c1", "c2"]


# ---------- 只做单轮:第二轮消息的合法性 ----------
#
# 上面那组测试钉的是事件序列;下面这组钉的是**发给模型的第二条请求
# 本身长什么样**。两者缺一不可:事件对了而消息不成对,线上就是 400,
# 而且只在真的触发工具调用时才复现。


def test_second_round_assistant_message_precedes_its_tool_message():
    """tool 消息前面必须紧跟带同名 tool_call_id 的 assistant 消息。

    OpenAI 兼容端点对此是硬校验:tool 消息的父 assistant 缺失或不相邻
    直接 400。上面那条 `test_tool_result_is_fed_back_as_tool_message`
    只数了 tool 消息的条数 —— 把配对的 assistant 消息整条删掉,它依然
    全绿。这条补上那个缺口。
    """
    from langchain.messages import AIMessage, ToolMessage

    model = ScriptedModel(
        [
            [FakeChunk(tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}])],
            [FakeChunk("已揽件。")],
        ]
    )
    _collect(model, RecordingSession(), {"query_logistics": query_logistics})

    second_round_messages = model.calls[1][1]
    tool_index = next(
        i for i, m in enumerate(second_round_messages) if isinstance(m, ToolMessage)
    )
    parent = second_round_messages[tool_index - 1]

    assert isinstance(parent, AIMessage)
    assert [tc["id"] for tc in parent.tool_calls] == ["c1"]
    assert parent.tool_calls[0]["name"] == "query_logistics"
    assert parent.tool_calls[0]["args"] == {"order_id": "1001"}


def test_tool_message_carries_full_content_not_truncated_summary():
    """回灌给模型的是**完整**结果,tool_result 事件里的才是截断摘要。

    两条都短的输出下 summary == content,所以上面所有编排测试都区分不出
    "拿 summary 回灌"这个 bug —— 模型会拿到被砍掉尾巴的数据,而且看不出来。
    """

    @tool
    async def query_order(order_id: str) -> str:
        """替身:返回超长结果。"""
        return "中" * 500

    model = ScriptedModel(
        [
            [FakeChunk(tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}])],
            [FakeChunk("已为您查到。")],
        ]
    )
    events = _collect(model, RecordingSession(), {"query_order": query_order})

    from langchain.messages import ToolMessage

    payload = next(p for kind, p in events if kind == "tool_result")
    assert len(payload["summary"]) == 201          # 200 字符 + 省略号
    tool_message = next(
        m for m in model.calls[1][1] if isinstance(m, ToolMessage)
    )
    assert len(tool_message.content) == 500        # 未被摘要截断


def test_tool_call_event_carries_the_model_supplied_args():
    """前端要靠 args 展示"正在查什么";退化成空 dict 时事件序列本身看不出来。"""
    model = ScriptedModel(
        [
            [FakeChunk(tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}])],
            [FakeChunk("已查询到。")],
        ]
    )
    events = _collect(model, RecordingSession(), {"query_logistics": query_logistics})

    assert events[0][1]["args"] == {"order_id": "1001"}


def test_tool_call_chunks_do_not_leak_into_tokens():
    """调工具的那一轮不发 token 帧 —— 模型还没说话,别把空串推给用户。"""
    model = ScriptedModel(
        [
            [FakeChunk(tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}])],
            [FakeChunk("已揽件。")],
        ]
    )
    events = _collect(model, RecordingSession(), {"query_logistics": query_logistics})

    assert [p["text"] for k, p in events if k == "token"] == ["已揽件。"]


# ---------- usage 与空 chunk ----------


def test_done_keeps_earlier_usage_when_a_later_chunk_has_none():
    """后一个 chunk 的 usage_metadata 为 None 时,不能把先前的 usage 冲掉。

    真实上游只有最后一帧带 usage;若实现写成 `usage = chunk.usage_metadata`
    (丢掉 `or usage`),这一条会退化成 None。
    """
    model = ScriptedModel(
        [
            [
                FakeChunk("您", usage={"input_tokens": 10, "output_tokens": 1}),
                FakeChunk("好", usage=None),
            ]
        ]
    )
    events = _collect(model, RecordingSession(), {})

    assert events[-1][1]["usage"] == {"input_tokens": 10, "output_tokens": 1}


def test_done_reports_second_round_usage():
    """第二轮才是产出答复的那一轮,usage 得取它的,不能继续用第一轮的。

    第一轮带 usage、第二轮也带,且两者不同 —— 只带其中一个的实现会露馅。
    """
    model = ScriptedModel(
        [
            [FakeChunk(tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}],
                       usage={"input_tokens": 5, "output_tokens": 1})],
            [FakeChunk("已揽件。", usage={"input_tokens": 20, "output_tokens": 3})],
        ]
    )
    events = _collect(model, RecordingSession(), {"query_logistics": query_logistics})

    assert events[-1][1]["usage"] == {"input_tokens": 20, "output_tokens": 3}


def test_stream_turn_skips_empty_token_chunks():
    model = ScriptedModel([[FakeChunk(""), FakeChunk("好")]])
    events = _collect(model, RecordingSession(), {})

    assert [e for e in events if e[0] == "token"] == [("token", {"text": "好"})]


def test_done_usage_is_none_when_absent():
    model = ScriptedModel([[FakeChunk("好")]])
    events = _collect(model, RecordingSession(), {})

    assert events[-1][1]["usage"] is None


# ---------- 消息组装与预算 ----------


def test_prepare_turn_returns_system_plus_input_for_empty_history():
    messages = prepare_turn(settings=_settings(), history=[], user_input="你好")

    assert len(messages) == 2
    assert messages[0].content  # 非空 system prompt
    assert messages[-1].content == "你好"
    # 只断长度和末条的话,[user_input, user_input] 这种实现也能过。
    assert isinstance(messages[0], SystemMessage)


def test_prepare_turn_includes_existing_history():
    messages = prepare_turn(
        settings=_settings(),
        history=[
            Message(role="user", content="我的订单是 20240915"),
            Message(role="assistant", content="好的,我为您查询"),
        ],
        user_input="我刚才说的订单号是多少？",
    )

    assert len(messages) == 4
    assert messages[1].content == "我的订单是 20240915"
    assert messages[-1].content == "我刚才说的订单号是多少？"


def test_prepare_turn_drops_history_that_does_not_fit_the_budget():
    """预算放不下历史时把它裁掉,但本轮输入照常发出。"""
    history = [
        Message(role="user", content="退" * 2000),
        Message(role="assistant", content="好" * 2000),
    ]
    messages = prepare_turn(
        settings=_settings(
            context_budget_tokens=1000,
            reserved_output_tokens=0,
            safety_margin_tokens=0,
        ),
        history=history,
        user_input="在吗",
    )

    assert len(messages) == 2
    assert messages[-1].content == "在吗"


def test_prepare_turn_raises_when_input_alone_exceeds_budget():
    with pytest.raises(ContextOverflowError):
        prepare_turn(
            settings=_settings(
                context_budget_tokens=200,
                reserved_output_tokens=0,
                safety_margin_tokens=0,
            ),
            history=[],
            user_input="退" * 5000,
        )
