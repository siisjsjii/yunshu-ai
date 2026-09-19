"""主力 Agent 的 ReAct 循环:收敛、工具回灌、停止条件、token 预算、流式。"""

import pytest
from langchain_core.messages import ToolMessage

from app.agent.nodes import make_agent_node
from app.config import Settings
from app.prompts import render_evidence
from app.schemas import Message
from app.tools.errors import ToolInfrastructureError
from app.tools.executor import SUMMARY_MAX_CHARS

REQUIRED = {
    "openai_base_url": "https://example.invalid/v1",
    "openai_api_key": "sk-test",
    "openai_model": "test-model",
    "database_url": "mysql+asyncmy://u:p@h:3306/db",
}


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **REQUIRED, **overrides)


def _text(msg) -> str:
    """1.x 里**流式 chunk** 的 content 是 block 列表;这里的 BaseMessage 是
    我们自己用字符串构造的,content 仍是 str —— 两种都兜住,免得断言靠猜。"""
    content = msg.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))


class FakeChunk:
    def __init__(self, text="", tool_calls=None, usage=None):
        self.text = text
        # 必须带 "type": "tool_call" —— 见 CLAUDE.md;缺键时 BaseTool.ainvoke
        # 会把整个 dict 当**参数**去校验 schema,每次调用都变成「参数不合法」。
        self.tool_calls = [{"type": "tool_call", **tc} for tc in (tool_calls or [])]
        self.usage_metadata = usage

    def __add__(self, other):
        # 累积 chunk 时**原样**拼接 tool_calls,**必须保留 `"type": "tool_call"`**。
        # 它是 `BaseTool.ainvoke` 判定「这是一个工具调用」的**唯一**依据
        # (CLAUDE.md 的硬约束);`execute_tool` 则原样把 dict 透传给
        # `tool.ainvoke`(`app/tools/executor.py:65`)。丢了它,真实 `@tool` 会把
        # 整个 dict 当**参数**去校验 schema,每次调用都退化成「参数不合法」的
        # 可恢复失败 —— 而本文件的替身工具不查这个键,所以**测试照样全绿**。
        #
        # 说明:本替身**不模拟** LangChain 真实的按 index 合并 + 分片 args 拼接
        # (真实模型会把一次工具调用拆成多个 chunk 流式吐出来),因为那些测试
        # 里每次工具调用都写在**单个** chunk 里,`__add__` 的合并分支用不到。
        # 真实多 chunk 的累积由 T11 的验收 5(真模型、要求 ReAct 不止一步)兜底。
        return FakeChunk(
            text=self.text + other.text,
            tool_calls=self.tool_calls + other.tool_calls,
            usage=other.usage_metadata or self.usage_metadata,
        )


class _BoundModel:
    def __init__(self, owner):
        self._owner = owner

    async def astream(self, messages):
        self._owner.bound_rounds += 1
        self._owner.bound_messages = list(messages)   # 断言入参用
        for chunk in self._owner.rounds.pop(0):
            yield chunk


class ScriptedModel:
    def __init__(self, rounds):
        self.rounds = list(rounds)
        self.bound_rounds = 0
        self.unbound_rounds = 0

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return _BoundModel(self)

    async def astream(self, messages):
        # 这里**刻意不记**入参:agent 每一轮都绑着工具问,未绑工具的入口只在
        # 步数用尽收尾时走一次 —— 把「回灌的 ToolMessage」记在这儿会永远看不到。
        # 要断回灌就读 `bound_messages`(Task 1 已踩过一次,见 ledger Ruling 6)。
        self.unbound_rounds += 1
        for chunk in self.rounds.pop(0):
            yield chunk


class FakeTool:
    def __init__(self, name="query_order", content='{"status": "已发货"}', error=None):
        self.name = name
        self.content = content
        self.error = error
        self.calls = []

    async def ainvoke(self, tool_call):
        self.calls.append(tool_call)
        if self.error is not None:
            raise self.error
        return type("_R", (), {"content": self.content})()


def _node(model, tools=(), registry=None, settings=None, frames=None):
    return make_agent_node(
        model=model, tools=list(tools), registry=registry or {},
        settings=settings or _settings(),
        emit=(frames.append if frames is not None else (lambda p: None)),
    )


def _state(**over):
    base = {"conversation_id": "c1", "user_input": "订单 1001 发货了吗",
            "resolved_input": "订单 1001 发货了吗", "history": []}
    base.update(over)
    return base


@pytest.mark.anyio
async def test_direct_text_answer_converges_without_tools():
    frames = []
    # 第二批(空)是**探针**:正确的实现根本不会去取它。若实现改成「无工具的一轮
    # 之后再走一次收尾」,它会吃掉这批空的 —— 于是红在下面的
    # `unbound_rounds == 0` 上(「多问了一次模型」);不摆它的话,红法会是
    # `rounds.pop(0)` 的 IndexError,把「多调了一次」报成「测试写错了」。
    model = ScriptedModel([[FakeChunk("你的"), FakeChunk("订单已发货。")], []])
    out = await _node(model, frames=frames).__call__(_state())
    assert out["reply"] == "你的订单已发货。"
    assert out["agent_steps"] == 1
    assert out["tool_calls_made"] == []
    assert frames == [
        {"frame": "token", "text": "你的"},
        {"frame": "token", "text": "订单已发货。"},
    ]
    # **只问一次模型**:没调工具就不该再走收尾那一轮。ch02 的
    # `test_no_tool_call_means_single_api_call`(`:163`,随 `stream_turn` 一起被
    # T8 删掉)断的是同一件事。少了这两行,「无工具的一轮多问一次模型」只会
    # 以**脚本耗尽**的样子红(`rounds.pop(0)` 抛 IndexError),而那个红指向的是
    # 「测试写错了」,盖掉真问题 —— 与 `_run` 里给 `agent_steps` 初值同一条理由。
    assert model.bound_rounds == 1
    assert model.unbound_rounds == 0


@pytest.mark.anyio
async def test_tool_call_round_emits_frames_and_feeds_result_back():
    frames = []
    tool_obj = FakeTool()
    model = ScriptedModel([
        [FakeChunk(tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}])],
        [FakeChunk("已发货。")],
    ])
    out = await _node(model, [tool_obj], {"query_order": tool_obj}, frames=frames).__call__(_state())

    assert out["reply"] == "已发货。"
    assert out["agent_steps"] == 2
    assert out["tool_calls_made"] == [{"name": "query_order", "ok": True}]
    assert {"frame": "tool_call", "name": "query_order",
            "args": {"order_id": "1001"}, "tool_call_id": "c1"} in frames
    assert {"frame": "tool_result", "tool_call_id": "c1", "ok": True,
            "summary": '{"status": "已发货"}'} in frames
    # **精确列表,不是 `in`**。调工具的那一轮模型还没说话,chunk.text 是空串 ——
    # `if chunk.text:` 是"不发空 token 帧"的**唯一**屏障。上面两条 `in` 断言
    # (以及 T8 删掉的 `test_tool_call_chunks_do_not_leak_into_tokens` /
    # `test_stream_turn_skips_empty_token_chunks` 的新家)都拦不住它:把
    # `if chunk.text:` 删掉(只留 `parts.append` / `emit`),空串 token 帧混进来,
    # 这组 `in` 断言照样全绿 —— 而前端会先画出一个空气泡。
    # 第二条空串来源:收尾那一轮也可能吐空 chunk。
    assert [p["text"] for p in frames if p["frame"] == "token"] == ["已发货。"]
    # 回灌的 ToolMessage 必须与 tool_call_id 配对。
    # 注意记的是**绑工具**那个入口:agent 每一轮都绑着工具问,未绑工具的入口
    # 只在步数用尽收尾时走一次。
    fed = [m for m in model.bound_messages if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in fed] == ["c1"]
    # 钉住 `msgs.append(acc)` 那一行:删掉它,上面 fed 的断言**依然全绿**,
    # 而真实链路上会退化成「有 tool 消息、没有前置的 assistant(tool_calls)消息」——
    # OpenAI 兼容 API 直接 400。这正是 CLAUDE.md 点名的「假绿」形态。
    calls_idx = [i for i, m in enumerate(model.bound_messages)
                 if getattr(m, "tool_calls", None)]
    tool_idx = [i for i, m in enumerate(model.bound_messages)
                if isinstance(m, ToolMessage)]
    assert calls_idx and tool_idx, "第二轮入参缺 assistant(tool_calls) 或 ToolMessage"
    assert tool_idx[0] == calls_idx[0] + 1   # 必须**紧邻**
    # trace 让验收 5「ReAct 不止一步」机械可断言
    assert out["trace"] == ["agent:step1 tool=query_order", "agent:converged"]


@pytest.mark.anyio
async def test_step_limit_converges_without_tools_bound():
    """步数用尽 → 最后一次调用**不绑 tools**,结构上保证收敛。"""
    # 脚本按消费顺序对齐:绑工具的 2 轮(== max_agent_steps)各弹一条,
    # 第 3 条留给收尾那一轮(未绑工具)。
    looping = [
        [FakeChunk(tool_calls=[{"name": "query_order", "args": {"order_id": str(i)}, "id": f"c{i}"}])]
        for i in range(2)
    ]
    model = ScriptedModel(looping + [[FakeChunk("收敛了")]])
    tool_obj = FakeTool()
    out = await _node(model, [tool_obj], {"query_order": tool_obj},
                      settings=_settings(max_agent_steps=2)).__call__(_state())
    assert out["agent_steps"] == 2
    assert model.bound_rounds == 2
    assert model.unbound_rounds == 1
    assert out["reply"] == "收敛了"


@pytest.mark.anyio
async def test_token_budget_stops_further_tool_rounds():
    """预算超限后不再进入下一轮工具调用。"""
    big = {"total_tokens": 99999}
    model = ScriptedModel([
        [FakeChunk(tool_calls=[{"name": "query_order", "args": {"order_id": "1"}, "id": "c1"}], usage=big)],
        [FakeChunk("收敛了")],
    ])
    tool_obj = FakeTool()
    out = await _node(model, [tool_obj], {"query_order": tool_obj},
                      settings=_settings(agent_token_budget=100)).__call__(_state())
    assert out["usage"]["total_tokens"] >= 99999
    assert model.bound_rounds == 1      # 只走了一轮绑工具的
    assert out["reply"] == "收敛了"


@pytest.mark.anyio
async def test_unknown_tool_error_propagates_as_infrastructure_error():
    """未分类的异常由 executor 判成基础设施故障,**必须向上抛**(→502),
    绝不能被回灌给模型、伪装成「你的订单号查不到」。"""
    tool_obj = FakeTool(error=RuntimeError("boom"))
    model = ScriptedModel([
        [FakeChunk(tool_calls=[{"name": "query_order", "args": {"order_id": "1"}, "id": "c1"}])],
    ])
    with pytest.raises(ToolInfrastructureError):
        await _node(model, [tool_obj], {"query_order": tool_obj}).__call__(_state())


@pytest.mark.anyio
async def test_history_and_evidence_reach_the_first_model_call():
    """历史经 `prompts.to_lc_messages` 转、证据并进本轮 human 消息。

    **断言的是发给模型的真实入参**,不是「函数没抛异常」——后者恒真。
    """
    model = ScriptedModel([[FakeChunk("好")]])
    await _node(model).__call__(
        _state(
            history=[Message(role="user", content="在吗")],
            evidence=[{"section_path": "退货政策", "answer": "七天无理由", "category": "退换货"}],
        )
    )
    sent = model.bound_messages
    assert sent[0].type == "system"            # 品牌 system prompt
    assert _text(sent[1]) == "在吗"             # 历史转过来了(只经 to_lc_messages)
    assert "七天无理由" in _text(sent[-1])      # 证据块
    assert "订单 1001 发货了吗" in _text(sent[-1])   # 用户原话
    assert sent[-1].type == "human"


@pytest.mark.anyio
async def test_no_evidence_means_no_evidence_block_in_the_prompt():
    """业务数据类没有证据块 —— 不能留一个空标题在那儿诱导模型。"""
    model = ScriptedModel([[FakeChunk("好")]])
    await _node(model).__call__(_state(evidence=[]))
    assert "以下是知识库中" not in _text(model.bound_messages[-1])
    assert _text(model.bound_messages[-1]) == "订单 1001 发货了吗"


@pytest.mark.anyio
async def test_two_tool_calls_in_one_round_are_both_executed_and_paired():
    """一轮里模型发**两个** tool_call:两个都执行、都按序配对回灌。

    这不是假想 —— dev-notes/ch02.md 阶段 6 ④ 记着真实模型在物流提问上同轮发了
    两个 tool_call。本文件其余用例**从没执行过 `for call in tool_calls` 的第二次
    迭代**,于是 `tool_calls[:1]` 这种错误实现能全绿,而它产出的消息序列是
    「assistant 申请两个、只回灌一个」—— 上游 OpenAI 兼容 API **直接 400**。
    `tests/test_chat_service.py:319` 为 ch02 的链路立过同一条守卫,而那条链路
    正是本节点替代掉的(Task 8 会连同 `stream_turn` 一起删掉它)。

    本用例还走**两个 chunk 累积成一轮**的路径 —— 真实模型正是这么流的,
    所以它同时覆盖 `FakeChunk.__add__` 的合并分支。
    """
    frames = []
    order_tool = FakeTool(name="query_order")
    logistics_tool = FakeTool(name="query_logistics", content='{"location": "杭州"}')
    model = ScriptedModel([
        [   # 同一轮,两个 chunk,各带一个 tool_call
            FakeChunk(tool_calls=[{"name": "query_order",
                                   "args": {"order_id": "1001"}, "id": "c1"}]),
            FakeChunk(tool_calls=[{"name": "query_logistics",
                                   "args": {"order_id": "1001"}, "id": "c2"}]),
        ],
        [FakeChunk("订单已发货,目前在杭州。")],
    ])
    out = await _node(
        model,
        [order_tool, logistics_tool],
        {"query_order": order_tool, "query_logistics": logistics_tool},
        frames=frames,
    ).__call__(_state())

    # ① 两个工具**都真的被调用**了(`tool_calls[:1]` 死在这里)
    assert [c["id"] for c in order_tool.calls] == ["c1"]
    assert [c["id"] for c in logistics_tool.calls] == ["c2"]
    assert out["tool_calls_made"] == [
        {"name": "query_order", "ok": True},
        {"name": "query_logistics", "ok": True},
    ]
    # ② 两个 ToolMessage 都回灌,按序,且**紧邻**那条 assistant(tool_calls)
    fed = [m for m in model.bound_messages if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in fed] == ["c1", "c2"]
    calls_idx = [i for i, m in enumerate(model.bound_messages)
                 if getattr(m, "tool_calls", None)]
    tool_idx = [i for i, m in enumerate(model.bound_messages)
                if isinstance(m, ToolMessage)]
    assert calls_idx and len(tool_idx) == 2
    assert tool_idx == [calls_idx[0] + 1, calls_idx[0] + 2]
    # ③ 两对 tool_call / tool_result 帧都发了
    assert [p for p in frames if p["frame"] == "tool_call"] == [
        {"frame": "tool_call", "name": "query_order",
         "args": {"order_id": "1001"}, "tool_call_id": "c1"},
        {"frame": "tool_call", "name": "query_logistics",
         "args": {"order_id": "1001"}, "tool_call_id": "c2"},
    ]
    # ④ **交错,不是"先两个申请再两个结果"**。T8 实测过:把 emit 拆成两个循环
    # (先对所有 call 发 tool_call 帧,再执行并逐个发 tool_result 帧)—— 状态、
    # 消息序列、落库结果全都不变,只有帧序变了,而上面①~③与全仓其余用例
    # **一律照样全绿**(实测 `1 passed`)。前端是按帧渲染的:那会让"正在查订单"
    # 与"正在查物流"两个转圈同时亮起、再同时收到两条结果,配错对子。
    # ch02 的 `test_tool_event_order_is_call_then_result_then_answer`
    # (`tests/test_chat_service.py:173`,被 T8 连同 `stream_turn` 删掉)守的正是这条,
    # 本文件当时没有等价的**顺序**断言(③只钉了 tool_call 帧的内部顺序)。
    assert [(p["frame"], p["tool_call_id"]) for p in frames
            if p["frame"] in ("tool_call", "tool_result")] == [
        ("tool_call", "c1"), ("tool_result", "c1"),
        ("tool_call", "c2"), ("tool_result", "c2"),
    ]
    # 两者同属**同一次** ReAct 步 —— 步号都是 1
    assert out["trace"] == [
        "agent:step1 tool=query_order",
        "agent:step1 tool=query_logistics",
        "agent:converged",
    ]


@pytest.mark.anyio
async def test_full_tool_content_is_fed_back_not_the_truncated_summary():
    """回灌给模型的必须是**完整内容** `outcome.content`,不是展示用的 `summary`。

    `summary` 截断在 `SUMMARY_MAX_CHARS = 200`(`app/tools/executor.py:22`)。
    回灌截断版的话:`query_faq` 常态返回 400+ 字符,模型手里只有半截 JSON,
    读不出答案,只能回「暂未收录」—— 用户看到「知识库里明明有,客服却说没有」。
    而本文件其它用例的工具载荷只有十几字符,`summary == content`,**永远看不出区别**。
    """
    long_answer = "七天无理由退货。" + "详情见退货政策第三条。" * 40   # 远超 200 字符
    tool_obj = FakeTool(content=long_answer)
    model = ScriptedModel([
        [FakeChunk(tool_calls=[{"name": "query_order",
                                "args": {"order_id": "1"}, "id": "c1"}])],
        [FakeChunk("好的。")],
    ])
    frames = []
    await _node(model, [tool_obj], {"query_order": tool_obj},
                frames=frames).__call__(_state())

    fed = [m for m in model.bound_messages if isinstance(m, ToolMessage)]
    assert len(fed) == 1
    assert fed[0].content == long_answer          # 完整,一字未截
    assert len(fed[0].content) > SUMMARY_MAX_CHARS
    # 帧上仍走**截断**版 —— 那是给前端/日志看的,两条路径不能共用一个变量
    shown = [p for p in frames if p["frame"] == "tool_result"][0]["summary"]
    assert shown.endswith("…")
    assert len(shown) < len(fed[0].content)


@pytest.mark.anyio
async def test_evidence_block_numbering_is_one_based_and_matches_citations():
    """证据块编号必须从 **[1]** 起 —— 它就是 `citations[].n` 的来源。

    `make_retrieve_knowledge_node` 给 citations 编的是 `{"n": i + 1}`,
    前端 `app/static/index.html` 用 `citations.find(x => x.n === n)`
    把正文里的 `[n]` 变成可点开的来源。这里要是从 `[0]` 起,模型照着抄 `[0]`,
    前端**永远匹配不到第一个来源** —— 引用 UI 静默失效,零报错。
    """
    text = render_evidence([
        {"section_path": "退换货 > 退货政策", "category": "退换货", "answer": "七天无理由"},
        {"section_path": None, "category": "物流", "answer": "48 小时内发货"},
    ])
    assert "[1] (退换货 > 退货政策) 七天无理由" in text
    assert "[2] (物流) 48 小时内发货" in text     # section_path 为 None 时退回 category
    assert "[0]" not in text
    assert "[3]" not in text


@pytest.mark.anyio
async def test_agent_uses_the_resolved_input_not_the_raw_utterance():
    """Agent 读的是 `resolved_input`,不是 `user_input`。

    本章指代消解是**原样透传**,两个键今天恒等,所以这条断言今天不区分任何东西
    —— 它的价值在**下一步**:指代消解落地后(「它」「那个」被补全成具体商品/订单),
    读 `user_input` 的实现会让 Agent 看到**未消解的原话**,用户问「它还有货吗」,
    模型把「它」当成商品名去查。那时这条断言是唯一会红的东西。
    """
    model = ScriptedModel([[FakeChunk("好")]])
    await _node(model).__call__(
        _state(user_input="它还有货吗", resolved_input="那件连衣裙还有货吗")
    )
    sent = _text(model.bound_messages[-1])
    assert "那件连衣裙还有货吗" in sent
    assert "它还有货吗" not in sent
