"""指代消解节点:改写结果进 `resolved_input`;**失败/空输出一律原样透传**。

本节点还兼任「每轮重置」(见 `app/agent/nodes.py:make_resolve_references_node`
的 docstring)—— checkpointer 是进程级单例、thread_id = session_id,
不清零就会把上一轮的通道值报成本轮的,而且一路静默。加消解**不得**丢掉这个职责。

**失败模拟哪一类(说明)**:`classify_intent` 在本文件同族里接的是
`OutputParserException` / `ValidationError`(「模型出参不可用」),这里接的是
`OutputParserException`(同一类语义)。真机上消解走的是**自由文本**,
框架不会抛它;真机上**可达**的那种失败是**模型吐了空串**
(`_EmptyModel` 那条用例),两者都走「原样透传」这同一个出口。
**没有**用裸 `Exception` 兜底:那会把 `AttributeError`/`TypeError` 这类
**我自己的实现缺陷**也伪装成「这轮没改写」,本仓栽过「静默故障」太多次。
"""

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.nodes import make_resolve_references_node
from app.agent.state import ChatState
from app.prompts import RESOLVE_SYSTEM_PROMPT, build_resolve_messages
from app.schemas import Message


class _RewriteModel:
    """替身:按脚本回答 —— 模拟「这句有指代,需要改写」那一类输出。"""

    def __init__(self, text):
        self.text = text
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        return AIMessage(content=self.text)


class _EchoModel:
    """替身:原样回显用户这一轮的话 —— 模拟「本来就完整,原样输出」。"""

    async def ainvoke(self, messages):
        return AIMessage(content=messages[-1].content)


class _EmptyModel:
    """替身:吐空串。**这是真机上可达的那种消解失败**(见模块 docstring)。"""

    def __init__(self, raw=""):
        self.raw = raw

    async def ainvoke(self, messages):
        return AIMessage(content=self.raw)


class _BoomModel:
    """替身:模型调用**抛错**。默认抛本仓「模型出参不可用」那一族。"""

    def __init__(self, exc):
        self.exc = exc

    async def ainvoke(self, messages):
        raise self.exc


def _history():
    return [
        Message(role="user", content="我买的猫砂盆不想要了"),
        Message(role="assistant", content="好的,请问是哪一张订单?"),
    ]


# ---- 改写本身(最重要的一条:没有它,「永远透传」的实现也全绿) ----


@pytest.mark.anyio
async def test_rewritten_input_replaces_the_original():
    """有指代时 `resolved_input` 必须是**模型的改写**,不是原话。

    ⚠️ 这条是本文件的**可证伪性来源**:只留「重置」与「失败透传」两条用例的话,
    把整个模型调用删掉、`resolved_input` 永远等于 `user_input` 的实现**照样全绿**
    —— 而那正是 T5 要做的全部事情。期望值(`猫砂盆能退吗`)与回显值(`它能退吗`)
    **不同**,所以它对「有没有真的改写」是可判别的。
    """
    model = _RewriteModel("猫砂盆能退吗")
    node = make_resolve_references_node(model=model)
    out = await node({"user_input": "它能退吗", "history": _history()})
    assert out["resolved_input"] == "猫砂盆能退吗"


# ---- 每轮重置(brief ⚠️:加消解不得丢掉它) ----


@pytest.mark.anyio
async def test_reset_still_clears_per_turn_channels():
    """加了消解之后,每轮重置**不能丢** —— 丢了会静默串轮。"""
    node = make_resolve_references_node(model=_EchoModel())
    out = await node({"user_input": "这个能退吗", "gate_passed": True,
                      "agent_steps": 3, "order_no": "1001"})
    assert out["gate_passed"] is None
    assert out["agent_steps"] == 0


@pytest.mark.anyio
async def test_reset_covers_every_per_turn_channel_not_just_two():
    """重置要**逐个通道**都清 —— 只钉 `gate_passed` 会漏掉 `evidence`。

    `evidence` 尤其真实:它被 agent 节点读进 `build_messages`,业务轮跟在知识轮后
    会把**上一轮的检索结果**当本轮知识塞进 prompt。整图用例盖不住它(闲聊轮不读
    `evidence`),所以在这里逐条断言。

    `order_no` **故意不在里面**:它是 T7 才加进 `ChatState` 的通道,
    现在写它会被 LangGraph 静默丢弃(详见计划 PF-2 与 `test_returned_keys_...`)。
    """
    node = make_resolve_references_node(model=_EchoModel())
    out = await node({
        "user_input": "在吗",
        "evidence": [{"answer": "上一轮的旧知识"}],
        "gate_passed": True,
        "agent_steps": 3,
        "reply": "上一轮的回复",
        "choices": ["handoff"],
        "citations": [{"n": 1}],
        "tool_calls_made": [{"name": "query_order"}],
    })
    assert out["trace"] == ["resolve_references"]
    assert out["evidence"] == []
    assert out["gate_passed"] is None
    assert out["agent_steps"] == 0
    assert out["reply"] == ""
    assert out["choices"] == []
    assert out["citations"] == []
    assert out["tool_calls_made"] == []


def test_returned_keys_are_all_declared_channels():
    """**T4 的 Critical 的同型守卫**:返回了未声明的 key,图里会被静默丢弃。

    LangGraph 对未声明通道的写入只 `logger.warning(... ignoring it)`,**不抛** ——
    于是单测(直接调节点)全绿、真机上空转。这里逐个核对返回的 key 都在
    `ChatState` 里。

    它同时钉住 PF-2:本任务**不许**提前写 `order_no` / `order_data` /
    `refund_decision`(那是 T7 的通道,清零随通道一起加)。
    """
    import asyncio

    node = make_resolve_references_node(model=_EchoModel())
    out = asyncio.run(node({"user_input": "在吗"}))
    undeclared = set(out) - set(ChatState.__annotations__)
    assert not undeclared, (
        f"这些 key 不在 ChatState 里,真机写入会被静默丢弃:{sorted(undeclared)}"
    )


# ---- 失败:原样透传 ----


@pytest.mark.anyio
async def test_resolution_failure_passes_through_original():
    """消解失败**原样透传** —— 绝不因为改写失败就答不出话。

    抛的是 `classify_intent` 接的同一族异常(「模型出参不可用」)。
    `PydanticOutputParser.parse` 会把 `ValidationError` 包成它,
    所以这一个类也覆盖「将来改成结构化出参」那条路。
    """
    node = make_resolve_references_node(model=_BoomModel(OutputParserException("不是一句问题")))
    out = await node({"user_input": "这个能退吗", "history": _history()})
    assert out["resolved_input"] == "这个能退吗"


@pytest.mark.parametrize("raw", ["", "   ", "\n"])
@pytest.mark.anyio
async def test_empty_model_output_passes_through_original(raw):
    """**真机可达的那种失败**:模型吐了空串/空白 —— 同样原样透传。

    少了这条,「空输出」这条路上的实现(比如直接把 `""` 写进 `resolved_input`)
    没有任何断言看着,而它会让**整轮对话的输入变成空串**。
    """
    node = make_resolve_references_node(model=_EmptyModel(raw))
    out = await node({"user_input": "这个能退吗", "history": _history()})
    assert out["resolved_input"] == "这个能退吗"


@pytest.mark.anyio
async def test_failure_still_resets_and_still_reports_the_original():
    """失败路径也要**照常重置** —— 在 `try` 里做重置,失败时就整段跳过。"""
    node = make_resolve_references_node(
        model=_BoomModel(OutputParserException("不是一句问题"))
    )
    out = await node({"user_input": "这个能退吗", "gate_passed": True})
    assert out["resolved_input"] == "这个能退吗"
    assert out["gate_passed"] is None
    assert out["trace"] == ["resolve_references"]


# ---- 提示词装配(唯一出口是 prompts.build_resolve_messages) ----


@pytest.mark.anyio
async def test_node_sends_exactly_what_prompts_builds():
    """节点发给模型的,必须**逐条等于** `build_resolve_messages` 的产物。

    这一条钉两件事:(1) 提示词装配留在 `app/prompts.py` 这**唯一出口**
    (节点里另起一套 = 两份 prompt 漂移,而漂移不会报错);
    (2) 历史**真的**进了这次调用 —— 指代消解的原料就是它。
    只断「消息里有猫砂盆」是不够的:漏掉整段历史、只发本轮也可能命中。
    """
    model = _RewriteModel("猫砂盆能退吗")
    node = make_resolve_references_node(model=model)
    await node({"user_input": "它能退吗", "history": _history()})

    assert model.calls[0] == build_resolve_messages(
        history=_history(), user_input="它能退吗"
    )


def test_build_resolve_messages_shape():
    """形状契约:system 打头、历史按 `to_lc_messages` 转、本轮原话在最后一条。"""
    messages = build_resolve_messages(
        history=_history(), user_input="它能退吗"
    )
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == RESOLVE_SYSTEM_PROMPT
    assert [type(m).__name__ for m in messages[1:]] == [
        "HumanMessage", "AIMessage", "HumanMessage",
    ]
    assert messages[1].content == "我买的猫砂盆不想要了"
    assert messages[-1].content == "它能退吗"


def test_build_resolve_messages_tolerates_empty_history():
    """没有历史(本轮是会话第一句)也要能装配 —— 空输出/空历史不是异常。"""
    messages = build_resolve_messages(history=[], user_input="退货政策是怎么规定的")
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[-1], HumanMessage)
    assert messages[-1].content == "退货政策是怎么规定的"


# ---- 接缝:改写必须真的走到下游 ----


class _RetrievedChunk:
    """`KnowledgeRetriever.search` 的返回项,只带上节点会读的字段。"""

    def __init__(self, answer, score):
        self.chunk_id = 1
        self.section_path = "退货政策"
        self.question = "怎么退货"
        self.answer = answer
        self.category = "退换货"
        self.score = score


class _Retriever:
    def __init__(self):
        self.calls = []

    async def search(self, query):
        self.calls.append(query)
        return [_RetrievedChunk("七天无理由", 0.9)]


class _GraphModel:
    """整图替身:`ainvoke` 给改写(消解),`astream` 给答复(Agent)。"""

    def __init__(self, rewritten):
        self.rewritten = rewritten

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        return AIMessage(content=self.rewritten)

    async def astream(self, messages):
        yield AIMessage(content="好的")


class _Intent:
    def __init__(self, intent, confidence=0.9):
        self.intent = intent
        self.confidence = confidence


class _IntentModel:
    def __init__(self, intent):
        self.intent = intent

    def with_structured_output(self, schema, method=None):
        return self

    async def ainvoke(self, messages):
        return _Intent(self.intent)


class _Session:
    def __init__(self):
        self.added = []

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        pass


@pytest.mark.anyio
async def test_rewrite_travels_through_the_graph_to_the_retriever():
    """**接缝测试**:改写过的 `resolved_input` 必须真的走到下游。

    节点级用例证明不了这件事 —— 把 `resolved_input` 写错名字(或通道没在
    `ChatState` 里声明),节点级断言全绿,而图里那一写被 LangGraph **静默丢弃**
    (`wrote to unknown channel ... ignoring it`):T4 的 Critical 就是这个形态。
    所以这里跑**真图**,并且断言两处:检索器收到的 query 是**改写后的那句**
    (下游确实用了它),以及终态里的 `resolved_input`(通道里确实落了值)。

    期望值(`猫砂盆能退吗`)与「原样透传」会给出的值(`它能退吗`)**不同**,
    所以它对「改写有没有生效」是可判别的。
    """
    from langgraph.checkpoint.memory import InMemorySaver

    from app.agent.graph import build_graph
    from app.config import Settings

    retriever = _Retriever()
    graph = build_graph(
        model=_GraphModel("猫砂盆能退吗"),
        intent_model=_IntentModel("商品咨询"),
        tools=[], registry={},
        settings=Settings(
            _env_file=None,
            openai_base_url="https://example.invalid/v1", openai_api_key="sk-test",
            openai_model="m", database_url="mysql+asyncmy://u:p@h:3306/db",
        ),
        retriever=retriever, session=_Session(), conversation_id="conv-resolve",
        emit=lambda payload: None, checkpointer=InMemorySaver(),
    )
    out = await graph.ainvoke(
        {"conversation_id": "conv-resolve", "user_input": "它能退吗",
         "history": _history(), "trace": []},
        config={"configurable": {"thread_id": "resolve-seam"}},
    )

    assert out["resolved_input"] == "猫砂盆能退吗"
    assert retriever.calls == ["猫砂盆能退吗"]
