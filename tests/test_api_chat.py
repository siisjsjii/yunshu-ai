"""聊天接口测试。全部用替身:不联网、不碰 MySQL。

ch02 Task 11 整体改写:端点的历史来源从进程内 `SessionStore` 换成 MySQL,
并接上单轮工具编排。夹具随之从"真 SessionStore + 假模型"换成
"替身 DB 会话 + 假模型"。逐条去留见 task-11-report.md。
"""

import asyncio
import json
import logging

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain.tools import tool
from langchain_core.messages import AIMessage, AIMessageChunk
from sqlalchemy import Update, select, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.sql.elements import BinaryExpression

from app.agent.state import RefundJudgement
from app.api import chat as chat_api
from app.config import Settings, get_settings
from app.db.models import Conversation, ConversationSummary, MessageRecord
from app.db.session import get_session
from app.main import app
from app.memory import budget as memory_budget
from app.memory import layers, trim
from app.memory.store import SessionStore
from app.prompts import render_system_prompt
from app.schemas import Message
from app.refund.orders import DEMO_ORDERS
from app.retrieval.expand import ExpandQueries
from app.retrieval.search import RetrievedChunk
from app.tools import registry as tools_registry
from app.tools.errors import ToolNotFound
from app.tools.mock_data import order_record

REQUIRED = {
    "openai_base_url": "https://example.invalid/v1",
    "openai_api_key": "sk-test",
    "openai_model": "test-model",
    "database_url": "mysql+asyncmy://u:p@h:3306/db",
}


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **REQUIRED, **overrides)


# ---------- ch06 退款子流程的常量 ----------

#: 点选卡片时递上去的订单号。**必须不在 `DEMO_ORDERS` 里**(T7 报告点名的第一种
#: 假绿形态):取演示池里的号的话,「resume 真的生效了」与「端点根本没读 resume、
#: 卡片候选里的演示单被拿去查了」会给出**同一个观测值**,断言零判别力。
PICKED_ORDER = "83746592"

#: 判定「能退」时给的话术。**字面量**,且与 `refund_nodes` 的两个兜底文案
#: (OFFER_FALLBACK / EXPLAIN_FALLBACK)都不撞 —— 撞上的话「判定的话术真的流到
#: 线上了吗」这条断言就恒真(节点退回兜底文案也照样绿)。
JUDGE_YES_REPLY = "这一单还在七天无理由期内,可以申请退款。"

#: 检索到的条款块。判定 prompt 里的 [n] 编号与 `citations` 帧同源,靠它认出来。
CLAUSE_CHUNK = RetrievedChunk(
    "定制商品能退货吗", "定制类商品一经确认不支持七天无理由退货", "退换货",
    chunk_id=7, section_path="退货政策 > 例外", score=0.71,
)


# ---------- 替身 ----------


class FakeChunk(AIMessageChunk):
    """模拟 AIMessageChunk:支持 + 累加,累加后携带 tool_calls。

    `"type": "tool_call"` 这个键**必须有**:真实链路上模型流出的 tool_call
    chunk 经 langchain 的 default_tool_parser 解析后带着它,而
    `BaseTool.ainvoke` 判"这是不是一次工具调用"只看
    `x.get("type") == "tool_call"`。缺键时 langchain 会把整个 dict 当成
    参数去校验,每次调用都退化成"参数不合法"的可恢复失败 —— 事件序列
    照样长得像那么回事,却一次都没走到真实执行路径上。

    ch07:**基类从 `object` 换成真的 `AIMessageChunk`**。agent 节点现在把
    累积出来的 chunk **整个塞进 `state["messages"]`**,而那个通道的
    `add_messages` reducer 会对每个条目做消息强制转换 —— 一个裸对象的红法是
    `NotImplementedError: Unsupported message type: <class '...FakeChunk'>`,
    指向替身而不是实现。真实链路上这里流的**就是** `AIMessageChunk`
    (`create_react_agent` 的模型节点也这么写),所以这不是给实现兜底,
    是把替身补齐到生产形状(与 T3 给替身补 `flush` 同一条理由)。
    """

    def __init__(self, text="", tool_calls=None, usage=None):
        # content/`tool_calls` 都走真实字段;`.text` 由基类的属性给出(即 content 的
        # 纯文本视图),不再自己塞一个 —— 两个属性不会再有对不上的可能。
        super().__init__(
            content=text,
            tool_calls=[{"type": "tool_call", **tc} for tc in (tool_calls or [])],
        )
        self.usage_metadata = usage

    def __add__(self, other):
        # 与既有实现同义(text/tool_calls 相加、usage 取非空的那个)。
        # **刻意不换成基类的 `__add__`**:真实 chunk 的合并走的是
        # `tool_call_chunks` 的按 index 分片拼接,而本文件的替身是把
        # tool_call 整条塞进一批里 —— 那套分片语义在这里用不上,硬换会让
        # 「一轮里两个 chunk 各带一个 tool_call」这条用例变成另一种形状。
        return FakeChunk(
            text=self.text + other.text,
            tool_calls=self.tool_calls + other.tool_calls,
            usage=other.usage_metadata or self.usage_metadata,
        )


class _EchoAinvoke:
    """补上 `ainvoke`(消解)那半边入口。

    ch06 T5 起 `resolve_references` **每轮请求都会** `ainvoke` 一次,而本文件的
    替身原先只有 Agent 走的 `astream`/`bind_tools` 半边 —— 少了这一行,
    端点里的 `AttributeError` 会被 `generate()` 的 `except Exception` 收成
    **error 帧**,于是一整批用例红在「本该 done 却 error」上,指向的还是脚手架。

    回显用户这一轮的原话(=「这句本来就完整,原样输出」那一类回答):
    `resolved_input` 因此等于 `user_input`,既有的
    「模型收到的最后一条 human 消息」等断言仍在问同一件事。

    真实模型(`app.llm.create_chat_model` → `ChatOpenAI`)两种入口都有 ——
    这不是给实现兜底,是把替身补齐到生产形状。
    """

    async def ainvoke(self, messages):
        return AIMessage(content=messages[-1].content)


class _BoundModel:
    def __init__(self, inner):
        self._inner = inner

    async def astream(self, messages):
        self._inner.calls.append(("bound", list(messages)))
        for chunk in self._inner.batches.pop(0):
            yield chunk


class ScriptedModel(_EchoAinvoke):
    """按顺序回放预置 chunk 批次的替身。

    记录每次 astream 走的是**绑了工具**还是**未绑工具**的入口,以及
    绑上去的到底是哪批工具 —— 后者是"绑给模型的集合与注册表同源"的
    观测点,而事件序列本身看不出这件事。

    ch06 起它还兼任**退款子流程的判定/扩写模型**(子流程用的是同一个主力
    模型,不新起一个),所以补上 `with_structured_output` 那一半入口 ——
    与 `_EchoAinvoke` 补 `ainvoke` 同一条理由:替身必须与生产形状一致,
    少一半入口时红的是「本该 done 却 error」,而错误指向脚手架。
    """

    def __init__(self, batches, judge_reply=JUDGE_YES_REPLY, queries=("退款政策",)):
        self.batches = list(batches)
        self.calls = []
        self.bound_tools = None
        self.judge_reply = judge_reply
        self.queries = list(queries)
        #: 每次结构化出参的 (schema 名, messages)。**判定 prompt 里有没有
        #: 那一单的数据**只能从这里看 —— 帧上只能看到订单号,看不到取数结果。
        self.structured: list[tuple[str, list]] = []

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return _BoundModel(self)

    def with_structured_output(self, schema, method=None):
        # 与 `FakeIntentModel` 同一行断言:本项目端点上结构化出参只有 json_mode
        # 一条路(function_calling / json_schema 均返回 400)。
        assert method == "json_mode", "结构化出参只能用 json_mode(本项目硬约束)"
        return _ScriptedChain(self, schema)

    async def astream(self, messages):
        self.calls.append(("unbound", list(messages)))
        for chunk in self.batches.pop(0):
            yield chunk

    @property
    def last_messages(self):
        return self.calls[-1][1] if self.calls else None


class _ScriptedChain:
    """`with_structured_output(...)` 的返回物:一个只有 `ainvoke` 的对象。"""

    def __init__(self, model, schema):
        self._model, self._schema = model, schema

    async def ainvoke(self, messages):
        self._model.structured.append((self._schema.__name__, list(messages)))
        if self._schema is ExpandQueries:
            return _Queries(self._model.queries)
        if self._schema is RefundJudgement:
            return _Judgement(self._model.judge_reply)
        raise AssertionError(f"未预期的结构化出参 schema:{self._schema}")


class _Queries:
    def __init__(self, queries):
        self.queries = list(queries)


class _Judgement:
    def __init__(self, reply, can_refund=True):
        self.can_refund = can_refund
        self.reply = reply


class FakeRetriever:
    """检索器替身。

    **必须有**:端点每请求都会 `build_retriever(session)`(真实实现会连 Milvus
    并按需加载 BGE-M3 权重),而「单测全程不联网」是硬规矩。用例通过
    `client_factory(retriever=...)` 把 `chat_api.build_retriever` 换掉 ——
    与 `build_tools` 走同一个注入缝。
    """

    def __init__(self, chunks=()):
        self.chunks = list(chunks)
        self.calls: list[str] = []

    async def search(self, query):
        self.calls.append(query)
        return list(self.chunks)


class _Intent:
    """意图分类器出参替身。`confidence` 必须跟着补 —— 见
    `tests/test_agent_graph.py:_Intent`:少一个字段,替身就不是生产结果的形状,
    而节点那边一加 `getattr` 兜底,「字段缺失」就再也不会红了。
    """

    def __init__(self, intent, confidence=0.9):
        self.intent = intent
        self.confidence = confidence


class FakeIntentModel:
    """意图识别节点的替身,形状对齐 `tests/test_agent_intent.py:FakeStructuredModel`。

    **为什么必须有这个替身**:ch05 在模型**前面**插了意图识别节点,而它
    **每一个请求都会跑**、每一个请求都会真的 `.ainvoke` 一次。不替换掉它,
    进入端点/图的用例都会朝 `https://example.invalid/v1` 发一次真实请求 ——
    「单测全程不联网」是硬规矩,不是偏好。端点为它留了 `get_intent_model`
    这个 `Depends` 缝(与 `get_chat_model` 并排),下面一行 override 就接上了。

    `with_structured_output` 里断言 `method == "json_mode"`:`IntentResult` 是
    出参结构,本项目的端点上 `function_calling` / `json_schema` 都返回 400
    (CLAUDE.md 的硬约束,`app/services/extract.py` 同一条)。替身不看 schema
    就看不出这件事,所以这一行必须留着。
    """

    def __init__(self, intent="订单", confidence=0.9):
        self.intent = intent
        #: 可改。`test_done_frame_carries_the_intent_confidence` 用一个**非默认**
        #: 的值改它 —— 与默认值撞车时,「端点透传了 confidence」与「谁给了个默认
        #: 值」给出同一个观测值。
        self.confidence = confidence

    def with_structured_output(self, schema, method=None):
        assert method == "json_mode", "结构化出参只能用 json_mode(本项目硬约束)"
        return self

    async def ainvoke(self, messages):
        return _Intent(self.intent, self.confidence)


class _Result:
    """`session.execute(...)` 的返回值:支持 .scalars().all() / .one_or_none()。"""

    def __init__(self, rows):
        self._rows = list(rows)

    def scalars(self):
        return self

    def all(self):
        return list(self._rows)

    def one_or_none(self):
        return self._rows[0] if self._rows else None


def _where_value(stmt):
    """取出 `select(X).where(X.col == 值)` 里的那个值。

    替身只支持这一种查询形态 —— 取不出来时**直接报错**,而不是退化成
    "返回全部"。忽略 where 的替身会让"历史串了会话"这类缺陷无从观测。
    """
    clause = stmt.whereclause
    if not isinstance(clause, BinaryExpression):
        raise AssertionError(f"替身不支持的查询形态:{stmt}")
    return clause.right.value


class FakeSession:
    """替身 DB 会话,支撑 ensure_conversation / load_history / append_turn 的往返。

    刻意做成**有状态**的:一个恒返回空结果的替身会让"上一轮的历史读回来
    了吗"无从断言 —— 而第二轮带着第一轮上下文,正是本文件要钉的东西。
    """

    def __init__(self):
        self.conversations: dict[str, Conversation] = {}
        self.messages: list[MessageRecord] = []
        #: ch07 T7 起端点在流开始前读一次梗概(`load_summaries`)打 `history_ctx`。
        #: 替身不认识 `ConversationSummary` 的话,**每一条**非续跑用例都会红在
        #: 「替身不支持的实体」上 —— 而那指向脚手架、不指向实现。
        self.summaries: list[ConversationSummary] = []
        self.commits = 0
        self._next_id = 1

    async def execute(self, stmt, *args, **kwargs):
        # ch07 起端点还会发锚点推进用的 `update(...)`(降级路径,§7.2)。
        # **必须真的改内存里的那条 Conversation** —— 做成 no-op 的替身会让
        # 「降级真的持久化了吗」恒真(tests/test_api_chat.py 的 T10 那几条)。
        # 本任务(T5)的端点路径还不会走到这里(update 只在 T10 接上),
        # 先按计划把替身补齐:补它是替身的事,不是实现的事。
        if isinstance(stmt, Update):
            conversation = self.conversations[stmt.whereclause.right.value]
            for col, val in stmt._values.items():
                # ⚠️ `.values(layer1_from_msg_id=7)` 的 `_values` 里存的是
                # **`BindParameter` 对象**,不是 7 本身 —— 原样 `setattr` 会把
                # 绑定参数写进 ORM 属性,于是 T10 的断言
                # (`conv.layer1_from_msg_id > 0`)在**比较**那一步炸成 TypeError,
                # 读起来像实现坏了。下面这行同时兜住「绑定参数」与「已经是字面量」
                # 两种形态(自检见
                # `test_fake_session_applies_updates_and_still_rejects_unknown_queries`)。
                setattr(conversation, col.key, getattr(val, "value", val))
            return _Result([])
        entity = stmt.column_descriptions[0]["entity"]
        # ↓ 以下原样保留(Conversation / MessageRecord 两个分支),一行未改。
        value = _where_value(stmt)
        if entity is Conversation:
            conversation = self.conversations.get(value)
            return _Result([conversation] if conversation is not None else [])
        if entity is MessageRecord:
            rows = [m for m in self.messages if m.conversation_id == value]
            return _Result(sorted(rows, key=lambda m: m.id))
        if entity is ConversationSummary:
            rows = [s for s in self.summaries if s.conversation_id == value]
            return _Result(sorted(rows, key=lambda s: s.seq))
        raise AssertionError(f"替身不支持的实体:{entity}")

    def add(self, obj):
        # ch07 T10:真实 `AsyncSession` 在 flush/commit 时会把**标量 Python 侧默认值**
        # 落到对象上(`Conversation.summary_upto_msg_id` 的 `default=0` 就是这类,
        # 与表上的 `server_default="0"` 互为表里)。替身不模拟这一步的话,
        # `ensure_conversation` **新建**出来的会话两个锚点是 `None` ——
        # 而端点的分层要拿它们与 `messages.id` 比大小,红法是一句
        # `TypeError: '<' not supported between instances of 'NoneType' and 'int'`
        # 指向 `layers`,而真正缺的是替身的这一步。**补替身,不给实现加兜底**:
        # 生产路径上这两个值确实是 0(真实库读回来实测就是 `0 / 0`)。
        for column in obj.__table__.columns:
            if getattr(obj, column.key) is None and getattr(
                column.default, "is_scalar", False
            ):
                setattr(obj, column.key, column.default.arg)
        if isinstance(obj, Conversation):
            self.conversations[obj.id] = obj
        elif isinstance(obj, MessageRecord):
            obj.id = self._next_id
            self._next_id += 1
            self.messages.append(obj)
        elif isinstance(obj, ConversationSummary):
            self.summaries.append(obj)
        else:
            raise AssertionError(f"替身不支持的实体:{type(obj)}")

    async def flush(self):
        """ch07 起 `append_turn` 落库前会 flush 拿自增主键。

        替身里主键在 `add` 就分配好了,所以这里是空操作 —— 但**必须存在**:
        真实 AsyncSession 有这个方法,替身没有的话被观测的就不是
        「append_turn 返回了什么」,而是「替身少了哪个方法」。
        """
        pass

    async def commit(self):
        self.commits += 1


@pytest.fixture
def client_factory(monkeypatch):
    """造一个端点级测试客户端,替换掉模型、会话存储与 DB 会话。

    返回 `(client, model)`;client 上另挂了 `.db` / `.store` / `.model` /
    `.intent_model`,供测试在请求结束后观察落库内容与锁状态。

    `intent` 的**默认值必须是业务数据类(物流/订单/售后)**。本文件的用例把
    `ScriptedModel(batches)` 的批次当作「Agent 节点一定会来消费」来写;意图若是
    `其他`/`闲聊`/`投诉`,路由**根本不进 Agent 节点**,那些批次一个都消费不到 ——
    红法会是 `pop from empty list` 之类**看不出因果**的样子。定在业务类,现有
    批次脚本就仍由 Agent 节点照常消费;要单独覆盖路由行为的用例显式传
    `intent="闲聊"` / `intent="投诉"`。
    """

    def make(batches, session=None, registry=None, intent="订单", retriever=None,
             **settings_overrides):
        model = ScriptedModel(batches)
        intent_model = FakeIntentModel(intent)
        db = session if session is not None else FakeSession()
        store = SessionStore(ttl_seconds=60, max_sessions=10)

        # 端点每请求自己 `build_retriever(session)`,真实实现会连 Milvus 并按需
        # 加载 BGE-M3 权重 —— **默认也要换掉**,不是"传了才换":任何一条将来走到
        # 退款检索腿却没传 `retriever=` 的用例,都会在无人察觉的情况下真的出网
        # (「单测全程不联网」是硬规矩)。默认给一个**空结果**的哑检索器:
        # 走到检索腿的用例拿到"没命中"这个确定行为,而不是一次真实 IO。
        monkeypatch.setattr(
            chat_api,
            "build_retriever",
            lambda session: retriever if retriever is not None else FakeRetriever([]),
        )

        if registry is not None:
            # 端点用 build_tools 组装工具集、再 registry_for 建映射。
            # 换掉 build_tools 就是同时换掉"绑给模型的"和"能执行的"两批;
            # 端点若从别处取注册表,下面的断言会以"工具不存在"(ok=false)
            # 或"绑定内容不对"的形式变红。
            monkeypatch.setattr(
                chat_api,
                "build_tools",
                lambda *, session, conversation_id: list(registry.values()),
            )

        async def _session_override():
            yield db

        app.dependency_overrides[get_settings] = lambda: _settings(
            **settings_overrides
        )
        app.dependency_overrides[chat_api.get_store] = lambda: store
        app.dependency_overrides[chat_api.get_chat_model] = lambda: model
        # 意图识别是每请求都跑的一步 —— 这一行不做,进入端点/图的用例都会发真实请求。
        app.dependency_overrides[chat_api.get_intent_model] = lambda: intent_model
        app.dependency_overrides[get_session] = _session_override

        client = TestClient(app)
        client.db = db
        client.store = store
        client.model = model
        client.intent_model = intent_model
        return client, model

    yield make
    app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_fake_session_applies_updates_and_still_rejects_unknown_queries():
    """**替身自检**(计划要求的那一条)。

    两件事必须同时成立,少一件都会让 T10 的端点用例变成假绿:

    ① `update(...)` 分支**真的改内存里那条 Conversation** —— 做成 no-op 的替身
       会让「降级真的持久化了吗」恒真(断言读的是它刚写进去的值);
    ② `_where_value` 对不支持的查询形态**仍然直接抛** —— 顺手放宽成「返回全部」
       的话,「降级有没有写对行」就再也观测不到了(忽略 where 的替身会让
       「历史串了会话」这类缺陷无从观测);
    ③(ch07 T10 补)`add` 会替 SQLAlchemy 落**标量 Python 侧默认值**:
       `ensure_conversation` **新建**出来的会话两个锚点必须是 `0`,不是 `None`
       —— 真实库读回来实测就是 `0 / 0`(见 `test_ensure_conversation_...` 的说明),
       而 `None` 会让端点的分层在 `None < id` 上炸成 TypeError(指向 `layers`,
       而缺的是替身这一步)。

    ⚠️ 本任务(T5)的端点路径走不到 ①,它是给 T10 的降级路径预先铺好的 ——
    所以这里直接对替身做自检,而不是等 T10 用「碰巧跑到了」当证据。
    """
    db = FakeSession()
    db.conversations["c1"] = Conversation(
        id="c1", user="demo-user", status="active",
        summary_upto_msg_id=0, layer1_from_msg_id=0,
    )

    await db.execute(
        update(Conversation).where(Conversation.id == "c1").values(layer1_from_msg_id=7)
    )
    assert db.conversations["c1"].layer1_from_msg_id == 7

    with pytest.raises(AssertionError):
        await db.execute(select(Conversation))          # 没有 where ⇒ 替身不认识

    # ③ 新建的会话带着 ORM 的标量默认值(生产路径上 flush 会补,真实库读回来是 0)。
    fresh = Conversation(id="c-new", user="demo-user", status="active")
    assert fresh.summary_upto_msg_id is None             # ← 前提:ORM 默认值是**在 flush 时**落的
    db.add(fresh)
    assert (fresh.summary_upto_msg_id, fresh.layer1_from_msg_id) == (0, 0)


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    """把 SSE 响应体解析成 (event, data) 列表。"""
    events = []
    for block in body.strip().split("\n\n"):
        if not block.strip():
            continue
        name, payload = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                payload = json.loads(line[len("data: ") :])
        events.append((name, payload))
    return events


# ---------- 协议基础(沿用 ch01) ----------


def test_stream_emits_meta_then_tokens_then_done(client_factory):
    client, _ = client_factory(batches=[[FakeChunk("您"), FakeChunk("好")]])
    with client as c:
        resp = c.post("/api/chat/stream", json={"message": "你好"})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(resp.text)
    assert events[0][0] == "meta"
    assert events[-1][0] == "done"
    assert [d["text"] for name, d in events if name == "token"] == ["您", "好"]


def test_stream_generates_session_id_when_omitted(client_factory):
    client, _ = client_factory(batches=[[FakeChunk("您好")]])
    with client as c:
        resp = c.post("/api/chat/stream", json={"message": "你好"})

    meta = _parse_sse(resp.text)[0][1]
    assert isinstance(meta["session_id"], str)
    assert meta["session_id"]
    # 32 是 conversations.id 的列宽(见下面那条 422 测试)。生成器一旦改用
    # `str(uuid4())`(36 字符带横线),这里立刻变红 —— 而不是等到线上
    # INSERT 阶段。
    assert len(meta["session_id"]) == 32


def test_stream_echoes_provided_session_id(client_factory):
    client, _ = client_factory(batches=[[FakeChunk("您好")]])
    with client as c:
        resp = c.post(
            "/api/chat/stream", json={"session_id": "s1", "message": "你好"}
        )
    assert _parse_sse(resp.text)[0][1]["session_id"] == "s1"


def test_meta_reports_the_model_name(client_factory):
    client, _ = client_factory(batches=[[FakeChunk("您好")]])
    with client as c:
        resp = c.post("/api/chat/stream", json={"message": "你好"})
    assert _parse_sse(resp.text)[0][1]["model"] == "test-model"


def test_second_turn_receives_first_turn_context(client_factory):
    """第二轮必须带上第一轮的历史 —— 现在历史在 MySQL,不在进程内 dict。

    这条同时钉住读写两侧用的是同一个 conversation_id:append_turn 写错
    id、或 load_history 按错的 id 查,都会变红。
    """
    client, _ = client_factory(
        batches=[[FakeChunk("好的")], [FakeChunk("是 20240915")]]
    )
    with client as c:
        c.post(
            "/api/chat/stream",
            json={"session_id": "s1", "message": "订单是 20240915"},
        )
        c.post(
            "/api/chat/stream",
            json={"session_id": "s1", "message": "刚才那个订单号是多少？"},
        )

    contents = [m.content for m in client.model.last_messages]
    assert "订单是 20240915" in contents
    assert contents[-1] == "刚才那个订单号是多少？"

    # 两轮都落了库,且都挂在 s1 名下。
    assert [m.role for m in client.db.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert {m.conversation_id for m in client.db.messages} == {"s1"}


def _state_messages(session_id: str) -> list:
    """读**进程级 checkpointer** 里这个 thread 的 `messages` 通道。

    端点用的就是 `graph.get_checkpointer()` 那个单例(T5 起端点在流开始前
    读它做播种判据),所以这里读到的是真链路写进去的东西,不是替身的复述。
    """
    from app.agent.graph import get_checkpointer

    tup = get_checkpointer().get_tuple({"configurable": {"thread_id": session_id}})
    assert tup is not None, f"thread {session_id} 没有 checkpoint"
    return list(tup.checkpoint["channel_values"].get("messages") or [])


def test_second_turn_on_the_same_session_does_not_duplicate_history(client_factory):
    """**播种幂等** —— `add_messages` append-only 那个坑的落点。

    每轮无条件播种 ⇒ 重新构造的消息没有 id ⇒ 当场被赋全新 uuid ⇒
    整段历史被**再追加一遍**。第三轮时历史是三份,**而每一轮的回复看起来都正常**。

    观测点选**模型实际收到的消息条数** + **state 里 `messages` 的内容** ——
    帧、落库、HTTP 状态码在两种实现下**完全一样**。

    ⚠️ 与原计划的两处偏差:

    ① 计划写 `humans = [m for _, msgs in model.calls for m in msgs ...]`(跨
       **所有**调用累计)。那会数到**两轮**的入参:第一轮 1 条 + 第二轮 2 条 = 3,
       与它自己那句「上一轮 1 条 + 本轮 1 条 = 2」矛盾 —— 那条断言**永远红**。
       改成只看**第二轮**那一次调用(这正是它要观测的东西)。
    ② 只断「模型收到的条数」是**不够**的:模型收到的是 `history` 通道(端点从
       MySQL 读出来、按锚点分层后播种),而 `messages` 通道**不是模型入参的
       一部分**(它唯一的读边是播种判据本身)—— 于是「每轮重复播种」这个变异
       **一个模型入参都不改**,条数照样是 2。所以这里
       对 **checkpointer 里的 `messages`** 再断一次「没有一条内容出现两遍」:
       那才是播种真的坏掉时会变的地方(实测变异:去掉 `if not seeded` 守卫,
       下面第三组断言当场变红,见 task-5-report.md)。
    """
    client, model = client_factory(batches=[[FakeChunk("好")], [FakeChunk("的")]])
    with client as c:
        sid = _parse_sse(c.post("/api/chat/stream", json={"message": "第一句"}).text)
        sid = [d for e, d in sid if e == "meta"][0]["session_id"]
        c.post("/api/chat/stream", json={"session_id": sid, "message": "第二句"})

    # 第二轮模型收到的 human 消息:上一轮 1 条 + 本轮 1 条 = 2。
    #
    # ⚠️ 这条**不区分播种实现**,别把它当成播种的守卫(原始注释曾声称
    # 「重复播种会让上一轮那条被再追加一次 ⇒ 3 条」—— 那是错的):模型读的是
    # `history` 通道(端点从 MySQL 读出来再分层派生),**不是** `messages`,
    # 所以「每轮重复播种」一个模型入参都不改。判别力在下面第三组断言里。
    humans = [m for m in model.calls[-1][1] if type(m).__name__ == "HumanMessage"]
    assert len(humans) == 2
    assert [m.content for m in humans] == ["第一句", "第二句"]

    # state 里的完整历史:两轮之后必须是「问、答、问、答」四条 ——
    # 用户原话与客服回复**都在**,各只出现一次,且**按发生顺序**。
    #
    # 这一条同时钉住三件事(少任何一件它都会红):
    #   ① 「重复播种/重复并入」⇒ 多出一条(变异实测见 task-5-report.md §4);
    #   ② **用户原话根本没进 state** ⇒ 只剩 ['好', '的'] 两条 —— 这正是
    #      `resolve_references` 那一行存在的理由(缺了它,state 与 MySQL 从第二轮
    #      起就分叉,而帧、落库、状态码**完全正常**);
    #   ③ 顺序错(比如把本轮的用户消息并在助手回复之后)⇒ 序列对不上。
    #      顺序不是装饰:`add_messages` 并入的正是「模型看到的历史顺序」。
    contents = [m.content for m in _state_messages(sid)]
    assert contents == ["第一句", "好", "第二句", "的"]


@pytest.mark.anyio
async def test_resume_does_not_add_a_second_user_message_to_state(client_factory):
    """续跑**不重跑** `resolve_references` —— 那一轮的 user 消息只该有一条。

    这条守的是「用户消息只在一个地方加」这件事的**位置**选择:本节点是 START 的
    唯一出边、每轮只跑一次,而 `resume` 的图从**挂起的那个节点**继续,不回到
    START。若哪天有人把这行挪进一个 resume 会重跑的节点(或挪进端点),同一轮的
    用户原话就会在 state 里出现两次 —— 而**每一轮的回复看起来都正常**,
    落库那两条(user + assistant)也照样对(它们走的是 `turn_messages` / `user_input`)。

    判据是「挂起时就已经有了 1 条,续跑之后**仍然**是 1 条」—— 于是它同时钉住
    「挂起那一轮真的写进去了」(0 条的实现在前半段就红)。
    """
    retriever = FakeRetriever([CLAUSE_CHUNK])
    client, _ = client_factory(batches=[], intent="退款退货", retriever=retriever)
    with client as c:
        first = c.post(
            "/api/chat/stream",
            json={"session_id": "s-resume-msg", "message": "这个能退吗"},
        )
        assert "event: order_choice" in first.text
        suspended = [m.content for m in _state_messages("s-resume-msg")]

        second = c.post(
            "/api/chat/stream",
            json={"session_id": "s-resume-msg", "resume": {"order_no": PICKED_ORDER}},
        )
        resumed = [m.content for m in _state_messages("s-resume-msg")]

    assert [name for name, _ in _parse_sse(second.text)][-1] == "done"
    assert suspended.count("这个能退吗") == 1        # 挂起那一轮就写进去了
    assert resumed.count("这个能退吗") == 1          # 续跑没有再加一遍
    # 续跑那一轮照常产出客服回复(它由出口节点并入)。
    assert JUDGE_YES_REPLY in resumed


def test_first_request_seeds_the_history_that_is_already_in_mysql(client_factory):
    """**播种那一半**:state 为空(进程重启/新会话)时,历史必须**真的**被读进来。

    上一条只钉「不重复」——**不播种**的实现能让它全绿(0 条也是「没有重复」)。
    本条的会话在 MySQL 里**已经有历史**而同 thread 的 checkpoint 尚不存在,
    正是服务重启后的真实情形:`InMemorySaver` 是进程内的,重启后全空 ⇒
    下一次请求必须从 MySQL 补回来(spec §7.4 的「重启自愈」)。

    判据取「state 里出现了**库里那条**的内容」:把播种整段删掉,这里一条都看不到。
    """
    db = FakeSession()
    db.conversations["s-seed"] = Conversation(
        id="s-seed", user="demo-user", status="active",
        summary_upto_msg_id=0, layer1_from_msg_id=0,
    )
    db.add(MessageRecord(conversation_id="s-seed", role="user", content="上一轮的问题"))
    db.add(MessageRecord(conversation_id="s-seed", role="assistant", content="上一轮的回答"))

    client, _ = client_factory(batches=[[FakeChunk("本轮的回答")]], session=db)
    with client as c:
        resp = c.post("/api/chat/stream", json={"session_id": "s-seed", "message": "本轮的问题"})

    assert resp.status_code == 200
    msgs = _state_messages("s-seed")
    contents = [m.content for m in msgs]
    assert "上一轮的问题" in contents          # ← 播种真的发生了
    assert "上一轮的回答" in contents
    assert "本轮的回答" in contents            # ← 本轮那条照常并入
    # 稳定 id:播种进来的消息带的是 MySQL 主键(`str(1)` / `str(2)`),
    # **不是**当场生成的 uuid —— 「重播种幂等」全靠它。
    # 判据按**内容**挑出播种的那两条(而不是按位置/按"不是本轮那条"),
    # 免得将来「用户消息也进 state」时这条断言被误伤成假红。
    seeded_ids = [m.id for m in msgs if m.content in ("上一轮的问题", "上一轮的回答")]
    assert seeded_ids == ["1", "2"]


def test_unknown_session_id_is_created_silently(client_factory):
    """没有的会话不是 404,而是就地新建 —— 落点现在是 conversations 表。"""
    client, _ = client_factory(batches=[[FakeChunk("您好")]])
    with client as c:
        resp = c.post(
            "/api/chat/stream", json={"session_id": "brand-new", "message": "你好"}
        )

    assert resp.status_code == 200
    assert _parse_sse(resp.text)[0][1]["session_id"] == "brand-new"
    assert client.db.conversations["brand-new"].status == "active"


def test_empty_message_is_rejected(client_factory):
    client, _ = client_factory(batches=[])
    with client as c:
        resp = c.post("/api/chat/stream", json={"message": ""})
    assert resp.status_code == 422


def test_missing_message_is_rejected(client_factory):
    client, _ = client_factory(batches=[])
    with client as c:
        resp = c.post("/api/chat/stream", json={})
    assert resp.status_code == 422


def test_insufficient_budget_returns_400_before_streaming(client_factory):
    """预算不足必须在响应开始前报错。

    SSE 一旦 yield 过首帧,响应头就发出去了,状态码再也改不了 ——
    所以这里既要看 400,也要看它根本不是一条 SSE 流,且没有调用模型。

    ch07 换了判据的**来源**(用例原名 `test_oversized_input_...`,改名是因为
    它说的不再是要测的那件事):不再是「本轮输入顶穿 `context_budget_tokens`」,
    而是「从窗口倒推出来的历史预算为负」。于是这里的配置换成把窗口压到
    `ge=1024` 的下界 —— 默认的固定开销 + 单轮峰值远超它,任何输入都会 400。
    输入刻意用**短句**:长输入会让这条路径的触发点变成「输入超
    `max_user_input_tokens`」那条端点校验(spec §8,归 T10),那时它红/绿都
    不再说明历史预算这条路还在。
    """
    client, _ = client_factory(
        batches=[[FakeChunk("不会走到这里")]],
        model_context_window=1024,
        safety_margin_tokens=0,
    )
    with client as c:
        resp = c.post("/api/chat/stream", json={"message": "在吗"})

    assert resp.status_code == 400
    assert "tokens" in resp.text
    assert resp.headers["content-type"].startswith("application/json")
    assert client.model.calls == []


def test_overflow_400_releases_lock_so_the_session_stays_usable(client_factory):
    """`ContextOverflowError` 这条退出路径也要放锁 —— 它**不是**兜底那条例外。

    端点里 `except ContextOverflowError` 与 `except BaseException` 是两个
    并列子句,前者的 `lock.release()` 删掉后,后者**不会**兜住这个异常。
    后果不是"报错",而是该 session 从此永久 409:持锁的锁既不被 TTL 也不被
    LRU 回收,症状与"锁泄漏"毫无相似之处。另外两条非流式退出路径都有具名
    测试(test_validation_failure_releases_lock / test_tool_build_failure_releases_lock),
    唯独这条没有。

    断言取"第二个同样超预算的请求仍然拿到 400 而不是 409"(会话仍可用)—— 这才是
    真正要守的性质;等锁超时调成 0.15s,漏放锁时第二次请求会在 0.15s 内变红,
    而不是用默认 60s 把测试挂死。顺带断一次锁对象本身,便于定位。

    ch07:造 400 的配置换成「窗口压到 `ge=1024` 的下界」(判据的来源从
    「输入顶穿预算」变成「倒推出来的历史预算为负」);消息本身刻意用短句,
    理由同上一条 —— 长输入会把触发点换成端点那条 `max_user_input_tokens`
    校验(spec §8),这条用例就不再守着 `ContextOverflowError` 的放锁路径了。
    """
    client, _ = client_factory(
        batches=[],
        model_context_window=1024,
        safety_margin_tokens=0,
        session_lock_timeout_seconds=0.15,
    )
    body = {"session_id": "s1", "message": "在吗"}

    with client as c:
        first = c.post("/api/chat/stream", json=body)
        second = c.post("/api/chat/stream", json=body)

    assert [first.status_code, second.status_code] == [400, 400]
    assert client.store.lock_for("s1").locked() is False


def test_upstream_error_becomes_sse_error_event(client_factory):
    """上游炸了 → error 帧终止**且整轮不落库**(不留孤儿行)。

    两条断言缺一不可。只断 error 帧的话,把 `log_turn` 挪到 agent **之前**
    (或让异常路径也调一次 `append_turn`),流照样以 error 帧收尾、这条用例
    照样全绿 —— 而库里会躺着一条 user 消息(或者一条空回复的 assistant),
    下一轮 `load_history` 把它读回来当历史,用户看到客服"提过半句就哑了"。
    ch02 的 `test_history_is_not_written_when_second_round_breaks` 守的就是这条,
    它连同 `stream_turn` 一起被 T8 删掉 —— 新架构里「失败不落库」是**结构上**
    成立的(`log_turn` 是唯一写方且在下游),但零断言,所以在这里补上。
    """

    class ExplodingModel(_EchoAinvoke):
        """`ainvoke`(消解)照常应答,**`astream`(Agent)炸** —— 正是本用例要的:
        炸在 Agent 那一步,于是「失败不落库」这条断言仍在问原问题。"""

        def bind_tools(self, tools):
            return self

        async def astream(self, messages):
            yield FakeChunk("前半")
            raise RuntimeError("上游超时")

    client, _ = client_factory(batches=[])
    app.dependency_overrides[chat_api.get_chat_model] = lambda: ExplodingModel()

    with client as c:
        resp = c.post("/api/chat/stream", json={"message": "你好"})

    events = _parse_sse(resp.text)
    assert events[-1][0] == "error"
    assert "上游超时" in events[-1][1]["message"]
    # 半截回复绝不能污染历史:一行都没有。
    assert client.db.messages == []


def test_error_event_does_not_echo_the_configured_key(client_factory):
    """异常文本里带着真实密钥时必须被抹掉。

    这里让异常文本**自带**夹具配置的密钥值,才真正检验得到过滤逻辑;
    若断言的是"响应里没有 sk-test"而异常文本里根本没有它,那条断言恒真。
    """

    class LeakyModel(_EchoAinvoke):
        def bind_tools(self, tools):
            return self

        async def astream(self, messages):
            raise RuntimeError("Incorrect API key provided: sk-test")
            yield FakeChunk("永不产出")

    client, _ = client_factory(batches=[])
    app.dependency_overrides[chat_api.get_chat_model] = lambda: LeakyModel()

    with client as c:
        resp = c.post("/api/chat/stream", json={"message": "你好"})

    events = _parse_sse(resp.text)
    assert events[-1][0] == "error"
    assert "sk-test" not in resp.text
    assert "***" in events[-1][1]["message"]


@pytest.mark.anyio
async def test_concurrent_same_session_second_request_times_out_with_409(client_factory):
    """同 session 并发:一个拿到锁走完,另一个等锁超时 → 恰好一个 409。

    设计文档 §6 的最后一行,也是唯一没有测试覆盖的一行。必须真并发 ——
    顺序调用只会得到两个 200。

    用 httpx.AsyncClient + ASGITransport 而不是 TestClient:TestClient
    是同步的,两个线程里跑同一事件循环会带来额外的调度不确定性。
    """

    class SlowModel(_EchoAinvoke):
        """持有锁约 0.8s:远长于 0.15s 的等锁超时。

        慢的是 `astream`(**持锁的那一段**)。消解的 `ainvoke` 照常立刻返回 ——
        它也在锁内,但它不是本用例要制造的延迟来源。
        """

        def bind_tools(self, tools):
            return self

        async def astream(self, messages):
            await asyncio.sleep(0.4)
            yield FakeChunk("您")
            await asyncio.sleep(0.4)
            yield FakeChunk("好")

    client, _ = client_factory(batches=[], session_lock_timeout_seconds=0.15)
    app.dependency_overrides[chat_api.get_chat_model] = lambda: SlowModel()
    db = client.db

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as c:
            first, second = await asyncio.gather(
                c.post(
                    "/api/chat/stream",
                    json={"session_id": "s1", "message": "你好"},
                ),
                c.post(
                    "/api/chat/stream",
                    json={"session_id": "s1", "message": "在吗"},
                ),
            )
    finally:
        app.dependency_overrides.clear()

    assert sorted([first.status_code, second.status_code]) == [200, 409]
    loser = first if first.status_code == 409 else second
    assert "正在处理另一条消息" in loser.text

    # 输的那个请求一条历史都没写 —— 恰好一轮(user + assistant)。
    # ch01 这里断的是 store.history("s1")(历史在进程内),历史迁到 DB 后
    # 改在替身会话上观察同一件事。
    assert [m.role for m in db.messages] == ["user", "assistant"]


def test_validation_failure_releases_lock(client_factory, monkeypatch):
    """非流式退出路径必须释放锁,否则该会话永久 409。

    替身的签名必须与真实 prepare_turn 一致。ch01 那份还停在
    `(*, settings, store, session_id, user_input)`,而 T10 之后真实签名是
    `(*, settings, budget, user_input) -> None`(纯校验,不收历史)——
    签名对不上时它抛的是 TypeError(而且被 pytest.raises(RuntimeError) 拦下,
    测试仍会红但红在一个误导性的位置)。
    """

    def boom(*, settings, budget, user_input):
        raise RuntimeError("预算校验失败")

    client, _ = client_factory(batches=[])
    monkeypatch.setattr(chat_api, "prepare_turn", boom)

    # 未处理的异常在默认 TestClient(raise_server_exceptions=True)下会被
    # 重新抛出,而不是变成 500 响应 —— 生产环境下它才是 500。这里用
    # pytest.raises 断言"没有得到 200 SSE 流",再断言锁已释放。
    with client as c:
        with pytest.raises(RuntimeError):
            c.post(
                "/api/chat/stream", json={"session_id": "s1", "message": "你好"}
            )

    assert client.store.lock_for("s1").locked() is False


def test_tool_build_failure_releases_lock(client_factory, monkeypatch):
    """工具组装必须在放锁守卫**之内**。

    `make_query_faq` / `make_create_ticket` 会做导入并建闭包 —— 本章新加进
    "拿到锁之后"这段区域的代码。它们抛异常时漏放锁的后果不是"慢":持锁的
    锁既不被 TTL 也不被 LRU 回收,该 session_id 从此**永远 409**,症状看
    起来和"锁泄漏"毫无相似之处。

    所以这里不只断言"这次报错了",而是断言**这个会话之后还能用** ——
    把工具组装挪到守卫之外(见本条的变异记录),下面两条断言都会变红。
    """

    # 签名必须与 make_query_faq 一致(它现在多收一个 retriever)—— 桩函数
    # 少收一个参数时抛的是 TypeError,而下面 pytest.raises 等的是 RuntimeError,
    # 这条用例会以「没抛 RuntimeError」的样子假红。
    def boom(session, retriever):
        raise RuntimeError("工具组装失败")

    monkeypatch.setattr(tools_registry, "make_query_faq", boom)
    client, _ = client_factory(
        batches=[[FakeChunk("您好")]], session_lock_timeout_seconds=0.15
    )

    with client as c:
        with pytest.raises(RuntimeError):
            c.post(
                "/api/chat/stream", json={"session_id": "s1", "message": "你好"}
            )

        # 工具组装恢复正常。锁若没被放掉,下面这次请求会等锁超时 → 409。
        monkeypatch.undo()
        follow_up = c.post(
            "/api/chat/stream", json={"session_id": "s1", "message": "在吗"}
        )

    assert client.store.lock_for("s1").locked() is False
    assert follow_up.status_code == 200


def test_lock_is_released_after_a_successful_stream(client_factory):
    """流跑完之后锁必须放开。

    等锁超时调成 0.15s:真漏了 `finally: lock.release()`,第二次请求会
    在 0.15s 内以 409 变红,而不是用默认的 60s 把测试挂死。
    """
    client, _ = client_factory(
        batches=[[FakeChunk("您好")], [FakeChunk("在的")]],
        session_lock_timeout_seconds=0.15,
    )
    with client as c:
        first = c.post(
            "/api/chat/stream", json={"session_id": "s1", "message": "你好"}
        )
        second = c.post(
            "/api/chat/stream", json={"session_id": "s1", "message": "在吗"}
        )

    assert [first.status_code, second.status_code] == [200, 200]


def test_ensure_conversation_runs_under_the_lock(client_factory, monkeypatch):
    """建会话必须在**持锁之后**,这不是可有可无的细节。

    两个并发的首请求若都在锁外检查"会话存在吗",会同时看不到行,其中
    一个 INSERT 撞主键抛 IntegrityError。把 ensure_conversation 挪到
    `lock.acquire()` 之前,这条立刻变红。
    """
    client, _ = client_factory(batches=[[FakeChunk("您好")]])
    seen = {}

    async def spy(*, session, session_id, user_id):
        # 用同一个 store 问一次:此刻锁是否已被本请求持有。
        seen["locked"] = client.store.lock_for(session_id).locked()
        # 返回值必须**与生产形状一致**:`ensure_conversation` 从不返回 None
        # (新建时当场 commit,两个标量默认值随即落到属性上 —— 实测真实库读回来
        # 是 `0 / 0`)。返回 None 是形状违规,而 ch07 的降级与播种要读它的两个
        # 锚点 —— 那时这条用例会红在一个与「锁」毫无关系的地方。
        # 与 T3 给替身补 `flush` 是同一条理由:补替身,不是给实现加兜底。
        return Conversation(
            id=session_id, user=user_id, status="active",
            summary_upto_msg_id=0, layer1_from_msg_id=0,
        )

    monkeypatch.setattr(chat_api, "ensure_conversation", spy)

    with client as c:
        resp = c.post(
            "/api/chat/stream", json={"session_id": "s1", "message": "你好"}
        )

    assert resp.status_code == 200
    assert seen["locked"] is True


def test_session_id_wider_than_the_column_is_rejected_as_422(client_factory):
    """32 是 conversations.id 的列宽(与 uuid4().hex 同长)。

    放宽回 128 的话,33–128 字符的 id 会一路走到 INSERT,在 MySQL 严格
    模式下抛 DataError,按 §6.7 被当成不可恢复故障 → 502:一个参数问题
    被报成服务端故障。这里必须死在**流开始之前**的入参校验上。
    """
    client, _ = client_factory(batches=[[FakeChunk("您好")], [FakeChunk("您好")]])
    with client as c:
        too_long = c.post(
            "/api/chat/stream", json={"session_id": "s" * 33, "message": "你好"}
        )
        at_limit = c.post(
            "/api/chat/stream", json={"session_id": "s" * 32, "message": "你好"}
        )

    assert too_long.status_code == 422
    assert at_limit.status_code == 200


def test_user_id_wider_than_the_column_is_rejected_as_422(client_factory):
    """128 是 conversations.user 的列宽,理由同上。"""
    client, _ = client_factory(batches=[])
    with client as c:
        resp = c.post(
            "/api/chat/stream", json={"message": "你好", "user_id": "u" * 129}
        )
    assert resp.status_code == 422


def test_chat_stream_accepts_optional_user_id(client_factory):
    """user_id 必须真的落到 conversations.user。

    只断 200 的话,这个字段被整条丢掉(永远用缺省值)也一样通过。
    """
    client, _ = client_factory(batches=[[FakeChunk("您好")]])
    with client as c:
        resp = c.post(
            "/api/chat/stream", json={"message": "你好", "user_id": "alice"}
        )
        session_id = _parse_sse(resp.text)[0][1]["session_id"]

    assert resp.status_code == 200
    assert client.db.conversations[session_id].user == "alice"


def test_user_id_defaults_to_demo_user(client_factory):
    client, _ = client_factory(batches=[[FakeChunk("您好")]])
    with client as c:
        resp = c.post("/api/chat/stream", json={"message": "你好"})
        session_id = _parse_sse(resp.text)[0][1]["session_id"]

    assert resp.status_code == 200
    assert client.db.conversations[session_id].user == "demo-user"


# ---------- 工具事件(ch02 新增) ----------

#: 一个「已发货」的订单号 —— `ok=True` 要求工具**真的执行成功**。
#: 多数订单号(含 1001)是「未发货」,对它们查物流会正确地返回 ToolNotFound。
#: 「哪些号码有物流」由 tests/test_tools_random.py 的自洽断言守护;种子函数
#: 若变动,这里会以 `assert False is True` 硬失败,而不是被悄悄放宽。
_SHIPPED_ORDER = "1003"


def test_chat_stream_emits_tool_call_event(client_factory):
    """验收 1 的后端一半:必须推出 tool_call 帧,且 name / args 正确。

    这里不替换注册表 —— 走的是生产的那五个工具(query_logistics 是确定性
    伪随机、不碰 DB 的那个),顺带证明 build_tools 真被端点用上了。
    """
    client, model = client_factory(
        batches=[
            [
                FakeChunk(
                    tool_calls=[
                        {
                            "name": "query_logistics",
                            "args": {"order_id": _SHIPPED_ORDER},
                            "id": "c1",
                        }
                    ]
                )
            ],
            [FakeChunk("已揽件。")],
        ]
    )
    with client as c:
        resp = c.post(
            "/api/chat/stream", json={"message": f"订单 {_SHIPPED_ORDER} 的物流到哪了"}
        )
        events = _parse_sse(resp.text)

    kinds = [name for name, _ in events]
    assert "tool_call" in kinds
    assert "tool_result" in kinds
    payload = next(p for name, p in events if name == "tool_call")
    assert payload["name"] == "query_logistics"
    assert payload["args"] == {"order_id": _SHIPPED_ORDER}
    assert payload["tool_call_id"] == "c1"
    assert kinds[-1] == "done"

    # Task 13 的验收脚本按 tool_result → ok 取值,ok 必须来自真实执行结果。
    result = next(p for name, p in events if name == "tool_result")
    assert result["tool_call_id"] == "c1"
    assert result["ok"] is True

    # 绑给模型的就是生产工具集(绑一批、能执行另一批是本章的接线隐患;
    # "执行批"那一半由下面替换注册表的那条测试钉住)。
    assert {t.name for t in model.bound_tools} == {
        "query_order",
        "query_product",
        "query_logistics",
        "query_faq",
        "create_ticket",
    }

    # **调过工具的一轮落库的是完整往返**(ch07 起):user / assistant(tool_calls)
    # / tool / assistant。ch05–ch06 只落 user + assistant(reply)两条 —— 那句
    # 旧注释连同它守的东西一起在这里作废,因为**本章的层 2 要截的就是那条
    # tool 行**:不落它,「大块工具结果」在下一轮根本不存在。
    #
    # 四条各自钉住一件事,少一条就有一种实现能蒙混:
    #   ① 工具轮的 assistant(tool_calls)**落库且配对**(`tool_call_id` 与
    #      assistant 那条的 `tool_calls[].id` 一致)—— 落成两条无配对的
    #      assistant 也能让「角色列表」看起来对,但下一轮读历史转
    #      `ToolMessage` 时上游直接 400;
    #   ② tool 行的内容是**完整工具结果**,不是展示用的截断 summary;
    #   ③ 收尾那条 assistant 的内容来自**收尾那一轮**(`reply` 是各轮 parts 的
    #      拼接,工具轮的 token 是空的)。
    assert [m.role for m in client.db.messages] == ["user", "assistant", "tool", "assistant"]
    calls_row, tool_row, final_row = client.db.messages[1], client.db.messages[2], client.db.messages[3]
    assert [c["id"] for c in calls_row.tool_calls] == ["c1"]
    assert tool_row.tool_call_id == calls_row.tool_calls[0]["id"]
    # 落的是**完整**工具结果(JSON 文本),不是给前端看的截断 summary ——
    # 截断版是 JSON 前缀、`json.loads` 直接炸(而那正是层 2 截短前的原料)。
    assert json.loads(tool_row.content)["order_id"] == _SHIPPED_ORDER
    assert final_row.content == "已揽件。"


def test_recoverable_tool_failure_is_not_an_error_frame(client_factory):
    """可恢复失败(查无此单)回灌给模型,流正常 done —— 不是 error 帧。

    把它"改进"成 error 帧会让用户看到一条因订单号打错而中断的对话。
    """

    @tool
    async def query_order(order_id: str) -> str:
        """替身:总是找不到订单。"""
        raise ToolNotFound("未找到订单 9999")

    client, _ = client_factory(
        batches=[
            [
                FakeChunk(
                    tool_calls=[
                        {"name": "query_order", "args": {"order_id": "9999"}, "id": "c1"}
                    ]
                )
            ],
            [FakeChunk("没查到该订单,请核对单号。")],
        ],
        registry={"query_order": query_order},
    )
    with client as c:
        resp = c.post("/api/chat/stream", json={"message": "订单 9999 的状态"})
        events = _parse_sse(resp.text)

    kinds = [name for name, _ in events]
    assert "error" not in kinds
    assert kinds[-1] == "done"

    result = next(p for name, p in events if name == "tool_result")
    assert result["ok"] is False
    assert "未找到订单 9999" in result["summary"]


def test_failed_tool_summary_is_redacted(client_factory):
    """工具失败的 summary 也是出站文本,同样要过脱敏。

    spec §5.2:失败时 summary 为错误原因的一句话(**同样脱敏后**),
    ok=false。这里让替身把配置里的密钥写进错误文本 —— 不脱敏就直接
    出现在响应体里。
    """

    @tool
    async def query_order(order_id: str) -> str:
        """替身:错误文本里带着配置里的密钥。"""
        raise ToolNotFound("Incorrect API key provided: sk-test")

    client, _ = client_factory(
        batches=[
            [
                FakeChunk(
                    tool_calls=[
                        {"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}
                    ]
                )
            ],
            [FakeChunk("抱歉。")],
        ],
        registry={"query_order": query_order},
    )
    with client as c:
        resp = c.post("/api/chat/stream", json={"message": "订单 1001 的状态"})
        events = _parse_sse(resp.text)

    result = next(p for name, p in events if name == "tool_result")
    assert result["ok"] is False
    assert "sk-test" not in result["summary"]
    assert "***" in result["summary"]
    assert "sk-test" not in resp.text


def test_infrastructure_failure_emits_error_frame(client_factory):
    """基础设施故障 → error 帧终止流,不是 done。"""

    @tool
    async def query_order(order_id: str) -> str:
        """替身:DB 挂了。"""
        raise OperationalError("SELECT 1", {}, Exception("连接断开"))

    client, model = client_factory(
        batches=[
            [
                FakeChunk(
                    tool_calls=[
                        {"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}
                    ]
                )
            ]
        ],
        registry={"query_order": query_order},
    )
    with client as c:
        resp = c.post("/api/chat/stream", json={"message": "订单 1001 的状态"})
        events = _parse_sse(resp.text)

    # 恰好三帧:工具申请发出去、执行炸掉、流终止。没有 done。
    assert [name for name, _ in events] == ["meta", "tool_call", "error"]

    payload = next(p for name, p in events if name == "error")
    # 出站的是 executor 那句固定文案,不是原始 SQLAlchemy 异常文本。
    # (等价于"不回显密钥":原始文本里带什么都不会流出去 —— 而"异常文本
    # 自带密钥"那条路径由上面 LeakyModel 与 summary 脱敏两条覆盖。)
    assert payload["message"] == "数据服务暂时不可用"

    # 绑给模型的必须就是注册表里那批。端点若把 `tools` 与 `registry`
    # 取自两处(绑生产工具集、执行替身),这条会红。
    assert model.bound_tools == [query_order]


# ---------- 意图路由(默认值之外的出口) ----------
#
# 上面 25 条全部走 `intent="订单"`(业务数据类 → Agent 节点)。下面两条**显式**
# 传非默认意图,覆盖两件上面一条都看不见的事:
#   ① 路由真的按意图分叉(闲聊 / 投诉**不进** Agent 节点);
#   ② ch05 新增的 `choices` 帧真的走到了 SSE 线上 —— T10 的前端将消费这个帧
#      (「转人工 / 建工单」两个按钮),帧丢了后端一句话都不报,只是按钮永远不出现。


def test_chitchat_intent_returns_fixed_copy_without_calling_the_chat_model(client_factory):
    """闲聊出口:固定话术 + **零模型调用**,并把 `intent` / `trace` 折进 done 帧。

    `batches=[]` 是刻意的:脚本为空,Agent 节点一旦被走到就会 `pop from empty
    list` —— 这条因此同时是「闲聊不进 Agent」的结构探针,而不只是内容断言。

    done 帧那条断言钉的是**端点新增的折帧逻辑**:`trace` 帧在端点被折进 done、
    不外推;折的逻辑写错(比如忘了 `final = payload`)时,验收脚本依赖的
    「trace 里有 retrieve_knowledge / agent:converged」会**静默变空**,
    而上面所有只断 `events[-1][0] == "done"` 的用例照样全绿。
    """
    from app.agent.nodes import CHITCHAT_REPLY

    client, model = client_factory(batches=[], intent="闲聊")
    with client as c:
        resp = c.post("/api/chat/stream", json={"session_id": "s1", "message": "你好"})

    events = _parse_sse(resp.text)
    assert resp.status_code == 200
    assert [p["text"] for name, p in events if name == "token"] == [CHITCHAT_REPLY]
    assert model.calls == []                       # 闲聊出口一次模型都不调

    done = events[-1]
    assert done[0] == "done"
    assert done[1]["intent"] == "闲聊"
    assert "chitchat_reply" in done[1]["trace"]
    assert done[1]["agent_steps"] == 0


def test_complaint_intent_emits_choices_frame(client_factory):
    """投诉出口:安抚话术 + `choices` 帧(转人工 / 建工单两个选项)。

    选项是**后端给、前端渲染**的:`options` 的键名或结构改了,后端零报错、
    单测全绿,只有用户看不到按钮。
    """
    client, model = client_factory(batches=[], intent="投诉")
    with client as c:
        resp = c.post("/api/chat/stream", json={"session_id": "s1", "message": "我要投诉"})

    events = _parse_sse(resp.text)
    assert model.calls == []
    choices = next(p for name, p in events if name == "choices")
    assert choices["options"] == [
        {"key": "handoff", "label": "转人工"},
        {"key": "ticket", "label": "建工单"},
    ]
    assert [name for name, _ in events][-1] == "done"


# ---------- ch06:退款子流程(挂起 → 点卡片 → resume) ----------
#
# 图那一半(T7)由 `tests/test_agent_refund.py` 覆盖;这里只问端点这一半的三件事:
# **帧出得来吗**(F1)、**resume 请求体认不认**、**挂起的那一轮有没有脏写/占锁**。
#
# 本节所有用例都用 `intent="退款退货"` —— 上一节 27 条的默认意图是「订单」,
# 走的是 Agent 出口,**根本到不了子流程**,`batches` 也照常被消费。意图传错时
# 红法是「卡片的帧没出现」,与「流模式漏了 updates」长得一模一样 —— 所以每条
# 用例的注释里都写明了它到底在防什么。


@pytest.mark.anyio
async def test_suspended_turn_emits_the_order_choice_frame(client_factory):
    """F1 的回归防线:`stream_mode` 漏掉 `updates` 时,interrupt **被整个吞掉**。

    实测(langgraph 1.2.11):只给 `custom` 时 run 照常结束、`state.next` 停在
    待续节点、**一个帧都不吐、也不报错**。这个故障的形态极难定位 —— 用户侧看到
    的不是「卡片没出现」,是**除了 meta 什么都没有**;库里连一行痕都没有
    (log_turn 在下游,没跑)。所以这里两样都断。

    `["meta", "order_choice"]` 这条**逐帧相等**的断言是有意的:它同时钉住
    「挂起的一轮不发 done 帧」—— done 帧自报的是「这一轮跑完了」,而挂起时
    trace / intent / agent_steps 全是初值,发出去是在撒谎。
    """
    client, _ = client_factory(batches=[], intent="退款退货")
    with client as c:
        resp = c.post(
            "/api/chat/stream", json={"session_id": "s-card", "message": "这个能退吗"}
        )

    events = _parse_sse(resp.text)
    assert [name for name, _ in events] == ["meta", "order_choice"]

    options = events[1][1]["options"]
    # 卡片候选来自 `app.refund.orders` 的演示池(语料里一个号都没出现过)。
    # 断言的是**整份** options:少一项、顺序变了,前端渲染出来的卡片就跟着变。
    assert [o["order_no"] for o in options] == list(DEMO_ORDERS)
    # 前端读的是 `payload.options`,每个候选要么是裸串、要么带这三项 ——
    # 少了它们,卡片上只有一串数字(前端容忍,但用户看不到商品与金额)。
    assert all({"order_no", "status", "product", "amount"} <= set(o) for o in options)

    # **挂起的那一轮不落库**:log_turn 在下游,没跑。这里断的是「一行都没有」,
    # 而不是「没有 assistant」—— 半条 user 消息同样会毒化下一轮的 load_history。
    assert client.db.messages == []


@pytest.mark.anyio
async def test_suspended_turn_releases_the_lock(client_factory):
    """挂起即放锁 —— 用户可能很久才点那张卡片。

    锁若跟着挂起不放,该 session **永久 409**:持锁的锁既不被 TTL 也不被 LRU
    回收(见 `app/memory/store.py`),症状看起来与「锁泄漏」毫无相似之处。
    等锁超时调成 0.15s:真漏了 `finally: lock.release()`,第二次请求会在
    0.15s 内以 409 变红,而不是用默认 60s 把测试挂死。
    """
    client, _ = client_factory(
        batches=[], intent="退款退货", session_lock_timeout_seconds=0.15
    )
    with client as c:
        first = c.post(
            "/api/chat/stream", json={"session_id": "s-hold", "message": "这个能退吗"}
        )
        # 同一 session 的第二个请求:锁没放掉时它等锁超时 → 409。
        second = c.post(
            "/api/chat/stream", json={"session_id": "s-hold", "message": "在吗"}
        )

    assert first.status_code == 200
    assert client.store.lock_for("s-hold").locked() is False
    assert second.status_code == 200

    # 顺带钉住 spec F4「挂起期间用户改问别的」这条**用户路径**:新消息另起一轮,
    # 旧的挂起被丢弃 —— 不是 409、不是 error 帧(实测如此,见 dev-notes 的探针)。
    names = [name for name, _ in _parse_sse(second.text)]
    assert names[0] == "meta"
    assert "error" not in names


@pytest.mark.anyio
async def test_resume_request_continues_the_flow_to_the_refund_offer(client_factory):
    """带 `resume` 的请求走 `Command(resume=...)`,把流程推完**并落库**。

    两次 POST 之间是**跨请求**的:第一次挂起后 HTTP 响应就结束了、锁也放了,
    第二次靠 checkpointer 的断点续跑 —— 这正是浏览器里点一张卡片的路径
    (前端 `resumeWith` 发的就是 `{session_id, resume: {order_no}}`)。

    四条断言各有分工,少一条就有一种实现能蒙混过去:

    ① `refund_offer` 帧报的订单号 = resume 递上去的那个 —— 「resume 载荷被
       翻译成了槽位」;
    ② 判定 prompt 里带着**这一单**的订单号 —— 光看 ① 不够:`refund_offer` 报的是
       槽位里的号,而取数完全可能拿的是别的号(卡片候选/空串),那时 ① 照样绿;
    ③ token 帧拼回的文本 = 判定给的话术 —— 「判定的话术真的流出去了」;
    ④ 库里恰好一轮 user + assistant —— 「挂起的一轮没写、续跑的那一轮写了」
       (第一次请求若也落了库,这里会是四行)。
    """
    retriever = FakeRetriever([CLAUSE_CHUNK])
    client, model = client_factory(
        batches=[], intent="退款退货", retriever=retriever
    )
    with client as c:
        first = c.post(
            "/api/chat/stream",
            json={"session_id": "s-resume", "message": "这个能退吗"},
        )
        assert "event: order_choice" in first.text

        second = c.post(
            "/api/chat/stream",
            json={"session_id": "s-resume", "resume": {"order_no": PICKED_ORDER}},
        )

    events = _parse_sse(second.text)
    names = [name for name, _ in events]

    # 不再问一次(卡片只在缺号时弹);这一轮也不再挂起 → 有 done。
    assert "order_choice" not in names
    assert names[-1] == "done"

    # ①
    offer = next(p for name, p in events if name == "refund_offer")
    assert offer["order_no"] == PICKED_ORDER

    # ② 判定 prompt 的「【订单信息】」段是取数结果的 JSON,订单号在其中。
    judged = [msgs for schema, msgs in model.structured if schema == "RefundJudgement"]
    assert len(judged) == 1                      # 恰好判一次
    rendered = "\n".join(str(m.content) for m in judged[0])
    assert PICKED_ORDER in rendered

    # ③ 逐 token 推送会把一句话切成好几帧,必须拼回来再比(CLAUDE.md 平台陷阱)。
    assert "".join(p["text"] for name, p in events if name == "token") == JUDGE_YES_REPLY

    # ④
    assert [m.role for m in client.db.messages] == ["user", "assistant"]
    assert client.db.messages[1].content == JUDGE_YES_REPLY
    assert {m.conversation_id for m in client.db.messages} == {"s-resume"}


@pytest.mark.anyio
async def test_resume_runs_the_retrieval_leg_with_the_expanded_queries(client_factory):
    """续跑要真的走「扩写 → 多路检索 → citations 帧」这一腿(不是空转)。

    这条与上一条分开,是因为它们是**两条独立的腿**:检索腿断了(比如检索器
    注入错、扩写出来的查询没被用上),上一条的④条断言**全都照样绿** ——
    判定的 prompt 里那份证据为空也有 `NO_CLAUSE_NOTE` 兜着。
    """
    retriever = FakeRetriever([CLAUSE_CHUNK])
    client, _ = client_factory(batches=[], intent="退款退货", retriever=retriever)
    with client as c:
        c.post(
            "/api/chat/stream",
            json={"session_id": "s-retr", "message": "这个能退吗"},
        )
        resp = c.post(
            "/api/chat/stream",
            json={"session_id": "s-retr", "resume": {"order_no": PICKED_ORDER}},
        )

    events = _parse_sse(resp.text)
    # 扩写出来的每一路都真的搜了(帧里的命中来自 `multi_search` 的合并结果)。
    assert "退款政策" in retriever.calls
    citations = next(p for name, p in events if name == "citations")
    assert [c["chunk_id"] for c in citations["items"]] == [CLAUSE_CHUNK.chunk_id]


@pytest.mark.anyio
async def test_expansion_cap_really_comes_from_settings(client_factory):
    """`query_expansion_max_queries` 必须**真的被节点读**(spec §9 的接线)。

    只测「配置项存在 + 越界被拒」是不够的:把那行改回模块常量 3,所有别的用例
    照样全绿 —— 那就是一个「改了没反应」的配置项。这里把上限压到 1、让扩写
    替身吐三条,再看检索器**实际收到几条**。
    """
    retriever = FakeRetriever([CLAUSE_CHUNK])
    client, model = client_factory(
        batches=[], intent="退款退货", retriever=retriever,
        query_expansion_max_queries=1,
    )
    model.queries = ["退款政策", "退货时效", "运费谁出"]
    with client as c:
        c.post("/api/chat/stream", json={"session_id": "s-cap", "message": "这个能退吗"})
        c.post(
            "/api/chat/stream",
            json={"session_id": "s-cap", "resume": {"order_no": PICKED_ORDER}},
        )

    assert retriever.calls == ["退款政策"]


@pytest.mark.anyio
async def test_request_with_neither_message_nor_resume_is_rejected(client_factory):
    """两个字段都没有 = 请求语义错 → 422,**不是**把 None 递给下游。

    `message` 在 ch06 之前是必填(缺了本来就 422);加了 `resume` 之后它必须
    变成可选,于是「两个都没给」这条路径**新开出来了** —— 没人守着的话
    `user_input=None` 会一路带进图里,而那时**流已经开始**,只能变成一条
    error 帧(HTTP 200),请求语义错被报成服务端故障的样子。
    (早先这里写的是「在 `prepare_turn` 的 tiktoken 里炸成 500」—— **那句已不成立**:
    T10 之后 `prepare_turn` 不收历史、对 `user_input is None` 有守卫;
    校验本身仍然对,理由换成上面这条。)
    """
    client, _ = client_factory(batches=[], intent="退款退货")
    with client as c:
        neither = c.post("/api/chat/stream", json={"session_id": "s1"})
        both = c.post(
            "/api/chat/stream",
            json={"session_id": "s1", "message": "这个能退吗", "resume": {"order_no": "1"}},
        )
        # 反面:`resume` 单独给是合法的(ch06 新增的入口)。**先造一个真的挂起点**
        # —— 没有挂起点时 200 也能拿到,但那是失败路径产的:请求带着 `Command`
        # 从 START 重开,`resolve_references` 抛 `KeyError('user_input')`,端点的
        # `except Exception` 把它变成**一个 `error` 帧的 200**。只断状态码的写法
        # 区分不了「续跑成功」与「炸在流里」,审查实测确认过这条。
        c.post(
            "/api/chat/stream", json={"session_id": "s1", "message": "这个能退吗"}
        )
        only_resume = c.post(
            "/api/chat/stream", json={"session_id": "s1", "resume": {"order_no": "1"}}
        )

    assert neither.status_code == 422
    # 两个都给同样拒:resume 会**静默吞掉** message,用户那句原话既没被回答、
    # 也不会落库 —— 事后连查都查不到。
    assert both.status_code == 422
    # 续跑**走完了**:帧序列以 done 收尾,且中间没有 error。
    assert only_resume.status_code == 200
    assert [name for name, _ in _parse_sse(only_resume.text)][-1] == "done"
    assert "event: error" not in only_resume.text


@pytest.mark.anyio
async def test_resume_without_a_pending_flow_is_rejected_before_the_stream(client_factory):
    """没有挂起点就 resume → **流开始之前** 409 + 固定文案。

    首版实测的结局是 HTTP 200 + 一个 error 帧,里面是裸的 `'user_input'`
    (图带着 `Command` 从 START 重开,`resolve_references` 取不到键)——
    用户看不懂、我们也没法查。可达场景:服务重启(InMemorySaver 是进程内的)
    之后用户点一张还挂在页面上的旧卡片。

    三条断言缺一不可:状态码(不是 200)、**不是 SSE 流**(一旦开始流式就再也
    改不了状态码)、以及流里**没有 error 帧**(「因为报错所以 200」正是这条
    用例要防的假绿;句子也不能带任何 Python 标识符)。

    顺带钉住这条退出路径的锁:它在 `try` 守卫**之内**抛,靠 `except BaseException`
    放锁 —— 漏了的话该会话从此永久 409(本仓的既定规矩:每条退出路径都要有具名测试)。
    """
    client, _ = client_factory(
        batches=[], intent="退款退货", session_lock_timeout_seconds=0.15
    )
    with client as c:
        resp = c.post(
            "/api/chat/stream",
            json={"session_id": "s-no-pending", "resume": {"order_no": PICKED_ORDER}},
        )
        # 锁若没放掉,下面这次请求会等锁超时 → 409(文案不同,见断言)。
        follow_up = c.post(
            "/api/chat/stream", json={"session_id": "s-no-pending", "message": "在吗"}
        )

    assert resp.status_code == 409
    assert resp.headers["content-type"].startswith("application/json")
    assert "event:" not in resp.text
    assert "error" not in resp.text
    assert "user_input" not in resp.text          # 裸标识符绝不能出现
    assert resp.json()["detail"] == "该会话没有待处理的流程,请直接发送消息开始新一轮。"
    assert client.store.lock_for("s-no-pending").locked() is False
    assert follow_up.status_code == 200


@pytest.mark.anyio
async def test_resume_after_the_flow_finished_is_rejected(client_factory):
    """走完之后的会话再 resume 一次 → 同样 409(那条 thread **有** checkpoint,
    只是没有待续任务 —— 与上一条的"从来没有过"分开,两条走的是不同的分支)。"""
    client, _ = client_factory(batches=[], intent="退款退货")
    with client as c:
        c.post("/api/chat/stream", json={"session_id": "s-done", "message": "这个能退吗"})
        first = c.post(
            "/api/chat/stream",
            json={"session_id": "s-done", "resume": {"order_no": PICKED_ORDER}},
        )
        again = c.post(
            "/api/chat/stream",
            json={"session_id": "s-done", "resume": {"order_no": PICKED_ORDER}},
        )

    assert [name for name, _ in _parse_sse(first.text)][-1] == "done"
    assert again.status_code == 409


@pytest.mark.anyio
async def test_empty_resume_payload_is_not_a_new_turn(client_factory):
    """`{"resume": {}}` 走**续跑**分支,不是"没有 message 就炸"。

    这一条钉的是端点的分支判据必须是 `is not None` 而不是真值判断:写成
    `if request.resume:` 时,空 dict 会掉进"开新一轮"那一支,而那一支要
    `message` —— 拿到的却是 `None`,于是这一轮带着一个空输入进图。
    (早先这里写的是「`prepare_turn` 在 tiktoken 里炸成 500」—— **那句已不成立**:
    T10 之后 `prepare_turn` 不收历史、对 `user_input is None` 有守卫。)
    判据仍然必须是 `is not None`,理由换成上面这条。

    载荷本身没意义(槽位是空串),子流程的既定行为是**如实说明查不到**
    (T7 已记账:文案里那个空订单号略糙但无害)。这里只要求「不是服务端故障」。
    """
    client, _ = client_factory(batches=[], intent="退款退货")
    with client as c:
        c.post(
            "/api/chat/stream", json={"session_id": "s-empty", "message": "这个能退吗"}
        )
        resp = c.post("/api/chat/stream", json={"session_id": "s-empty", "resume": {}})

    assert resp.status_code == 200
    # **实测(langgraph 1.2.11)**:空 dict 会被 `_loop.py` 判成"**空的 resume 映射**"
    # (`all(is_xxh3_128_hexdigest(k) for k in {})` 恒为真),于是这一次续跑**没有**
    # 递任何 resume 值 —— 节点重跑、`interrupt()` 再次挂起,用户又拿到一次卡片。
    # 那是个可接受的结局(不是服务端故障),所以这里只断"没有 error 帧 + 卡片还在"。
    assert [name for name, _ in _parse_sse(resp.text)] == ["meta", "order_choice"]


@pytest.mark.anyio
async def test_done_frame_carries_the_intent_confidence(client_factory):
    """`confidence` 要进 done 帧(spec §4.2)—— 它是这条链路唯一的出口。

    T4 把 confidence 写进了 state 与 `trace` 帧的载荷,端点这一半在 T8:
    `done` 帧少了这个键,值就在端点被丢掉,而**除了这条用例没有任何东西看得见**
    (前端不读它,验收脚本也不读)。期望值刻意用 0.77 而不是 `_Intent` 的默认
    0.9 —— 与替身默认值撞车时,「端点真把值透出来了」与「谁给了个默认值」
    给出同一个观测值。
    """
    client, _ = client_factory(batches=[], intent="闲聊")
    client.intent_model.confidence = 0.77
    with client as c:
        resp = c.post("/api/chat/stream", json={"session_id": "s1", "message": "你好"})

    done = _parse_sse(resp.text)[-1]
    assert done[0] == "done"
    assert done[1]["confidence"] == 0.77


# ---------- ch07:端点接线(降级 → 起任务 → 播种精简版 → 两个观测面) ----------
#
# 本节验的**只有接线**:降级、分层截短、摘要触发、组装、观测面各自都有单测,
# 而把它们接错时**那些单测一条都不会红**(每一件都还在,只是不在请求路径上)。
# 所以这里一律走**端点**,观测点取「模型真的收到了什么」与「日志里真的写了什么」。

SCRATCH_CONV = "c" * 32

#: 层 2 里那条工具结果。远超 `layer2_tool_chars=60`,所以**一定会被截短** ——
#: 短于阈值的输入会让「截短生效」与「截短失效」在这条用例里长得一模一样
#: (本章第五种假绿形态:测试输入小到触发不了被测行为)。
_LONG_TOOL_RESULT = json.dumps(
    {"order_no": "1002", "status": "已取消",
     "note": "用户已申请退款,等待仓库确认。" * 12},
    ensure_ascii=False,
)

#: 层 2 里那条客服答复。同理,要长过 `layer2_assistant_chars=50`。
_LONG_REPLY = "您的订单 1002 当前状态为已取消,已为您登记退款申请。" * 5

#: 降级用例的默认行数。**标定值,不是随手写的数**(计划给的是 20)。
_DEGRADE_ROWS = 24
#: 摘要用例的行数 —— 必须长到层 2 的**截短后** token 数超过 `layer2_budget`。
_SUMMARY_ROWS = 40

LAYERED_CONV = "d" * 32


def _log_payload(caplog, prefix: str) -> dict:
    """从日志里取出最后一条 `"<prefix> <json>"` 的 JSON 体。

    取不到**直接抛** —— 退化成「返回 {}」会让断言全变成 KeyError 或恒真,
    而本节的用例问的正是「那一行到底有没有、里面是什么」。
    """
    for record in reversed(caplog.records):
        if record.message.startswith(f"{prefix} "):
            # 按**前缀长度**切,不是 `split(" ", 1)`:`tasks._emit` 的前缀是
            # `"summary <event>"`(两个词),按空格切会切掉半个前缀、
            # 留下 `'trigger {...}'` 这种不是 JSON 的东西,而红法会是
            # 一句没头没脑的 JSONDecodeError。
            return json.loads(record.message[len(prefix) :].lstrip())
    raise AssertionError(f"日志里没有 {prefix} 行")


def _stuffed_session(rows: int = _DEGRADE_ROWS) -> FakeSession:
    """造一个已存在、且历史长到会触发降级的会话。

    ⚠️ `rows` 是**标定过的**(计划给的 20 有一个致命问题:它触发不了摘要,
    于是「摘要任务被起起来了」那条用例会以 `fired == []` 的形式失败,而
    红法看起来像「接线没做」)。实测(默认配置:窗口 18000、固定开销 5463、
    单轮峰值 8000 ⇒ `history_budget=4537`、层 1 预算 3175、层 2 预算 1362;
    每行「很长的历史内容」×20 = 140 字 ≈ 160 token):

    | rows | 降级后层 1 起点 | 层 1 token | 层 2 token | 触发摘要 |
    |---|---|---|---|---|
    | 20 | 3 | 2880 | 219 | 否 |
    | **24** | 7 | 2880 | 657 | 否 |   ← 默认
    | 40 | 23 | 2880 | 2409 | **是** |

    默认取 24:降级**确定发生**(那才是 `test_degrade_...` 要验的),
    而层 2 离触发线还远 —— 免得那条用例在无人察觉的情况下起一个真的后台线程
    (它自带 engine 与 HTTP 客户端,而「单测全程不联网」是硬规矩)。
    """
    db = FakeSession()
    db.conversations[SCRATCH_CONV] = Conversation(
        id=SCRATCH_CONV, user="demo-user", status="active",
        summary_upto_msg_id=0, layer1_from_msg_id=0,
    )
    for i in range(rows):
        db.add(MessageRecord(
            conversation_id=SCRATCH_CONV,
            role="user" if i % 2 == 0 else "assistant",
            content="很长的历史内容" * 20,
        ))
    return db


def _layered_session() -> FakeSession:
    """一个**已经降过级**的会话:两个锚点非零 ⇒ 层 2 非空。

    层 2 里放一条长工具结果与一段长客服答复 —— 它们正是验收 4b 要看见的
    「截短后的形态」(`[工具结果] …` / `…`)。那两样**只有** `layers.truncate`
    产得出来:早先那条线传的是单层裁剪的输出(`trim.select_history`,只整轮
    丢弃、从不标注内容 —— 该函数已在 T10 删除),所以分层接上之前,
    `history_ctx` 那一行在结构上承载不了 4b。
    """
    db = FakeSession()
    db.conversations[LAYERED_CONV] = Conversation(
        id=LAYERED_CONV, user="demo-user", status="active",
        summary_upto_msg_id=0, layer1_from_msg_id=5,
    )
    db.add(MessageRecord(conversation_id=LAYERED_CONV, role="user", content="订单 1002 能退吗"))
    db.add(MessageRecord(
        conversation_id=LAYERED_CONV, role="assistant", content="",
        tool_calls=[{"id": "c1", "name": "query_order", "args": {"order_id": "1002"}}],
    ))
    db.add(MessageRecord(
        conversation_id=LAYERED_CONV, role="tool",
        content=_LONG_TOOL_RESULT, tool_call_id="c1",
    ))
    db.add(MessageRecord(conversation_id=LAYERED_CONV, role="assistant", content=_LONG_REPLY))
    db.add(MessageRecord(conversation_id=LAYERED_CONV, role="user", content="那运费退吗"))
    db.add(MessageRecord(
        conversation_id=LAYERED_CONV, role="assistant", content="运费在退款时一并退还。"
    ))
    return db


def test_oversized_user_input_returns_400_json_before_streaming(client_factory):
    """超 `max_user_input_tokens` ⇒ 400 且**是普通 JSON 不是 SSE**。

    一旦 yield 过首帧,响应头就发出去了、状态码再也改不了 ——
    这正是预算校验必须在流开始前的原因,也是这条断言存在的理由。
    """
    client, model = client_factory(batches=[[FakeChunk("好")]], max_user_input_tokens=5)
    with client as c:
        resp = c.post("/api/chat/stream", json={"message": "这是一句明显超过五个 token 的话"})

    assert resp.status_code == 400
    assert resp.headers["content-type"].startswith("application/json")   # ← 不是 SSE
    assert "event:" not in resp.text          # 确认真的没走流
    # 模型一次都没被调用 —— 「400 但已经把这一轮跑了一遍」也是一种实现。
    assert model.calls == []


def test_degrade_persists_the_new_anchor_and_does_not_touch_messages(
    client_factory, monkeypatch
):
    """降级只写一个整数,**一行 messages 都不动**。

    「不搬数据」是本章的核心卖点,必须有断言钉住 —— 否则「顺手把旧消息截短了
    写回 messages 表」这种实现能让其余**所有**用例照样通过。

    后台任务在这里**挡掉**(本用例不测它):这段历史是按 `_DEGRADE_ROWS`
    标定的 —— 触发降级、但不触发摘要。顺带把「没触发」也断下来,
    标定一旦漂移(提示词变长、阈值改动)这里会**响亮**地红,而不是默默起一个
    连真 MySQL、真上游客户端的线程,或者默默不验降级。
    """
    fired = []
    monkeypatch.setattr(
        chat_api, "run_summary_in_background",
        lambda **kw: (fired.append(kw), False)[1],
    )
    db = _stuffed_session()
    before = [(m.role, m.content) for m in db.messages]
    client, _ = client_factory(batches=[[FakeChunk("好")]], session=db)

    with client as c:
        c.post("/api/chat/stream", json={"session_id": SCRATCH_CONV, "message": "现在这句"})

    conv = db.conversations[SCRATCH_CONV]
    assert conv.layer1_from_msg_id > 0                      # ← 降级真的写进去了
    assert fired == []                                      # 见 docstring:这个尺度不触发摘要
    after = [(m.role, m.content) for m in db.messages[:len(before)]]
    assert after == before                                  # ← 旧行一行没改
    # 新增的两行是本轮的 user + assistant,不是被搬过来的历史
    assert len(db.messages) == len(before) + 2


def test_degrade_is_logged_with_both_the_old_and_the_new_anchor(
    client_factory, monkeypatch, caplog
):
    """**级联的第一环**:`layer1 降级` 那行必须报出**真的**旧值与新值。

    它是 spec §10.5 验收 2 要 grep 的那一行,而**只有这一个接缝**同时握着两个值:
    `layers.degrade` 只返回新值,`advance_anchors` 只收新值。没有它,
    「层 1 超预算就降级一批」在日志里没有生产者 —— 验收那条断言要么失败,
    要么被写成一条恒真的 grep。

    四条断言各有分工:

    ① `from` 是**请求开始时**那个锚点(用例预置的 10,非默认值 —— 期望值撞上
       默认的 0 时,「报了真值」与「谁填了个 0」给出同一个观测值);
    ② `to` 与**库里最终那个值**一致 —— 报一个没落库的边界等于说谎;
    ③ `from != to != 0` 是前提:真的挪了、且两个值都不是默认值;
    ④ `layer1_tokens` **对着生产代码现算的层 1 用量**比。原先这里只断
       `0 < layer1_tokens <= layer1_budget`,而那个尺度下 `layer2_tokens` 也满足
       它(**两个数都是正的、都小于预算**)⇒ 断言分不出两个字段(终审 Minor 10)。
       改成与 `summary trigger` 那条同款的 cross-check:拿**写进库的锚点**现切一次,
       逐字比。写死成常量同样不行 —— 那样「报的是挪**之前**的层 1(必然超预算)」
       也过得了。
    """
    fired: list[str] = []
    monkeypatch.setattr(
        chat_api, "run_summary_in_background",
        lambda **kw: (fired.append(kw["conversation_id"]), True)[1],
    )
    db = _stuffed_session(rows=_SUMMARY_ROWS)
    db.conversations[SCRATCH_CONV].layer1_from_msg_id = 10   # 上一次降级的产物
    settings = _settings()

    client, _ = client_factory(batches=[[FakeChunk("好")]], session=db)
    with caplog.at_level(logging.INFO):
        with client as c:
            c.post("/api/chat/stream", json={"session_id": SCRATCH_CONV, "message": "现在这句"})

    payload = _log_payload(caplog, "layer1 降级")
    assert payload["conversation_id"] == SCRATCH_CONV
    assert payload["from"] == 10                                        # ①
    assert payload["to"] == db.conversations[SCRATCH_CONV].layer1_from_msg_id   # ②
    assert payload["from"] != payload["to"] != 0                        # ③
    assert payload["layer1_budget"] == memory_budget.derive(
        settings=settings, system_prompt=render_system_prompt(settings.brand_name)
    ).layer1_budget
    # ④ 日志里那个数必须**等于**生产代码对同一批消息、同一对锚点切出来的层 1 用量。
    #    (`db.messages[:_SUMMARY_ROWS]` 是这次请求**开始前**库里的全部历史 ——
    #     本轮新增的那两条由 `log_turn` 在流里追加,晚于这一行日志。)
    got = layers.split(
        [Message(id=m.id, role=m.role, content=m.content)
         for m in db.messages[:_SUMMARY_ROWS]],
        summary_upto_msg_id=0,
        layer1_from_msg_id=db.conversations[SCRATCH_CONV].layer1_from_msg_id,
        settings=settings,
    )
    # 前提:`layer1_tokens` 在这个尺度上**真的与 `layer2_tokens` 不同** ——
    # 不然「报错了字段」与「报对了」会给出同一个观测值(那正是上面那句旧断言的问题)。
    assert got.layer1_tokens != got.layer2_tokens
    assert got.layer1_tokens <= payload["layer1_budget"]      # 前提:「装下了」
    assert payload["layer1_tokens"] == got.layer1_tokens


def test_no_degrade_means_no_degrade_log_line(client_factory, monkeypatch, caplog):
    """装得下就**不打**那行日志 —— 与「装得下就什么都不做」同一条纪律。

    少了这条反向断言,一个「每轮都打一行(旧=新)」的实现能通过上面那条
    (`from == to` 的日志同样说得通),而它在日志里制造的是噪音:
    读的人会以为降级在持续发生。
    """
    monkeypatch.setattr(
        chat_api, "run_summary_in_background", lambda **kw: False
    )
    db = _stuffed_session(rows=2)
    client, _ = client_factory(batches=[[FakeChunk("好")]], session=db)

    with caplog.at_level(logging.INFO):
        with client as c:
            c.post("/api/chat/stream", json={"session_id": SCRATCH_CONV, "message": "现在这句"})

    # 前提:这段历史**确实**装得下(否则「没有日志」与「没跑过降级」分不开)。
    assert db.conversations[SCRATCH_CONV].layer1_from_msg_id == 0
    assert [r for r in caplog.records if r.message.startswith("layer1 降级")] == []


def test_summary_task_is_fired_without_blocking_the_reply(client_factory, monkeypatch):
    """摘要任务被起起来了,而**回复不 await 它**。

    spec 验收 4「摘要生成没有阻塞该轮用户回复」。测法是**换掉起任务的函数**,
    让它记录调用后立刻返回,再断言 done 帧照样到 ——
    而不是去测时间差(那会退化成一条不稳定断言,且在快机器上恒真)。
    """
    fired: list[str] = []
    monkeypatch.setattr(
        chat_api, "run_summary_in_background",
        lambda **kw: (fired.append(kw["conversation_id"]), True)[1],
    )
    # ↑ 这要求 `app/api/chat.py` 里是 `from app.memory.tasks import
    #   run_summary_in_background`(模块级名字),而不是 `from app.memory import
    #   tasks` 再 `tasks.run_summary_in_background(...)`。后者 patch 不到,
    #   而红法会是「明明起了、断言说没起」—— 指向测试而不是实现。
    #   同理,T4 的 `layers` 与 T8 的 `summarize` 在端点里也要是模块级名字。
    db = _stuffed_session(rows=_SUMMARY_ROWS)
    client, _ = client_factory(batches=[[FakeChunk("好")]], session=db)

    with client as c:
        resp = c.post("/api/chat/stream", json={"session_id": SCRATCH_CONV, "message": "现在这句"})

    assert fired == [SCRATCH_CONV]                    # ← 起了
    assert _parse_sse(resp.text)[-1][0] == "done"     # ← 而回复没被它挡住


def test_summary_trigger_is_logged_with_the_layer2_counts(
    client_factory, monkeypatch, caplog
):
    """级联的**第一环**:`summary trigger` 必须带上层 2 的截短后 token 与预算。

    它**只能**在端点这一处打:T9 的 `log_trigger` 拿不到那两个数(算分层的那边
    才有)。不调它,验收 2 要 grep 的那一行在日志里**根本不存在** ——
    而摘要照跑、落库照写、回复照出,一切都「看起来正常」。

    两个数都对着**库里读回来的锚点**现算(cross-check 的是**接线**:
    端点有没有把降级后的锚点那一份交出去),而不是写死成常量 ——
    写死的话,「日志报的是降级**前**的层 2(恒为 0)」这种实现也能过。
    """
    fired: list[str] = []
    monkeypatch.setattr(
        chat_api, "run_summary_in_background",
        lambda **kw: (fired.append(kw["conversation_id"]), True)[1],
    )
    db = _stuffed_session(rows=_SUMMARY_ROWS)
    client, _ = client_factory(batches=[[FakeChunk("好")]], session=db)

    with caplog.at_level(logging.INFO):
        with client as c:
            c.post("/api/chat/stream", json={"session_id": SCRATCH_CONV, "message": "现在这句"})

    assert fired == [SCRATCH_CONV]
    payload = _log_payload(caplog, "summary trigger")

    settings = _settings()
    got = layers.split(
        [Message(id=m.id, role=m.role, content=m.content) for m in db.messages],
        summary_upto_msg_id=0,
        # **降级之后**的锚点(写进库的那个),不是请求开始时的 0。
        layer1_from_msg_id=db.conversations[SCRATCH_CONV].layer1_from_msg_id,
        settings=settings,
    )
    b = memory_budget.derive(
        settings=settings, system_prompt=render_system_prompt(settings.brand_name)
    )
    assert got.layer2_tokens > 0                      # 前提:层 2 真的非空
    assert payload["layer2_tokens"] == got.layer2_tokens
    assert payload["layer2_budget"] == b.layer2_budget
    assert payload["layer2_tokens"] > payload["layer2_budget"]   # 触发判据(严格 >)


def test_history_ctx_carries_the_truncated_layer2_forms(client_factory, caplog):
    """**验收 4b 的单元版**:`history_ctx` 那一行里必须看得见截短后的形态。

    分层接上之前,这条线拿的是端点交给单层裁剪(`trim.select_history`)的输出
    —— 那个函数**只整轮丢弃、从不标注内容**,所以 `…` 与 `[工具结果] `
    **不可能出现**(T7 审查的 Important 2)。这条用例是它的收口:
    `history=` 换成分层后的精简版之后,形态才真的出现在那一行里。

    用 `intent="闲聊"`:`history_ctx` 在**路由之前**打,这条断言与 Agent 无关,
    也就不该被 Agent 的批次脚本干扰(「每轮必打」由 T7 那条覆盖)。
    """
    db = _layered_session()
    client, _ = client_factory(batches=[], session=db, intent="闲聊")

    with caplog.at_level(logging.INFO):
        with client as c:
            c.post("/api/chat/stream", json={"session_id": LAYERED_CONV, "message": "在吗"})

    sliding = _log_payload(caplog, "history_ctx")["sliding"]
    assert [m["role"] for m in sliding] == ["user", "assistant", "tool", "assistant", "user", "assistant"]
    assert sliding[0]["content"] == "订单 1002 能退吗"          # 层 2 的 user 原话不动
    assert sliding[1]["content"] == ""                         # 带 tool_calls 的那条(结构原样)
    assert sliding[2]["content"].startswith("[工具结果] ")      # ← 4b 的形态
    assert sliding[2]["content"].endswith("…")
    assert sliding[3]["content"].endswith("…")                 # 客服答复只留开头
    assert sliding[4]["content"] == "那运费退吗"                # 层 1 是原文,一个字不动


def test_model_ctx_describes_the_messages_actually_sent(client_factory, caplog):
    """`model_ctx` 必须描述**真正发出去的那批消息**,`bounds` 必须是真的锚点。

    ① `sliding` 逐条**等于**模型实际收到的历史。这一条钉住「重切用 `layers.split`」
       的实现:精简版里的层 2 已经是截短过的形态,**再截一次**会给工具结果叠上
       第二个 `[工具结果] ` 前缀、内容也随之变短 —— 日志从此描述的是**另一次
       切分**,而它看起来完全正常、没有任何断言覆盖这种分叉。
    ② `bounds` 是降级之后真的用过的两个锚点。它们靠 state 的两个通道传下来,
       而**没在 `ChatState` 里声明的通道会被 LangGraph 静默丢弃**(ch06 的教训)
       —— 那时这里读到的是 `0`/`0`,也就是「一次没发生过的切分」。
    """
    db = _layered_session()
    client, model = client_factory(batches=[[FakeChunk("好的")]], session=db)

    with caplog.at_level(logging.INFO):
        with client as c:
            c.post("/api/chat/stream", json={"session_id": LAYERED_CONV, "message": "在吗"})

    payload = _log_payload(caplog, "model_ctx")
    sent = model.last_messages
    # ① 逐条比内容(系统提示词与末尾那条 human 不在 sliding 里)。
    assert [m.content for m in sent[1:-1]] == [m["content"] for m in payload["sliding"]]
    assert payload["rounds"] == len(payload["sliding"]) == 6
    assert payload["sliding"][2]["content"].startswith("[工具结果] ")
    assert not payload["sliding"][2]["content"].startswith("[工具结果] [工具结果] ")
    # ② 锚点是**降级之后**的那一对(层 1 从第 5 条起),不是默认的 0/0。
    assert payload["bounds"] == {"summary_upto_msg_id": 0, "layer1_from_msg_id": 5}
    assert payload["tokens"]["layer2"] > 0


def test_the_joined_summary_is_seeded_into_the_model_context(client_factory):
    """多段梗概**拼成一段**后注入,引导语与拼法在真链路上对得上。

    `_SUMMARY_HEADER`(prompts)、`join_summaries`(summarize)与
    `history_ctx` 里那个计数口径分别写在三处 —— 这条用例走**真链路**把它们
    对上:库里两段梗概,模型看到的是**一段**带引导语的背景,两段都在、按 seq 序、
    且排在用户原话**之后**。拼法或顺序改了,这里会红。
    """
    db = _layered_session()
    db.summaries.append(ConversationSummary(
        conversation_id=LAYERED_CONV, seq=1, upto_msg_id=2, content="第一段:用户问过订单 1002",
    ))
    db.summaries.append(ConversationSummary(
        conversation_id=LAYERED_CONV, seq=2, upto_msg_id=4, content="第二段:要求退运费",
    ))
    client, model = client_factory(batches=[[FakeChunk("好的")]], session=db)

    with client as c:
        c.post("/api/chat/stream", json={"session_id": LAYERED_CONV, "message": "在吗"})

    content = model.last_messages[-1].content
    assert content.startswith("在吗")
    assert "第一段:用户问过订单 1002" in content
    assert "第二段:要求退运费" in content
    # 按 seq 序拼成**一段**,不是两条消息、也不是倒序。
    assert content.index("第一段") < content.index("第二段")
    assert content.index("第一段:用户问过订单 1002") < content.rindex("第二段:要求退运费")
    assert sum(1 for m in model.last_messages if content == m.content) == 1


def test_an_over_budget_last_round_is_dropped_rather_than_blowing_the_window(
    client_factory,
):
    """层 1 的选择真的在干活:**装不下的那一轮整轮不留**。

    这条咬人的场合是 `degrade` 的另一个出口 —— 「只剩一轮还超预算就停在原地」
    (再挪就把层 1 挪空了)。一轮工具密集的 ReAct(单条工具结果封顶 1200 token,
    最多 5 步)真的能比 `layer1_budget` 还大,那时只有 `select_layer1` 能收口:
    **宁可这一轮的历史一条都不发,也不把窗口顶穿**。

    「整轮丢、不留半轮」是 `start_on="human"` + `allow_partial=False` 一起给的
    —— 半轮 = 一条没有提问的回答,或者 tool 消息与它的 assistant 被拆开
    (上游直接 400,且只在历史长到触发分层时才复现)。

    前提断在最前面:这一轮的原文 token **确实**超过层 1 预算。少了它,这条用例
    在别的配置下会退化成「什么都没发生」还照样绿。
    """
    settings = _settings()
    b = memory_budget.derive(
        settings=settings, system_prompt=render_system_prompt(settings.brand_name)
    )
    db = FakeSession()
    db.conversations[SCRATCH_CONV] = Conversation(
        id=SCRATCH_CONV, user="demo-user", status="active",
        summary_upto_msg_id=0, layer1_from_msg_id=0,
    )
    db.add(MessageRecord(conversation_id=SCRATCH_CONV, role="user", content="这一轮的开头"))
    for i in range(22):     # 一轮里塞 22 次工具往返 —— 全部落在**同一轮**
        db.add(MessageRecord(
            conversation_id=SCRATCH_CONV, role="assistant", content="",
            tool_calls=[{"id": f"c{i}", "name": "query_order", "args": {"order_id": "1002"}}],
        ))
        db.add(MessageRecord(
            conversation_id=SCRATCH_CONV, role="tool",
            content="很长的历史内容" * 20, tool_call_id=f"c{i}",
        ))
    round_tokens = sum(
        trim.count_tokens(m.content) for m in db.messages
    )
    assert round_tokens > b.layer1_budget          # ← 前提:这一轮装不下

    client, model = client_factory(batches=[[FakeChunk("好")]], session=db)
    with client as c:
        c.post("/api/chat/stream", json={"session_id": SCRATCH_CONV, "message": "现在这句"})

    # 窗口里只剩 [system, 本轮 human] —— 那一轮整轮被丢在外面,而不是顶穿窗口。
    assert [type(m).__name__ for m in model.last_messages] == ["SystemMessage", "HumanMessage"]
    # 而它**没有**被写进库的锚点(`degrade` 挪不动:挪了就空了)。
    assert db.conversations[SCRATCH_CONV].layer1_from_msg_id == 0


def test_layer1_within_budget_triggers_neither_degrade_nor_summary(client_factory, monkeypatch):
    """**验收 3 的单元版**:装得下就一个动作都不做。

    「压缩是成本不是美德」。这条防的是「保守起见每次都压一点」的实现 ——
    那种实现能让验收 1/2/4 **全部通过**,而它在默认窗口下白白把历史压没了。
    """
    fired = []
    monkeypatch.setattr(
        chat_api, "run_summary_in_background",
        lambda **kw: (fired.append(kw), False)[1],
    )
    db = FakeSession()
    db.conversations[SCRATCH_CONV] = Conversation(
        id=SCRATCH_CONV, user="demo-user", status="active",
        summary_upto_msg_id=0, layer1_from_msg_id=0,
    )
    db.add(MessageRecord(conversation_id=SCRATCH_CONV, role="user", content="你好"))
    db.add(MessageRecord(conversation_id=SCRATCH_CONV, role="assistant", content="你好呀"))
    client, _ = client_factory(batches=[[FakeChunk("好")]], session=db)   # 默认窗口

    with client as c:
        c.post("/api/chat/stream", json={"session_id": SCRATCH_CONV, "message": "现在这句"})

    assert fired == []                                        # 没起任务
    assert db.conversations[SCRATCH_CONV].layer1_from_msg_id == 0   # 也没降级
