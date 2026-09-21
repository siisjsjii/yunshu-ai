"""聊天接口测试。全部用替身:不联网、不碰 MySQL。

ch02 Task 11 整体改写:端点的历史来源从进程内 `SessionStore` 换成 MySQL,
并接上单轮工具编排。夹具随之从"真 SessionStore + 假模型"换成
"替身 DB 会话 + 假模型"。逐条去留见 task-11-report.md。
"""

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain.tools import tool
from langchain_core.messages import AIMessage
from sqlalchemy.exc import OperationalError
from sqlalchemy.sql.elements import BinaryExpression

from app.agent.state import RefundJudgement
from app.api import chat as chat_api
from app.config import Settings, get_settings
from app.db.models import Conversation, MessageRecord
from app.db.session import get_session
from app.main import app
from app.memory.store import SessionStore
from app.refund.orders import DEMO_ORDERS
from app.retrieval.expand import ExpandQueries
from app.retrieval.search import RetrievedChunk
from app.tools import registry as tools_registry
from app.tools.business import _order_record
from app.tools.errors import ToolNotFound

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


class FakeChunk:
    """模拟 AIMessageChunk:支持 + 累加,累加后携带 tool_calls。

    `"type": "tool_call"` 这个键**必须有**:真实链路上模型流出的 tool_call
    chunk 经 langchain 的 default_tool_parser 解析后带着它,而
    `BaseTool.ainvoke` 判"这是不是一次工具调用"只看
    `x.get("type") == "tool_call"`。缺键时 langchain 会把整个 dict 当成
    参数去校验,每次调用都退化成"参数不合法"的可恢复失败 —— 事件序列
    照样长得像那么回事,却一次都没走到真实执行路径上。
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
        self.commits = 0
        self._next_id = 1

    async def execute(self, stmt, *args, **kwargs):
        entity = stmt.column_descriptions[0]["entity"]
        value = _where_value(stmt)
        if entity is Conversation:
            conversation = self.conversations.get(value)
            return _Result([conversation] if conversation is not None else [])
        if entity is MessageRecord:
            rows = [m for m in self.messages if m.conversation_id == value]
            return _Result(sorted(rows, key=lambda m: m.id))
        raise AssertionError(f"替身不支持的实体:{entity}")

    def add(self, obj):
        if isinstance(obj, Conversation):
            self.conversations[obj.id] = obj
        elif isinstance(obj, MessageRecord):
            obj.id = self._next_id
            self._next_id += 1
            self.messages.append(obj)
        else:
            raise AssertionError(f"替身不支持的实体:{type(obj)}")

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


def test_oversized_input_returns_400_before_streaming(client_factory):
    """预算不足必须在响应开始前报错。

    SSE 一旦 yield 过首帧,响应头就发出去了,状态码再也改不了 ——
    所以这里既要看 400,也要看它根本不是一条 SSE 流,且没有调用模型。
    """
    client, _ = client_factory(
        batches=[[FakeChunk("不会走到这里")]],
        context_budget_tokens=200,
        reserved_output_tokens=0,
        safety_margin_tokens=0,
    )
    with client as c:
        resp = c.post("/api/chat/stream", json={"message": "退" * 5000})

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

    断言取"第二个超大请求仍然拿到 400 而不是 409"(会话仍可用)—— 这才是
    真正要守的性质;等锁超时调成 0.15s,漏放锁时第二次请求会在 0.15s 内变红,
    而不是用默认 60s 把测试挂死。顺带断一次锁对象本身,便于定位。
    """
    client, _ = client_factory(
        batches=[],
        context_budget_tokens=200,
        reserved_output_tokens=0,
        safety_margin_tokens=0,
        session_lock_timeout_seconds=0.15,
    )
    body = {"session_id": "s1", "message": "退" * 5000}

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
    `(*, settings, store, session_id, user_input)`,端点改用 `history=`
    之后它抛的是 TypeError(而且被 pytest.raises(RuntimeError) 拦下,
    测试仍会红但红在一个误导性的位置)—— 这里同步改掉。
    """

    def boom(*, settings, history, user_input):
        raise RuntimeError("组装消息失败")

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
        return None

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

    # **调过工具的一轮也要落库,且落的是最终答复**。ch05 起 `log_turn` 是唯一
    # 写方,写的只有 user + assistant(reply)两条 —— ch02 那四条(user /
    # assistant(tool_calls) / tool / assistant)按设计不再落库。剩下这条**残留
    # 保证**没人钉:`reply` 是 agent 各轮 `parts` 的拼接,工具轮的 token 是空的,
    # 所以落库的 assistant 内容必须来自**收尾那一轮**。把 `log_turn` 里那次
    # `append_turn` 删掉、或让它写 `state["user_input"]` 当回复,下面两条变红
    # (前者由 `test_successful_turn_is_persisted_to_mysql_history` 一并覆盖,
    # 后者只有这里看得见)。
    assert [m.role for m in client.db.messages] == ["user", "assistant"]
    assert client.db.messages[1].content == "已揽件。"


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
    变成可选,于是「两个都没给」这条路径**新开出来了** —— 没人守着的话它会
    一路走到 `prepare_turn(user_input=None)`,在 tiktoken 里炸成一个 500。
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
    `message`(None)—— 于是 `prepare_turn` 在 tiktoken 里炸成一个 500。

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
