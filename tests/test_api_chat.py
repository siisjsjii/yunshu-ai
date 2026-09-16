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
from sqlalchemy.exc import OperationalError
from sqlalchemy.sql.elements import BinaryExpression

from app.api import chat as chat_api
from app.config import Settings, get_settings
from app.db.models import Conversation, MessageRecord
from app.db.session import get_session
from app.main import app
from app.memory.store import SessionStore
from app.tools import registry as tools_registry
from app.tools.errors import ToolNotFound

REQUIRED = {
    "openai_base_url": "https://example.invalid/v1",
    "openai_api_key": "sk-test",
    "openai_model": "test-model",
    "database_url": "mysql+asyncmy://u:p@h:3306/db",
}


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **REQUIRED, **overrides)


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


class _BoundModel:
    def __init__(self, inner):
        self._inner = inner

    async def astream(self, messages):
        self._inner.calls.append(("bound", list(messages)))
        for chunk in self._inner.batches.pop(0):
            yield chunk


class ScriptedModel:
    """按顺序回放预置 chunk 批次的替身。

    记录每次 astream 走的是**绑了工具**还是**未绑工具**的入口,以及
    绑上去的到底是哪批工具 —— 后者是"绑给模型的集合与注册表同源"的
    观测点,而事件序列本身看不出这件事。
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

    @property
    def last_messages(self):
        return self.calls[-1][1] if self.calls else None


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

    返回 `(client, model)`;client 上另挂了 `.db` / `.store` / `.model`,
    供测试在请求结束后观察落库内容与锁状态。
    """

    def make(batches, session=None, registry=None, **settings_overrides):
        model = ScriptedModel(batches)
        db = session if session is not None else FakeSession()
        store = SessionStore(ttl_seconds=60, max_sessions=10)

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
        app.dependency_overrides[get_session] = _session_override

        client = TestClient(app)
        client.db = db
        client.store = store
        client.model = model
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


def test_upstream_error_becomes_sse_error_event(client_factory):
    class ExplodingModel:
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


def test_error_event_does_not_echo_the_configured_key(client_factory):
    """异常文本里带着真实密钥时必须被抹掉。

    这里让异常文本**自带**夹具配置的密钥值,才真正检验得到过滤逻辑;
    若断言的是"响应里没有 sk-test"而异常文本里根本没有它,那条断言恒真。
    """

    class LeakyModel:
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

    class SlowModel:
        """持有锁约 0.8s:远长于 0.15s 的等锁超时。"""

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

    def boom(session):
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
                            "args": {"order_id": "1001"},
                            "id": "c1",
                        }
                    ]
                )
            ],
            [FakeChunk("已揽件。")],
        ]
    )
    with client as c:
        resp = c.post("/api/chat/stream", json={"message": "订单 1001 的物流到哪了"})
        events = _parse_sse(resp.text)

    kinds = [name for name, _ in events]
    assert "tool_call" in kinds
    assert "tool_result" in kinds
    payload = next(p for name, p in events if name == "tool_call")
    assert payload["name"] == "query_logistics"
    assert payload["args"] == {"order_id": "1001"}
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
