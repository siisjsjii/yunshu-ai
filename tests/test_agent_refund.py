"""退款子流程:interrupt → 点选订单 → resume 走完。

**为什么每条都走整图(`build_graph`)而不是一个子图**:REFUND 出口本身是本任务
的交付物之一 —— 只测子图的话,`routing.py` 里少一行映射、`graph.py` 里少一条
边,全部用例照样绿(T4 的 Critical 就是这么溜过去的:通道没声明,单测全绿、
生产恒 null)。子图能覆盖到的只是「节点自己有没有毛病」。

**interrupt 的读法**(F1/F2,本机 langgraph 1.2.11 实测):`stream_mode="custom"`
会把 interrupt **整个吞掉** —— 一个帧都不吐、run 直接结束、`state.next` 停在待续
节点,而且**不报任何错**。所以下面每一条都用 `["custom", "updates"]`,从 `updates`
里认 `__interrupt__` 键。把端点改成双模式是 T8 的事,这里钉的是图的行为。

**假绿的三种形态,本文件的每个断言都按它们自查过**(见 CLAUDE.md「写测试的规矩」):

1. **期望值与兜底值撞车**:凡是断言「resume 回填的订单号被用上了」的地方,
   订单号一律取 `PICKED = "83746592"` —— 它**不在** `DEMO_ORDERS` 里。
   用 brief 里那个 `20240915` 的话,「resume 生效」与「根本没读 resume、拿了
   卡片候选里的演示单」这两件事给出**同一个观测值**,断言零判别力。
2. **注入了「已经处理过」的值**:每轮的 `history` 都是**本轮现给**的,
   且第 2 轮用例刻意给空历史 —— 不这样的话「从历史里取号」与「状态残留」
   分不开。
3. **脚手架改了别的行**:本文件不 import 生产常量来当期望值(除了
   `REFUND_REASON_CATEGORIES` / `DEMO_ORDERS` 这两个**单一来源**本身),
   文案断言一律用**字面量**。
"""

import asyncio
import json

import pytest
from langchain.tools import tool
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from sqlalchemy.exc import SQLAlchemyError

from app.agent.emit import make_emitter
from app.agent.graph import build_graph
from app.agent.nodes import make_resolve_references_node
from app.agent.state import RefundJudgement
from app.config import Settings
from app.refund.categories import REFUND_REASON_CATEGORIES
from app.refund.orders import DEMO_ORDERS
from app.retrieval.expand import ExpandQueries
from app.retrieval.search import RetrievedChunk
from app.schemas import Message
from app.tools.errors import ToolInfrastructureError, ToolNotFound

CONV = "conv-refund"

#: resume 回填的订单号。**必须不在 `DEMO_ORDERS` 里**(见模块 docstring 第 1 条)。
PICKED = "83746592"
#: 出现在问题/历史里的订单号,同样避开演示池。
IN_CONTEXT = "63514087"
IN_HISTORY = "94720158"

#: 检索到的条款原文。断言它**进了判定 prompt**时用字面量(接缝测试)。
CLAUSE = "定制类商品一经确认不支持七天无理由退货"
CHUNK = RetrievedChunk(
    "定制商品能退货吗", CLAUSE, "退换货",
    chunk_id=7, section_path="退货政策 > 例外", score=0.71,
)

#: 判定的「能退」答案。刻意与任何兜底文案都不撞(第 1 条自查)。
YES_REPLY = "这一单还在七天无理由期内,可以申请退款。"
NO_REPLY = "这一单已拆封使用,不在七天无理由范围内,建议联系人工客服核实。"


# ---- 替身 ----------------------------------------------------------------


class _Intent:
    def __init__(self, intent, confidence=0.93):
        self.intent = intent
        self.confidence = confidence


class _Judgement:
    def __init__(self, can_refund, reply):
        self.can_refund = can_refund
        self.reply = reply


class _Queries:
    def __init__(self, queries):
        self.queries = list(queries)


class _Chunk:
    """Agent 那一轮用的最小 chunk 替身。

    只有「物流轮的出口不是子流程」那条用例会碰到它 —— 退款子流程**不经过**
    Agent 节点(见 spec §3.3:判定是一次 `ainvoke`,不进 ReAct)。
    """

    def __init__(self, text):
        self.text = text
        self.tool_calls = []
        self.usage_metadata = None

    def __add__(self, other):
        return _Chunk(self.text + other.text)


class _Outcome:
    """`with_structured_output(...)` 的返回物:一个只有 `ainvoke` 的对象。

    **计数器在这里,不在节点里**:本文件要证明「订单查不到时**没去判**」,
    而"判"这个动作唯一可观测的落点就是这条链的 `ainvoke`。
    `result` 是异常实例时照抛 —— 那正是「模型出参不可用」的替身形态。
    """

    def __init__(self, model, key, result, sink=None):
        self._model, self._key, self._result, self._sink = model, key, result, sink

    async def ainvoke(self, messages):
        self._model.calls[self._key] += 1
        if self._sink is not None:
            self._sink.append(list(messages))
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result


class RefundModel:
    """退款子流程里的**主力模型**替身。真模型有三个入口,替身也分三个:

    | 入口 | 谁在用 | 替身行为 |
    |---|---|---|
    | `ainvoke` | 指代消解 | 回显本轮原话(`resolved_input == user_input`) |
    | `with_structured_output(ExpandQueries)` | Query 扩写 | 回给定 queries |
    | `with_structured_output(RefundJudgement)` | 退款判定 | 回给定判定,或抛给定异常 |

    合成一个入口的话,「判定到底调没调」就没有观测点了 —— 而
    「订单查不到时**不许**判」正是本文件要钉的一条。
    """

    def __init__(self, *, judgement=None, queries=("退款政策", "退货时效"),
                 resolve_text=None):
        self.judgement = _Judgement(True, YES_REPLY) if judgement is None else judgement
        self.queries = _Queries(queries)
        # 消解默认回显;给了 resolve_text 就吐那一段(模拟改写把订单号改掉/吃掉)。
        self.resolve_text = resolve_text
        self.calls = {"judge": 0, "expand": 0, "resolve": 0}
        self.judge_messages: list[list] = []

    async def ainvoke(self, messages):
        self.calls["resolve"] += 1
        if self.resolve_text is not None:
            return AIMessage(content=self.resolve_text)
        return AIMessage(content=messages[-1].content)

    def with_structured_output(self, schema, method=None):
        if schema is RefundJudgement:
            return _Outcome(self, "judge", self.judgement, self.judge_messages)
        if schema is ExpandQueries:
            return _Outcome(self, "expand", self.queries)
        raise AssertionError(f"未预期的结构化出参 schema:{schema}")

    # 主力模型同时是 Agent 那一轮用的模型(ch05 起 `resolve_references` 也用它)。
    # 退款子流程走不到这两个入口,但「物流轮不弹卡片」那条用例会走到 Agent。
    def bind_tools(self, tools):
        return self

    async def astream(self, messages):
        yield _Chunk("物流回复")


class IntentModel:
    """意图分类替身:固定回一个 intent(图必须真的把它路由到 REFUND)。"""

    def __init__(self, intent, confidence=0.93):
        self.result = _Intent(intent, confidence)
        self.calls = {"intent": 0}

    def with_structured_output(self, schema, method=None):
        return _Outcome(self, "intent", self.result)


class FakeRetriever:
    def __init__(self, chunks=(), fail=None):
        self.chunks = list(chunks)
        self.fail = fail
        self.calls: list[str] = []

    async def search(self, query):
        self.calls.append(query)
        if self.fail is not None:
            raise self.fail
        return list(self.chunks)


class RecordingSession:
    def __init__(self):
        self.added = []

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        # ch07 起 `append_turn` 落库前会 flush 拿自增主键(真实 AsyncSession 有)。
        pass

    async def commit(self):
        pass


def _order_tool(*, calls, fail=None, delay=0.0):
    """`query_order` 的替身。

    **必须是真 `@tool`**:`execute_tool` 走的是 `BaseTool.ainvoke(tool_call)`,
    参数校验与 `{"type": "tool_call"}` 那一套语义只有真工具才有(MagicMock
    会把整条链变成「替身自己跟自己玩」)。名字也必须叫 `query_order` ——
    注册表按名字查。

    `delay` 用来制造**超时**(执行器按 `settings.tool_timeout_seconds` 掐);
    计数照旧在进函数体时先记 —— 被取消之前它已经进来过了。
    """

    @tool
    async def query_order(order_id: str) -> str:
        """查询订单详情:状态、商品、金额、下单时间。"""
        calls.append(order_id)
        if delay:
            await asyncio.sleep(delay)
        if fail is not None:
            raise fail
        return json.dumps(
            {"order_id": order_id, "status": "已发货", "product": "无线耳机",
             "amount": "129.50"},
            ensure_ascii=False,
        )

    return query_order


class _Harness:
    """一次构建的全部可观测面。图是**每请求现编**的,所以每个用例自己一份。"""

    def __init__(self, *, graph, session, retriever, calls, model, frames, history):
        self.graph = graph
        self.session = session
        self.retriever = retriever
        self.calls = calls
        self.model = model
        self.frames = frames          # 从 custom 流收到的帧(累积)
        self.history = history
        self.interrupts: list = []    # 从 updates 流里认出的 interrupt 载荷(累积)

    async def start(self, user_input, *, thread):
        """开一轮,跑到挂起(或跑完)。返回**本次**出现的 interrupt 载荷列表。"""
        return await self._drive(
            {"conversation_id": CONV, "user_input": user_input,
             "history": self.history, "trace": []},
            thread=thread,
        )

    async def resume(self, value=PICKED, *, thread):
        """从挂起点续跑。"""
        return await self._drive(Command(resume=value), thread=thread)

    async def _drive(self, payload, *, thread):
        seen = []
        async for mode, chunk in self.graph.astream(
            payload,
            config={"configurable": {"thread_id": thread}},
            stream_mode=["custom", "updates"],
        ):
            if mode == "custom":
                self.frames.append(chunk)
            elif "__interrupt__" in chunk:
                seen.append(chunk["__interrupt__"][0].value)
        self.interrupts.extend(seen)
        return seen

    async def state(self, *, thread):
        snapshot = await self.graph.aget_state(
            {"configurable": {"thread_id": thread}}
        )
        return snapshot.values

    def tokens(self) -> str:
        """把所有 token 帧拼回一段文本。

        验收/断言**不许**直接 grep 原始帧流:回复是逐 token 推的,一个订单号
        会被切成三个独立帧(CLAUDE.md 平台陷阱里那条,ch02 咬过三次)。
        """
        return "".join(f["text"] for f in self.frames if f.get("frame") == "token")

    def frames_of(self, name) -> list[dict]:
        return [f for f in self.frames if f.get("frame") == name]


def _settings(**over):
    return Settings(
        _env_file=None,
        openai_base_url="https://example.invalid/v1", openai_api_key="sk-test",
        openai_model="m", database_url="mysql+asyncmy://u:p@h:3306/db", **over,
    )


def _build_refund_graph(*, intent="退款退货", judgement=None, order_fail=None,
                        order_delay=0.0, retriever=None, history=(),
                        queries=("退款政策", "退货时效"), resolve_text=None,
                        settings=None, registry=None):
    calls: list[str] = []
    frames: list[dict] = []
    session = RecordingSession()
    model = RefundModel(judgement=judgement, queries=queries, resolve_text=resolve_text)
    # **只建一个检索器**:图与台账各拿一份的话,`h.retriever.calls` 永远是空的,
    # 而「扩写出来的几条查询都真的搜了」那条断言会因为「拿到的是另一个对象」
    # 而看起来像绿的。
    retriever = FakeRetriever([CHUNK]) if retriever is None else retriever
    # `registry=None` 才用默认那份;显式传 `{}` 要保留(那是「注册表里没有
    # query_order」这个接线 bug 的探针,`registry or {...}` 会把它换掉)。
    if registry is None:
        registry = {"query_order": _order_tool(calls=calls, fail=order_fail,
                                               delay=order_delay)}
    graph = build_graph(
        model=model,
        intent_model=IntentModel(intent),
        tools=[],
        registry=registry,
        settings=settings or _settings(),
        retriever=retriever,
        session=session,
        conversation_id=CONV,
        # collector 直接**炸**:帧只许从图内(`get_stream_writer`)出来。
        # 退化成 collector 的话本文件的帧断言会全绿 —— 而那是 emit 坏了、
        # 前端一帧都收不到的形态(见 `app/agent/emit.py` 的实测记录)。
        emit=make_emitter(lambda payload: pytest.fail(f"帧没进图:{payload}")),
        checkpointer=InMemorySaver(),
    )
    return _Harness(graph=graph, session=session, retriever=retriever,
                    calls=calls, model=model, frames=frames, history=list(history))


# ---- 1. 缺订单号 → interrupt 弹卡片 ---------------------------------------


@pytest.mark.anyio
async def test_missing_order_no_interrupts_with_cards():
    """缺订单号 → interrupt,载荷是 order_choice 帧的形状。"""
    h = _build_refund_graph()
    got = await h.start("这个能退吗", thread="t-refund-1")

    assert got, "没有出现 interrupt —— 订单卡片永远不会显示"
    payload = got[0]
    assert payload["frame"] == "order_choice"
    assert payload["options"]
    # 卡片要能显示「是哪一单」:spec §5.2 给的是对象数组。
    assert [o["order_no"] for o in payload["options"]] == list(DEMO_ORDERS)
    assert all(set(o) == {"order_no", "status", "product", "amount"}
               for o in payload["options"])
    # **挂起的那一轮不落库**(spec §5.1):`log_turn` 根本没跑。
    assert h.session.added == []
    # 取订单在下游节点 —— interrupt 之前**一次都没有**发生过(见 F3 那条用例)。
    assert h.calls == []
    # 卡片帧由**端点**从 `__interrupt__` 转出(spec §5.2),节点自己不发 ——
    # 节点也发一次的话前端会渲染出两组卡片。
    assert h.frames_of("order_choice") == []


@pytest.mark.anyio
async def test_pick_order_node_does_not_emit_frames():
    """挂起前的最后一个动作是 `interrupt()`,不是发帧。

    节点里发帧的话那张卡会**发两遍**(一次在挂起前、一次由端点转出),
    而前端每次都会 append 一组卡片。
    """
    h = _build_refund_graph()
    await h.start("这个能退吗", thread="t-refund-emit")
    assert h.frames == []


# ---- 2. resume 回填订单号 → 走完并给出退款入口 -----------------------------


@pytest.mark.anyio
async def test_resume_with_order_no_drives_flow_to_offer():
    """resume 回填订单号 → 子流程走完并给出可提交退款的信号。"""
    h = _build_refund_graph(judgement=_Judgement(True, YES_REPLY))
    assert await h.start("这个能退吗", thread="t-refund-2")

    got = await h.resume(PICKED, thread="t-refund-2")
    assert got == []                                  # 续跑后不再挂起

    offers = h.frames_of("refund_offer")
    assert len(offers) == 1
    assert offers[0]["order_no"] == PICKED
    # 类目**从帧里来**,且与单一来源逐字一致(前端不硬编码这份清单)。
    assert list(offers[0]["categories"]) == list(REFUND_REASON_CATEGORIES)
    # 传给 query_order 的就是用户点的那一单。
    assert h.calls == [PICKED]
    # 话术以 token 帧流出(前端是「累积 token 画气泡」的,只写 state 会空白)。
    assert YES_REPLY in h.tokens()
    # 走完了 → 落库(spec §5.1:挂起的那一轮不落库,resume 走完才落)。
    assert [m.role for m in h.session.added] == ["user", "assistant"]


@pytest.mark.anyio
async def test_resume_accepts_the_request_body_shape_too():
    """端点的请求体是 `{"resume": {"order_no": …}}`(spec §5.1),而本节点的
    brief 写的是裸串。**两种都得能用** —— 这是跨任务的接缝:本文件的其他用例
    一律传裸串,端点若把整个 resume 对象递进来,只有这条用例看得见。

    形状收窄成裸串时的表现**不是报错**,而是用户点完卡片收到
    「没能查到订单 {'order_no': '83746592'}」—— 内部载荷漏给了用户。
    """
    h = _build_refund_graph(judgement=_Judgement(True, YES_REPLY))
    await h.start("这个能退吗", thread="t-refund-16")

    await h.resume({"order_no": PICKED}, thread="t-refund-16")

    assert h.calls == [PICKED]
    assert h.frames_of("refund_offer")[0]["order_no"] == PICKED


# ---- 3. 问题里 / 历史里已经带了订单号 → 不弹卡片 ---------------------------


@pytest.mark.anyio
async def test_context_already_has_order_no_does_not_interrupt():
    """问题里已经带了订单号 → 不弹卡片,直接走。"""
    h = _build_refund_graph(judgement=_Judgement(True, YES_REPLY))
    got = await h.start(f"订单 {IN_CONTEXT} 能退吗", thread="t-refund-3")

    assert got == []                                  # 没打断
    assert h.frames_of("order_choice") == []
    assert h.calls == [IN_CONTEXT]                    # 用的就是问题里那个号
    assert h.frames_of("refund_offer")                # 且确实走完了
    assert h.session.added


@pytest.mark.anyio
async def test_raw_user_input_outranks_the_rewrite_when_both_carry_a_number():
    """两段语料**各带一个号码**时,用**原话**那个(次序即优先级)。

    上一条只证明原话**被包含进**语料,证不了次序 —— 把语料顺序倒过来
    (`[raw, resolved]`),上一条照样绿,而这里会是 `[IN_HISTORY]`。
    """
    h = _build_refund_graph(judgement=_Judgement(True, YES_REPLY),
                            resolve_text=f"订单 {IN_HISTORY} 能退吗")
    got = await h.start(f"订单 {IN_CONTEXT} 能退吗", thread="t-refund-19")

    assert got == []
    assert h.calls == [IN_CONTEXT]        # 原话那个
    assert IN_HISTORY not in h.calls      # 改写稿那个**没有**被用


@pytest.mark.anyio
async def test_raw_user_input_wins_over_a_rewrite_that_drops_the_order_no():
    """改写**可能把订单号改掉**(消解是自由文本,模型随时可能顺手"润色")。

    槽位因此两段语料都扫,且**本轮原话优先**:拿改写后那一版当唯一依据的话,
    这个用例里用户报了单号、却只会拿到一组演示订单卡片 —— 点进去还不是他的单。
    `resolved_input` 在本用例里**不含订单号**,所以它一旦被当成唯一来源,
    断言立刻红。
    """
    h = _build_refund_graph(judgement=_Judgement(True, YES_REPLY),
                            resolve_text="这个订单能退吗")
    got = await h.start(f"订单 {IN_CONTEXT} 能退吗", thread="t-refund-15")

    assert got == []
    assert h.calls == [IN_CONTEXT]


@pytest.mark.anyio
async def test_history_order_no_fills_the_slot_without_cards():
    """会话历史里出现过订单号 → 用它,**不弹卡片**(spec §3.3 来源②)。

    历史里那个号不是演示池里的号:用演示号的话「从历史取到」与「回落到演示集」
    会给出同一个观测值。
    """
    h = _build_refund_graph(
        judgement=_Judgement(True, YES_REPLY),
        history=[Message(role="user", content=f"我买的 {IN_HISTORY} 到哪了")],
    )
    got = await h.start("这个能退吗", thread="t-refund-4")

    assert got == []
    assert h.calls == [IN_HISTORY]


# ---- 4. F3 回归防线:取数恰好一次 -----------------------------------------


@pytest.mark.anyio
async def test_fetch_runs_exactly_once_across_resume():
    """resume 会把节点从头重跑 —— 取订单**不能**因此执行两次。

    两段断言各有分工,都不是多余的:

    - `calls == []`(挂起后)**证不了**「取数在下游」:把取数放进 pick 节点里,
      第一次跑就会查一次 —— 那时它已经非空了。所以这条断的是
      「**interrupt 之前**没干过任何事」;
    - `calls == [PICKED]`(续跑后)才是「恰好一次」:取数若在 pick 节点里、
      且摆在 `interrupt()` **之前**,节点重跑会执行**第二遍**,这里是 2。
      (摆在 interrupt **之后**的取数本来就是单次 —— 那种摆法不是 F3 那个故障,
      本条断言因此**管不着**它;本节点真正的规矩是「除了 interrupt 什么都不干」,
      比「别执行两遍」更严。)

    实测本机 langgraph 1.2.11:resume 时 `interrupt()` 之前的代码确实会再执行
    一遍(探针里 `[pick] top` 打印了两次),而 `interrupt()` **之后**的节点
    (取数)只跑一次。
    """
    h = _build_refund_graph(judgement=_Judgement(True, YES_REPLY))
    await h.start("这个能退吗", thread="t-refund-5")
    assert h.calls == []                      # interrupt 之前一次都没有

    await h.resume(PICKED, thread="t-refund-5")
    assert h.calls == [PICKED]                # resume 之后恰好一次(不是两次)


# ---- 5. 每轮清零:漏了它就静默串轮 ----------------------------------------


@pytest.mark.anyio
async def test_resolve_references_resets_refund_channels():
    """每轮开头必须把这三个通道清回初值(通道与它的清零同处一地)。"""
    node = make_resolve_references_node(model=RefundModel())
    out = await node({
        "user_input": "在吗",
        "order_no": PICKED,
        "order_data": {"order_id": PICKED, "status": "已发货"},
        "refund_decision": True,
    })

    assert out["order_no"] == ""
    assert out["order_data"] == {}
    assert out["refund_decision"] is None


@pytest.mark.anyio
async def test_second_turn_on_same_thread_clears_refund_slots():
    """**跨轮串扰**:checkpointer 是进程级单例、thread_id = session_id,
    未写的通道**保留上一轮的值**。

    漏了清零的后果是静默的:第 1 轮的订单号会被第 2 轮当成本轮槽位 ——
    用户明明没提订单号,却既不弹卡片、又拿着**上一单**去判能不能退。
    断言必须落在「第 2 轮还弹不弹卡片」上,那是唯一能区分两者的观测点。
    """
    h = _build_refund_graph(judgement=_Judgement(True, YES_REPLY))
    # 第 1 轮:订单号就在问题里 → 不弹卡片,走完(状态里留下 order_no)。
    assert await h.start(f"订单 {IN_CONTEXT} 能退吗", thread="t-refund-6") == []
    assert h.calls == [IN_CONTEXT]

    # 第 2 轮:同一 thread,同样问退款,但**这一轮没有订单号**。
    got = await h.start("这个能退吗", thread="t-refund-6")

    assert got, "第 2 轮没弹卡片 —— 上一轮的订单号被当成了本轮槽位(清零丢了)"
    assert got[0]["frame"] == "order_choice"
    # 第 2 轮只许查它自己选的那一单。
    assert h.calls == [IN_CONTEXT]


# ---- 6. 查不到单:如实报告,**不判** ---------------------------------------


@pytest.mark.anyio
async def test_order_not_found_is_reported_and_never_judged():
    """查不到单不是「不能退」—— 不许进判定,也不许给退款入口。"""
    h = _build_refund_graph(
        judgement=_Judgement(True, YES_REPLY),
        order_fail=ToolNotFound("未找到订单 x,请如实告知用户,不要自行编造"),
    )
    got = await h.start(f"订单 {IN_CONTEXT} 能退吗", thread="t-refund-7")

    assert got == []
    assert h.model.calls["judge"] == 0                # 没判
    assert h.frames_of("refund_offer") == []          # 也没给入口
    text = h.tokens()
    assert IN_CONTEXT in text                         # 点出是哪个号查不到
    assert "人工" in text                             # 给出去处
    # 面向用户的话术是**我们自己写的**,不是工具那句话的复读:工具的 `ToolNotFound`
    # 文案是写给**模型**看的(同族的 query_logistics 那条里写着「不要自行编造物流
    # 信息」),原样吐出来就是把提示词漏给用户。替身里那句哨兵因此必须不出现。
    assert "不要自行编造" not in text
    # 如实报告 → 这一轮照样落库(不是挂起)。
    assert h.session.added
    # 判定的载荷里不该出现"能不能退"的结论。
    assert (await h.state(thread="t-refund-7"))["refund_decision"] is False


@pytest.mark.anyio
async def test_lookup_timeout_is_not_reported_as_a_missing_order():
    """**超时不是「你的订单号查不到」。**

    这条是本节点最容易犯的错:`execute_tool` 在**超时**时也返回 `ok=False`,
    照单全收就会把 MySQL 卡住说成「请核对订单号」—— 拿服务端的故障指责用户
    输入,而这正是 `ToolInfrastructureError` 那条边界要防的事。
    """
    h = _build_refund_graph(
        judgement=_Judgement(True, YES_REPLY),
        order_delay=0.5,
        settings=_settings(tool_timeout_seconds=0.01, tool_retry_attempts=0),
    )
    got = await h.start(f"订单 {IN_CONTEXT} 能退吗", thread="t-refund-17")

    assert got == []
    assert h.calls == [IN_CONTEXT]                    # 确实试过(且只试了一次)
    text = h.tokens()
    assert "稍后再试" in text                          # 不指责用户的说法
    assert "核对订单号" not in text                     # ← 真正的判别点
    assert h.frames_of("refund_offer") == []
    assert (await h.state(thread="t-refund-17"))["refund_decision"] is False


@pytest.mark.anyio
async def test_registry_miss_raises_instead_of_blaming_the_user():
    """注册表里没有 `query_order` = **接线 bug**,不是「查无此单」。

    这个工具名是本文件写死的,所以它只会因为**我们自己**改了名字/注册表而落空。
    编一句面向用户的「没能查到订单」会让一次接线错误伪装成用户报错了号码 ——
    而谁都查不出来。上抛 → 端点 502 + 固定文案。
    """
    h = _build_refund_graph(judgement=_Judgement(True, YES_REPLY), registry={})
    with pytest.raises(ToolInfrastructureError):
        await h.start(f"订单 {IN_CONTEXT} 能退吗", thread="t-refund-18")

    assert h.calls == []                              # 工具压根没被调
    assert h.frames == []                             # 一句面向用户的话都没说出去


# ---- 7. 判定 ---------------------------------------------------------------


@pytest.mark.anyio
async def test_judge_no_goes_to_explain_without_offer():
    """替身说「不能退」→ 走 refund_explain,且**不发** refund_offer。

    这条是本任务对「判据必须可测」的落点:判据换成什么都行,但一个明确的
    「不能退」必须路由到解释出口,而不是照样给退款入口。
    """
    h = _build_refund_graph(judgement=_Judgement(False, NO_REPLY))
    await h.start(f"订单 {IN_CONTEXT} 能退吗", thread="t-refund-8")

    assert h.model.calls["judge"] == 1
    assert h.frames_of("refund_offer") == []
    assert NO_REPLY in h.tokens()
    assert (await h.state(thread="t-refund-8"))["refund_decision"] is False


@pytest.mark.anyio
async def test_unparseable_judgement_says_so_instead_of_guessing():
    """判不出结构 → 如实说「判不了」,不给入口、也不假装是不能退。

    形态:模型出参不可用(`OutputParserException`),与 `classify_intent` /
    `expand_queries` 那一族同源。判别的落点是 `refund_decision is None` ——
    **不是 False**:False 的含义是「这一单不能退」,那是个结论。
    """
    h = _build_refund_graph(judgement=OutputParserException("不是 JSON"))
    await h.start(f"订单 {IN_CONTEXT} 能退吗", thread="t-refund-9")

    assert h.model.calls["judge"] == 1
    assert h.frames_of("refund_offer") == []
    assert (await h.state(thread="t-refund-9"))["refund_decision"] is None
    text = h.tokens()
    assert "判断不了" in text or "判不了" in text
    assert "人工" in text


# ---- 8. 接缝:条款真的进了判定的 prompt -----------------------------------


@pytest.mark.anyio
async def test_retrieved_clauses_reach_the_judge_prompt():
    """**接缝测试**:检索到的条款必须真的走到判定那一步的 prompt 里。

    两半各自都有测试也照样漏:扩写节点断言「检索到了」、判定节点断言「路由对了」,
    而中间那根线(节点写 `evidence`、判定读 `evidence`)没有任何人看 ——
    这正是 T4 那条 Critical 的形态(通道没声明,两端各自都绿)。
    """
    h = _build_refund_graph(judgement=_Judgement(True, YES_REPLY))
    await h.start(f"订单 {IN_CONTEXT} 能退吗", thread="t-refund-10")

    assert h.model.calls["expand"] == 1
    assert h.model.calls["judge"] == 1
    rendered = "\n".join(str(m.content) for m in h.model.judge_messages[-1])
    assert CLAUSE in rendered                      # 条款原文进了 prompt
    assert IN_CONTEXT in rendered                  # 这一单的信息也进了
    # 命中编号与 citations 帧对齐([1]) —— 话术里能引用得动。
    assert "[1]" in rendered
    items = h.frames_of("citations")[0]["items"]
    assert [i["n"] for i in items] == [1]
    assert items[0]["answer"] == CLAUSE
    # 扩写出来的多条查询都真的搜了(多路检索去重合并)。
    assert h.retriever.calls == ["退款政策", "退货时效"]


@pytest.mark.anyio
async def test_no_evidence_still_reaches_a_verdict_path():
    """检索一条都没命中时,流程仍必须走完(不许静默停在半路)。

    知识库里没有条款不是「能退」的依据,也不该让整轮崩掉 —— 判定照跑,
    由 prompt 要求模型如实说判不了(这里替身回的是「能退」,断言的是
    **流程没有断**、且落库)。
    """
    h = _build_refund_graph(judgement=_Judgement(True, YES_REPLY),
                            retriever=FakeRetriever([]))
    await h.start(f"订单 {IN_CONTEXT} 能退吗", thread="t-refund-11")

    assert h.model.calls["judge"] == 1
    assert h.frames_of("citations") == []          # 没有命中就不发引用帧
    assert h.session.added
    # 空证据**不能**渲染成「以下是知识库中与该问题相关的资料:」后面跟着空白 ——
    # 那句邀请标注编号的话还在,模型就会凭空编一条 [1] 出来。所以断言两件事:
    # 说明句在,邀请句不在。(只断前一句的话,把 `if evidence else NO_CLAUSE_NOTE`
    # 换成无条件 `render_evidence(evidence)` 照样绿。)
    rendered = "\n".join(str(m.content) for m in h.model.judge_messages[-1])
    assert "没有检索到" in rendered
    assert "以下是知识库中" not in rendered


@pytest.mark.anyio
async def test_empty_judge_reply_falls_back_to_a_fixed_line_on_the_offer_path():
    """判定只填了 can_refund、话术是空串 → 必须有一句固定话术兜住。

    前端是「累积 token 画气泡」的:没有 token 帧 = 一个**空气泡**,而退款表单
    会单独出现在下面(看起来像坏了)。
    """
    h = _build_refund_graph(judgement=_Judgement(True, ""))
    await h.start(f"订单 {IN_CONTEXT} 能退吗", thread="t-refund-20")

    assert "请选择退款原因后提交" in h.tokens()
    assert h.frames_of("refund_offer")


@pytest.mark.anyio
async def test_empty_judge_reply_falls_back_to_a_fixed_line_on_the_explain_path():
    h = _build_refund_graph(judgement=_Judgement(False, "   "))
    await h.start(f"订单 {IN_CONTEXT} 能退吗", thread="t-refund-21")

    assert "建议联系人工客服进一步核实" in h.tokens()
    assert h.frames_of("refund_offer") == []


# ---- 9. 基础设施故障:一路抛出去,绝不伪装 --------------------------------


@pytest.mark.anyio
async def test_order_infrastructure_failure_propagates():
    """数据库故障绝不能被伪装成「你的订单号查不到」。"""
    h = _build_refund_graph(order_fail=SQLAlchemyError("db down"))
    with pytest.raises(ToolInfrastructureError):
        await h.start(f"订单 {IN_CONTEXT} 能退吗", thread="t-refund-12")


@pytest.mark.anyio
async def test_retrieval_infrastructure_failure_propagates():
    """检索侧同理(向量库/嵌入挂了 → 502,不是「没搜到」)。"""
    h = _build_refund_graph(
        judgement=_Judgement(True, YES_REPLY),
        retriever=FakeRetriever(fail=ToolInfrastructureError("向量库不可用")),
    )
    with pytest.raises(ToolInfrastructureError):
        await h.start(f"订单 {IN_CONTEXT} 能退吗", thread="t-refund-13")


# ---- 10. 出口没接错 --------------------------------------------------------


@pytest.mark.anyio
async def test_refund_flow_is_not_reached_for_other_intents():
    """退款子流程只对 REFUND 出口开 —— 物流轮不该弹卡片。

    (只测「退款轮走子流程」的话,把 `route_by_intent` 里**所有**意图都改成
    REFUND 也全绿。)
    """
    h = _build_refund_graph(intent="物流")
    got = await h.start("订单 83746592 到哪了", thread="t-refund-14")

    assert got == []
    assert h.model.calls["judge"] == 0
    assert h.session.added
