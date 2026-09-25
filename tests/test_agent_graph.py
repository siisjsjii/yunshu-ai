"""整图行为:五条出口各走一遍,断言 trace(验收 1/5 的可检查性来源)。"""

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.emit import make_emitter
from app.agent.graph import build_graph, get_checkpointer
from app.agent.nodes import make_resolve_references_node
from app.agent.routing import INTENT_TO_ROUTE
from app.config import Settings
from app.retrieval.search import RetrievedChunk
from app.tools.builtin.handoff import build as build_handoff
from app.tools.registry import _spec_from_tool


class _Intent:
    """意图分类器的出参替身。

    `confidence` **必须跟着补**(T4 起 `IntentResult` 有这个字段):不补的话
    替身不是生产结果的形状,而 `classify_intent` 那边一加 `getattr` 兜底
    「缺字段」就再也不会红 —— T4 初版正是这样让「通道没声明」溜过去的。
    """

    def __init__(self, intent, confidence=0.9):
        self.intent = intent
        self.confidence = confidence


class FakeChunk(AIMessageChunk):
    """ch07:**基类是真正的 `AIMessageChunk`**。

    agent 节点现在把累积出来的 chunk 整个塞进 `state["messages"]`,而这个通道的
    `add_messages` reducer 会对条目做消息强制转换 —— 裸对象的红法是
    `NotImplementedError: Unsupported message type: <class '...FakeChunk'>`,
    指向替身而不是实现。真实链路上流的**就是** `AIMessageChunk`,所以这是把替身
    补齐到生产形状(与 T3 给替身补 `flush` 同一条理由)。
    """

    def __init__(self, text="", tool_calls=None, usage=None):
        super().__init__(
            content=text,
            tool_calls=[{"type": "tool_call", **tc} for tc in (tool_calls or [])],
        )
        self.usage_metadata = usage

    def __add__(self, other):
        # 与 T6 的替身同一条理由:**累积时不能丢 tool_calls,更不能丢
        # `"type": "tool_call"` 键**(见 Ruling 21)。丢了前者,「模型在一轮里
        # 分多次吐完一次工具调用」在替身里就永远累积不起来;丢了后者,
        # `BaseTool.ainvoke` 会把 dict 当**参数**去校验 schema(CLAUDE.md 的硬约束)。
        # 本文件的用例目前每轮只有一个 chunk,所以这条分支**走不到** ——
        # 但替身必须忠实于真实 chunk 的形状,否则它会教后来的人写错的形状,
        # 而且错得**不会红**。真实的多 chunk 累积由 T11 的验收 5 兜底。
        return FakeChunk(
            self.text + other.text,
            tool_calls=self.tool_calls + other.tool_calls,
        )


class ScriptedModel:
    """每次 astream 都从 next(脚本)取一批。"""

    def __init__(self, rounds):
        self.rounds = list(rounds)

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema, method=None):
        return self

    async def ainvoke(self, messages):
        batch = self.rounds.pop(0)
        return batch[0] if isinstance(batch, list) else batch

    async def astream(self, messages):
        batch = self.rounds.pop(0)
        for chunk in (batch if isinstance(batch, list) else [batch]):
            yield chunk


class _EchoModel:
    """消解那一步的替身:回显用户这一轮的原话。

    回显 = 「这句本来就完整,原样输出」那一类回答 ⇒ `resolved_input == user_input`,
    于是本文件既有的断言(如「检索器收到的 query 就是用户那句话」)仍然在问同一件事。
    """

    async def ainvoke(self, messages):
        return AIMessage(content=messages[-1].content)


class _EchoOnAinvoke(_EchoModel):
    """把 `ainvoke`(消解)与 `bind_tools`/`astream`(Agent)分给两个入口的替身。

    ch06 T5 起 `resolve_references` **每轮都会** `ainvoke` 一次,而本文件的脚本
    是按 Agent 的 `astream` 批次排的:消解若也去 pop 批次,`rounds=[]` 的探针
    会以 `IndexError: pop from empty list` 红在半路,有脚本的那些则被**错位消费**
    —— 两种红法指向的都是脚手架,不是实现。

    (真实模型两种入口都有,所以这里**不是**在替实现兜底:
    `graph.build_graph` 给消解传的就是同一个 model 实例。)
    """

    def __init__(self, inner):
        self._inner = inner

    def bind_tools(self, tools):
        return self._inner.bind_tools(tools)

    async def astream(self, messages):
        async for chunk in self._inner.astream(messages):
            yield chunk


class FakeRetriever:
    def __init__(self, chunks=()):
        self.chunks = list(chunks)
        self.calls = []

    async def search(self, query):
        self.calls.append(query)
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


def _settings(**over):
    return Settings(
        _env_file=None,
        openai_base_url="https://example.invalid/v1", openai_api_key="sk-test",
        openai_model="m", database_url="mysql+asyncmy://u:p@h:3306/db", **over,
    )


def _graph(intent, *, retriever=None, rounds=None, settings=None, frames=None,
           confidence=0.9, tools=(), registry=None):
    session = RecordingSession()
    # rounds is None 才用默认;显式传 [] 要保留成空脚本 ——
    # 那是「碰模型就炸」的探针,`rounds or [...]` 会把空列表换掉、探针失效。
    graph = build_graph(
        model=_EchoOnAinvoke(
            ScriptedModel(rounds if rounds is not None else [[FakeChunk("模型回复")]])
        ),
        intent_model=ScriptedModel([[_Intent(intent, confidence)]]),
        tools=list(tools),
        # `tools` 给的是**工具**,注册表要的是 `name → ToolSpec`;转换收在这里,
        # 与 `tests/test_agent_node.py::_reg` 同款。放在调用点的话每个用到工具的
        # 用例都要自己写一遍,而写错的样子是 `execute_tool` 报「工具不存在」
        # —— 指向的是测试脚手架,不是实现。
        registry={name: _spec_from_tool(tool, source="builtin")
                  for name, tool in (registry or {}).items()},
        settings=settings or _settings(),
        retriever=retriever or FakeRetriever(),
        session=session, conversation_id="conv-1",
        emit=(frames.append if frames is not None else (lambda p: None)),
        # **每个测试一个全新的 checkpointer**,不用 `get_checkpointer()` 那个
        # 进程级单例 —— 真机已验证:同一个 thread 上重复 ainvoke,带
        # `operator.add` 的 `trace` 会**跨调用累积**(实测 `['a','b','a','b']`)。
        # 而所有测试都用默认 thread "t",于是**测试之间互相污染**:
        # `test_business_route_skips_retrieval_and_gate` 断言的
        # `"confidence_gate:pass" not in out["trace"]` 会被前面那个知识类测试
        # 留下的同一句打红 —— 而它红得毫无道理,指向的是测试脚手架而非实现。
        # 单例本身由 `test_checkpointer_is_a_process_level_singleton` 单独覆盖,
        # 生产路径由 T8 的端点测试覆盖。
        checkpointer=InMemorySaver(),
    )
    return graph, session


async def _run(graph, user_input, thread="t"):
    # `agent_steps` 必须在入参里**显式给初值**:真机已验证,一个从未被写过的
    # 通道在返回的 state 里**根本不存在**(实测 `"agent_steps" in out` 为 False),
    # 于是 `assert out["agent_steps"] == 0` 抛的是 KeyError ——
    # 那个红指向「测试写错了」,而不是「闸没拦住」,把真问题盖掉。
    # 给了初值之后,这条断言才真的在问「Agent 到底有没有跑」。
    return await graph.ainvoke(
        {"conversation_id": "conv-1", "user_input": user_input, "history": [],
         "agent_steps": 0, "trace": []},
        config={"configurable": {"thread_id": thread}},
    )


@pytest.mark.anyio
async def test_knowledge_intent_forces_retrieval_before_agent():
    """验收 1:政策类问题,trace 里必须看到强制检索节点被走到。"""
    retriever = FakeRetriever([RetrievedChunk("怎么退货", "七天无理由", "退换货",
                                              chunk_id=1, section_path="退货政策", score=0.9)])
    graph, _ = _graph("商品咨询", retriever=retriever)
    out = await _run(graph, "怎么退货")
    assert retriever.calls == ["怎么退货"]
    assert "retrieve_knowledge:1 hits top=0.90" in out["trace"]
    assert "confidence_gate:pass" in out["trace"]


@pytest.mark.anyio
async def test_weak_evidence_skips_agent_and_records_low_confidence():
    retriever = FakeRetriever([RetrievedChunk("q", "a", "c", chunk_id=1,
                                              section_path=None, score=0.10)])
    frames = []
    graph, session = _graph("商品咨询", retriever=retriever, frames=frames)
    out = await _run(graph, "冷门问题")
    assert out["gate_passed"] is False
    assert out["agent_steps"] == 0          # **没进 Agent**
    assert session.added[0].entry_point == "置信度闸"
    # 落库的那条记录必须用的是**当轮会话 id**,不是闭包随手给的值。闭包与
    # state 各持一份 `conversation_id`,是同一个事实的两个来源 —— 不匹配时
    # 问题会被记到别的会话名下,而此前**没有任何断言看得见**(审查实测)。
    assert session.added[0].source_conversation_id == "conv-1"


@pytest.mark.anyio
async def test_business_route_skips_retrieval_and_gate():
    """业务数据类不预检索、不过闸(没有检索证据,证据强弱无从谈起)。"""
    retriever = FakeRetriever()
    graph, _ = _graph("物流", retriever=retriever)
    out = await _run(graph, "订单 1001 的物流到哪了")
    assert retriever.calls == []
    assert not any(t.startswith("confidence_gate") for t in out["trace"])
    assert "agent:converged" in out["trace"]


@pytest.mark.anyio
async def test_chitchat_returns_fixed_copy_and_never_calls_the_model():
    """闲聊出口的答复是**固定话术**:不调模型。

    ⚠️ ch06 T5 起,这个 `rounds=[]` 探针的射程**收窄了**:探针本身是替身里的
    `astream`/`bind_tools` 那条路,而 `resolve_references` 每轮都会先 `ainvoke`
    一次(走的是 `_EchoOnAinvoke` 的回显,不消费批次)。所以它现在断言的是
    「**Agent 那一步**没被调用」,不再是「整轮一次模型都没调」——
    后者从 T5 起就不成立了(消解本身要调一次)。
    """
    graph, _ = _graph("闲聊", rounds=[])     # 脚本为空:Agent 那一步碰模型就会 IndexError
    out = await _run(graph, "你好")
    assert "客服小猫" in out["reply"]
    assert "classify_intent:闲聊" in out["trace"]


@pytest.mark.anyio
async def test_complaint_emits_choices_and_returns_soothing_copy():
    frames = []
    graph, _ = _graph("投诉", rounds=[], frames=frames)
    out = await _run(graph, "我要投诉")
    assert out["choices"] == ["handoff", "ticket"]
    assert {"frame": "choices", "options": [
        {"key": "handoff", "label": "转人工"}, {"key": "ticket", "label": "建工单"}]} in frames


@pytest.mark.anyio
async def test_fallback_route_for_out_of_vocabulary_intent():
    graph, _ = _graph("其他", rounds=[])
    out = await _run(graph, "帮我写首诗")
    assert "没太理解" in out["reply"]
    assert "classify_intent:其他" in out["trace"]


@pytest.mark.anyio
async def test_handoff_intent_reaches_the_agent_and_calls_transfer_to_human():
    """ch10-A 第九类「转人工」的**图级**接线:标签 → 路由值 → 节点 → 工具。

    本文件此前那五条出口用例(商品咨询 / 物流 / 闲聊 / 投诉 / 其他)**一条都碰不到
    「转人工」** —— 而它的接线是**两个字典各写一半的约定**:
    `app/agent/routing.py` 把标签「转人工」映成路由值 `HANDOFF`,
    `app/agent/graph.py` 再把 `HANDOFF` 映到节点 `"agent"`。
    两处之间**没有任何东西把它们绑在一起**:把 graph.py 那一行删掉(或指到别的节点),
    每一轮「转人工」都会在图这一层失败,而**没有一条快测试会红** ——
    eval 集抓不到(分类器判得**对**),只有打网络的验收脚本会看见(审查 I1)。

    ⚠️ **一定要真的走到 `agent` 节点**,不能只看 `route_by_intent` 的返回值:
    后者是 `tests/test_agent_routing.py` 那条用例的事,它对「图怎么接」零判别力。
    这里的证据是 `agent` 节点的 trace(`agent:step1 tool=...`,由
    `app/agent/nodes.py` 在工具**真的执行过**之后追加)+ `tool_calls_made`
    + 帧里的 `tool_call` / `tool_result`。
    """
    frames = []
    retriever = FakeRetriever()
    # 用**真的**内置工具(不是替身):这条要钉的名字 `transfer_to_human` 与
    # `app/tools/builtin/handoff.py` 里那个是**同一个契约** —— 改名时这里必须跟着动。
    # 换成替身的话,「工具改名了、而 Agent 调不到它」会**静默**过去。
    tool = build_handoff(session=None, conversation_id="conv-1", retriever=None)[0]
    graph, _ = _graph(
        "转人工",
        retriever=retriever, frames=frames,
        tools=[tool], registry={"transfer_to_human": tool},
        rounds=[
            # 第 1 轮:模型调工具。`FakeChunk` 会替每条 `tool_calls` 补上
            # `"type": "tool_call"`(缺了它 `BaseTool.ainvoke` 会把整个 dict
            # 当成**参数**去校验,于是每次调用都返回一个假的「参数不合法」)。
            [FakeChunk("", tool_calls=[{
                "name": "transfer_to_human",
                "args": {"reason": "我要转人工"},
                "id": "call_h1",
            }])],
            # 第 2 轮:不绑 tools 的收尾轮(ch05 的结构保证)⇒ 这一段就是回复。
            [FakeChunk("已为您转接人工客服,工号 A102。")],
        ],
    )
    out = await _run(graph, "我要转人工")

    # 转人工**不是**知识类:不检索、不过置信度闸(spec §11.3 的「不开新出口」)。
    assert retriever.calls == []
    assert not any(t.startswith("confidence_gate") for t in out["trace"])
    assert "classify_intent:转人工" in out["trace"]
    # **走到了 agent 节点,并且工具真的执行了**。两半都断,因为只看后者的话,
    # 「路由到了别的节点、那儿碰巧也调了这个工具」也会绿。
    assert "agent:step1 tool=transfer_to_human" in out["trace"]
    assert [c["name"] for c in out["tool_calls_made"]] == ["transfer_to_human"]
    assert out["tool_calls_made"][0]["ok"] is True
    assert out["agent_steps"] == 2
    # 帧这一层(验收 ②/②b 读的就是它):call 发出去过、result 是 ok。
    assert {"frame": "tool_call", "name": "transfer_to_human",
            "args": {"reason": "我要转人工"}, "tool_call_id": "call_h1"} in frames
    results = [f for f in frames if f["frame"] == "tool_result"]
    assert len(results) == 1 and results[0]["ok"] is True
    assert results[0]["tool_call_id"] == "call_h1"
    # 收尾那一轮的文本就是给用户的回复,而且这一轮走完了整张图。
    assert out["reply"] == "已为您转接人工客服,工号 A102。"
    assert out["trace"][-1] == "log_turn"


def test_every_route_value_is_wired_to_a_node_in_the_graph():
    """**完整性守卫**:`INTENT_TO_ROUTE` 的每个路由值都必须在 `classify_intent`
    的条件边字典里当键出现(审查 I1 的第 2 半)。

    为什么不靠上面那条「转人工走 agent」的用例:它只盯**一个**路由值。
    将来加第十类意图、`INTENT_TO_ROUTE` 多一行、而忘了改 graph.py 时,
    新增的那一类**每一次**都会在图里炸(路由值不在 path_map 里)—— 而那时红的
    是运行时的整轮对话,不是测试。计划点名当这个守卫的
    `test_outlets_are_still_exactly_five`(在 `tests/test_agent_routing.py`)守的是
    **另一件事**(出口数没变多),对「加了意图却没接线」**零判别力**
    —— 那一条按复审的意见**保持原样**,这条新的才是这个用途的守卫。

    读的是**编译后的图**(`builder.branches[...].ends`),也就是 langgraph 真正
    拿到的那份映射 —— 不是把 graph.py 的字典再抄一遍(抄一遍等于两处同源,
    改了哪边都测不出来)。**不**用 `get_graph()`:它会把「键与值相同的 path_map」
    折成 `data=None`(实测 `refund_fetch_order` 那两条边就是这样),于是
    「路由值恰好与节点名同名」时这条断言会**少一个键** —— 而那不是缺陷。
    装置本身失效时(取不到那条分支/映射)会**红在半路**,不会静默变成「集合相等」。
    """
    graph, _ = _graph("闲聊", rounds=[])
    branches = graph.builder.branches["classify_intent"]
    ends = [spec.ends for spec in branches.values()]
    assert len(ends) == 1 and ends[0], (
        f"读不到 classify_intent 的条件边映射(拿到 {ends})—— 这条守卫的装置坏了"
    )
    wired = set(ends[0])

    expected = set(INTENT_TO_ROUTE.values())
    missing = sorted(expected - wired)
    extra = sorted(wired - expected)
    assert not missing, (
        f"这些路由值在 graph.py 的 classify_intent 条件边里**没有接线**:{missing} "
        f"(实际接了 {sorted(wired)})—— 命中这些路由值的那一类意图,"
        f"每一次都会在 langgraph 里炸,而没有任何快测试会先红"
    )
    assert not extra, (
        f"这些条件边键不是任何意图的路由值:{extra} —— 死边(写错了或意图被删了)"
    )


@pytest.mark.anyio
async def test_successful_turn_is_persisted_to_mysql_history():
    graph, session = _graph("闲聊", rounds=[])
    out = await _run(graph, "你好")
    roles = [m.role for m in session.added]
    assert roles == ["user", "assistant"]
    # 只断言角色的话,把两条消息的**内容对调**(把 user_input 存成 assistant)
    # 照样绿 —— 断言必须咬住内容。闲聊是固定话术,确定性,可以安全断言。
    assert session.added[0].content == "你好"
    assert session.added[1].content == out["reply"]


@pytest.mark.anyio
async def test_checkpointer_is_a_process_level_singleton():
    """每请求新建 checkpointer 的话,下一轮 thread 状态就没了。"""
    assert get_checkpointer() is get_checkpointer()


@pytest.mark.anyio
async def test_log_turn_emits_trace_frame_for_end_to_end_acceptance():
    """`trace` 要能被验收脚本从 SSE 里读到(验收 1/5 靠它机械可断言)。

    `log_turn` 用同一个 emit 通道发一帧 `trace`;端点把它折进 `done` 帧、
    不外推给前端。走同一条发出路径 = 不需要第二个 stream_mode。
    """
    frames = []
    graph, _ = _graph("闲聊", rounds=[], frames=frames)
    out = await _run(graph, "你好")
    kinds = [p["frame"] for p in frames]
    assert "trace" in kinds
    trace = next(p for p in frames if p["frame"] == "trace")["trace"]
    assert "resolve_references" in trace
    assert "classify_intent:闲聊" in trace
    assert "chitchat_reply" in trace
    assert trace[-1] == "log_turn"
    # 上面那条断言的是 **emit 的载荷**,而载荷是 `[*(state.get("trace") or []),
    # "log_turn"]` 拼的 —— 里面**天然**有 "log_turn"。`log_turn` 的**返回值**
    # 也要断言,否则把 `return {"trace": ["log_turn"]}` 改成 `{}` 全绿。
    assert out["trace"][-1] == "log_turn"


@pytest.mark.anyio
async def test_confidence_travels_from_classifier_through_the_graph_to_the_trace_frame():
    """**接缝测试**:分类器的 confidence 必须真的**在图里**走到 `trace` 帧。

    T4 初版漏了这条,而两半各自都有测试:
    - `tests/test_agent_intent.py` 断言节点的返回值;
    - 同文件另一条断言 `log_turn` 的帧载荷 —— 但它**把 confidence 直接注进 state**,
      LangGraph 全程没参与。
    于是「`ChatState` 里根本没声明 `confidence` 这个通道」溜了过去:通道集合由
    `StateGraph(ChatState)` 的注解决定,LangGraph 对未声明通道的写入是**静默丢弃**
    (`wrote to unknown channel ..., ignoring it`,只 warning 不抛)—— 真机上
    `log_turn` 每帧发出去的 confidence **恒为 null**,而两条测试全绿。

    本用例的 confidence **由替身分类器给**(0.87,与任何默认值都不撞),经
    `classify_intent` → state 通道 → `log_turn`,两端都断言。
    """
    frames = []
    graph, _ = _graph("闲聊", rounds=[], frames=frames, confidence=0.87)
    out = await _run(graph, "你好")

    payload = next(f for f in frames if f["frame"] == "trace")
    assert payload["confidence"] == pytest.approx(0.87)   # T8 要折进 done 帧的那份载荷
    assert out["confidence"] == pytest.approx(0.87)       # 通道里确实落了值


@pytest.mark.anyio
async def test_second_turn_on_same_thread_reports_only_its_own_turn():
    """**跨轮证据链**:同一个 thread 连跑两轮,第二轮只许报第二轮。

    checkpointer 是**进程级单例**、`thread_id = session_id`、而 `trace` 是
    `operator.add` 累积通道、其余通道**未写就保留旧值**。三件事叠起来:
    第 N 轮的 trace 帧是「第 1..N 轮」的拼接,于是**验收 1 的
    「trace 里有 retrieve_knowledge」会在一条根本没检索的轮上通过**(上一轮
    留下的),逐轮的 `gate_passed` / `agent_steps` 同理。这是本章最高价值
    证据链上的假绿通道,所以两条修法**(逐轮重置 + trace 切片)各自都要有
    断言钉住**,不能只修一条。

    本文件其余 10 条用例每条都新建 `InMemorySaver`、只跑一轮 —— **没有一条
    能看见这个**。
    """
    retriever = FakeRetriever([RetrievedChunk("怎么退货", "七天无理由", "退换货",
                                              chunk_id=1, section_path="退货政策", score=0.9)])
    frames = []
    session = RecordingSession()
    graph = build_graph(
        model=_EchoOnAinvoke(ScriptedModel([[FakeChunk("模型回复")]])),
        # 意图替身按**调用顺序**回放(ainvoke 是 rounds.pop(0)):第 1 轮知识类,
        # 第 2 轮闲聊 —— 两轮走**不同分支**,残留才看得见。
        intent_model=ScriptedModel([
            [_Intent("商品咨询", 0.93)],
            [_Intent("闲聊", 0.97)],
        ]),
        tools=[], registry={}, settings=_settings(),
        retriever=retriever, session=session, conversation_id="conv-1",
        emit=frames.append,
        # 两轮**共用**同一个 checkpointer,且 thread_id 相同 —— 这正是要测的场景。
        checkpointer=InMemorySaver(),
    )

    first = await _run(graph, "怎么退货")
    one_turn = len(frames)
    # 第二轮的入参**刻意不给 `agent_steps` 初值**(第一轮仍走 `_run`)。
    # 原因**实测**得到,不是推理:入参里的值会**覆写**非归约通道,所以
    # `agent_steps: 0` 一给,就等于替实现把上一轮的残留抹掉了 ——
    # 变异实测(只重置 `gate_passed`、`agent_steps` 不清)下:
    #   第二轮入参**给** `agent_steps: 0` → 帧里报 0 → 断言**假绿**;
    #   第二轮入参**不给**         → 帧里报 1(上一轮的值)→ 断言红。
    # 而真机端点(T8)的入参里**没有** `agent_steps`(见计划 T8 Step 5),
    # 残留会一路进 `done` 帧 —— 所以这里必须真的能红。
    second = await graph.ainvoke(
        {"conversation_id": "conv-1", "user_input": "你好", "history": [], "trace": []},
        config={"configurable": {"thread_id": "t"}},
    )
    second_frames = frames[one_turn:]

    # 前置条件:第 1 轮**确实**检索了、过了闸、进了 Agent —— 这样第二轮的
    # 「没有」才是在说「被清掉了」,而不是「第 1 轮本来就没有」。
    assert retriever.calls == ["怎么退货"]
    assert "retrieve_knowledge:1 hits top=0.90" in first["trace"]
    assert "confidence_gate:pass" in first["trace"]
    assert "agent:converged" in first["trace"]

    # 前提:累积通道**确实**累积了 —— 这正是本用例存在的理由。
    # 没有这一条,下面那些断言可能只是因为"根本没累积"而通过。
    assert len(second["trace"]) > len(first["trace"])

    payload = second_frames[-1]                    # 第二轮的 trace 帧
    assert payload["frame"] == "trace"
    assert payload["trace"][0] == "resolve_references"
    assert not any(t.startswith("confidence_gate") for t in payload["trace"])
    assert "retrieve_knowledge" not in " ".join(payload["trace"])
    assert payload["gate_passed"] is None          # 第 1 轮过闸了;第 2 轮没进闸
    assert payload["agent_steps"] == 0             # 第 1 轮进过 Agent;第 2 轮没有


@pytest.mark.anyio
async def test_resolve_references_resets_every_per_turn_channel():
    """每轮开头必须清掉上一轮的**全部**逐轮通道。

    为什么值得单钉:**删除其中一个键不会有任何测试变红** —— 而后果按通道
    不同而不等。`evidence` 尤其真实:它被 agent 节点读进 `build_messages`
    (`app/agent/nodes.py:234` 附近),业务轮跟在知识轮后会把**上一轮的检索
    结果**当本轮知识塞进 prompt —— 用户看到的是上一轮的知识,且完全静默。

    两轮整图用例(闲聊轮)盖不住它:闲聊不读 `evidence`。

    ch07:`turn_messages` 也进了这份清单,而且是**唯一不会自动被覆盖**的一个 ——
    其余通道每轮都被节点写一遍,这个只在「本轮真的产生了消息」时才写。所以
    「这轮没产生消息」的路径(挂起的那一轮)会**原样继承上一轮的值**,
    上一轮的 assistant 消息就会被**再写一遍**。入参里**显式给一个非空旧值**:
    不给的话它本来就是空的,断言恒真(与被测实现无关)。
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
        "turn_messages": [AIMessage(content="上一轮的回复")],
    })

    assert out["resolved_input"] == "在吗"
    assert out["trace"] == ["resolve_references"]
    assert out["evidence"] == []
    assert out["gate_passed"] is None
    assert out["agent_steps"] == 0
    assert out["reply"] == ""
    assert out["choices"] == []
    assert out["citations"] == []
    assert out["tool_calls_made"] == []
    assert out["turn_messages"] == []


@pytest.mark.anyio
async def test_emitter_sends_frames_through_astream_custom_mode():
    """**这条最关键**:证明 emit 在真实的图运行中把帧送到了 astream 消费端。

    缺了它,把 `make_emitter()` 写成「一次性 `get_stream_writer()`」的错误版本
    也能全绿 —— 而那种写法下前端**一帧都收不到**(图外必然抛 RuntimeError,
    静默退化成 no-op)。这正是本项目最怕的「假绿 + 静默故障」。
    """
    collected = []                       # 图外兜底:真跑到这里就说明没拿到 writer
    emit = make_emitter(collected.append)
    session = RecordingSession()
    graph = build_graph(
        model=_EchoOnAinvoke(ScriptedModel([])),
        intent_model=ScriptedModel([[_Intent("投诉", 0.95)]]),
        tools=[], registry={}, settings=_settings(), retriever=FakeRetriever(),
        session=session, conversation_id="conv-emit", emit=emit,
        checkpointer=get_checkpointer(),
    )

    got = []
    async for payload in graph.astream(
        {"conversation_id": "conv-emit", "user_input": "我要投诉", "history": [], "trace": []},
        config={"configurable": {"thread_id": "emit-test"}},
        stream_mode="custom",
    ):
        got.append(payload)

    assert {"frame": "choices", "options": [
        {"key": "handoff", "label": "转人工"}, {"key": "ticket", "label": "建工单"}]} in got
    assert collected == []               # 图内拿到了真 writer,没退化成 collector
