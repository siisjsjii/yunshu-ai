"""`agent` 节点的知识轮协议(ch09 §5.1–§5.6)。

协议是怎么一回事:知识轮多一条**协议消息**,文本流出过一次
`JsonAnswerDecoder` —— 一次调用里既作答、又自评。业务轮**一个字节都不改**。

调用方契约是 T9 交付的**三行表**(task-9-report.md §11-③,含 §11-⑩ 与 §11-⑪-D
两条注)。本文件的用例逐行对着那张表写,重点三处:

- **`useful is False`** ⇒ 兜底话术;只有知识类落池。**不降级、不显示原文。**
- **违约 / `useful is None` / 括号没配平** ⇒ 降级:把 `raw` 当纯文本发一遍
  + `agent:protocol_violation`;**唯一的例外是 `mode == PLAIN`**(§11-⑪-D)——
  那条路的 `raw` 已经在 lead 阶段逐片透出过,再发一遍就是把用户刚看过的话
  重说一次,而它**不是违约**(「先说我查一下再调工具」正是最常见的那一轮)。
- 其余 ⇒ 正常,已经流出去的就是答案。

⚠️ 本文件里**最不能省的是 `test_business_turn_is_byte_identical`** —— 它是
「业务轮零回归」这句话的**唯一**硬证据。它比对的是**帧序列**,不是最终 reply:
两次运行的 reply 可以一模一样,而 token 帧的**切法**不同就是回归(前端是按帧
画气泡的,逐字符帧会变成逐字符上屏)。**它的期望值是从改动前的实现上抄下来的**
(见 task-10-report.md 的 RED 一节),不是照着新实现反推的。
"""

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.agent.nodes import FALLBACK_REPLY, PROTOCOL_MESSAGE, make_agent_node
from app.config import Settings
from app.memory import budget
from app.prompts import render_system_prompt
from app.tools.registry import _spec_from_tool

REQUIRED = {
    "openai_base_url": "https://example.invalid/v1",
    "openai_api_key": "sk-test",
    "openai_model": "test-model",
    "database_url": "mysql+asyncmy://u:p@h:3306/db",
}


def _settings(**overrides) -> Settings:
    # `_env_file=None`:仓库根有真实 `.env`,不传的话缺字段的用例会被它静默补上。
    return Settings(_env_file=None, **REQUIRED, **overrides)


class _Chunk:
    """最小 chunk 替身。

    ⚠️ 两个字段都**必须**与生产同形:
    - `text` 是 1.x 里取流式文本的那一个(`content` 是 block 列表);
    - 工具调用条目要带 `"type": "tool_call"` —— `BaseTool.ainvoke` 判
      「这是不是工具调用」**只看**这一个键(CLAUDE.md 的硬约束)。
    """

    def __init__(self, text="", tool_calls=None, usage=None):
        self.text = text
        self.content = text
        self.tool_calls = [{"type": "tool_call", **tc} for tc in (tool_calls or [])]
        self.usage_metadata = usage

    def __add__(self, other):
        return _Chunk(
            text=self.text + getattr(other, "text", ""),
            tool_calls=list(self.tool_calls) + list(getattr(other, "tool_calls", []) or []),
            usage=getattr(other, "usage_metadata", None) or self.usage_metadata,
        )


class _Model:
    """脚本化模型。`rounds` 记下**每一轮的入参** —— 协议消息在不在那批消息里,
    只能从这里看(它在 `msgs` 上,不在 `state` 上)。
    """

    def __init__(self, rounds):
        # 两种写法都收:一整个脚本 `[[a, b], [c]]`,或单轮 `[a, b]`。
        if rounds and all(isinstance(r, str) for r in rounds):
            rounds = [rounds]
        self._rounds = [list(r) for r in rounds]
        self.rounds: list[list] = []
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def astream(self, msgs, **kw):
        self.calls += 1
        self.rounds.append(list(msgs))
        for item in self._rounds.pop(0):
            # 片段可以是纯文本(常见的形态),也可以是一个**已经造好的 chunk**
            # (带 tool_calls 那一轮要用它)。
            yield item if isinstance(item, _Chunk) else _Chunk(text=item)


class _Tool:
    """假工具。`args_schema = None` ⇒ 空 schema ⇒ 校验闸一律放行。

    本文件的工具都**不真的被调用**(`execute_tool` 那条路已被
    `tests/test_agent_node.py` 覆盖),这里只要能造出 tool_calls 那一轮。
    """

    args_schema = None
    description = ""

    def __init__(self, name="query_faq", content='{"answer": "七天无理由"}'):
        self.name = name
        self.content = content

    async def ainvoke(self, tool_call):
        return type("_R", (), {"content": self.content})()


class _Session:
    """假 session:`record_low_confidence` 只调 `add` + `commit`。"""

    def __init__(self):
        self.added: list = []

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        pass


def _make(model, *, settings=None, frames=None, session=None, tools=(), registry=None):
    settings = settings or _settings()
    return make_agent_node(
        model=model,
        tools=list(tools),
        registry={
            name: _spec_from_tool(tool, source="builtin")
            for name, tool in (registry or {}).items()
        },
        settings=settings,
        emit=(frames.append if frames is not None else (lambda p: None)),
        session=(session if session is not None else _Session()),
        # R4:`context_budget` **绝不传 `None`** —— 它会被喂给
        # `journal.model_ctx`,而那里读 `budget.layer1_budget`。生产路径由端点
        # 推一次传下来(`build_graph`),单测这里现算:`budget.derive` 是纯函数,
        # 同一份 settings + 同一个 system prompt ⇒ 同一个结果。
        context_budget=budget.derive(
            settings=settings, system_prompt=render_system_prompt(settings.brand_name)
        ),
    )


def _state(**over):
    base = {
        "conversation_id": "c1",
        "user_input": "退货政策是什么",
        "resolved_input": "退货政策是什么",
        "history": [],
        "evidence": [],
    }
    base.update(over)
    return base


def _tokens(frames) -> list[str]:
    return [f["text"] for f in frames if f["frame"] == "token"]


#: 一条**违规**的 `useful=false` 流:协议第 3 条要求 `answer` 为空串,这里故意非空。
#: 用它才验得动「`useful=false` 之后一个 delta 都不许出去」(正常的空串流没有
#: 可吐的东西,断言恒真 —— 那是 T9 边界 11 存在的理由,调用侧也要按它测)。
INSUFFICIENT_WITH_ANSWER = (
    '{"useful": false, "confidence": 0.2, "answer": "这段文本一个字都不该出现在用户面前"}'
)


@pytest.mark.anyio
async def test_knowledge_turn_streams_answer_only_after_useful():
    """不变量 1/2 的调用侧落点:`useful` 之前、`useful=false` 之后,都没有字节出去。

    合法协议流,跨 chunk 切开(真实网关就是这么流的)。
    """
    frames = []
    model = _Model(['{"useful": tru', 'e, "confidence": 0.9, "answer": "退货',
                    '政策是 7 天"}'])
    out = await _make(model, frames=frames).__call__(_state(intent="商品咨询"))

    texts = _tokens(frames)
    assert "".join(texts) == "退货政策是 7 天"
    assert out["reply"] == "退货政策是 7 天"
    # 协议字节不许泄漏给用户(前两片是纯协议,一个字符都不该以 token 帧出去)
    assert "useful" not in "".join(texts)
    assert "{" not in "".join(texts) and "confidence" not in "".join(texts)
    # 协议消息**确实挂上了**(不加的话上面几条同样绿,而模型根本不知道要吐协议)
    sent = model.rounds[0]
    assert PROTOCOL_MESSAGE in [m.content for m in sent]
    assert sent[-1].content[-len(PROTOCOL_MESSAGE):] == PROTOCOL_MESSAGE


@pytest.mark.anyio
async def test_business_turn_is_byte_identical():
    """**业务轮的帧序列与改动前逐字节相同** —— 「零回归」唯一的硬证据。

    期望值 = 改动前那版 `_stream_round` 的输出(**每个非空 `chunk.text` 出且只出
    一个 token 帧,原文逐字**),空片不发。

    ⚠️ **脚本必须是「协议形状」的,否则这条断言没有牙** —— 这是本轮实测出来的:
    解码器在 `plain` 态是**逐片原样**透出的,与「不过解码器」**一模一样**。
    先写成一句普通中文时,把 `is_knowledge` 打成 `True`(协议泄漏到业务轮)
    **这条帧断言照样绿**(实测),真正红的是下面那条 system 消息断言。
    换成 `{` 打头的输入,两条都红了:那半截一旦过解码器就被判成
    「首键不是 `useful`」的违约并**截断**,帧序列当场变成一条残缺的帧 ——
    而用户看到的就是**被吞掉一半的回答**。

    业务轮本来就该把模型说的**任何**东西原样交给用户(它没挂协议消息,
    模型也没有理由吐 JSON;可**万一它吐了**,我们更不该去解释它)。
    """
    frames = []
    model = _Model([['{"status": ', '"已发货"}', "", "已送达。"]])
    out = await _make(model, frames=frames).__call__(_state(intent="物流"))

    assert frames == [
        {"frame": "token", "text": '{"status": '},
        {"frame": "token", "text": '"已发货"}'},
        {"frame": "token", "text": "已送达。"},
    ]
    assert out["reply"] == '{"status": "已发货"}已送达。'
    # 业务轮不挂协议消息:消息序列就是「system + human」两条,与今天一字不差。
    assert [m.type for m in model.rounds[0]] == ["system", "human"]


@pytest.mark.anyio
async def test_business_tool_round_frames_are_byte_identical():
    """带工具往返的业务轮同样逐帧不变 —— ch08 的确认流正是长在这条路上。

    ⚠️ 这条断言的是**完整帧序列**(含 tool_call / tool_result 与它们的交错),
    不是「有没有发过某一帧」:把两帧的次序调换、或把收尾轮多问一次,前者
    在 `in` 式断言下照样绿(spec §12.1)。
    """
    tool = _Tool()
    frames = []
    model = _Model([
        [_Chunk(tool_calls=[{"name": "query_faq", "args": {"q": "退货"}, "id": "c1"}])],
        ["七天", "无理由"],
    ])
    out = await _make(model, frames=frames, tools=[tool],
                      registry={"query_faq": tool}).__call__(_state(intent="物流"))

    assert frames == [
        {"frame": "tool_call", "name": "query_faq",
         "args": {"q": "退货"}, "tool_call_id": "c1"},
        {"frame": "tool_result", "tool_call_id": "c1", "ok": True,
         "summary": '{"answer": "七天无理由"}'},
        {"frame": "token", "text": "七天"},
        {"frame": "token", "text": "无理由"},
    ]
    assert out["reply"] == "七天无理由"
    assert [m.type for m in model.rounds[0]] == ["system", "human"]
    # 第二轮入参的尾巴:带 tool_calls 的 assistant **紧邻** 它的 tool 结果
    # (少回灌一条、或把两者拆开,上游 OpenAI 兼容 API 直接 400)。
    second = model.rounds[1]
    assert [m.type for m in second[:2]] == ["system", "human"]
    assert [tc["name"] for tc in getattr(second[2], "tool_calls", [])] == ["query_faq"]
    assert isinstance(second[3], ToolMessage) and second[3].tool_call_id == "c1"


@pytest.mark.anyio
async def test_knowledge_tool_round_keeps_working_and_protocol_is_appended_once():
    """知识轮的**工具往返零损失**:协议只改「作答轮」,工具轮照常跑。

    并且钉住一处最容易写错的形状:**协议消息是每轮追加一次**(它进的是
    `msgs`,而 `msgs` 跨轮累积),**不是每轮重复追加一条** —— 写在循环里就会
    变成第二轮两条、第三轮三条,而那**不影响任何其余断言**(模型照样答得出)。
    """
    tool = _Tool()
    frames = []
    model = _Model([
        [_Chunk(tool_calls=[{"name": "query_faq", "args": {"q": "退货"}, "id": "c1"}])],
        ['{"useful": true, "confidence": 0.9, "answer": "七天无理由。"}'],
    ])
    out = await _make(model, frames=frames, tools=[tool],
                      registry={"query_faq": tool}).__call__(_state(intent="商品咨询"))

    assert "".join(_tokens(frames)) == "七天无理由。"
    assert out["reply"] == "七天无理由。"
    assert out["agent_steps"] == 2
    # 第一轮就已经挂上了(工具轮也看得见协议),而且**恰好一条**
    assert [m.content for m in model.rounds[0]].count(PROTOCOL_MESSAGE) == 1
    # 第二轮(作答轮)仍然**恰好一条** —— 不是两条、也不是三条
    assert [m.content for m in model.rounds[1]].count(PROTOCOL_MESSAGE) == 1
    assert "agent:protocol_violation" not in out["trace"]


@pytest.mark.anyio
async def test_useful_false_falls_back_and_records_one_pool_row():
    """契约表第 1 行:兜底话术 + 知识类落池(带召回片段快照)。

    用的是**违规流**(`useful=false` 却带非空 answer)—— 只有这样
    「一个 answer_delta 都没出去」才是可判别的:合法的空串流上,
    「不发」与「没东西可发」长得一模一样。
    """
    session = _Session()
    settings = _settings()
    evidence = [
        {"chunk_id": 10 + i, "section_path": f"节 {i}", "score": round(0.9 - i / 10, 2),
         "answer": "七天无理由退货。" * 60,   # 480 字 > snapshot_answer_chars(400)
         "question": f"问 {i}", "category": "退换货"}
        for i in range(6)                      # 6 条 > snapshot_top_n(5)
    ]
    frames = []
    model = _Model([INSUFFICIENT_WITH_ANSWER])
    out = await _make(model, settings=settings, frames=frames,
                      session=session).__call__(
        _state(intent="商品咨询", user_input="退货政策是什么", evidence=evidence)
    )

    # 用户只看到兜底话术:**一个 answer 帧都没有**,而且**只有这一帧**
    assert frames == [{"frame": "token", "text": FALLBACK_REPLY}]
    assert out["reply"] == FALLBACK_REPLY
    assert "这段文本一个字都不该出现在用户面前" not in "".join(_tokens(frames))

    assert len(session.added) == 1
    row = session.added[0]
    assert row.entry_point == "生成自评"
    assert row.question == "退货政策是什么"
    assert row.source_conversation_id == "c1"
    assert "证据不足" in row.reject_reason
    # 快照:两条上限(settings 的默认值)与四个键都由这里钉住 —— 少一个键、
    # 或者把截断写成「不截」,审核页拿到的东西就不一样了。
    snap = row.evidence_snapshot
    assert len(snap) == settings.snapshot_top_n == 5
    assert set(snap[0]) == {"chunk_id", "score", "section_path", "answer"}
    assert snap[0]["chunk_id"] == 10 and snap[0]["score"] == 0.9
    assert snap[0]["section_path"] == "节 0"
    assert len(snap[0]["answer"]) == settings.snapshot_answer_chars == 400


@pytest.mark.anyio
async def test_useful_false_on_a_business_turn_records_nothing():
    """契约表第 1 行的**范围**:落池只发生在知识类(§5.5)。

    业务轮的模型**照今天的规矩处理** —— 它没挂协议消息,那条 JSON 对用户
    来说就是一句普通文本,逐片原样透出去(与改动前逐字节相同),**不落池**。
    """
    session = _Session()
    frames = []
    model = _Model([[INSUFFICIENT_WITH_ANSWER, "还有什么可以帮您?"]])
    out = await _make(model, frames=frames, session=session).__call__(
        _state(intent="物流", evidence=[
            {"chunk_id": 1, "section_path": "s", "score": 0.9, "answer": "a"}])
    )

    assert session.added == []
    assert out["reply"] == INSUFFICIENT_WITH_ANSWER + "还有什么可以帮您?"
    # 逐片原样 —— 业务轮连「像协议」的输入都不过解码器
    assert _tokens(frames) == [INSUFFICIENT_WITH_ANSWER, "还有什么可以帮您?"]


@pytest.mark.anyio
async def test_protocol_violation_degrades_to_the_raw_text_flushed_once():
    """契约表第 2 行:`raw` **整段发一次** + `trace` 记违约 + **不落池**。

    违约形态取「首键不是 `useful`」—— 它判在键名的收尾引号上,结构上不可能
    先吐过东西(§5.6 里唯一「零 emit」的那种违约)。**分两片喂**:断言里
    「一次」才有内容 —— 逐片重发也是一种「发过了」。
    """
    script = ['{"answer": ', '"您好", "useful": true}']
    raw = "".join(script)
    session = _Session()
    frames = []
    model = _Model([script])
    out = await _make(model, frames=frames, session=session).__call__(
        _state(intent="商品咨询", evidence=[
            {"chunk_id": 1, "section_path": "s", "score": 0.9, "answer": "a"}])
    )

    assert frames == [{"frame": "token", "text": raw}]
    assert out["reply"] == raw
    assert "agent:protocol_violation" in out["trace"]
    assert session.added == []          # 协议不合**不是**证据不足:不许造假池记录


@pytest.mark.anyio
async def test_truncated_protocol_degrades_to_the_raw_text():
    """契约表第 2 行的第三个入口:流已结束、`done is False`、`answer` 为空。

    截断(`}` 从没来过)与「对象闭合」是两回事 —— 收尾判据是 `done`。
    顺手把「连 `useful` 都没解出来」也压在同一条上:两者动作相同(降级)。
    """
    script = ['{"useful": tru', 'e, "confidence": 0.9']
    raw = "".join(script)
    frames = []
    model = _Model([script])
    out = await _make(model, frames=frames).__call__(_state(intent="商品咨询"))

    assert frames == [{"frame": "token", "text": raw}]
    assert out["reply"] == raw
    assert "agent:protocol_violation" in out["trace"]


@pytest.mark.anyio
async def test_plain_mode_neither_reflushes_nor_records_a_violation():
    """⚠️ **契约表第 2 行的例外**(§11-⑪-D):`mode == PLAIN` **两个动作都不做**。

    `plain` 的终态**正好符合**第 2 行的字面描述(`useful is None`、`done is False`、
    `answer == ""`),但它是**今天的行为**、是**被认可的正常路径**
    ——「模型先说一句『让我查一下』再调工具」正是最常见的那一轮。照字面实现会在
    **每一个非协议轮**上 (i) 把已经实时透出的整段回复用 `raw` 重发一遍、
    (ii) 把这条正常路径记成违约。

    ⇒ 帧序列必须是**逐片一次**(第二片若被当成 `raw` 补发,这里会多出第三帧)。
    """
    frames = []
    model = _Model([["好的,我来查一下", "退货政策是 7 天"]])
    out = await _make(model, frames=frames).__call__(_state(intent="商品咨询"))

    assert frames == [
        {"frame": "token", "text": "好的,我来查一下"},
        {"frame": "token", "text": "退货政策是 7 天"},
    ]
    assert out["reply"] == "好的,我来查一下退货政策是 7 天"
    assert "agent:protocol_violation" not in out["trace"]


@pytest.mark.anyio
async def test_resumed_write_round_carries_the_protocol_too():
    """续跑路径也带协议 —— 它是**同一轮**的最终作答(§Step 4 点名的那个调用点)。

    少带这一处:`decode=False` 会让模型吐出的协议 JSON **原样给用户看**,
    而其余断言(工具往返、`turn_messages` 追加、`agent_steps` 不归零)全部照绿。
    """
    prior = AIMessage(content="", tool_calls=[
        {"name": "create_ticket", "args": {"description": "x"}, "id": "call_1",
         "type": "tool_call"},
    ])
    result = ToolMessage(content='{"ticket_no": "T-1"}', tool_call_id="call_1")
    frames = []
    model = _Model([['{"useful": true, "confidence": 0.8, "answer": "已为您建单 T-1。"}']])
    out = await _make(model, frames=frames).__call__(
        _state(intent="商品咨询", turn_messages=[prior, result],
               pending_write={}, write_decision="approved")
    )

    assert out["reply"] == "已为您建单 T-1。"
    assert _tokens(frames) == [*"已为您建单 T-1。"]     # 逐字流出,没有 JSON 原文
    assert "agent:write_resumed" in out["trace"]
    assert "agent:protocol_violation" not in out["trace"]
    # 只问一次模型:续跑续的是同一轮,不该重跑 ReAct 循环
    assert model.calls == 1
