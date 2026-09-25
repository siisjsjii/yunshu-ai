"""意图识别节点:解析失败/越界必须降级为「其他」,**不许抛异常**。"""

import json
import re

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser

from app.agent.nodes import make_classify_intent_node, make_log_turn_node
from app.agent.routing import OTHER
from app.agent.state import IntentResult


class FakeStructuredModel:
    """替身:with_structured_output 返回一个可注入结果或异常的链。"""

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def with_structured_output(self, schema, method=None):
        assert method == "json_mode", "抽取类出参只能用 json_mode(本项目硬约束)"
        self.schema = schema
        return self

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        if self.error is not None:
            raise self.error
        return self.result


class _Intent:
    def __init__(self, intent, confidence=0.9):
        self.intent = intent
        # 节点改读 `.confidence` 之后,替身也必须带上它:不带的话 `result.confidence`
        # 抛 AttributeError —— 那是**替身不忠实**,不是产品缺陷,照抄旧形状的人会
        # 把它误读成回归。
        self.confidence = confidence


class _FakeIntentModel:
    """替身:「模型吐了什么 JSON」→ **真解析器** → `IntentResult`。

    刻意**不**用鸭子类型。真机路径是
    `model.with_structured_output(IntentResult, method="json_mode")`,而
    langchain-openai 1.4.0 在 `json_mode` 分支末尾挂的是
    `PydanticOutputParser(pydantic_object=schema)`(读源码 + 离线实测过)。这里用
    **同一个解析器**,于是两条支路的结论都来自产品代码的解析/校验/默认值,
    而不是替身自己编的:

    - `confidence=0.87`  → 模型输出 `{"intent": ..., "confidence": 0.87}`;
    - `confidence=None`  → 模型**压根没输出这个字段**,即真机上 `{"intent": "物流"}`
      那一份 JSON,靠 `IntentResult.confidence` 的 `default=0.0` 兜住
      (实测:`PydanticOutputParser.parse('{"intent": "物流"}')` 返回
      `IntentResult(intent='物流', confidence=0.0)`,**不报错**;把 schema 的
      `default` 拿掉同一句抛 `OutputParserException`)。
    """

    def __init__(self, intent, confidence=None):
        self.intent = intent
        self.confidence = confidence
        self.parser = None

    def with_structured_output(self, schema, method=None):
        assert method == "json_mode", "抽取类出参只能用 json_mode(本项目硬约束)"
        self.parser = PydanticOutputParser(pydantic_object=schema)
        return self

    async def ainvoke(self, messages):
        payload = {"intent": self.intent}
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        return self.parser.parse(json.dumps(payload, ensure_ascii=False))


@pytest.mark.anyio
@pytest.mark.parametrize(
    "intent", ["物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊", "转人工"]
)
async def test_every_declared_label_passes_through(intent):
    model = FakeStructuredModel(result=_Intent(intent))
    node = make_classify_intent_node(model=model)
    out = await node({"user_input": "随便问点什么"})
    assert out["intent"] == intent
    assert out["trace"] == [f"classify_intent:{intent}"]


@pytest.mark.anyio
async def test_out_of_vocabulary_intent_degrades_to_other():
    """模型吐了七类之外的标签 → 「其他」,由路由送兜底。"""
    model = FakeStructuredModel(result=_Intent("退款"))
    node = make_classify_intent_node(model=model)
    assert (await node({"user_input": "x"}))["intent"] == OTHER


@pytest.mark.anyio
async def test_parse_failure_degrades_to_other_instead_of_raising():
    """解析失败**不许**抛 —— 意图识别是骨架第一步,它的失败不该毁掉整轮对话。"""
    model = FakeStructuredModel(error=OutputParserException("模型输出不是 JSON"))
    node = make_classify_intent_node(model=model)
    out = await node({"user_input": "x"})
    assert out["intent"] == OTHER
    assert out["trace"] == [f"classify_intent:{OTHER}"]


@pytest.mark.anyio
async def test_structured_output_is_bound_to_the_intent_schema():
    """绑错 schema 时上面 10 条全绿(替身不看 schema,只看返回对象的 .intent)。

    出参 schema 是**下游契约**:换掉它,json_mode 的真机路径会直接失效,
    而所有替身用例照过 —— 所以这里必须钉住。
    """
    model = FakeStructuredModel(result=_Intent("闲聊"))
    make_classify_intent_node(model=model)
    assert model.schema is IntentResult


@pytest.mark.anyio
async def test_prompt_carries_the_user_utterance():
    model = FakeStructuredModel(result=_Intent("闲聊"))
    node = make_classify_intent_node(model=model)
    await node({"user_input": "你好呀"})
    assert "你好呀" in model.calls[0][-1].text


def _enumerated_labels(prompt: str) -> set[str]:
    """取 Prompt 里**九类枚举段**的标签(冒号前的名字)。

    **必须限定段落,不能整篇跑正则**:输出契约段(ch06 起)也是
    `- intent: ...` / `- confidence: ...` 的形状,整篇扫会把 `intent`/
    `confidence` 当成两个「标签」,于是 stale 断言在一份**完全正确**的
    prompt 上报红(实测:stale=['confidence','intent'])—— 假红会诱人
    去改 prompt,而真正的守卫(标签名两处对齐)反而被放过。

    两个边界 = `九类` → `**输出一个 JSON 对象`(契约段标题),都 `assert` 在。
    这两条 assert 只是**让失效时报得清楚**(改名/删段时给一句人话),
    **不是**为了防「空集恒真假绿」—— 解析成空集会被下面的 `missing` 断言挡下
    (`INTENT_LABELS` 非空)。这里真正的风险是**解析得太多**(把契约段当标签),
    也就是上面那段说的假红。

    另外,下边界 `partition("九类")` 命中的是**介绍句**里的「九类」
    (`判断用户这一句话属于下面九类中的哪一类…`),不是段头那一行。
    介绍句里没有 `- 标签:` 形状的行,所以结果不受影响 —— 但这是**偶然**,
    不是设计;真要换文案时留意这里。
    """
    _, sep, tail = prompt.partition("九类")
    assert sep, "Prompt 里找不到「九类」段标题,标签守卫已失效"
    body, contract, _ = tail.partition("**输出一个 JSON 对象")
    assert contract, "Prompt 里找不到输出契约段标题,标签守卫已失效"
    return set(re.findall(r"^\s*-\s*(.+?):", body, re.MULTILINE))


def test_every_label_in_the_prompt_matches_the_routing_table():
    """标签名在两处各写一份,必须自动对齐。

    `INTENT_SYSTEM_PROMPT` 里的八个标签是**手写**的;`INTENT_TO_ROUTE` 是代码里的
    权威表。谁改了 `app/agent/routing.py` 的键名,模型仍会吐**旧名**,
    `nodes.py` 的越界检查把**整个桶**静默降级成「其他」—— 全部进兜底,零报错。

    既有守卫 `tests/test_agent_routing.py:35`(`INTENT_LABELS == tuple(INTENT_TO_ROUTE)`)
    是恒真式(定义即如此),**正好抓不到这个**;这条才是真的守卫。

    **必须匹配「枚举行」而不是子串**:Prompt 正文里散落着这些词(如末段的
    「不属于寒暄」)。真机反例:把 `routing.py` 的「闲聊」改成「寒暄」,
    `"寒暄" in INTENT_SYSTEM_PROMPT` 照样为真 —— 而模型永远不会吐「寒暄」,
    闲聊桶静默降级。子串版在那次改动上是**假绿**。
    """
    from app.agent.routing import INTENT_LABELS
    from app.prompts import INTENT_SYSTEM_PROMPT

    enumerated = _enumerated_labels(INTENT_SYSTEM_PROMPT)

    missing = [label for label in INTENT_LABELS if label not in enumerated]
    assert not missing, (
        f"Prompt 的枚举行里缺这些标签:{missing};实际枚举了 {sorted(enumerated)} —— "
        f"缺的那个桶会静默降级成「其他」"
    )

    stale = sorted(enumerated - set(INTENT_LABELS))
    assert not stale, f"Prompt 里枚举了不在 INTENT_TO_ROUTE 里的标签:{stale}"


def test_chitchat_fewshot_keeps_the_smalltalk_example():
    """裁定项:闲聊的 few-shot 必须留着「今天天气不错」。

    ch06 重写 prompt 时把这条例子删了,而它不是新加的用例(从 ch05 就在评估集里)。
    实测(各 8 次采样):删掉后该句 1/8 落「其他」、**判对时 conf 只有 0.60–0.85**;
    补回后 8/8 且 0.95–0.97。评估集只能**概率性**发现这种回退(6 轮里 3 轮才红),
    所以这里要一条确定的守卫。
    """
    from app.prompts import INTENT_SYSTEM_PROMPT

    assert "今天天气不错" in INTENT_SYSTEM_PROMPT


# ---- ch06 T4:confidence ----


@pytest.mark.anyio
async def test_confidence_is_written_to_state():
    """confidence 必须进 state(本章只用于日志/后续降级路,不改变路由)。"""
    node = make_classify_intent_node(model=_FakeIntentModel("物流", 0.87))
    out = await node({"user_input": "包裹到哪了"})
    assert out["intent"] == "物流"
    assert out["confidence"] == pytest.approx(0.87)


@pytest.mark.anyio
async def test_confidence_defaults_when_model_omits_it():
    """模型没给 confidence 时不得炸 —— 落一个保守值。

    这条断的是**真机支路**:模型返回的 JSON 里没有 `confidence` 字段时,
    `default=0.0` 兜住(替身走的是产品同款解析器,见 `_FakeIntentModel`)。
    把 schema 里的 `default` 拿掉,本条**两个断言都会红**(解析抛异常 →
    降级成「其他」),不是恒真式。
    """
    node = make_classify_intent_node(model=_FakeIntentModel("物流", None))
    out = await node({"user_input": "包裹到哪了"})
    assert out["intent"] == "物流"
    assert out["confidence"] == 0.0


@pytest.mark.anyio
async def test_parse_failure_also_carries_zero_confidence():
    """解析失败的降级路也要带 confidence —— 少了它下游 `state["confidence"]` 直接 KeyError。"""
    model = FakeStructuredModel(error=OutputParserException("模型输出不是 JSON"))
    node = make_classify_intent_node(model=model)
    out = await node({"user_input": "x"})
    assert out["intent"] == OTHER
    assert out["confidence"] == 0.0


class _RecordingSession:
    """`append_turn` 只用到 `add` + `flush` + `commit`(ch07 起多了 `flush`)。"""

    def __init__(self):
        self.added = []

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        pass

    async def commit(self):
        pass


@pytest.mark.anyio
async def test_log_turn_trace_frame_carries_confidence():
    """spec §4.2:confidence 要进 done 帧(T8 透传);T4 这半是**进 `trace` 帧**。

    没有这条,把 `"confidence"` 从那个帧载荷里删掉**没有任何测试会红** ——
    state 里有、帧里没有,而 `classify_intent` 那几条断言照样绿。
    """
    frames = []
    node = make_log_turn_node(session=_RecordingSession(), emit=frames.append)
    await node(
        {
            "conversation_id": "conv-1",
            "user_input": "包裹到哪了",
            "reply": "已发货",
            "intent": "物流",
            "confidence": 0.87,
            "gate_passed": None,
            "agent_steps": 0,
            "trace": ["resolve_references", "classify_intent:物流"],
        }
    )
    payload = next(f for f in frames if f["frame"] == "trace")
    assert payload["confidence"] == pytest.approx(0.87)


def test_complaint_and_handoff_are_kept_apart():
    """「投诉」与「转人工」不能揉成一个 —— 揉了的后果是**投诉出口被绕过**。

    投诉出口除了安抚话术,还会发一个 `choices` 帧把「转人工 / 建工单」两个选项
    交给用户。如果模型把「我要投诉」直接判成转人工,用户就**再也拿不到那两个选项**,
    而 `complaint_reply` 那段固定话术永远不会出现 —— 没有任何东西会报错。

    这条只断「提示词里同时写着这两条边界样例」,不断模型行为(模型行为由
    `evals/intent_cases.jsonl` 的边界负例覆盖,见任务 3)。

    ⚠️ **「同时」两个字曾经是假的**(审查 M1):负例那一半在,而**正例**那一半
    (`- 「我要转人工」「找真人客服」→ 转人工`,教模型「这个桶存在」的那一行)
    当时**没有任何断言看着它** —— 删掉它,本文件全绿,而模型从此学不到「转人工」
    这个词(第九类于是只能靠分类器**碰巧**判对)。所以下面两条断言缺一不可:
    正例问「这个桶被教过没有」,负例问「它没被投诉抢走没有」。
    """
    from app.prompts import INTENT_SYSTEM_PROMPT

    # ---- 正例:把「转人工」这一桶教给模型的那一行 ----
    assert "「我要转人工」「找真人客服」→ 转人工" in INTENT_SYSTEM_PROMPT, (
        "提示词里缺少「转人工」的**正例**样例 —— 删掉它模型就学不到这一桶"
        "(分类仍可能碰巧判对,所以别的测试看不出来)"
    )
    # ---- 负例:把「投诉」与「转人工」分开的那一句 ----
    # `or` 那第二支**今天永远不会触发**(正文是 `(**不是**转人工`,粗体星号夹在
    # 中间)。**刻意留着**:它表达的是「这句话在场」这个语义,而重写文案时最容易
    # 掉的就是那对星号 —— 留着它,一次「去粗体」的重排不会得到一条假红。
    # 它不是恒真的那一支(第一支缺席时它仍**可能**不成立),所以留着不会把这条
    # 断言变成一条自己声明自己为真的空话。
    assert "不是**转人工" in INTENT_SYSTEM_PROMPT or "不是转人工" in INTENT_SYSTEM_PROMPT, (
        "提示词里缺少「投诉 ≠ 转人工」的边界样例"
    )
