"""意图识别节点:解析失败/越界必须降级为「其他」,**不许抛异常**。"""

import re

import pytest
from langchain_core.exceptions import OutputParserException

from app.agent.nodes import make_classify_intent_node
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
    def __init__(self, intent):
        self.intent = intent


@pytest.mark.anyio
@pytest.mark.parametrize("intent", ["物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊"])
async def test_seven_labels_pass_through(intent):
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


def test_every_label_in_the_prompt_matches_the_routing_table():
    """标签名在两处各写一份,必须自动对齐。

    `INTENT_SYSTEM_PROMPT` 里的七个标签是**手写**的;`INTENT_TO_ROUTE` 是代码里的
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

    # Prompt 用 `- 标签:说明` 的形式枚举七类,取冒号前的标签名。
    enumerated = set(re.findall(r"^\s*-\s*(.+?):", INTENT_SYSTEM_PROMPT, re.MULTILINE))

    missing = [label for label in INTENT_LABELS if label not in enumerated]
    assert not missing, (
        f"Prompt 的枚举行里缺这些标签:{missing};实际枚举了 {sorted(enumerated)} —— "
        f"缺的那个桶会静默降级成「其他」"
    )

    stale = sorted(enumerated - set(INTENT_LABELS))
    assert not stale, f"Prompt 里枚举了不在 INTENT_TO_ROUTE 里的标签:{stale}"
