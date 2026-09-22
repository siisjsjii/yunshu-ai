"""ch07 T8:摘要 —— prompt + 触发判定 + 原子落库。**不联网、不碰 db、不碰 LangChain。**

本文件验三样,每一样都对应一个「错了也不报错」的失效:

1. **触发**(`should_summarize`)看的是**层 2 截短后**的用量,且是严格的 `>`
   —— 数条数看不见工具结果涨(本章最大的一块),`>=` 则等于「每轮都压一次」,
   而摘要**不可逆**;
2. **渲染**(`render_turns`)给摘要模型的是**原文**,含 `role="tool"` 那几行 ——
   拿截短版去提炼等于把截断损失焊进**永久背景**;
3. **落库**(`summarize_range`)只走一次 `append_summary_and_advance`,且**只在
   模型成功之后**。反过来(先推边界再落库)会让那段历史**永久消失**。

`SUMMARY_SYSTEM_PROMPT` 的**措辞**不在这里断言「输出对不对」(真实模型输出非确定,
按工作要求 1 走 `evals/summary_cases.jsonl` 标注样例);这里只钉住它**必须写出
那几条硬约束** —— 这是 ch03「挖知识 prompt 首版没写什么不算知识」留下的教训。

`asyncio_run` 是本仓既有写法(`tests/test_history.py` 同款),不引 pytest-asyncio。
"""

import asyncio
import json
import logging

import pytest
from langchain.messages import AIMessage
from langchain_core.messages import convert_to_messages

from app.config import Settings
from app.memory import journal, layers, summarize, trim
from app.memory.summarize import (
    SUMMARY_SYSTEM_PROMPT,
    build_summary_messages,
    join_summaries,
    render_turns,
    should_summarize,
    summarize_range,
)
from app.schemas import Message


#: `_env_file=None` 是本仓硬规矩:仓库根有真实 `.env`,不传的话
#: 「缺字段应报错」这类用例会因为 `.env` 把值补上而静默通过。
REQUIRED = dict(
    _env_file=None,
    openai_base_url="https://example.invalid/v1",
    openai_api_key="sk-test",
    openai_model="m",
    database_url="mysql+asyncmy://u:p@127.0.0.1:3306/x",
)


def _settings(**over):
    return Settings(**{**REQUIRED, **over})


def asyncio_run(coro):
    return asyncio.run(coro)


def _layers_with(layer2_tokens: int) -> layers.Layers:
    """只关心 layer2_tokens 这一个字段时的最小构造。"""
    return layers.Layers(layer2=[], layer1=[], layer2_tokens=layer2_tokens,
                         layer1_tokens=0)


# ------------------------------------------------------------------ 触发判定


def test_should_summarize_compares_against_the_layer2_budget():
    """触发看**层 2 截短后**的用量,不数条数。

    数条数看不见工具结果涨 —— 而工具结果是本章最大的一块。
    """
    assert should_summarize(_layers_with(101), layer2_budget=100) is True
    assert should_summarize(_layers_with(99), layer2_budget=100) is False


def test_should_summarize_is_false_when_exactly_at_budget():
    """`>` 不是 `>=` —— 「装得下就不压」(验收 3 的反向断言)。

    边界上差一个等号,就是「每轮都压一次」与「一次都不压」的区别。
    """
    assert should_summarize(_layers_with(100), layer2_budget=100) is False


def test_should_summarize_reads_layer2_not_layer1():
    """**取错版本**是这一处最可能的错法(两个字段就差一个词)。

    `_layers_with` 只给 layer1_tokens=0,所以「读层 1」的实现会在
    `layer2_tokens=101` 时报 False —— 上一条用例的 `is True` 已经把它判死,
    这条用**两层都超**的形态再钉一次:值必须来自 layer2。
    """
    both_over = layers.Layers(layer2=[], layer1=[], layer2_tokens=101,
                              layer1_tokens=999)
    assert should_summarize(both_over, layer2_budget=100) is True
    lt_only = layers.Layers(layer2=[], layer1=[], layer2_tokens=0,
                            layer1_tokens=999)
    assert should_summarize(lt_only, layer2_budget=100) is False


def test_should_summarize_takes_the_budget_from_its_argument_not_from_settings():
    """**同一份 `Layers`,只换预算参数,答案必须翻面。**

    这条才是「它不读 settings」的判据(「同一入参调两次结果相同」那种写法
    什么也没证明:读 settings 的实现两次答案也一样)。读 settings 的话,
    两次调用的比较对象是**同一个固定的数**,于是下面这两条一真一假不可能同时成立
    —— 这里 101 对 100 超、对 200 不超,只有「预算来自参数」才做得到。
    """
    got = _layers_with(101)
    assert should_summarize(got, layer2_budget=100) is True
    assert should_summarize(got, layer2_budget=200) is False


def test_should_summarize_is_false_for_an_empty_layer2():
    """层 2 空(新会话 / 刚压完)时**不可能触发** —— 压不出东西来。"""
    assert should_summarize(_layers_with(0), layer2_budget=100) is False


# ------------------------------------------------------------------ 渲染原文


def test_render_turns_includes_tool_rows():
    """渲染给摘要模型看的**原文**必须含工具结果。

    否则梗概里存不下「用户报过的订单号」—— 而验收 2 的落点正是它
    (「最开始那个订单后来怎么说」要能靠梗概答对)。

    注意这里喂的是**原文**(`role="tool"` 那条的完整 JSON),不是截短版:
    拿截短文本去提炼,等于把截断损失焊进梗概,而梗概是**永久背景**。
    """
    text = render_turns([
        Message(id=1, role="user", content="订单 1002 能退吗"),
        Message(id=5, role="tool", content='{"order_no":"1002","status":"已取消"}',
                tool_call_id="call_1"),
    ])
    assert "1002" in text
    assert "已取消" in text          # 截短版只留 60 字也会含它,但完整 JSON 也含
    assert "{" in text               # ← 这条才把「原文 vs 截短版」区分开


def test_render_turns_renders_the_raw_text_not_a_truncated_one():
    """**上一条用例其实区分不了「原文」与「截短版」**,这条才是那件事的判据。

    上一条喂的工具结果只有 29 字符,比 `layer2_tool_chars`(默认 60)短 ——
    截短根本不会发生,于是 `{` 与 `已取消` 在**两种实现下都在**,三条断言全绿。
    真正的判据是「**超出截短上限的那一段**还在不在」:这里把标记放在
    正文**末尾**,任何截短实现都会把它连同 `…` 一起截掉。
    """
    tail = "尾部标记-退款原路返回"
    raw = '{"order_no":"1002","items":[' + "填充字段," * 40 + tail + "]}"
    assert len(raw) > 60              # 比默认上限长,否则这条用例又是恒真

    text = render_turns([
        Message(id=5, role="tool", content=raw, tool_call_id="call_1"),
    ])
    assert tail in text
    assert layers.ELLIPSIS not in text     # 截短版会带这个记号


def test_render_turns_keeps_every_message_in_order_with_its_role():
    """每条一行、按原序、带角色 —— 摘要模型要知道哪句是谁说的。"""
    text = render_turns([
        Message(id=1, role="user", content="第一句"),
        Message(id=2, role="assistant", content="第二句"),
    ])
    assert text.index("第一句") < text.index("第二句")
    assert "user" in text and "assistant" in text


# ------------------------------------------------------------------ 摘要 prompt


def test_summary_prompt_says_what_counts_as_fact():
    """四样提炼物必须写进 prompt(少一样就丢上下文)。"""
    assert "订单号" in SUMMARY_SYSTEM_PROMPT
    assert "诉求" in SUMMARY_SYSTEM_PROMPT
    assert "还没解决" in SUMMARY_SYSTEM_PROMPT


def test_summary_prompt_says_what_does_not_count():
    """**「什么不算」必须写出来** —— ch03 的挖知识 prompt 首版漏了这条。

    漏掉的后果不是「少提炼一点」:客服的「抱歉查不到运费」这种**非答案**被压进
    梗概,然后**每轮都注入一遍**,等于教模型继续失败。同款失效在梗概上
    只会更糟 —— 原文已经不喂了,编出来的事实**无法追溯**。
    """
    assert "编" in SUMMARY_SYSTEM_PROMPT          # 不许编
    assert "寒暄" in SUMMARY_SYSTEM_PROMPT or "客套" in SUMMARY_SYSTEM_PROMPT


def test_build_summary_messages_is_system_plus_the_raw_turns():
    """system 是那段 prompt,user 是渲染出来的原文;两条,顺序固定。"""
    msgs = build_summary_messages(turns=[
        Message(id=1, role="user", content="订单 1002 能退吗"),
    ])
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == SUMMARY_SYSTEM_PROMPT
    assert msgs[1]["role"] == "user"
    assert "订单 1002 能退吗" in msgs[1]["content"]


def test_summary_messages_are_plain_dicts_the_model_layer_accepts():
    """`app/memory/` 不依赖 LangChain,所以这里**只能**给纯 dict。

    给 LangChain 消息对象 = 把 LangChain 拖进 `memory/`;而纯 dict 能不能被
    真实模型层吃下,是本条要回答的问题:过一遍 `convert_to_messages`(真机
    路径上那一步),拿不下来的形状在这里就红,而不是等生产里炸。
    """
    msgs = build_summary_messages(turns=[Message(id=1, role="user", content="你好")])
    assert all(isinstance(m, dict) for m in msgs)
    converted = convert_to_messages(msgs)
    assert [m.type for m in converted] == ["system", "human"]


# ------------------------------------------------------------------ 多段拼接


def test_join_summaries_of_nothing_is_an_empty_string():
    """没有梗概 ⇒ 空串(不是 `None`)—— `summary_text` 通道与 `build_context_messages`
    都按空串判「没有」。"""
    assert join_summaries([]) == ""


def test_join_summaries_single_row_is_itself():
    """只有一段时**不加任何前后缀** —— 加个「第 1 段:」之类的包装会让
    单段会话的注入文本与多段时长得不一样,而两者都是正常形态。"""
    assert join_summaries([(1, "只此一段")]) == "只此一段"


def test_join_summaries_joins_with_a_blank_line(caplog):
    """段间用 `"\\n\\n"` 拼 —— **与 `journal.history_ctx` 的计数口径同源**。

    这条不是「拼得好看」:`history_ctx` 里 `tokens.summary` 是按
    `"\\n\\n".join(...)` 算的,而模型实际收到的是 `join_summaries` 的产物。
    两把尺子的话,日志报的数**不是**模型收到的那段文本的 token 数 ——
    一个看起来正常、其实什么也没说的观测面(本模块最该避免的那种)。
    """
    rows = [(1, "第一段"), (2, "第二段")]
    assert join_summaries(rows) == "第一段\n\n第二段"

    # 与 journal 的口径对一次:日志里的 tokens.summary 必须等于
    # 「模型真正收到的那段文本」的 token 数。
    from app.memory import budget as budget_module

    s = _settings()
    b = budget_module.derive(settings=s, system_prompt="你是客服。")
    with caplog.at_level(logging.INFO):
        journal.history_ctx(
            conversation_id="c1", summaries=rows,
            history=[Message(id=9, role="user", content="最近一句")], budget=b,
        )
    payload = None
    for record in reversed(caplog.records):
        if record.message.startswith("history_ctx "):
            payload = json.loads(record.message.split(" ", 1)[1])
            break
    assert payload is not None, "日志里没有 history_ctx 行"
    assert payload["tokens"]["summary"] == trim.count_tokens(join_summaries(rows))


# ------------------------------------------------------------------ 原子落库


class _FakeModel:
    """最小的模型替身:`ainvoke` 返回一条 `AIMessage`。

    用真的 `AIMessage` 而不是 `SimpleNamespace(text=...)` —— `.text` 是
    1.x 的属性(nodes.py 同款用法),替身自己造一个同名属性就把
    「形状对不对」这件事替成了「我以为的形状对不对」。
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=self.text)


class _AppendSpy:
    """记下 `append_summary_and_advance` 的每一次调用,并**返回一个非 1 的段号**。

    返回 `7` 而不是 `1` 是**故意的**:`summarize_range` 要把这个值原样带出去,
    而「写死 1」与「根本没带出来」在段号为 1 时输出完全一样 —— 本仓第 (a) 种
    假绿形态(期望值等于缺省/兜底产生的值)。
    """

    def __init__(self, seq: int = 7) -> None:
        self.calls: list[dict] = []
        self.seq = seq

    async def __call__(self, **kwargs) -> int:
        self.calls.append(kwargs)
        return self.seq


def test_summarize_range_returns_none_for_an_empty_range():
    """空区间不开任务、不落空梗概。"""
    assert asyncio_run(summarize_range(
        model=None, session=None, conversation_id="c1", turns=[], upto_msg_id=0,
    )) is None


def test_empty_range_calls_neither_the_model_nor_the_anchor(monkeypatch):
    """上一条的 `model=None` 只在「真去调模型」时才炸,字面上看不出
    「有没有碰边界」。这条把两件事都记下来 —— 空区间**两样都不做**。"""
    spy = _AppendSpy()
    model = _FakeModel("不该被调用")
    monkeypatch.setattr(summarize, "append_summary_and_advance", spy)

    assert asyncio_run(summarize_range(
        model=model, session=object(), conversation_id="c1", turns=[],
        upto_msg_id=7,
    )) is None
    assert model.calls == []
    assert spy.calls == []


def test_summarize_range_does_not_advance_the_anchor_on_failure():
    """**失败等于什么都没发生。**

    边界只在**成功后**推进 ⇒ 失败时层 2 原封不动、下次再触发即可,所以不重试。
    反过来(先推边界再落库)会让那段历史**永久消失**。

    注入的模型**真的抛**(不是「返回空串」那种)—— 返回空串是「在处理之后注入」
    的变体:它跳过了「模型调用失败」那一步,而那一步正是本用例要验的。

    `session=None` 也是**故意的**:断言的是「抛在落库**之前**」。真去落库会撞
    `AttributeError`,而那个错会盖掉 `RuntimeError` —— 于是这条用例就红在
    一个指向脚手架的地方,盖住了真问题。
    """
    class _Boom:
        async def ainvoke(self, messages):
            raise RuntimeError("上游炸了")

    with pytest.raises(RuntimeError):
        asyncio_run(summarize_range(
            model=_Boom(), session=None, conversation_id="c1",
            turns=[Message(id=1, role="user", content="你好")], upto_msg_id=1,
        ))


def test_failure_writes_nothing_at_all(monkeypatch):
    """上一条证明「抛出来了」,这条证明的是**边界那一下没发生**。

    把落库函数换成替身:它记下每一次调用。模型炸了之后,替身必须**一次都没被调**
    —— `summarize_range` 里推进边界的唯一途径就是它(T3 里两步是原子的),
    所以「没调它」= 「边界没动」。
    """
    spy = _AppendSpy()
    monkeypatch.setattr(summarize, "append_summary_and_advance", spy)

    class _Boom:
        async def ainvoke(self, messages):
            raise RuntimeError("上游炸了")

    with pytest.raises(RuntimeError):
        asyncio_run(summarize_range(
            model=_Boom(), session=object(), conversation_id="c1",
            turns=[Message(id=1, role="user", content="你好")], upto_msg_id=1,
        ))
    assert spy.calls == []


def test_summarize_range_writes_once_and_advances_to_the_given_id(monkeypatch):
    """成功路径:模型 → **一次**原子落库 → 返回 `(段号, 梗概文本)`。

    `turns` 的边界由调用方给(`upto_msg_id`),这里不自己算 —— 算的话就与
    重读锚点的那一步(T9)有两处答案。
    """
    spy = _AppendSpy()
    monkeypatch.setattr(summarize, "append_summary_and_advance", spy)
    model = _FakeModel("  用户问过订单 1002 能不能退。  ")
    session = object()

    seq, text = asyncio_run(summarize_range(
        model=model, session=session, conversation_id="c1",
        turns=[
            Message(id=3, role="user", content="订单 1002 能退吗"),
            Message(id=5, role="tool", content='{"order_no":"1002"}',
                    tool_call_id="call_1"),
        ],
        upto_msg_id=6,
    ))

    # 段号**原样**来自落库那一步(替身给的是 7,不是 1 —— 写死的实现在这里红)
    assert seq == spy.seq
    assert text == "用户问过订单 1002 能不能退。"   # 去掉了首尾空白
    assert len(spy.calls) == 1                      # **只落一次**
    assert spy.calls[0]["session"] is session
    assert spy.calls[0]["conversation_id"] == "c1"
    assert spy.calls[0]["upto_msg_id"] == 6
    assert spy.calls[0]["content"] == text          # 落进去的就是返回的那段
    # 模型看到的是**原文**(含工具行),不是截短版
    assert len(model.calls) == 1
    assert "1002" in model.calls[0][-1]["content"]


def test_summarize_range_writes_nothing_when_the_model_returns_nothing(monkeypatch):
    """模型吐空串/空白 ⇒ **不落、不推边界**,返回 `None`。

    这条不是形式主义:空梗概落进去 + 边界推过去 = 那段历史**永久消失**,
    而库里留下的替换物是一段空文本 —— 验收 2(靠梗概答出订单号)会静默失败,
    没有任何东西报错。
    """
    spy = _AppendSpy()
    monkeypatch.setattr(summarize, "append_summary_and_advance", spy)

    for blank in ("", "   \n  "):
        assert asyncio_run(summarize_range(
            model=_FakeModel(blank), session=object(), conversation_id="c1",
            turns=[Message(id=1, role="user", content="你好")], upto_msg_id=1,
        )) is None
    assert spy.calls == []
