"""ch07 分层与截短。纯函数 —— 不联网、不碰 db、不碰 LangChain。"""

import pytest

from app.config import Settings
from app.memory import layers
from app.schemas import Message

REQUIRED = dict(
    _env_file=None,
    openai_base_url="https://example.invalid/v1",
    openai_api_key="sk-test",
    openai_model="m",
    database_url="mysql+asyncmy://u:p@127.0.0.1:3306/x",
)


def _settings(**over):
    return Settings(**{**REQUIRED, **over})


def _history():
    """三轮,id 1..9。第 2 轮带一次工具调用(assistant + tool 两条)。"""
    return [
        Message(id=1, role="user", content="你好"),
        Message(id=2, role="assistant", content="你好呀,有什么可以帮您的吗?"),
        Message(id=3, role="user", content="订单 1002 能退吗"),
        Message(id=4, role="assistant", content="", tool_calls=[
            {"name": "query_order", "args": {"order_id": "1002"},
             "id": "call_1", "type": "tool_call"}
        ]),
        Message(id=5, role="tool", content='{"order_no":"1002","status":"已取消"}',
                tool_call_id="call_1"),
        Message(id=6, role="assistant", content="您的订单 1002 当前状态为已取消,可以申请退款。"),
        Message(id=7, role="user", content="那就退吧"),
        Message(id=8, role="assistant", content="好的,已经为您登记。"),
        Message(id=9, role="user", content="多久到账"),
    ]


def test_split_puts_three_segments_end_to_end_without_gap_or_overlap():
    """三段不重不漏 —— off-by-one 会**静默丢消息**或**重复注入**。"""
    s = _settings()
    got = layers.split(
        _history(), summary_upto_msg_id=2, layer1_from_msg_id=7, settings=s
    )
    assert [m.id for m in got.layer2] == [3, 4, 5, 6]
    assert [m.id for m in got.layer1] == [7, 8, 9]
    # 1、2 已被梗概覆盖,两层都不含它们
    assert all(m.id not in (1, 2) for m in got.layer2 + got.layer1)


def test_split_with_zero_anchors_puts_everything_in_layer1():
    s = _settings()
    got = layers.split(
        _history(), summary_upto_msg_id=0, layer1_from_msg_id=0, settings=s
    )
    assert [m.id for m in got.layer1] == list(range(1, 10))
    assert got.layer2 == []


def test_messages_without_ids_land_in_layer1_only():
    """没有 MySQL id 的消息(手工构造 / 尚未落库)**只进层 1**,绝不两层都占。

    它的失效形态**正是** spec §12.1① 修掉的那个重复注入:组装是
    `layer2 + layer1`,同一条落进两层就是**注入两遍** —— 不报错、不丢消息,
    只是上下文凭空翻倍,而每一轮的回复看起来都正常(终审 Minor 11:这条规则
    此前**没有任何用例**)。

    三个锚点组合都覆盖:`0/0`(新会话,层 2 为空)、`0/7`(层 2 非空)、
    `2/7`(两个锚点都非零)。
    """
    s = _settings()
    pending = Message(role="user", content="刚发出去、还没落库的一句原话")
    for upto, from_ in ((0, 0), (0, 7), (2, 7)):
        got = layers.split(
            [*_history(), pending],
            summary_upto_msg_id=upto, layer1_from_msg_id=from_, settings=s,
        )
        # 「恰好出现一次」按 id 是否为 None 数 —— 比 `list.count(msg)` 稳:
        # 后者依赖 pydantic 的等值语义,而这里要问的就是**这一条**的去向。
        assert sum(m.id is None for m in got.layer2 + got.layer1) == 1, (upto, from_)
        assert all(m.id is not None for m in got.layer2)
        assert [m.content for m in got.layer1 if m.id is None] == [pending.content]


def test_layer2_tokens_are_counted_on_the_truncated_form():
    """**本任务的核心断言**:层 2 的计数必须按截短后。

    按原文数的话,截短就退化成纯渲染装饰 —— 层 2 该何时触发摘要还是何时触发,
    而**所有输出看起来都正常**。这条用「截短前后计数必须不同」把它区分开。
    """
    s = _settings(layer2_assistant_chars=5, layer2_tool_chars=5)
    source = _history()
    raw = layers.split(source, summary_upto_msg_id=0, layer1_from_msg_id=7, settings=s)
    assert [m.id for m in raw.layer2] == [1, 2, 3, 4, 5, 6]
    assert raw.layer2_tokens < sum(
        len(m.content) for m in raw.layer2
    ) + 1  # 便宜的存在性护栏,真正的判别在下一句
    # 截短确实发生了:助手回复被截到 5 字
    assert any(m.content.endswith("…") for m in raw.layer2 if m.role == "assistant")
    # 而同样一段历史,若按**原文**数会显著更大。
    # 拿 `source`(**未经截短**的那一份)来数才构成判别:把 `split` 的计数换成
    # 未截短的 `layer2_raw`,下面两个数会**相等**,这一句立刻红。
    from app.memory import trim
    layer2_ids = {m.id for m in raw.layer2}
    raw_tokens = sum(trim.count_tokens(m.content) for m in source if m.id in layer2_ids)
    assert raw.layer2_tokens < raw_tokens


def test_truncate_leaves_user_content_untouched():
    s = _settings(layer2_assistant_chars=3)
    msg = Message(id=1, role="user", content="这是一句很长的用户原话,一个字都不该动")
    assert layers.truncate(msg, settings=s).content == msg.content


def test_truncate_keeps_tool_calls_intact_for_pairing():
    """只截 `tool` 的 content,**不动 `tool_calls`**。

    截断 `tool_calls` 就是把 assistant 与它的 tool 消息拆开 ⇒ 上游 400,
    而且**只在历史长到触发分层时才复现**。
    """
    s = _settings(layer2_assistant_chars=3, layer2_tool_chars=3)
    a = Message(id=4, role="assistant", content="很长的一句解释" * 10, tool_calls=[
        {"name": "query_order", "args": {"order_id": "1002"}, "id": "call_1",
         "type": "tool_call"}
    ])
    out = layers.truncate(a, settings=s)
    assert out.tool_calls == a.tool_calls
    t = Message(id=5, role="tool", content="x" * 500, tool_call_id="call_1")
    assert layers.truncate(t, settings=s).tool_call_id == "call_1"
    assert len(layers.truncate(t, settings=s).content) < 500


def test_degrade_moves_the_boundary_forward_only_and_lands_on_a_round_start():
    """降级只把 layer1_from 往后挪,且必须落在**轮的起点**(user 消息)。

    不落在 user 上就会把一轮切开,连带把 tool 与它的 assistant 拆到两层里。
    """
    h = _history()
    s = _settings()
    new_from = layers.degrade(
        h, summary_upto_msg_id=0, layer1_from_msg_id=0, layer1_budget=1, settings=s
    )
    # `> 0` 而不是 `>= 0`:后者在「边界压根没挪」(返回入参 0 或任何负数)时
    # 同样成立 —— 它是一条**恒真**断言,还顺手让下面那句查不到消息(终审 Minor 10)。
    assert new_from > 0
    moved = next(m for m in h if m.id == new_from)
    assert moved.role == "user"


def test_degrade_returns_the_same_value_when_already_within_budget():
    """装得下就不动 —— 「压缩是成本不是美德」在纯函数层也要成立。"""
    h = _history()
    s = _settings()
    assert layers.degrade(
        h, summary_upto_msg_id=0, layer1_from_msg_id=0,
        layer1_budget=10 ** 6, settings=s,
    ) == 0


def test_degrade_never_crosses_below_summary_upto():
    """`layer1_from >= summary_upto` 是不变量;而**返回值永不小于入参**(单调)。

    ⚠️ 这条用例原先**驱动不到它命名的那条分支**:合法入参下
    `if nxt <= summary_upto_msg_id` 是**不可达**的(`summary_upto <= layer1_from
    <= cur`,而那里 `nxt > cur`,三者不可能同时成立),所以旧写法
    `return summary_upto_msg_id` 永远跑不到,断言 `>= 6` 也永远只走正常出口
    (终审 Minor 6)。要驱动它,必须喂一对**相对不变量不可能出现**的锚点
    (`summary_upto > layer1_from`)—— 那正是这个防御分支存在的理由:
    调用方送来坏值时的兜底。

    **为什么兜底值必须是 `cur`**:`return summary_upto_msg_id` 会返回一个
    **比入参更小**的值,而端点只在 `layer1_from != conv.layer1_from_msg_id`
    时才写库 ⇒ 那个更小的值会被写进去:层 2 当场清空、已进梗概的原文被拉回
    层 1 再注入一遍(spec §12.1① 的重复注入形状,**不报错**)。
    """
    h = _history()
    s = _settings()

    # ① 正常出口:合法入参下它只往前走,不越过 summary_upto。
    assert layers.degrade(
        h, summary_upto_msg_id=6, layer1_from_msg_id=7,
        layer1_budget=0, settings=s,
    ) >= 6

    # ② 防御出口:入参故意违反不变量(8 > 1)⇒ 边界**原样不动**,不是倒退。
    #    期望值取 1(= 入参)而不是 `>= 1` —— 后者对「返回 8」的旧实现同样成立。
    assert layers.degrade(
        h, summary_upto_msg_id=8, layer1_from_msg_id=1,
        layer1_budget=0, settings=s,
    ) == 1


def test_degrade_lets_the_last_round_overflow_rather_than_emptying_layer1():
    """连一层 1 都只剩**一轮**且它还超预算时,降级停在原地 —— 不许把这一轮也让出去。

    **这条分支在别的用例里走不到**:`_history()` 的最后一条是独立的 user 消息,
    `to_rounds` 把它单独算一轮,所以前面几条用例的层 1 永远是两轮以上。
    (实测:把那句 `len(rounds) <= 1` 的分支改成"把最后一条也挪出去",
     `pytest tests/test_memory_layers.py` 仍然是 9 passed —— 也就是说这条路径
     原本**没有任何断言**。)

    为什么要钉住它:再挪一次,层 1 就空了,而**没有任何东西报错** ——
    模型从此看不到最近说过什么。这是本仓的"静默丢消息"族。
    """
    h = [
        Message(id=1, role="user", content="很长的一句问话" * 100),
        Message(id=2, role="assistant", content="很长的一句回答" * 100),
    ]
    s = _settings()
    new_from = layers.degrade(
        h, summary_upto_msg_id=0, layer1_from_msg_id=0, layer1_budget=1, settings=s
    )
    got = layers.split(h, summary_upto_msg_id=0, layer1_from_msg_id=new_from, settings=s)
    assert new_from == 0
    assert [m.id for m in got.layer1] == [1, 2]


def test_degrade_loops_until_it_converges():
    """降级是**循环到收敛**,不是一次判断 —— 挪一次会同时改变两层的大小。"""
    h = _history()
    s = _settings()
    tight = _settings(layer2_assistant_chars=1000, layer2_tool_chars=1000)
    loose = layers.degrade(h, summary_upto_msg_id=0, layer1_from_msg_id=0,
                           layer1_budget=10 ** 6, settings=tight)
    tight_from = layers.degrade(h, summary_upto_msg_id=0, layer1_from_msg_id=0,
                                layer1_budget=1, settings=tight)
    # `loose == 0` 是前提(预算足够大 ⇒ 边界停在原地;`_history()` 的轮数又不止
    # 一轮,所以"装得下就返回入参"这条与"压根没挪"在这里由它区分)。
    # 而 `tight_from >= loose` 是 `9 >= 0` —— **恒真**,任何返回值都过得了(终审 Minor 10)。
    assert loose == 0
    assert tight_from > loose
    assert layers.split(h, summary_upto_msg_id=0, layer1_from_msg_id=tight_from,
                        settings=tight).layer1_tokens <= 1 or tight_from == 9


# ------------------------------------------------------------ resplit(T10)

#: 一条**长过** `layer2_tool_chars` 的工具结果。短的话截短不触发,
#: 「二次截短」与「不再截短」在那个输入上**完全一样**(本章第五种假绿:
#: 测试输入小到触发不了被测行为)。
_LONG_TOOL_RESULT = "订单 1002 已取消,用户可以申请退款。" * 20


def _long_tool_history():
    return [
        Message(id=1, role="user", content="订单 1002 能退吗"),
        Message(id=2, role="assistant", content="", tool_calls=[
            {"name": "query_order", "args": {"order_id": "1002"},
             "id": "call_1", "type": "tool_call"}
        ]),
        Message(id=3, role="tool", content=_LONG_TOOL_RESULT, tool_call_id="call_1"),
        Message(id=4, role="user", content="那运费退吗"),
    ]


def test_resplit_is_the_identity_on_an_already_truncated_window():
    """`resplit` 给**已经截短过**的精简版重新分类,**不再截短一次**。

    为什么不能用 `split` 代替 —— 这不是洁癖,是一个可以算出来的差:

    ```
    工具结果截完 = "[工具结果] " + content[:60] + "…" = 67 字 > 阈值 60
    ⇒ 再走一遍 `split` 会二次截短,并叠上第二个 "[工具结果] " 前缀
    ```

    日志里描述的因此**不是真正发出去的那批消息**,而它看起来完全正常
    (端点那行 `model_ctx` 的全部价值就是描述发出去的那批消息)。
    这条用例把两种行为**并排**钉住:先是 `split` 的二次截短(前提前半),
    再是 `resplit` 的恒等 —— 任何一边被改坏都会红。
    """
    s = _settings()
    anchors = dict(summary_upto_msg_id=0, layer1_from_msg_id=4)
    got = layers.split(_long_tool_history(), settings=s, **anchors)
    once = next(m for m in got.layer2 if m.role == "tool").content
    assert once.startswith("[工具结果] ") and once.endswith("…")
    assert len(once) > 60                      # ← 前提:截短后的形态仍然超阈值

    trimmed = got.layer2 + got.layer1
    # 前提前半:再走一遍 `split` 确实会二次截短(所以不能拿它当 resplit)。
    again = layers.split(trimmed, settings=s, **anchors)
    assert again.layer2[2].content.startswith("[工具结果] [工具结果] ")

    # `resplit` 对同一批消息、同一对锚点是**恒等**的:内容、token 数、锚点全等。
    assert layers.resplit(trimmed, **anchors) == got
