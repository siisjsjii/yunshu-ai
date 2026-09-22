from app.memory.trim import (
    ContextOverflowError,
    count_tokens,
    to_rounds,
)
from app.schemas import Message


def test_count_tokens_is_positive_for_nonempty_text():
    assert count_tokens("你好") > 0
    assert count_tokens("") == 0


def test_count_tokens_grows_with_length():
    assert count_tokens("退货退款流程是什么" * 5) > count_tokens("退货退款流程是什么")


def test_budget_leaves_no_room_for_history_when_overhead_eats_the_window():
    """原来由 compute_available_tokens 直接覆盖的语义:预算可以是负的。

    删掉旧函数不等于删掉这条不变量 —— 它现在由 budget.derive 承担,
    而「负预算」正是端点返回 400 的判据,不能没有覆盖。

    判别力:把 `history_budget` **夹到 0**(例如 `max(0, …)`)或者只留
    `keep_rounds × per_round_steady` 那一支,这条都会变红 —— 而 400 那条
    路径正是在这两种改法下静默消失的(端点再也等不到负预算)。
    逐项算术的敏感性由 `tests/test_memory_budget.py` 单独钉住,这里只钉符号。
    """
    from app.config import Settings
    from app.memory import budget

    s = Settings(
        _env_file=None,
        openai_base_url="https://example.invalid/v1",
        openai_api_key="sk-test",
        openai_model="m",
        database_url="mysql+asyncmy://u:p@127.0.0.1:3306/x",
        model_context_window=1024,
        max_output_tokens=2000,
        tool_def_tokens=5000,
    )
    b = budget.derive(settings=s, system_prompt="你是客服。")
    assert b.history_budget < 0


# ---- ch07 T10:`select_history` 已删除,它守着的不变量搬到这里 ----
#
# 为什么删:上下文从「裁一条单层的」换成 `layers.split` 分三层(层 2 按**截短后**
# 计数),而 `select_history` 按**原文** token 整轮丢弃 —— 叠在分层前面会让层 2
# 少算 ⇒ 摘要永不触发,级联的第二环在日志里平静地缺席。它的**函数选择**
# (预算为 0 → 空、先丢最老的、按时间正序、放不下就停不回退)随函数一起退场;
# 其中「按预算取最近若干轮」那一族**换了家**:`prompts.select_layer1`
# (用例在 `tests/test_prompts.py`)。
#
# 留下来的是它当年真正守住的那个**不变量**:轮的边界必须落在 `user` 上,
# 否则 `tool` 消息会与它的 assistant 父亲被切开 ⇒ 上游 400(ch01–ch02 用真实
# 故障换来的)。今天依赖这条边界的是 `layers.degrade`(它只把边界挪到轮的起点),
# 所以这些用例对着 `to_rounds` 写 —— 那是它唯一的作用点。


def _pairing_intact(messages) -> bool:
    """每条 tool 消息前面都必须有带对应 tool_call_id 的 assistant 消息。"""
    pending: set[str] = set()
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            pending |= {tc["id"] for tc in m.tool_calls}
        elif m.role == "tool":
            if m.tool_call_id not in pending:
                return False
    return True


def test_rounds_never_start_with_a_tool_message():
    """**轮的边界落在 user 上** —— 这是 `to_rounds` 存在的全部理由。

    换成 ch01 的旧规则(遇 assistant 收轮)时,一轮工具往返会被切成
    `[user, assistant(tool_calls)]` 与 `[tool, assistant]` 两半:后半段**以 tool
    打头**,而它的 assistant 父亲在前半段里 —— 按轮裁剪/挪边界都会把这对切开,
    上游 OpenAI 兼容 API 直接 400,且只在历史长到触发分层时偶发。
    `_pairing_intact` 逐轮断言这件事,而不是只数轮数。
    """
    call = {"id": "call_1", "name": "query_order", "args": {"order_id": "1001"}}
    history = [
        Message(role="user", content="订单 1001 到哪了"),
        Message(role="assistant", content="", tool_calls=[call]),
        Message(role="tool", content="已揽件", tool_call_id="call_1"),
        Message(role="assistant", content="您的包裹已揽件。"),
        Message(role="user", content="那什么时候到"),
        Message(role="assistant", content="预计明天送达。"),
    ]

    rounds = to_rounds(history)

    assert len(rounds) == 2
    assert all(r[0].role == "user" for r in rounds)      # ← 边界只落在 user 上
    # 每一轮**内部**的配对都是完整的(逐轮查,而不是把整段拼回去查 ——
    # 拼回去会把「切开又被相邻两轮拼回来」这种假绿放过去)。
    assert all(_pairing_intact(r) for r in rounds)
    assert [m.role for m in rounds[0]] == ["user", "assistant", "tool", "assistant"]


def test_rounds_keep_every_message_in_order_without_gap_or_overlap():
    """切轮**不重不漏、保持原序** —— 它是分层与降级的共同前提。

    少一条 ⇒ 那条消息从此不在任何一层里(静默丢历史);多一条 ⇒ 同一句被注入
    两遍(上下文凭空翻倍)。两者都不报错,所以这里断的是**拼回去等于原序列**。
    """
    history = [
        Message(role="user", content="q1"),
        Message(role="assistant", content="", tool_calls=[
            {"id": "c1", "name": "t", "args": {}},
        ]),
        Message(role="tool", content="r1", tool_call_id="c1"),
        Message(role="assistant", content="a1"),
        Message(role="user", content="q2"),
        Message(role="assistant", content="a2"),
    ]
    rounds = to_rounds(history)

    assert [m for r in rounds for m in r] == history           # 不重不漏、原序
    assert all(r for r in rounds)                              # 没有空轮


def test_a_trailing_user_message_is_its_own_round():
    """末尾孤立的 user(本轮刚发、还没有回答)**单独成轮**,不并进上一轮。

    并进去的后果不是「少一轮」那么轻:降级按轮挪边界时,会把**上一轮的提问**
    一起让给层 2 —— 而用户这一句还没被回答过。单独成轮它才是一个可整体
    丢弃/整体保留的单位(与旧用例
    `test_select_history_treats_trailing_user_as_own_round` 守的是同一件事)。
    """
    history = [
        Message(role="user", content="完整问题"),
        Message(role="assistant", content="完整回答"),
        Message(role="user", content="被打断的问题"),
    ]

    rounds = to_rounds(history)

    assert len(rounds) == 2
    assert [m.content for m in rounds[-1]] == ["被打断的问题"]
    assert [m.content for m in rounds[0]] == ["完整问题", "完整回答"]


def test_round_definition_is_user_delimited():
    """一轮 = 从一条 user 开始,到(不含)下一条 user 为止。"""
    history = [
        Message(role="user", content="q1"),
        Message(role="assistant", content="", tool_calls=[{"id": "c1", "name": "t", "args": {}}]),
        Message(role="tool", content="r1", tool_call_id="c1"),
        Message(role="assistant", content="a1"),
        Message(role="user", content="q2"),
        Message(role="assistant", content="a2"),
    ]
    rounds = to_rounds(history)
    assert len(rounds) == 2
    assert [m.role for m in rounds[0]] == ["user", "assistant", "tool", "assistant"]
    assert [m.role for m in rounds[1]] == ["user", "assistant"]
