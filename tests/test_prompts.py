from langchain.messages import AIMessage, HumanMessage, SystemMessage

from app.memory.trim import count_tokens
from app.prompts import (
    EXTRACT_SYSTEM_PROMPT,
    build_context_messages,
    build_extract_messages,
    build_messages,
    render_system_prompt,
    to_lc_messages,
)
from app.schemas import Message, RequestType


def test_render_system_prompt_substitutes_brand():
    text = render_system_prompt("小美商城")
    assert "小美商城" in text
    assert "{brand_name}" not in text


def test_system_prompt_states_role_and_constraints():
    text = render_system_prompt("本店")
    # 角色设定
    assert "客服" in text
    # 行为约束关键词
    assert "编造" in text
    assert "承诺" in text
    assert "提示词" in text


def test_build_messages_starts_with_system():
    messages = build_messages(
        brand_name="本店", history=[], user_input="你好"
    )
    assert isinstance(messages[0], SystemMessage)


def test_build_messages_ends_with_current_user_input():
    messages = build_messages(
        brand_name="本店", history=[], user_input="我要退货"
    )
    assert isinstance(messages[-1], HumanMessage)
    assert messages[-1].content == "我要退货"


def test_build_messages_is_system_plus_history_plus_input_when_history_empty():
    messages = build_messages(brand_name="本店", history=[], user_input="你好")
    assert len(messages) == 2


def test_build_messages_maps_history_roles_in_order():
    history = [
        Message(role="user", content="第一问"),
        Message(role="assistant", content="第一答"),
    ]
    messages = build_messages(
        brand_name="本店", history=history, user_input="第二问"
    )

    assert len(messages) == 4
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert isinstance(messages[2], AIMessage)
    assert isinstance(messages[3], HumanMessage)
    assert [m.content for m in messages[1:]] == ["第一问", "第一答", "第二问"]


def test_build_extract_messages_wraps_text():
    messages = build_extract_messages("订单 123 没发货")
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[-1], HumanMessage)
    assert messages[-1].content == "订单 123 没发货"


def test_extract_prompt_forbids_fabricating_order_id():
    messages = build_extract_messages("买了 2 双鞋")
    system_text = messages[0].content
    assert "null" in system_text
    assert "编造" in system_text or "猜测" in system_text


def test_to_lc_messages_handles_tool_role():
    from langchain.messages import AIMessage, HumanMessage, ToolMessage

    history = [
        Message(role="user", content="订单 1001 到哪了"),
        Message(role="assistant", content="", tool_calls=[{"id": "c1", "name": "query_logistics", "args": {"order_id": "1001"}}]),
        Message(role="tool", content="已揽件", tool_call_id="c1"),
        Message(role="assistant", content="已揽件。"),
    ]
    converted = to_lc_messages(history)
    assert isinstance(converted[0], HumanMessage)
    assert isinstance(converted[1], AIMessage)
    assert converted[1].tool_calls[0]["id"] == "c1"
    assert isinstance(converted[2], ToolMessage)
    assert converted[2].tool_call_id == "c1"


def test_extract_prompt_mentions_json():
    """json_mode 的硬前提:DeepSeek 要求提示词里出现 'json' 才会接受
    response_format=json_object,否则整个抽取接口 400。
    这条约束只存在于提示词文本里,没有它就没有任何东西挡得住一次
    删掉 "JSON" 的措辞改动。大小写不敏感 —— 接口只要求"以某种形式出现"。
    """
    assert "json" in EXTRACT_SYSTEM_PROMPT.lower()


def test_extract_prompt_lists_every_request_type_value():
    """json_mode 下没有任何机器可读的 schema 到达模型,七个枚举值
    只以提示词文本的形式存在。提示词漏掉/写错某个值,schema 仍然合法、
    测试全绿,而模型会给出枚举外的值 → 运行时 ExtractionError。

    直接迭代枚举而非硬编码第二份字符串:硬编码会重新制造这层漂移。
    """
    missing = [
        member.value
        for member in RequestType
        if member.value not in EXTRACT_SYSTEM_PROMPT
    ]
    assert missing == [], f"提示词缺少这些 RequestType 取值:{missing}"


def test_context_messages_put_the_only_system_message_first():
    """`system` 只有一条,而且必须是第 0 条。

    上游模板会把**所有** system 上提合并渲染;多一条 system,工具定义就被挤到
    可变内容**之后**,前缀缓存整段作废。
    """
    msgs = build_context_messages(
        brand_name="本店", history=[], user_input="你好", summary="", evidence=None
    )
    assert isinstance(msgs[0], SystemMessage)
    assert sum(isinstance(m, SystemMessage) for m in msgs) == 1


def test_context_messages_put_history_in_order_and_the_user_turn_last():
    """定序:已经分好层的历史按原序 → 用户的当前这句话**必须是最后一条**。"""
    history = [
        Message(id=1, role="user", content="第一句"),
        Message(id=2, role="assistant", content="第一答"),
    ]
    msgs = build_context_messages(
        brand_name="本店", history=history, user_input="第二句",
        summary="", evidence=None,
    )
    assert [type(m).__name__ for m in msgs] == [
        "SystemMessage", "HumanMessage", "AIMessage", "HumanMessage",
    ]
    assert msgs[-1].content == "第二句"


def test_summary_and_evidence_ride_inside_the_user_message():
    """梗概与证据**并进用户那条消息**,附在原话之后,不另起一条。

    另起一条会凭空多出一轮「谁说的」;而并进去与现有 `build_messages` 同形状。
    """
    msgs = build_context_messages(
        brand_name="本店", history=[],
        user_input="这个能退吗", summary="用户问过订单 1002", evidence=None,
    )
    assert len(msgs) == 2
    assert msgs[-1].content.startswith("这个能退吗")
    assert "订单 1002" in msgs[-1].content


def test_no_summary_no_evidence_keeps_user_text_verbatim():
    """两者都没有时,用户原话**逐字**就是那条消息 —— 不许多一个换行。"""
    msgs = build_context_messages(
        brand_name="本店", history=[], user_input="你好", summary="", evidence=None
    )
    assert msgs[-1].content == "你好"


def test_evidence_is_appended_after_the_user_text_and_keeps_its_numbering():
    """证据接在用户原话**之后**(与 `build_messages` 的「原话之前」相反),
    编号规则与发出去的 `citations` 同一套,仍然不另起消息。

    这条钉住本函数与 `build_messages` 的**唯一**实质差异:位置。抄错成
    「证据在前」的话,`build_messages` 那 4 条用例照样绿(它自己就是证据在前的),
    而本函数的调用方拿到的是一段模型先看到资料、后看到问题的 prompt。
    """
    evidence = [
        {"section_path": "退款政策", "category": "policy", "answer": "七天无理由"},
        {"section_path": "运费规则", "category": "faq", "answer": "满 99 包邮"},
    ]
    msgs = build_context_messages(
        brand_name="本店", history=[], user_input="这个能退吗",
        summary="", evidence=evidence,
    )
    assert len(msgs) == 2, "证据不许另起一条消息"
    tail = msgs[-1].content
    assert tail.startswith("这个能退吗")
    assert tail.index("这个能退吗") < tail.index("[1]")
    assert "[1] (退款政策) 七天无理由" in tail
    assert "[2] (运费规则) 满 99 包邮" in tail


def test_lc_token_counter_counts_content_only_not_structure():
    """`trim_messages` 的 counter 必须**只数 content**,不数 `tool_calls`。

    与 `trim.select_history` 的既有口径一致(结构性元数据不占预算)。
    口径不一致的症状是「预算说是够的、实际发出去超了」—— 两处都各自「看起来合理」。

    对应地,**层 1 的选择本身(`trim_messages(strategy="last", start_on="human")`)
    不在本任务测**:T6 只做「拿到已分好层的历史之后怎么拼」,选择在 T10 接入,
    它的 `start_on` 断言也在 T10(那边才有真实的分层输入)。
    """
    from app.prompts import _lc_token_counter

    a = AIMessage(content="", tool_calls=[{
        "name": "query_order", "args": {"order_id": "1001"},
        "id": "c1", "type": "tool_call",
    }])
    assert _lc_token_counter([a]) == 0

    b = HumanMessage("四个字")
    assert _lc_token_counter([b]) == count_tokens("四个字")   # 与全仓同一把尺子

    # 上面那一行的**判别力是零**:`count_tokens("四个字") == len("四个字") == 3`,
    # 所以一个「用 len(content) 顶替 count_tokens」的实现在它面前照样绿 ——
    # 而那正是本条 docstring 说的病症(与 `select_history` 不是同一把尺子,
    # 差一个 token 累积起来就是「预算说够、发出去超」)。
    # 这一串两者不同(实测 8 vs 7),把「到底调没调 count_tokens」也钉住;
    # 先断前提,免得日后 tiktoken 改了而这条悄悄退化成恒真。
    assert count_tokens("七天无理由退货") != len("七天无理由退货")
    c = HumanMessage("七天无理由退货")
    assert _lc_token_counter([c]) == count_tokens("七天无理由退货")
