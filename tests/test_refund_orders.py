import pytest

from app.refund.orders import DEMO_ORDERS, MAX_CANDIDATES, candidate_orders
from app.schemas import Message


def _msg(role, content):
    return Message(role=role, content=content)


def test_history_order_numbers_come_first():
    history = [_msg("user", "我刚才问的是订单 20240915 的物流")]
    got = candidate_orders(history, "s1")
    assert got[0] == "20240915"
    # ⚠️ 这条**钉不住「由近及远」**:期望值恰好是回退也会给的那个串
    # (把 reversed 改成正序它照样绿)。真正钉住顺序的是
    # test_more_recent_history_wins。


def test_history_scan_finds_numbers_in_assistant_messages_too():
    history = [_msg("assistant", "您的订单 12345678 已发货")]
    assert "12345678" in candidate_orders(history, "s1")


def test_no_history_order_numbers_falls_back_to_demo_set():
    got = candidate_orders([_msg("user", "这个能退吗")], "s1")
    assert got == list(DEMO_ORDERS)


def test_candidates_are_deduped_and_capped():
    history = [_msg("user", "订单 111111 和 111111 还有 222222")]
    got = candidate_orders(history, "s1")
    assert len(got) == len(set(got))
    assert len(got) <= 4
    # ⚠️ 这条**区分不出**「有没有截断」:它的输入最多产出 2 个候选。
    # 真正钉住截断的是 test_candidates_are_capped_at_the_card_size。


def test_demo_set_is_session_stable():
    """同一会话两次调用给出同一批 —— 否则卡片上的号码会跳。

    ⚠️ 在「回落是常量」的实现下这条**恒真**(纯函数同参数不可能不同)。
    后半句才是能证伪的:换一个会话 id 也必须拿到**同一批** ——
    计划原文的 `sha256(conversation_id) % len(_DEMO_POOL)` 轮转会让它红。
    """
    a = candidate_orders([], "conv-A")
    b = candidate_orders([], "conv-A")
    assert a == b
    assert candidate_orders([], "conv-B") == a


def test_demo_numbers_are_usable_order_numbers():
    """演示集里的号码必须能真的被 query_order 查出来(4-32 位 ASCII 数字)。"""
    for no in DEMO_ORDERS:
        assert no.isascii() and no.isdigit() and 4 <= len(no) <= 32


# --- 以下为 T1 实现者自补:brief 里那组用例**必要但不充分** ---------------


def test_amounts_and_dates_are_not_candidates():
    """金额与日期不得被当成订单号 —— 这条 brief 里没有,是本任务补的。

    brief 给的用例**全部**用纯数字串构造(「订单 111111 和 111111」),
    所以「把 `\\b\\d{4,32}\\b` 的边界条件全部忽略」的实现照样全绿。

    ⚠️ 干扰数字必须取 **4 位以上**:brief 举例的「花了 99 元」里的 `99`
    只有两位,本来就不满足 `\\d{4,32}`,拿它当干扰项是**恒真的假断言**。
    """
    history = [_msg("user", "订单 20240915 那个,一共 1299 元。2024 年 9 月下的单")]
    got = candidate_orders(history, "s1")
    assert got == ["20240915"]  # 不是 `in`:多出来的干扰项必须**看得见**


def test_amount_only_history_falls_back_to_demo_set():
    """整段历史只有金额/日期时,一条候选都不该造出来,应回落到演示集。

    错误实现会把 `1299` 当成候选订单交给前端 —— 用户拿到一张点开查无此单的卡片。
    """
    history = [_msg("user", "标价 1299 元。2024 年 9 月买的,能退吗?")]
    assert candidate_orders(history, "s1") == list(DEMO_ORDERS)


def test_decimal_prices_and_quantities_are_not_candidates():
    """小数价格、带货币符号的标价与数量词同样不是订单号。"""
    history = [_msg("assistant", "这批一共 1234.50 元,共 1000 件,订单 20240818 已发出")]
    assert candidate_orders(history, "s1") == ["20240818"]

    # 没有量词兜底的标价:只有货币符号可依,`¥1299` 不能变成候选。
    price_tag = [_msg("user", "吊牌价 ¥1299,能退吗")]
    assert candidate_orders(price_tag, "s1") == list(DEMO_ORDERS)


def test_full_width_digits_are_not_candidates():
    """全角数字不是订单号:`_require_order_no` 要 `isascii()`,全角号查无此单。

    `\\d` 是 Unicode 感知的,写成 `\\d{4,32}` 时「２０２４０９１５」会被选进卡片,
    而工具侧拒收它 —— 又一张点开查无此单的卡。
    """
    history = [_msg("user", "订单 ２０２４０９１５ 到哪了")]
    assert candidate_orders(history, "s1") == list(DEMO_ORDERS)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("订单 20240915 包裹到哪了", ["20240915"]),
        ("订单 20240915 只发了一件", ["20240915"]),
        ("订单 20240915 分两笔付的", ["20240915"]),
        ("订单 20240915 钱付了", ["20240915"]),
        ("订单 20240915 日期还能改吗", ["20240915"]),
        ("我要退订单 12345678 分两笔付的", ["12345678"]),
        ("订单 20240915,1234 元的那个", ["20240915", "1234"]),
    ],
)
def test_unit_like_word_starts_do_not_eat_the_order_number(text, expected):
    """单位字**只在词尾**才算单位 —— 否则会把人真实的订单号整条吃掉。

    `_UNIT_WORDS` 与高频词首字大量重叠(包/只/分/钱/日…),「下一个字是单位字
    就否掉」的写法会让 `分两笔付的` / `只发了一件` 这类普通句子把 20240915
    整个丢掉。丢号码的后果是**回退到演示集:用户看到的是别人的订单**,
    比多给一个候选严重得多(审查裁定:冲突时取「找到真实订单」)。

    ⚠️ 断言必须**精确等于列表**,不能用 `got[0] == "20240915"`:那样前 5 条
    在错误实现下会**靠回退蒙对** —— 回退的第一个号码正好就是 20240915
    (第一版就是这么写的,变异 M-H 下只有第 6 条变红,其余照绿)。

    最后一条里的 `1234` 是**记账过的已知泄漏**(`元` 后面跟着 `的`,`元` 不处在
    词尾),不是期望行为;它一旦变红且 `1234` 消失,说明收紧成功,改期望值即可。
    """
    assert candidate_orders([_msg("user", text)], "s1") == expected


def test_more_recent_history_wins():
    """候选按**由近及远**——把 `reversed(history)` 改成正序时必须红。

    原有那条钉不住它:它的期望值恰好是「回退也会给」的第一个串。这里两条消息
    各带一个号码,且断言**两个都在、近的在前**(顺带挡住「只看最后一条」的实现)。
    """
    history = [
        _msg("user", "早先问的是订单 20240101"),
        _msg("user", "现在要退订单 20240915"),
    ]
    assert candidate_orders(history, "s1") == ["20240915", "20240101"]


def test_candidates_are_capped_at_the_card_size():
    """候选**真的被截断**到 4 张卡。

    只断 `len(got) <= 4` 是**空断言**:把 `MAX_CANDIDATES` 调到 10、
    或干脆不在 `_scan` 里截断,`<= 4` 那一族断言都照样绿。所以这里
    ①历史必须产出**多于 4** 个不同号码;②断言精确等于前 4 个。
    """
    history = [_msg("user", "订单 111111 222222 333333 444444 555555 666666")]
    got = candidate_orders(history, "s1")
    assert len(got) == 4 == MAX_CANDIDATES
    assert got == ["111111", "222222", "333333", "444444"]
