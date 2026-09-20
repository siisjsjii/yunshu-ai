from app.refund.orders import DEMO_ORDERS, candidate_orders
from app.schemas import Message


def _msg(role, content):
    return Message(role=role, content=content)


def test_history_order_numbers_come_first():
    history = [_msg("user", "我刚才问的是订单 20240915 的物流")]
    got = candidate_orders(history, "s1")
    assert got[0] == "20240915"


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


def test_demo_set_is_session_stable():
    """同一会话两次调用给出同一批 —— 否则卡片上的号码会跳。"""
    a = candidate_orders([], "conv-A")
    b = candidate_orders([], "conv-A")
    assert a == b


def test_demo_numbers_are_usable_order_numbers():
    """演示集里的号码必须能真的被 query_order 查出来(4-32 位数字)。"""
    for no in DEMO_ORDERS:
        assert no.isdigit() and 4 <= len(no) <= 32


# --- 以下为 T1 实现者自补:brief 里那组用例**必要但不充分** ---------------


def test_amounts_and_dates_are_not_candidates():
    """金额与日期不得被当成订单号 —— 这条 brief 里没有,是本任务补的。

    brief 给的用例**全部**用纯数字串构造(「订单 111111 和 111111」),
    所以「把 `\\b\\d{4,32}\\b` 的边界条件全部忽略」的实现照样全绿。

    ⚠️ 干扰数字必须取 **4 位以上**:brief 举例的「花了 99 元」里的 `99`
    只有两位,本来就不满足 `\\d{4,32}`,拿它当干扰项是**恒真的假断言**
    (这条测试就差点这么写了)。
    """
    history = [_msg("user", "订单 20240915 那个,1299 元买的,2024 年 9 月下的单")]
    got = candidate_orders(history, "s1")
    assert got == ["20240915"]  # 不是 `in`:多出来的干扰项必须**看得见**


def test_amount_only_history_falls_back_to_demo_set():
    """整段历史只有金额/日期时,一条候选都不该造出来,应回落到演示集。

    错误实现会把 `1299` 当成候选订单交给前端 —— 用户拿到一张点开查无此单的卡片。
    """
    history = [_msg("user", "那个 1299 元的能退吗?我 2024 年买的")]
    assert candidate_orders(history, "s1") == list(DEMO_ORDERS)


def test_decimal_prices_and_quantities_are_not_candidates():
    """小数价格、带货币符号的标价与数量词同样不是订单号。"""
    history = [_msg("assistant", "这批一共 1234.50 元,共 1000 件,订单 20240818 已发出")]
    assert candidate_orders(history, "s1") == ["20240818"]

    # 没有量词兜底的标价:只有货币符号可依,`¥1299` 不能变成候选。
    price_tag = [_msg("user", "吊牌价 ¥1299,能退吗")]
    assert candidate_orders(price_tag, "s1") == list(DEMO_ORDERS)
