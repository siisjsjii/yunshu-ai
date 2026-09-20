"""候选订单 —— 补图里没有的一环。

系统里**没有 orders 表**:订单是 `app/tools/business.py` 的 `_order_record()`
用 hashlib 按订单号**现算**的,不存在「某用户的订单」这个概念。
所以订单选择器的候选要另找来源,顺序如下:

1. **会话历史里出现过的订单号**(用户之前提过的);
2. 一个都没有时,给**会话固定的一组演示订单**。

⚠️ 第 2 条是**演示数据,不是真实用户订单** —— 本仓没有用户-订单关系表,
这里只是让「不带订单号问退款」这条验收路径能走完。号码本身是真能查出单的
(订单由哈希派生,任何 4-32 位数字都有确定数据)。
"""

import re

from app.schemas import Message

#: 演示订单号池。取 8 位数字,与既有演示号(如 20240915)同量级。
_DEMO_POOL: tuple[str, ...] = (
    "20240915",
    "20240901",
    "20240818",
    "20240808",
    "20240726",
    "20240712",
)

#: 卡片最多给几个候选。
MAX_CANDIDATES = 4

#: 兼容测试与外部引用的名字。
DEMO_ORDERS = _DEMO_POOL[:MAX_CANDIDATES]

#: 紧跟数字后面就说明它**不是订单号**的词:金额/数量/时间单位。
#: 不含「号」—— 「订单 111111 号」这种写法里它是订单的一部分。
_UNIT_WORDS = "元块角分钱年月日时天周个件台条张只次折米克斤岁人份盒套瓶包箱"

#: 订单号形态:4–32 位纯数字(`app/tools/business.py` 的 `_order_record` 同此契约)。
#:
#: ⚠️ 裸 `\b\d{4,32}\b` 会**把金额、日期、数量一起捞进来** —— 用户会看到一张
#: 点开查无此单的卡片。两处否定环视各自挡一类:
#:
#: - 前视 `(?<![\d.,¥￥$])` 与后视里的 `(?!\d|[.,\-/]\d)`:`1234.50` 的小数两半、
#:   `2024-09-15` 的年份、`¥1299` 这种带货币符号的标价都不是订单号,同时
#:   不允许把一个更长的数字串**截断**成 32 位;
#: - 后视里的 `(?!\s*[元块角…])`(类目即 `_UNIT_WORDS`):`1299 元`、`1000 件`、
#:   `2024 年` 是金额/数量/时间。
#:
#: 注意 `\d{4,32}` 的长度下界**本身**就挡掉了两位数的「花了 99 元」——
#: 所以拿两位数字当干扰项的测试是**恒真**的,挡不住任何错误实现
#: (见 `tests/test_refund_orders.py` 里那条金额用例,干扰项特意取 4 位以上)。
_ORDER_RE = re.compile(
    rf"(?<![\d.,¥￥$])"
    rf"\d{{4,32}}"
    rf"(?!\d|[.,\-/]\d|\s*[{_UNIT_WORDS}])"
)


def _scan(history: list[Message], limit: int) -> list[str]:
    """从**由近及远**的历史里取订单号,去重保序。"""
    found: list[str] = []
    for msg in reversed(history):          # 越近的越可能是用户当下关心的
        for no in _ORDER_RE.findall(msg.content or ""):
            if no not in found:
                found.append(no)
            if len(found) >= limit:
                return found
    return found


def _demo_for(conversation_id: str) -> list[str]:
    """回落用的演示订单。

    **固定一组、与会话 id 无关** —— 「同一会话每次拿到同一批」由「它是常量」
    以最强形式满足(连跨会话都不变)。

    计划里原本写的是 `sha256(conversation_id) % len(_DEMO_POOL)` 轮转取
    `MAX_CANDIDATES` 个,**实测做不到**:`sha256("s1") % 6 == 4`,于是
    「无号码时回落」拿到的第 0 个是 `20240712` 而不是 `DEMO_ORDERS[0]` 的
    `20240915`,`candidate_orders(...) == list(DEMO_ORDERS)` 这条断言直接红。
    轮转对任何用到它的断言都只能**靠撞对 id 才绿**(换一个会话 id 就翻),
    那种绿是运气不是保证,故不采用。

    `conversation_id` 参数**保留但不使用**:签名是 T7 `refund_pick_order`
    定死的接口,改签名会连带改调用点。
    """
    return list(DEMO_ORDERS)


def candidate_orders(history: list[Message], conversation_id: str) -> list[str]:
    """给订单卡片用的候选订单号(已去重、已限长)。"""
    from_history = _scan(history, MAX_CANDIDATES)
    if from_history:
        return from_history
    return _demo_for(conversation_id)
