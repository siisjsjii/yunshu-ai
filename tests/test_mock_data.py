"""mock 数据源的抽取与**跨进程稳定性**。

跨进程那条是本文件的重点:内置工具与两个 MCP Server 是**三个不同的进程**,
种子实现只要有一点不同,同一订单号就会在三处得到不同数据。
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.tools.mock_data import (
    ECHO_LIMIT,
    LOGISTICS_BY_STATUS,
    logistics_record,
    order_record,
    require_order_no,
    rng,
)
from app.tools.errors import ToolNotFound

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_rng_is_deterministic_within_process():
    assert rng("order", "1002").random() == rng("order", "1002").random()


def test_rng_differs_across_partitions():
    """同一订单号、不同前缀 ⇒ 两条独立流(这是**刻意的**:订单与物流各自演化)。"""
    assert rng("order", "1002").random() != rng("logistics", "1002").random()


def test_order_record_fields_are_stable():
    rec = order_record("1002")
    assert rec["order_id"] == "1002"
    assert set(rec) == {"order_id", "status", "product", "amount", "created_at"}


def test_logistics_exists_exactly_when_the_order_status_says_so():
    """ch02 的既有不变量:物流记录**从订单状态派生**,不是另起一条随机流。

    两个工具各自 `rng(不同前缀, 同一订单号)` 时是两条独立随机流,
    同一个订单可以同时是「已取消」和「已签收」。

    ⚠️ **这一条的初稿是同义反复,零判别力** —— 写成了
    `if status in TABLE: continue` 后面跟 `assert status not in TABLE`,
    对任何实现都恒真(实现者在 T1 上报,已订正)。现在它**真的**去调
    `logistics_record`,于是「按状态派生」与「独立抽一条」这两种实现
    会在这里分叉。
    """
    shipped = unsent = 0
    for no in (str(1000 + i) for i in range(1, 60)):
        status = order_record(no)["status"]
        if status in LOGISTICS_BY_STATUS:
            assert logistics_record(no)["status"] in LOGISTICS_BY_STATUS[status]
            shipped += 1
        else:
            with pytest.raises(ToolNotFound):
                logistics_record(no)
            unsent += 1
    # **两个分支都要真的走到过** —— 否则这条测试可能整段被跳过
    # (本仓记过的第 (e) 类假绿:输入小到触发不了被测行为)。
    assert shipped > 0 and unsent > 0


#: 订单 1008 的物流记录**黄金值**。**真跑一次打印出来照抄的,不是手写的。**
#:
#: 为什么挑 1008:它是实现里 `shipped` 分支(状态落在 `LOGISTICS_BY_STATUS` 里)
#: 在 1001–1059 区间里第一个 **`已发货`** 的订单,候选集是三元素
#: (`已揽件/运输中/派送中`),三种物流状态都能走到。
#:
#: ⚠️ **初选是 1003,实测后换掉了。** 1003 同样是「有物流」的单,但它落在
#: 约 15% 的**对抽取次序不敏感**的号码里 —— 把 `CITIES` 与 `candidates` 两次
#: 抽取互换(2000 个订单里 646 个的记录会变),1003 的记录**一个字都不变**,
#: 黄金断言照样绿。换 1008 之后同一条变异必红(见报告 M4)。
#: **挑黄金号码要看它真的会随实现变,不能只看它「有物流」。**
#:
#: 为什么需要它:`logistics_record` 的**轨迹与时间**此前只被「状态落在候选集里」
#: 间接盖到,而 T7 之后物流 MCP Server 走的正是这条路 —— 那时它是**唯一**的
#: 实现。这条断言把 `order_record` 的抽取顺序(状态→商品→金额→时间)与
#: `logistics_record` 的(状态→城市→发货偏移→最新偏移)**逐值**钉死。
_GOLDEN_LOGISTICS_1008 = {
    "order_id": "1008",
    "status": "已揽件",
    "location": "成都分拨中心",
    "traces": [
        {"time": "2026-05-21 05:06", "desc": "成都分拨中心 已发出"},
        {"time": "2026-05-24 08:06", "desc": "成都分拨中心 已揽件"},
    ],
}


def test_logistics_record_matches_the_golden_value():
    """整条物流记录逐值比对 —— 「抽取顺序不可改动」的显式守卫。

    比的是**值**不是形状:另外两个「有物流」的单(1003 / 1005)记录形状与它
    完全相同,值不同,断言对它们不成立(见报告里的实测)。形状断言
    (`set(rec) == {...}`)对这一层毫无判别力。
    """
    assert logistics_record("1008") == _GOLDEN_LOGISTICS_1008


@pytest.mark.parametrize("bad", ["", "12", "abc", "١٢٣٤", "²²²²"])
def test_require_order_no_rejects(bad):
    """`isascii() and isdigit()` **两个条件**都要。

    单独 `isdigit()` 是 Unicode 感知的:阿拉伯-印度数字与上标都为 True,
    这类输入会**通过**校验并拿到一张凭空编造的订单。
    """
    with pytest.raises(ToolNotFound):
        require_order_no(bad)


def test_require_order_no_strips_and_returns():
    assert require_order_no("  1002  ") == "1002"


def test_echo_limit_is_a_positive_int():
    assert isinstance(ECHO_LIMIT, int) and ECHO_LIMIT > 0


def test_order_record_is_stable_across_processes():
    """**跨进程**:子进程里算一遍,与本进程比对。

    必须加 `-X utf8` —— 本机 locale 是 cp936,管道上的 stdout 按 GBK 编码
    而父进程按 UTF-8 解码,报错会表现为 `proc.stdout is None`。

    写法(REPO_ROOT / `cwd` / `text=True, encoding="utf-8"`)与
    `tests/test_tools_random.py::test_seed_is_stable_across_processes` **逐字对齐**
    —— 全仓只该有**一个**跨进程约定,不然两处各修各的,迟早只修好一半。
    """
    code = (
        "import json, sys;"
        "sys.path.insert(0, '.');"
        "from app.tools.mock_data import order_record;"
        "sys.stdout.buffer.write("
        "json.dumps(order_record('1002'), ensure_ascii=False).encode('utf-8'))"
    )
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == order_record("1002")
