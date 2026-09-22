"""确定性伪随机工具的测试。不联网、不碰 DB。

**物流那半的调用点搬过一次家**(ch08 T7):`query_logistics` 不再是内置工具,
它现在跑在**独立进程**的物流 MCP Server 里(`mcp_servers/logistics.py`,
经 Streamable HTTP 调用)。单测全程不联网,够不到那条 HTTP 路,所以下面这些
用例一律走 `mock_data.logistics_record` —— 它就是**内置与两个 MCP Server 的
唯一真相源**(`app/tools/mock_data.py` 的模块 docstring),物流 Server 的工具体
只是一行 `json.dumps(logistics_record(...))` 的透传。Server 那一侧(入参校验、
序列化、抛出形状)的进程内验证在 `tests/test_mcp_servers.py`。
"""

import asyncio
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from app.tools.builtin.orders import query_order, query_product
from app.tools.errors import ToolNotFound
from app.tools.mock_data import logistics_record, require_order_no

REPO_ROOT = Path(__file__).resolve().parents[1]


def _call(tool, args: dict) -> str:
    tool_call = {"name": tool.name, "args": args, "id": "call_1", "type": "tool_call"}
    return asyncio.run(tool.ainvoke(tool_call)).content


def _logistics(order_id: str) -> str:
    """物流查询的**当前提供者**(见模块 docstring)。

    走的顺序与物流 Server 的工具体**逐字一致**:先 `require_order_no`(入参
    不合法 ⇒ `ToolNotFound`,而不是编一张单),再 `logistics_record` 取数,
    最后按 `ensure_ascii=False` 序列化(中文不转义成 \\uXXXX,白烧 token)。
    """
    return json.dumps(
        logistics_record(require_order_no(order_id)), ensure_ascii=False
    )


#: 订单状态 → 该状态下**可能**出现的物流状态。
#:
#: 期望值在测试里**独立声明**,不 import 实现里的同名常量 —— import 的话
#: 实现把表改错时期望值跟着一起改,断言就退化成同义反复。
_ALLOWED_LOGISTICS = {
    "已发货": {"已揽件", "运输中", "派送中"},
    "已完成": {"已签收"},
}

#: 派生演示订单号用的扫描区间。必须同时存在「已发货/已完成」与「未发货」
#: 两种订单,否则下面两个 helper 会硬失败。
_SCAN = range(1000, 1040)


def _status_of(order_id: str) -> str:
    return json.loads(_call(query_order, {"order_id": order_id}))["status"]


def _order_ids(*, shipped: bool) -> list[str]:
    """按「有没有物流」筛订单号。

    物流类断言**不能**写死 1001:它是「已取消」,对它查物流会抛
    ToolNotFound —— 那是**正确行为**,不是"正常订单"。写死别的号码同样不行:
    种子函数一变它就静默失效,而失败信息会指向"工具抛错了"而不是"号码选错了"。
    """
    ids = [str(i) for i in _SCAN if (_status_of(str(i)) in _ALLOWED_LOGISTICS) is shipped]
    if not ids:
        kind = "有" if shipped else "没有"
        raise AssertionError(
            f"{_SCAN.start}-{_SCAN.stop - 1} 里找不到{kind}物流的订单 —— 种子函数改坏了"
        )
    return ids


def test_same_order_id_gives_same_result():
    """同一订单号永远返回同样数据。"""
    oid = _order_ids(shipped=True)[0]
    assert _logistics(oid) == _logistics(oid)


def test_different_order_ids_differ():
    """不同订单号应有不同数据,否则工具等于常量。"""
    first, second = _order_ids(shipped=True)[:2]
    assert _logistics(first) != _logistics(second)


def test_result_is_json_with_chinese_not_escaped():
    """返回 JSON 字符串,且中文不被转义成 \\uXXXX(白烧 token)。"""
    raw = _logistics(_order_ids(shipped=True)[0])
    payload = json.loads(raw)
    assert set(payload) >= {"order_id", "status", "location"}
    assert "\\u" not in raw
    assert any("一" <= ch <= "鿿" for ch in raw)


def test_seed_is_stable_across_processes():
    """跨进程确定性。

    这条是本章最容易写错的断言 —— 内置 hash() 对 str 每进程随机化
    (PYTHONHASHSEED),用它会让同一订单号在重启后返回不同数据,
    而同进程内的任何测试都测不出来。故必须另起两个进程比对。

    ⚠️ **调用点在 ch08 T7 换了,这条测试没有**(它钉的性质与调用点无关):
    搬到了 `mock_data.logistics_record`。这个真相源现在被**三个进程**共用 ——
    客服服务(内置 `query_order`)、物流 Server、售后 Server ——
    种子在每个进程里都必须一样,否则「同一订单号处处同数据」当场不成立。
    """
    oid = _order_ids(shipped=True)[0]
    # 拼字符串而不是 f-string:下面这段代码里全是花括号,用 f-string 得逐个
    # 翻倍转义,读起来全是 `{{`。
    #
    # ⚠️ **订单号必须是上面派生的那个,不能写死一个演示号码**:T7 的 brief 里
    # 这一段的字面值是 `logistics_record('1002')`,而 **1002 是「已取消」** ——
    # `logistics_record` 对它抛 `ToolNotFound`,子进程非 0 退出,这条会直接红。
    # (实测确认过。)派生出来的 `oid` 是 `_order_ids(shipped=True)` 筛过的。
    code = (
        "import json, sys; sys.path.insert(0, r'.');"
        "from app.tools.mock_data import logistics_record;"
        "sys.stdout.buffer.write("
        "json.dumps(logistics_record('" + oid + "'), ensure_ascii=False).encode('utf-8'))"
    )
    # 子进程的输出编码必须显式钉成 UTF-8。本机 locale 是 cp936,Python 会把管道
    # 上的 stdout 按 GBK 编码,而下面按 UTF-8 解码 —— 不钉的话子进程输的中文会
    # 在解码时炸掉,表现为 proc.stdout 为 None。这与工具逻辑无关,纯属平台差异。
    outs = []
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, "-X", "utf8", "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(REPO_ROOT),
        )
        assert proc.returncode == 0, proc.stderr
        outs.append(proc.stdout)
    assert outs[0] == outs[1]
    assert outs[0].strip()  # 非空,防止两边都空而"通过"


def test_malformed_order_id_raises_tool_not_found():
    """不像订单号的输入 → ToolNotFound(可恢复),不是随机编一个结果。"""
    with pytest.raises(ToolNotFound):
        _logistics("abc")


def test_query_product_uses_keyword():
    a = _call(query_product, {"keyword": "无线耳机"})
    b = _call(query_product, {"keyword": "保温杯"})
    assert a != b
    assert "无线耳机" in a


# ---- 以下两条是 Task 5 修的 T4 遗留缺陷的钉子 ----


def test_query_product_name_and_spec_never_contradict():
    """name 里的规格与 spec 字段必须来自同一次抽取。

    抽两次的话两者相互独立,四次里只有一次对得上 —— 工具会把自相矛盾的
    数据喂给模型,而本章验收全靠模型如实转述工具结果。

    为什么不只测一个关键词:单次抽中同一规格的概率是 1/4,一个关键词
    有 1/4 的概率**漏报**(假绿)。20 个相互独立的入参把假绿压到 4^-20。
    """
    for i in range(20):
        keyword = f"商品{i}"
        payload = json.loads(_call(query_product, {"keyword": keyword}))
        assert payload["name"] == f"{keyword}({payload['spec']})"


def test_non_ascii_digits_are_not_order_numbers():
    """`isdigit()` 是 Unicode 感知的:阿拉伯-印度数字、上标、全角数字都为 True。

    只判 `isdigit()` 的话这些入参会**通过**校验,拿到一张凭空编造的订单,
    而不是 ToolNotFound —— 即"查无此单"被伪装成"查到了"。
    """
    for bad in ("١٢٣٤", "²²²²", "１２３４"):  # 阿拉伯-印度、上标、全角
        with pytest.raises(ToolNotFound):
            _logistics(bad)


def test_error_message_does_not_echo_unbounded_input():
    """回显给模型的错误文本必须有界。

    入参是模型给的,长度不受我们控制;原样回灌等于把上下文预算交给它。
    """
    huge = "9" * 5000
    with pytest.raises(ToolNotFound) as exc:
        _call(query_order, {"order_id": huge})
    assert huge not in str(exc.value)
    assert len(str(exc.value)) < 200


# ---- 订单与物流的数据自洽 ----


def test_unshipped_order_has_no_logistics():
    """未发货的订单查物流必须查**不到**,而不是编一条出来。

    这条钉的是耦合的反向那半:只把物流状态限制在订单状态的候选集里还不够 ——
    「待付款」的候选集是空的,实现要么抛错,要么就得凭空造一条物流记录,
    而凭空造的那条必然与订单状态矛盾。
    """
    with pytest.raises(ToolNotFound):
        _logistics(_order_ids(shipped=False)[0])


def test_order_and_logistics_status_never_contradict():
    """同一订单号下,订单状态与物流状态必须自洽。

    两个工具各自 `rng(不同前缀, 同一订单号)` 时是**两条独立随机流**,
    于是同一个订单可以同时是「已取消」和「已签收」。实测 1000 个订单里
    807 个自相矛盾(80.7%),连当时演示用的 1001 都落在里面。

    扫一段区间而不是抽查:80.7% 的矛盾率下,抽查两三个也大概率全中,
    那条断言区分不出实现。
    """
    contradictions = []
    for i in range(1000, 1400):
        oid = str(i)
        order = json.loads(_call(query_order, {"order_id": oid}))
        try:
            logistics = json.loads(_logistics(oid))
        except ToolNotFound:
            # 未发货的订单**必须**查不到物流。走得到这个分支本身就是耦合的
            # 证据:两条独立随机流下,任何合法订单号都查得到物流,永远不会抛。
            if order["status"] in _ALLOWED_LOGISTICS:
                contradictions.append((oid, order["status"], "查不到物流"))
            continue
        allowed = _ALLOWED_LOGISTICS.get(order["status"], set())
        if logistics["status"] not in allowed:
            contradictions.append((oid, order["status"], logistics["status"]))

    assert contradictions == [], (
        f"订单与物流自相矛盾 (订单号, 订单状态, 物流状态):{contradictions[:5]}"
    )


def test_logistics_trace_never_predates_the_order():
    """物流轨迹时间必须晚于下单时间。

    同一个根因的另一面:轨迹时间原本也从 `rng("logistics", ...)` 独立抽,
    于是能出现「9 月 5 日已发出」而订单「9 月 20 日下单」的包裹先于订单存在。

    这条的矛盾率远低于状态那条(量级 1%),故扫描区间要够宽 —— 区间窄了
    在旧实现下可能一个都撞不上,断言就成了恒真。
    """
    bad = []
    for i in range(1000, 3000):
        oid = str(i)
        order = json.loads(_call(query_order, {"order_id": oid}))
        try:
            logistics = json.loads(_logistics(oid))
        except ToolNotFound:
            continue
        created = datetime.strptime(order["created_at"], "%Y-%m-%d %H:%M")
        trace = datetime.strptime(logistics["traces"][0]["time"], "%Y-%m-%d %H:%M")
        if trace < created:
            bad.append((oid, order["created_at"], logistics["traces"][0]["time"]))

    assert bad == [], f"轨迹早于下单 (订单号, 下单时间, 轨迹时间):{bad[:5]}"


def test_logistics_traces_are_a_coherent_timeline():
    """轨迹必须是**时间递增**的事件序列,且末条描述当前状态。

    原先末条写死 `"time": "当前"` —— 不是时间戳,整条轨迹因此无法排序。
    模型读到的是"最后一次扫描停在『已发出』",于是 status 为「已签收」时它
    会当场指出"两者信息不太一致"并追问用户是否收到货:验收 4 的真实回复
    就是这么写的。断言按"能不能排序 + 末条是否描述当前状态"来钉。
    """
    fmt = "%Y-%m-%d %H:%M"
    for oid in _order_ids(shipped=True)[:5]:
        payload = json.loads(_logistics(oid))

        times = []
        for entry in payload["traces"]:
            try:
                times.append(datetime.strptime(entry["time"], fmt))
            except ValueError:
                pytest.fail(f"{oid} 的轨迹条目 time 不是时间戳,轨迹无法排序:{entry!r}")

        assert times == sorted(times), f"{oid} 轨迹时间未递增:{times}"
        assert payload["status"] in payload["traces"][-1]["desc"], (
            f"{oid} 末条轨迹没有描述当前状态 {payload['status']}:{payload['traces'][-1]!r}"
        )
