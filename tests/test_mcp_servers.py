"""两个 MCP Server 的**进程内**验证(不起进程、不走 HTTP)。

`FastMCP.list_tools()` / `call_tool()` 是纯内存方法 —— 这就是本章能让
MCP Server 也进单测的原因(单测全程不联网是硬规矩)。
"""

import json

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from mcp_servers.aftersales import mcp as aftersales
from mcp_servers.logistics import mcp as logistics


@pytest.mark.anyio
async def test_logistics_exposes_exactly_one_tool():
    names = [t.name for t in await logistics.list_tools()]
    assert names == ["query_logistics"]


@pytest.mark.anyio
async def test_aftersales_exposes_the_two_declared_tools():
    names = sorted(t.name for t in await aftersales.list_tools())
    assert names == ["query_return_progress", "query_warranty"]


@pytest.mark.anyio
async def test_every_tool_has_a_description_and_an_input_schema():
    """服务端这三样齐备,客户端那边才有东西可登记(注册中心的第 1 条要求)。"""
    for server in (logistics, aftersales):
        for t in await server.list_tools():
            assert t.description and t.description.strip(), t.name
            schema = t.inputSchema
            assert isinstance(schema, dict) and schema.get("properties"), t.name


@pytest.mark.anyio
async def test_logistics_returns_the_same_record_as_the_builtin_data_source():
    """**这是本任务最重要的一条。**

    Server 与内置工具**必须共用** `app.tools.mock_data` —— 各带一套随机数的话,
    同一个订单号在「订单查询」与「物流查询」之间会说两套话
    (订单说「已发货」、物流说「待付款」),演示时一眼穿帮。
    """
    from app.tools.mock_data import logistics_record, order_record

    # 挑一个**真的发了货**的订单号,否则物流侧应当抛 ToolNotFound
    shipped = next(
        no for no in (str(1000 + i) for i in range(1, 60))
        if order_record(no)["status"] in ("已发货", "已完成")
    )
    result = await logistics.call_tool("query_logistics", {"order_id": shipped})
    # ⚠️ **`call_tool()` 返回的是 2-tuple `(list[ContentBlock], dict)`,不是列表。**
    # 函数签名上的返回注解写的是 `Sequence[ContentBlock] | dict[str, Any]`,
    # **与实测不符** —— 照注解写 `result[0].text` 会得到
    # `TypeError: Object of type TextContent is not JSON serializable`。
    # (T6 的实现者实测到并订正了;这个注解是个陷阱。)
    assert json.loads(result[0][0].text) == logistics_record(shipped)


@pytest.mark.anyio
async def test_logistics_says_not_found_for_an_unsent_order():
    """未发货 ⇒ **没有**物流记录,如实说,不编一条出来。

    与 `app/tools/builtin/orders.py` 那条同源:`LOGISTICS_BY_STATUS` 里
    不在的状态就是查不到,不是上游故障。
    """
    from app.tools.mock_data import order_record

    unsent = next(
        no for no in (str(1000 + i) for i in range(1, 60))
        if order_record(no)["status"] in ("待付款", "已付款", "已取消")
    )
    # ⚠️ 断**具体的异常类**并断**文案**,不是 `pytest.raises(Exception)` 加一个 `or`。
    # `ToolError` 是 mcp 1.30.0 里 **进程内与经 HTTP 两条路都会产生**的那个类
    # (`mcp/server/fastmcp/tools/base.py` 的 `Tool.run` 把任何异常包成
    #  `ToolError(f"Error executing tool {name}: {e}")`),所以它可以钉。
    with pytest.raises(ToolError) as exc:
        await logistics.call_tool("query_logistics", {"order_id": unsent})
    assert "尚未发货" in str(exc.value)


@pytest.mark.anyio
async def test_logistics_rejects_a_malformed_order_id():
    """⚠️ **这条是 T6 定稿后补的**(审查员实测上报)。

    **本 Server 上 `require_order_no` 没有任何看守** —— 把它从
    `query_logistics` 里删掉,**全部测试照样绿**。而 `order_record("abc")`
    **真的会返回一条完整的伪造订单**(审查员只读地跑过),于是 Server 会对
    「abc」这种垃圾入参**凭空编一张订单出来** ——
    而那正是 `mock_data.py::require_order_no` 的 docstring 明令禁止的:
    「不符合视为查无此单,**而不是编一个结果**」。

    内置那一半的同一条规矩是**有**看守的(`tests/test_tools_random.py` 对
    `"abc"` 与非 ASCII 数字都断了 `ToolNotFound`);MCP 这一半原先没有。
    """
    with pytest.raises(ToolError) as exc:
        await logistics.call_tool("query_logistics", {"order_id": "abc"})
    assert "未找到订单" in str(exc.value)


@pytest.mark.anyio
async def test_warranty_rejects_a_malformed_order_id():
    """同上,售后那两个工具各要一条 —— 它们**各自**调了一次 `require_order_no`。"""
    with pytest.raises(ToolError) as exc:
        await aftersales.call_tool("query_warranty", {"order_id": "abc"})
    assert "未找到订单" in str(exc.value)

    with pytest.raises(ToolError) as exc2:
        await aftersales.call_tool("query_return_progress", {"order_id": "abc"})
    assert "未找到订单" in str(exc2.value)


@pytest.mark.anyio
async def test_warranty_product_comes_from_the_shared_order_record():
    """⚠️ **这条是 T6 定稿后补的**(实现者实测上报)。

    初稿只守住了**物流**那一半的「两个 Server 必须共用 `mock_data`」——
    实现者把「改一个 aftersales 的种子前缀」这个变异跑了一遍,**6 条测试全绿**:
    变异**确实生效了**(输出从「保修中」变成「已过保」),但**没有任何断言看得见它**。
    也就是说,把 `query_warranty` 里那句 `order_record(order_no)["product"]`
    换成它自己的随机流,**一条测试都不会红** ——
    而它坏掉的表现与物流那半**一模一样**:
    **同一个订单号,`query_order` 说「无线耳机」、`query_warranty` 说「运动鞋」。**

    这里断的是**一致性**(而不是钉一个会随种子漂移的黄金值):
    售后报的商品必须与订单真相源报的**是同一个**。
    """
    from app.tools.mock_data import order_record

    order_no = "1008"
    result = await aftersales.call_tool("query_warranty", {"order_id": order_no})
    payload = json.loads(result[0][0].text)
    assert payload["order_id"] == order_no
    assert payload["product"] == order_record(order_no)["product"]


@pytest.mark.anyio
async def test_aftersales_is_deterministic():
    """同一入参两次调用必须同结果 —— 否则验收写不了会失败的断言。"""
    a = await aftersales.call_tool("query_warranty", {"order_id": "1002"})
    b = await aftersales.call_tool("query_warranty", {"order_id": "1002"})
    assert json.dumps(a, ensure_ascii=False, default=str) == json.dumps(
        b, ensure_ascii=False, default=str
    )

    # ⚠️ **`query_return_progress` 也要断**,别只测 `query_warranty`。
    # 审查员指出:那个工具有**任何**内容断言都没有 —— 返回常量、或者用了
    # 未播种的随机流,都看不出来。而它恰好是本章数据**明知与 `refund_requests`
    # 无关**的那一个,更没有别的地方兜着。
    c = await aftersales.call_tool("query_return_progress", {"order_id": "1002"})
    d = await aftersales.call_tool("query_return_progress", {"order_id": "1002"})
    assert json.dumps(c, ensure_ascii=False, default=str) == json.dumps(
        d, ensure_ascii=False, default=str
    )
    # 断它**真的按订单号取值**(不是一个常量):换个订单号,内容必须不同。
    #
    # ⚠️ **必须先把 `order_id` 摘掉再比**(T6 修复轮 2 实测):`order_id` 只是入参的
    # 回显 —— 任何实现都会把它原样写回去,**光靠它就能让两个 JSON 不等**。
    # 实测把 `query_return_progress` 改成**返回常量**(`stage`/`updated_at` 两种订单号
    # 一模一样),整条用例**照样绿** —— 也就是说直接比整个 JSON 的写法,
    # 断的根本不是「按号取值」。摘掉回显后,比的才是**载荷**。
    def _payload(call_result):
        body = json.loads(call_result[0][0].text)
        body.pop("order_id", None)  # 回显,不是证据
        return body

    e = await aftersales.call_tool("query_return_progress", {"order_id": "1003"})
    assert _payload(c) != _payload(e)
