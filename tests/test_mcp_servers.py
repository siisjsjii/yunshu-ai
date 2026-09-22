"""两个 MCP Server 的**进程内**验证(不起进程、不走 HTTP)。

`FastMCP.list_tools()` / `call_tool()` 是纯内存方法 —— 这就是本章能让
MCP Server 也进单测的原因(单测全程不联网是硬规矩)。
"""

import json

import pytest

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
    from app.tools.errors import ToolNotFound

    # 挑一个**真的发了货**的订单号,否则物流侧应当抛 ToolNotFound
    shipped = next(
        no for no in (str(1000 + i) for i in range(1, 60))
        if order_record(no)["status"] in ("已发货", "已完成")
    )
    result = await logistics.call_tool("query_logistics", {"order_id": shipped})
    # ⚠️ 1.30.0 的 `call_tool` 返回的是**二元组** `(list[TextContent], dict)`,
    # **不是** `list[TextContent]` —— 形参里的 `Sequence[ContentBlock] | dict`
    # 说的是**两个返回位**各自的类型。所以是 `result[0][0].text`,不是 `result[0].text`。
    text = result[0][0].text
    assert json.loads(text) == logistics_record(shipped)


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
    with pytest.raises(Exception) as exc:
        await logistics.call_tool("query_logistics", {"order_id": unsent})
    assert "物流" in str(exc.value) or "尚未发货" in str(exc.value)


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
