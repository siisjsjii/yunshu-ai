# ch08 工具系统 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `app/tools/registry.py` 里硬编码的五个 `@tool` 升级成
**注册中心 + 统一 JSON Schema 校验 + 权限门 + 单一执行引擎 + 审计留痕 + MCP 接入**,
并新增一条**建工单确认流**(LangGraph `interrupt()` + resume)。

**Architecture:** 工具一律登记成 `ToolSpec`(名 / 用途 / **原始 JSON Schema** / 读写类别 /
来源 / 绑给模型的那份 `BaseTool`),内置从 `app/tools/builtin/` **包内自动发现**,
MCP 从两个自建 Server **每请求现问现拿**。所有调用走 `execute_tool` 一处,
它依次过权限闸、JSON Schema 校验、执行、结果格式化,并在固定两处落审计。
工单确认拆成「只有 `interrupt()` 的节点」+「执行/拒绝的节点」,
`agent` 节点撞到未确认的写调用时**停在半路**、决议后再**续跑一轮不绑 tools** 的收尾。

**Tech Stack:** LangGraph 1.2.11 / LangChain 1.x / FastAPI / SQLAlchemy 2.x /
MySQL 8.0.46(主机端口 **3307**)/ `mcp>=1.24,<2`(FastMCP,Streamable HTTP)/
`langchain-mcp-adapters==0.3.2` / `jsonschema`

**Spec:** `docs/superpowers/specs/2026-09-22-ecommerce-cs-ch08-tools-design.md`

---

## Global Constraints

> 以下每一条都**逐字取自 spec 或 `CLAUDE.md`**,是每一任务的需求的一部分。
> 与具体任务冲突时,以本节为准并记账。

### 依赖与版本

- **`mcp` 钉 `>=1.24,<2`(解析到 1.30.0)。不要装 2.x。**
  `langchain-mcp-adapters==0.3.2` 的 `Requires-Dist` 写死 `mcp<2.0.0,>=1.24.0`。
- **mcp 1.x 的入口是 `from mcp.server import FastMCP`**(不是 v2 的
  `mcp.server.mcpserver.MCPServer`)。Context7 站点已整站迁到 v2,连标着 v1 的
  library id 也返回 v2 内容 ⇒ **这一处以锁定版本的轮子源码为准,不信文档。**
- **`convert_mcp_tool_to_langchain_tool(None, t, connection=conn, server_name=name,
  handle_tool_errors=False)`** —— 传 `connection` **不是** `session`;
  `handle_tool_errors` **必须显式关**(默认 `True` 会把 MCP 故障包成一条**正常的**
  工具返回,于是执行器眼里「物流服务连不上」是成功)。
- **`LangGraph==1.2.11` / `langchain==1.4.0` 等既有版本一律不动。**

### 平台陷阱(Windows + Git Bash,本机 locale cp936)

- **含中文的请求体不能走 `curl` 的 argv** —— MSYS2 按 CP936 重编码,服务端只回
  `error parsing the body`。一律走 **stdin heredoc**,或 `httpx` 这类替你处理编码的客户端。
- **子进程输出要显式钉编码**:跨进程测试给子进程加 `-X utf8`。
- **脚本打印非 ASCII 用 `sys.stdout.buffer.write(....encode("utf-8"))`**,不要 `print`。
- **验收断言不能直接 grep 原始 SSE 流**:回复逐 token 推送,`20240915` 会被切成独立帧。
  先 `join_tokens` 拼回再比对。
- **不要用 `grep '[一-龥]'` 检查中文完好性**(C locale 下退化成字节区间,恒真)。
  用 Python 码点判断。
- **起服务前先查端口**;见到多个残留 uvicorn **全部清掉**再起。
- **不要往命令行加 `-q`**:`pytest.ini` 的 `addopts` 已有一个,叠加成 `-qq` 后整行不打印 `N passed`。

### 测试规矩

- **单测全程不联网。** MCP Server 那侧用 `mcp.list_tools()` / `call_tool()`
  **进程内**验(mcp 1.30.0 的源码里它们是纯内存方法),不起进程、不走 HTTP。
- **`Settings(...)` 构造必须传 `_env_file=None`**(仓库根有真实 `.env`)。
- **db 测试读真实 `.env`,不加 `_env_file=None`**,并带 `@pytest.mark.db`。
- **别为自由文本写字符串断言**(`deepseek-flash` 在 `temperature=0` 下仍非确定)。
- **变量名不等于语义 —— 断言一个字段之前,先去读它是怎么被赋值的。**
- **「在处理之后注入」是本项目最高频的假绿形态**:注入**处理之前**的形态。

### 错误语义(既有边界,本章不新立规矩)

- **`ToolInfrastructureError` 必须向上抛,不能回灌给模型** —— 数据库故障绝不能被
  伪装成「你的订单号查不到」。
- **`422` 只表示「模型输出无法解析为约定结构」**;上游故障一律 `502` + 固定文案。
- **所有出站错误文本必须过 `app/sanitize.py:redact_api_key`**。
- **`create_ticket` 永不重试**;本章起这是**由 `kind == "write"` 推出的结构保证**。

### 建库顺序

- **`init_db.py` 永不加列。** 带 `db/chNN.sql` 的章都必须在 `init_db` 之外**再执行那份 DDL**。
  本章的 `db/ch08.sql` 同样**不带 `IF NOT EXISTS`**(重复执行要响亮失败)。

---

## File Structure

**新建**

| 路径 | 职责 |
|---|---|
| `app/tools/mock_data.py` | 订单/商品/物流的**唯一 mock 真相源**。内置工具与两个 MCP Server **共用**(T1) |
| `app/tools/spec.py` | `ToolSpec` 数据类 + `validate_args()`(唯一的 JSON Schema 校验器)(T2) |
| `app/tools/policy.py` | 本地权限声明:`工具名 → read\|write` 的**唯一真相源**(T2) |
| `app/tools/builtin/__init__.py` | `discover()` —— `pkgutil` 走遍子模块收集 `SPECS`(T3) |
| `app/tools/builtin/orders.py` | `query_order` / `query_product`(T3) |
| `app/tools/builtin/knowledge.py` | `make_query_faq`(T3) |
| `app/tools/builtin/tickets.py` | `make_create_ticket`(`kind="write"`)(T3) |
| `app/tools/audit.py` | `record_audit()` —— 审计的**唯一写口**(T5) |
| `mcp_servers/__init__.py` | 空包(T6) |
| `mcp_servers/logistics.py` | 物流 MCP Server:`query_logistics`(T6) |
| `mcp_servers/aftersales.py` | 售后 MCP Server:`query_warranty` / `query_return_progress`(T6) |
| `app/mcp/__init__.py` | 空包(T7) |
| `app/mcp/client.py` | `discover_mcp_tools()` —— 每请求发现 + 单 Server 降级(T7) |
| `app/agent/confirm_nodes.py` | `confirm_write`(只有 interrupt)/ `apply_write_decision`(T8) |
| `db/ch08.sql` | `tool_audit_logs`(T5) |
| `scripts/acceptance_ch08.sh` | 端到端验收 1–6(T11) |

**修改**

| 路径 | 改什么 |
|---|---|
| `requirements.txt` | 三个新依赖(T1) |
| `app/tools/business.py` | 先瘦身成 mock 数据源的转发(T1),**T3 删除** |
| `app/agent/refund_nodes.py` | `_order_record` 的 import 落点(T1);`registry` 的类型(T4) |
| `app/tools/registry.py` | `build_registry()` 产出 `dict[str, ToolSpec]`(T3、T7) |
| `app/tools/executor.py` | 权限闸 + 校验前置 + 六类分诊 + 结果格式化 + 重试规则(T4、T5) |
| `app/db/models.py` | `ToolAuditLog` 只做映射(T5) |
| `app/agent/state.py` | 两个新通道(T8) |
| `app/agent/nodes.py` | 每轮清零 + `agent` 停循环 / 续跑(T8、T9) |
| `app/agent/graph.py` | 两个新节点 + 条件边(T9) |
| `app/api/chat.py` | 注册表构造 + 把两个上下文通道播种(T9) |
| `app/static/index.html` | 工单预览卡片(T10) |
| `app/config.py` | MCP 三项 + 重试默认值 1→2(T11) |
| `CLAUDE.md` / `AGENTS.md` | 章级同步(T12) |

**被删除**

| 路径 | 何时 | 为什么 |
|---|---|---|
| `app/tools/business.py` | T3 末尾 | 五个工具全部迁进 `builtin/`,只留一处登记机制 |

---

### Task 1: 依赖与 mock 数据源的抽取

**Files:**
- Modify: `requirements.txt`
- Create: `app/tools/mock_data.py`
- Modify: `app/tools/business.py`(改成从 `mock_data` 取数据)
- Modify: `app/agent/refund_nodes.py:55` 与同处注释
- Modify: `tests/test_api_chat.py:36`
- Test: `tests/test_mock_data.py`(新建)

**Interfaces:**
- Consumes: 无(第一个任务)
- Produces:
  - `mock_data.rng(*parts: str) -> random.Random`
  - `mock_data.ECHO_LIMIT: int = 32`
  - `mock_data.require_order_no(order_id: str) -> str`
  - `mock_data.order_record(order_no: str) -> dict`
  - `mock_data.ORDER_STATUS: list[str]` / `PRODUCT_NAMES` / `PRODUCT_SPECS` / `CITIES`
  - `mock_data.LOGISTICS_BY_STATUS: dict[str, tuple[str, ...]]`

**为什么先做这个**:`query_logistics` 在 T7 要搬进 **独立进程** 的 MCP Server,
而 ch02 已确立「`_order_record()` 是订单的**唯一真相源**」。数据源不先抽出来共用,
独立 Server 就会自带一套随机数 ⇒ **同一个订单号,内置 `query_order` 与 MCP
`query_logistics` 会说两套话**(订单说「已发货」,物流说「待付款」)。

- [ ] **Step 1: 写失败测试**

新建 `tests/test_mock_data.py`:

```python
"""mock 数据源的抽取与**跨进程稳定性**。

跨进程那条是本文件的重点:内置工具与两个 MCP Server 是**三个不同的进程**,
种子实现只要有一点不同,同一订单号就会在三处得到不同数据。
"""

import json
import subprocess
import sys

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
    """
    code = (
        "import json, sys;"
        "sys.path.insert(0, r'.');"
        "from app.tools.mock_data import order_record;"
        "sys.stdout.buffer.write("
        "json.dumps(order_record('1002'), ensure_ascii=False).encode('utf-8'))"
    )
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", code],
        capture_output=True,
        cwd=".",
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    assert json.loads(proc.stdout.decode("utf-8")) == order_record("1002")
```

- [ ] **Step 2: 跑测试,确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_mock_data.py`
Expected: FAIL —— `ModuleNotFoundError: No module named 'app.tools.mock_data'`

- [ ] **Step 3: 新建 `app/tools/mock_data.py`**

把 `app/tools/business.py` 里第 21–163 行那批**数据源**整体搬过来,
**私有名改成公开名**(它现在是被三个进程共用的接口,不再是模块内部实现)
—— 但 **`_rng` 的抽取顺序一个字节都不许动**,改动会改变每个订单号的具体取值。

```python
"""订单 / 商品 / 物流的 **mock 唯一真相源**。

**谁在用**:内置工具(`app/tools/builtin/`)与两个业务 MCP Server
(`mcp_servers/`)。后两者是**独立进程** —— 这正是本模块存在的理由:
数据源不共用的话,同一个订单号在内置 `query_order` 与 MCP `query_logistics`
之间会**说两套话**(订单说「已发货」、物流说「待付款」),演示时一眼穿帮。

**不接真实系统、不建表** —— 全部由入参确定性派生,同一入参永远得到同样结果,
所以验收可以写**会失败的**断言。
"""

import hashlib
import random
from datetime import datetime, timedelta


def rng(*parts: str) -> random.Random:
    """由入参派生稳定种子。

    **绝不能用内置 hash()** —— 它对 str 每进程随机化(PYTHONHASHSEED),
    会让"同一订单号永远返回同样数据"在进程重启后失效,而同进程内的
    测试完全测不出来。sha256 跨进程、跨平台稳定。
    """
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest, "big"))


#: 回显给模型的入参最多截这么长 —— 模型给的输入不受我们控制,原样回灌
#: 等于让它自己决定往上下文里塞多少 token。
ECHO_LIMIT = 32


def require_order_no(order_id: str) -> str:
    """订单号须为 4-32 位 ASCII 数字。不符合视为查无此单,而不是编一个结果。

    必须是 `isascii() and isdigit()` 两个条件:单独一个 `isdigit()` 是
    Unicode 感知的,`"١٢٣٤".isdigit()`(阿拉伯-印度数字)与 `"²²²²".isdigit()`
    (上标)都为 True —— 这类输入会**通过**校验并拿到一张凭空编造的订单,
    而不是 ToolNotFound。
    """
    cleaned = order_id.strip()
    if not (cleaned.isascii() and cleaned.isdigit()) or not (4 <= len(cleaned) <= 32):
        raise ToolNotFound(f"未找到订单 {cleaned[:ECHO_LIMIT]},请核对订单号后重试")
    return cleaned


ORDER_STATUS = ["待付款", "已付款", "已发货", "已完成", "已取消"]
PRODUCT_NAMES = ["无线耳机", "运动鞋", "双肩包", "保温杯", "机械键盘"]
PRODUCT_SPECS = ["标准版", "Pro 版", "家用款", "经典款"]
CITIES = ["广州分拨中心", "上海分拨中心", "北京分拨中心", "成都分拨中心"]

#: 订单状态 → 该状态下**可能**出现的物流状态。
#:
#: 这是本模块唯一的「状态耦合」定义:物流状态不是自己抽的,而是从订单状态
#: 派生出的候选里抽。反向的那半同样重要 ——「待付款 / 已付款 / 已取消」不在
#: 表里,没发货的单子就是**没有**物流记录,查物流应当查不到,而不是编一条出来。
LOGISTICS_BY_STATUS = {
    "已发货": ("已揽件", "运输中", "派送中"),
    "已完成": ("已签收",),
}


def order_record(order_no: str) -> dict:
    """订单的唯一真相源 —— `query_order` 与 `query_logistics` 都必须经它取值。

    两个工具各自 `rng(不同前缀, 同一订单号)` 是本模块最容易犯的错:那是
    **两条相互独立**的随机流,于是同一个订单可以同时是「已取消」和「已签收」。
    实测 1000 个订单里 807 个状态矛盾、2000 个里 217 个轨迹早于下单时间。
    共用同一条记录之后,这类矛盾在结构上不可能出现。

    **抽取顺序不可改动** —— 改动会改变每个订单号的具体取值。
    """
    r = rng("order", order_no)
    return {
        "order_id": order_no,
        "status": r.choice(ORDER_STATUS),
        "product": r.choice(PRODUCT_NAMES),
        "amount": f"{r.randint(49, 999)}.{r.randint(0, 99):02d}",
        "created_at": (
            f"2026-{r.randint(1, 9):02d}-{r.randint(10, 28):02d} "
            f"{r.randint(9, 21):02d}:{r.randint(0, 59):02d}"
        ),
    }


def logistics_record(order_no: str) -> dict:
    """物流轨迹。**必须经 `order_record` 取状态**(见它的 docstring)。"""
    order = order_record(order_no)
    candidates = LOGISTICS_BY_STATUS.get(order["status"])
    if candidates is None:
        # 未发货的单子**没有**物流记录 —— 这是"查无此物",不是上游故障,
        # 所以走 ToolNotFound(可恢复),不是 ToolInfrastructureError。
        raise ToolNotFound(
            f"订单 {order_no} 当前状态是「{order['status']}」,尚未发货、没有物流记录,"
            f"请如实告知用户,不要自行编造物流信息"
        )

    r = rng("logistics", order_no)
    status = r.choice(candidates)
    city = r.choice(CITIES)
    # 轨迹时间必须**从下单时间往后推**。另起一条随机流去抽 2026-09-xx 会得到
    # 早于下单的「已发出」时间 —— 那是与状态矛盾同一类的自相矛盾,实测 2000 个
    # 订单里 217 个中招。
    shipped = datetime.strptime(order["created_at"], "%Y-%m-%d %H:%M") + timedelta(
        days=r.randint(1, 3), hours=r.randint(1, 20)
    )
    # 末条轨迹必须带**真实时间戳**并描述当前状态,不能写成 {"time": "当前"}:
    # 那样整条轨迹无法排序,模型读到的是"最后一次扫描停在『已发出』",于是
    # status 为「已签收」时它会当场指出"两者信息不太一致"并追问用户是否收到货
    # —— 验收 4 的真实回复就是这么写的,演示看起来像坏了。
    latest = shipped + timedelta(days=r.randint(1, 4), hours=r.randint(1, 12))
    fmt = "%Y-%m-%d %H:%M"
    return {
        "order_id": order_no,
        "status": status,
        "location": city,
        "traces": [
            {"time": shipped.strftime(fmt), "desc": f"{city} 已发出"},
            {"time": latest.strftime(fmt), "desc": f"{city} {status}"},
        ],
    }
```

**同时**在文件顶部补上它自己的 import:

```python
from app.tools.errors import ToolNotFound
```

> ⚠️ `logistics_record` 是本任务**新增**的一个函数(原来这段逻辑写在
> `query_logistics` 工具体里)。搬出来的理由和 `order_record` 一样:
> T7 之后**只有 MCP Server 需要它**,而 Server 不该自己重写一遍。
> 注意新函数里 `r = rng("logistics", order_no)` **仍然在 `candidates` 判空之后** ——
> 保持与原文**完全相同的抽取顺序**,否则每个订单号的物流数据都会变。

- [ ] **Step 4: 把 `business.py` 改成从 `mock_data` 取**

`app/tools/business.py` 里删掉搬走的那批定义,改成 import;
`query_order` / `query_product` / `query_logistics` 三个工具的**函数体一字不改**,
只把 `_rng(...)` → `rng(...)`、`_order_record(...)` → `order_record(...)`、
`_require_order_no(...)` → `require_order_no(...)`、`_ECHO_LIMIT` → `ECHO_LIMIT`。

`query_logistics` 的工具体**保持原样**(它仍定义在 `business.py`,T3 才搬走),
只把里面对数据源的两处调用换成新名字。

- [ ] **Step 5: 更新两个调用点**

`app/agent/refund_nodes.py`:

```python
# 原:from app.tools.business import _order_record
from app.tools.mock_data import order_record
```

并把该文件里 `_order_record(` 的调用与第 153 行注释里的
`app/tools/business.py` 的 `_order_record`(**私有**,同模块的…)改成新落点
—— **注释也要改**:这份代码库最贵的一类缺陷就是「注释还指着已经搬走的东西」。

`tests/test_api_chat.py`:

```python
# 原:from app.tools.business import _order_record
from app.tools.mock_data import order_record
```

同文件里对 `_order_record(` 的调用一并改名。

- [ ] **Step 6: 三个依赖加进 `requirements.txt`**

在文件末尾追加(顺序无所谓,但**版本要照抄**):

```
mcp>=1.24,<2
langchain-mcp-adapters==0.3.2
jsonschema==4.26.0
```

装:

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -c "import mcp, langchain_mcp_adapters, jsonschema; \
import mcp.server as s; print(s.FastMCP, langchain_mcp_adapters.__name__)"
```

Expected: 打印出 `<class 'mcp.server.fastmcp.server.FastMCP'> langchain_mcp_adapters`
—— **不是** `AttributeError: module 'mcp.server' has no attribute 'FastMCP'`。
看到后者就是装成了 mcp 2.x,回退到 `mcp>=1.24,<2`。

- [ ] **Step 7: 跑全量测试**

Run: `.venv/Scripts/python.exe -m pytest`
Expected: 全绿,通过数 ≥ 607(新增了 `test_mock_data.py` 的若干条)。
**任何既有测试变红都说明搬迁改动了行为** —— 最可能是抽取顺序被动了,
回去逐字比对 `order_record` 与 `logistics_record` 的抽取次序。

- [ ] **Step 8: 提交**

```bash
git add requirements.txt app/tools/mock_data.py app/tools/business.py \
        app/agent/refund_nodes.py tests/test_api_chat.py tests/test_mock_data.py
git commit -m "refactor(ch08): 抽出 mock 数据源 mock_data.py —— 三进程共用的唯一真相源"
```

---

### Task 2: `ToolSpec` / 权限策略 / JSON Schema 校验器

**Files:**
- Create: `app/tools/spec.py`
- Create: `app/tools/policy.py`
- Test: `tests/test_tool_spec.py`(新建)

**Interfaces:**
- Consumes: 无
- Produces:
  - `spec.ToolSpec`(frozen dataclass:`name` / `description` / `input_schema` /
    `kind` / `source` / `tool`)
  - `spec.validate_args(spec: ToolSpec, args: dict) -> list[str]`
  - `spec.READ: str = "read"` / `spec.WRITE: str = "write"`
  - `policy.kind_of(tool_name: str) -> str` —— 本地声明表是唯一真相源;
    **未声明的名字一律返回 `READ`**
  - `policy.WRITE_TOOLS: frozenset[str]`(`{"create_ticket"}`)

**为什么这条要单独一个任务**:它是本章**唯一的纯函数层**,没有 IO、没有图、
没有网络 —— 校验闸与权限判定的**全部判别力**都在这里,单独一个评审关口最划算。

- [ ] **Step 1: 写失败测试**

新建 `tests/test_tool_spec.py`:

```python
"""`ToolSpec` 与统一校验器 —— 本章唯一的纯函数层。

⚠️ 本文件的断言**不许**写成「抛了异常就算过」:校验闸的价值在于
**回灌给模型的文案点名到字段**,一个只会返回空列表的实现必须让这里变红。
"""

import pytest

from app.tools.policy import WRITE_TOOLS, kind_of
from app.tools.spec import READ, WRITE, ToolSpec, validate_args


class _FakeTool:
    """只当占位 —— `validate_args` 不碰 `tool`,不必是 BaseTool。"""


def _spec(schema: dict, *, name: str = "demo", kind: str = READ) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="测试用",
        input_schema=schema,
        kind=kind,
        source="builtin",
        tool=_FakeTool(),
    )


SCHEMA = {
    "type": "object",
    "properties": {
        "order_id": {"type": "string"},
        "quantity": {"type": "integer", "minimum": 1},
        "channel": {"type": "string", "enum": ["web", "app"]},
    },
    "required": ["order_id"],
}


def test_valid_args_pass():
    assert validate_args(_spec(SCHEMA), {"order_id": "1002"}) == []


def test_missing_required_names_the_field():
    """断言的是**字段名出现在文案里**,不是「列表非空」。"""
    problems = validate_args(_spec(SCHEMA), {})
    assert problems
    assert any("order_id" in p for p in problems)


def test_wrong_type_is_reported():
    """⚠️ 初稿只写了 `assert problems` —— **对 `type` 分支零判别力**
    (实现者的常量探针证实:把 `_readable` 换成常量它照样绿)。
    与本文件头一条 docstring 的说法自相矛盾,已订正为字段级断言。
    """
    problems = validate_args(_spec(SCHEMA), {"order_id": 1002})
    assert problems
    assert any("order_id" in p for p in problems)


def test_below_minimum_is_reported():
    """**这条是判别力最强的一条**:只看必填与类型的实现会在这里变红。

    spec §3.3 明确要求 MCP 的原始 `inputSchema` **不经过 pydantic 转换** ——
    正是因为转换会把 `minimum` / `enum` 这类约束丢掉,闸就形同虚设。
    """
    problems = validate_args(_spec(SCHEMA), {"order_id": "1002", "quantity": 0})
    assert problems
    assert any("quantity" in p for p in problems)


def test_value_outside_enum_is_reported():
    problems = validate_args(
        _spec(SCHEMA), {"order_id": "1002", "channel": "fax"}
    )
    assert problems
    assert any("channel" in p for p in problems)


def test_problems_are_human_readable_not_pydantic_dumps():
    """文案是**给模型看**的:不许出现 pydantic 的堆栈式原文。"""
    problems = validate_args(_spec(SCHEMA), {})
    joined = " ".join(problems)
    for noise in ("Traceback", "pydantic", "validation error", "1 validation"):
        assert noise not in joined


def test_empty_schema_accepts_anything():
    """空 schema 是合法 JSON Schema(等价于「什么参数都行」),不是「校验失败」。"""
    assert validate_args(_spec({}), {"whatever": 1}) == []


def test_unexpected_extra_field_is_allowed():
    """**刻意不禁止**多余字段。

    JSON Schema 的默认语义就是「额外的键不校验」;加了
    `additionalProperties: false` 的话,模型多传一个它自己编的字段
    就会被拦下 —— 而那是个**无害**的行为,拦它只会白白浪费一轮对话。
    """
    assert validate_args(_spec(SCHEMA), {"order_id": "1", "extra": "x"}) == []


# ---- 权限策略 ----------------------------------------------------------


def test_declared_write_tool_is_write():
    assert kind_of("create_ticket") == WRITE


@pytest.mark.parametrize(
    "name", ["query_order", "query_product", "query_logistics", "query_faq"]
)
def test_declared_read_tools_are_read(name):
    assert kind_of(name) == READ


def test_undeclared_tool_defaults_to_read():
    """spec §5.2(**用户 2026-09-22 拍板**):未知 MCP 工具默认**只读**。

    这是验收 3 成立的前提 —— 在 Server 侧加工具、只重启该 Server,
    客服系统这边代码不动、服务不重启就能用上。默认拒绝的话,
    新工具还要回来加一行声明,直接与需求 1 / 验收 3 冲突。

    同时它也是安全的:**写只认我们本地的表**,外部 Server 无法靠改自己的
    用途声明拿到写权限。
    """
    assert kind_of("some_tool_we_never_declared") == READ
    assert kind_of("mcp__logistics__anything") == READ


def test_write_tools_is_exactly_create_ticket():
    """这份集合是**写操作的全集**,不是「已知写操作的一部分」。

    断言写成精确相等而不是 `in`:将来往表里加东西必须**显式改这条测试**,
    而 `assert "create_ticket" in WRITE_TOOLS` 对新增的写工具完全无感。
    """
    assert WRITE_TOOLS == frozenset({"create_ticket"})
```

- [ ] **Step 2: 跑测试,确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tool_spec.py`
Expected: FAIL —— `ModuleNotFoundError: No module named 'app.tools.spec'`

- [ ] **Step 3: 写 `app/tools/policy.py`**

```python
"""工具权限的**本地**声明表 —— 能不能调只认这份表。

**刻意不看 MCP Server 的用途声明**:那是对方自己写的,不可信。
**也刻意不让模型临场判断**:模型只负责「调不调」,不负责「能不能调」。

未声明的工具名一律按**只读**放行(spec §5.2,用户 2026-09-22 拍板):

- 它是验收 3 成立的前提 —— 在 Server 侧加工具、只重启该 Server,
  客服系统这边代码不动、服务不重启就能用上;
- 同时它是安全的 —— **写只认这份表**,外部 Server 无法靠改自己的
  用途声明拿到写权限。
"""

#: 写操作的全集。**今天只有一条。**
#:
#: ⚠️ 测试 `test_write_tools_is_exactly_create_ticket` 断言的是**精确相等**,
#: 加一条就必须去改那条测试 —— 这是刻意的:**多一个写工具是个需要被看见的决定**。
WRITE_TOOLS = frozenset({"create_ticket"})


def kind_of(tool_name: str) -> str:
    """工具名 → `"read"` / `"write"`。**未声明 = 只读**(见模块 docstring)。"""
    from app.tools.spec import READ, WRITE

    return WRITE if tool_name in WRITE_TOOLS else READ
```

- [ ] **Step 4: 写 `app/tools/spec.py`**

```python
"""`ToolSpec` —— 一条工具登记项 + 全章**唯一**的 JSON Schema 校验器。

内置与 MCP 来的工具**一视同仁**:名 / 用途描述 / JSON Schema 参数定义三样齐备。
"""

from dataclasses import dataclass

import jsonschema

READ = "read"
WRITE = "write"


@dataclass(frozen=True)
class ToolSpec:
    """一条工具登记项。

    `input_schema` 是**原始 JSON Schema**,不是 pydantic 模型:
    - 内置那份从 `tool.args_schema.model_json_schema()` 派生;
    - MCP 那份从 Server 的 `inputSchema` **原样取**(spec §3.3)。

    后者刻意**不经 adapters 的 pydantic 转换** —— 转换会削平
    `minimum` / `maxLength` / `enum` 这类约束,于是「统一按 JSON Schema 校验」
    退化成「只查必填和类型」,闸看起来在工作、实际漏掉一半。

    `tool` 是绑给模型的那份 `BaseTool`;执行时才用它。
    """

    name: str
    description: str
    input_schema: dict
    kind: str          # READ | WRITE
    source: str        # "builtin" | "mcp:logistics" | "mcp:aftersales"
    tool: object       # BaseTool(此处不 import LangChain,见 app/tools/registry.py)


def _readable(error: jsonschema.ValidationError) -> str:
    """一条 `ValidationError` → **给模型看**的一句话。

    ⚠️ 文案是给**模型**看的,不是给人看的调试信息:它会被当作工具结果回灌,
    模型据此**追问用户**或**重新组织调用**。所以:
    - 点名到字段(`order_id`),不要 `$['order_id']` 这种取值路径;
    - 说清楚**为什么**(必填未提供 / 类型不对 / 取值越界);
    - **不带** pydantic 或 jsonschema 的堆栈式原文 ——
      `test_problems_are_human_readable_not_pydantic_dumps` 钉着这条。
    """
    path = ".".join(str(p) for p in error.absolute_path) or "(根)"
    if error.validator == "required":
        missing = error.message.split("'")[1] if "'" in error.message else error.message
        return f"缺少必填字段 `{missing}`"
    if error.validator == "type":
        return f"`{path}` 类型不对:{error.message}"
    if error.validator == "enum":
        allowed = "、".join(str(v) for v in error.validator_value)
        return f"`{path}` 取值必须是 {allowed} 之一"
    if error.validator in ("minimum", "maximum", "minLength", "maxLength"):
        return f"`{path}` 超出取值范围:{error.message}"
    return f"`{path}` 不合法:{error.message}"


def validate_args(spec: ToolSpec, args: dict) -> list[str]:
    """按 `spec.input_schema` 校验 `args`。返回**给模型看**的问题列表,空 = 通过。

    **空 schema 是合法的**(等价于「什么参数都行」),不是「校验失败」——
    `jsonschema.validate` 对 `{}` 一律放行,本函数不额外加限制。

    **刻意不禁止多余字段**:JSON Schema 的默认语义就是「额外的键不校验」。
    加 `additionalProperties: false` 的话,模型多传一个它自己编的字段就会被
    拦下,而那是个**无害**行为,拦它只会白白浪费一轮对话。

    ⚠️ **不抛异常**:校验失败是**可恢复**的,由执行器包成一条工具结果回灌给模型。
    抛异常会让它走 `except Exception` → `ToolInfrastructureError` → 502,
    把「模型参数写错了」伪装成「服务挂了」。
    """
    if not spec.input_schema:
        return []
    validator_cls = jsonschema.validators.validator_for(spec.input_schema)
    try:
        validator_cls.check_schema(spec.input_schema)
    except jsonschema.SchemaError:
        # schema 本身是坏的(我们的 bug,不是模型的):**响亮地抛**,
        # 由执行器归到基础设施那一支。静默返回 [] 等于把闸关掉而没人知道。
        raise
    validator = validator_cls(spec.input_schema)
    return [_readable(e) for e in validator.iter_errors(args)]
```

- [ ] **Step 5: 跑测试,确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tool_spec.py`
Expected: PASS(全绿)

- [ ] **Step 6: 提交**

```bash
git add app/tools/spec.py app/tools/policy.py tests/test_tool_spec.py
git commit -m "feat(ch08): ToolSpec + 本地权限声明表 + 统一 JSON Schema 校验器"
```

---

### Task 3: 内置工具包(自动发现)+ 注册表改造

**Files:**
- Create: `app/tools/builtin/__init__.py`
- Create: `app/tools/builtin/orders.py`
- Create: `app/tools/builtin/knowledge.py`
- Create: `app/tools/builtin/tickets.py`
- Delete: `app/tools/business.py`
- Modify: `app/tools/registry.py`
- Modify: `tests/test_tools_random.py`(两处 import,含第 94 行的子进程字符串)
- Modify: `tests/test_tools_db.py:11`
- Modify: `tests/test_tools_query_faq.py:19`
- Modify: `tests/test_registry.py`
- Modify: `app/agent/graph.py:5` 与 `app/agent/nodes.py:3` 的注释落点
- Test: `tests/test_builtin_discovery.py`(新建)

**Interfaces:**
- Consumes: `spec.ToolSpec` / `spec.validate_args`(T2);`mock_data.*`(T1)
- Produces:
  - 每个 `builtin/` 模块导出 `build(*, session, conversation_id, retriever) -> list[BaseTool]`
  - `builtin.discover(*, session, conversation_id, retriever) -> list[BaseTool]`
  - `registry.build_registry(*, session, conversation_id, settings) -> dict[str, ToolSpec]`
  - `registry.build_tools(*, session, conversation_id, settings) -> list[BaseTool]`(**投影**,给评估脚本)
  - `registry.registry_for(tools) -> dict[str, BaseTool]`(不变)
  - `registry.build_retriever(session) -> KnowledgeRetriever`(不变)

**验收 1 的直接依据就是本任务**:新写一个工具 = 在本包内**新增一个文件**。

> ⚠️ **`retriever` 为什么要一路传下来**:`make_query_faq` 需要它,而
> `build_retriever` 住在 `registry.py` 里。让 `builtin/knowledge.py` 自己去 import
> `registry` 会**成环**(registry → builtin → registry)。
> 所以由 registry **先造 retriever、再喂给 discover** —— 静态工具忽略这个参数即可。

- [ ] **Step 1: 写失败测试**

新建 `tests/test_builtin_discovery.py`:

```python
"""内置工具的**自动发现** + 注册表的顺序与去重。

验收 1 是「新写一个简单工具,只做注册动作、不动核心代码,Agent 就能用上」——
本文件是它的单测版:**动态造一个 builtin 模块,断言它自己进了表**。
"""

import sys
import textwrap
from pathlib import Path

import pytest

from app.tools import builtin
from app.tools.policy import WRITE, kind_of
from app.tools.registry import build_registry


class _FakeRetriever:
    async def search(self, keyword):  # pragma: no cover - 本文件用不到
        return []


def _registry(session=None):
    return build_registry(
        session=session, conversation_id="c1", settings=None,
    )


def test_five_builtin_tools_are_registered():
    reg = _registry()
    assert set(reg) == {
        "query_order",
        "query_product",
        "query_logistics",
        "query_faq",
        "create_ticket",
    }


def test_query_logistics_is_still_builtin_before_t7():
    """⚠️ **T7 会把这条测试删掉** —— 那时 `query_logistics` 已搬进物流 MCP Server。

    留它的理由:本任务结束时它必须还在内置里,否则 T3 与 T7 之间
    「物流查询」会有一段**谁都提供不了**的空窗,而验收 2 的题面在 T7 之前
    就已经被人手动跑过。
    """
    assert _registry()["query_logistics"].source == "builtin"


def test_every_spec_carries_the_three_things():
    """名 / 用途描述 / JSON Schema —— 注册中心的要求就是这三样齐备。"""
    for name, spec in _registry().items():
        assert spec.name == name
        assert spec.description.strip()
        assert isinstance(spec.input_schema, dict) and spec.input_schema


def test_input_schema_declares_the_parameters():
    """**判别力所在**:空 schema 的实现会让这条变红。"""
    props = _registry()["query_order"].input_schema["properties"]
    assert "order_id" in props


def test_write_kind_comes_from_the_policy_table():
    """`kind` 由**策略表**给,不由模块自己声明 —— 一处真相源。"""
    reg = _registry()
    assert reg["create_ticket"].kind == WRITE
    assert reg["query_order"].kind != WRITE
    for name, spec in reg.items():
        assert spec.kind == kind_of(name)


def test_order_is_stable_across_two_builds():
    """顺序稳定 = 工具定义块逐字节相同 = 前缀缓存命中(spec §3.4)。"""
    assert list(_registry()) == list(_registry())


def test_new_module_is_picked_up_without_touching_core_code(tmp_path, monkeypatch):
    """**验收 1 的单测版。**

    往 `app/tools/builtin/` 里丢一个模块(不碰任何既有文件),再 build 一次,
    新工具必须已经在表里。
    """
    pkg_dir = Path(builtin.__file__).parent
    new_module = pkg_dir / "zz_scratch_probe.py"
    new_module.write_text(
        textwrap.dedent(
            '''
            """临时探测模块 —— 本测试自己不碰任何核心代码。"""

            from langchain.tools import tool


            @tool
            async def echo_probe(text: str) -> str:
                """把入参原样回显。测试用。"""
                return text


            def build(*, session, conversation_id, retriever):
                return [echo_probe]
            '''
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "modules",
        {k: v for k, v in sys.modules.items() if "zz_scratch_probe" not in k},
    )
    try:
        reg = _registry()
        assert "echo_probe" in reg, "新增 builtin 模块没有被自动发现"
        assert reg["echo_probe"].description.strip()
    finally:
        new_module.unlink()
        sys.modules.pop("app.tools.builtin.zz_scratch_probe", None)


def test_duplicate_tool_name_raises_loudly(monkeypatch):
    """重名**必须响亮地失败**。

    静默的去重会让「其中一个胜出」,而两个实现谁胜出取决于排序 ——
    表现是「工具偶尔返回另一种数据」,没人查得出来。
    """
    import app.tools.builtin.orders as orders

    original = orders.build

    def duplicated(*, session, conversation_id, retriever):
        return original(
            session=session, conversation_id=conversation_id, retriever=retriever
        ) + [original(
            session=session, conversation_id=conversation_id, retriever=retriever
        )[0]]

    monkeypatch.setattr(orders, "build", duplicated)
    # ⚠️ **`match` 里是 `query_order` 不是 `query_product`** —— 初稿写错了。
    # `duplicated` 复制的是 `original(...)[0]`,而 `orders.build()` 返回的**第一个**
    # 是 `query_order`(build 的返回顺序,不是排序后的顺序)。照初稿写这条测试
    # **永远不可能通过**,而它看起来只是「断言写得具体一点而已」。
    with pytest.raises(ValueError, match="query_order"):
        _registry()
```

- [ ] **Step 2: 跑测试,确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_builtin_discovery.py`
Expected: FAIL —— `ModuleNotFoundError: No module named 'app.tools.builtin'`

- [ ] **Step 3: 建 `app/tools/builtin/__init__.py`**

```python
"""内置工具包 —— **包内自动发现**。

**新写一个工具 = 在本包内新增一个模块,核心代码零改动**(验收 1)。

每个模块导出同一个接口:

    def build(*, session, conversation_id, retriever) -> list[BaseTool]

静态工具(不需要会话的)忽略后三个参数即可 —— 接口统一比省那几个字值钱:
`discover()` 不必分辨模块「是工厂还是常量」。

⚠️ 内置工具是 `import` 进来的,**新增内置工具需要重启客服服务**。
MCP 工具那条路是**每请求现问现拿**,不需要重启(验收 3)。两句不矛盾,是两条通道。
"""

import importlib
import pkgutil

from langchain_core.tools import BaseTool


def discover(*, session, conversation_id, retriever) -> list[BaseTool]:
    """走遍本包的所有子模块,收集它们导出的工具。

    顺序按 `(模块名, 工具名)` 排序 —— **稳定**是硬要求:工具定义块每轮都要
    逐字节相同,否则前缀缓存整段作废(spec §3.4)。
    """
    found: list[tuple[str, BaseTool]] = []
    for info in pkgutil.iter_modules(__path__):
        module = importlib.import_module(f"{__name__}.{info.name}")
        for tool in module.build(
            session=session, conversation_id=conversation_id, retriever=retriever
        ):
            found.append((info.name, tool))
    found.sort(key=lambda pair: (pair[0], pair[1].name))
    return [tool for _, tool in found]
```

- [ ] **Step 4: 建 `app/tools/builtin/orders.py`**

把 `business.py` 的 `query_order` / `query_product` / `query_logistics` **原样搬过来**
(只把数据源的名字换成 `mock_data` 的)。文件头:

```python
"""订单 / 商品 / 物流三个「假装有上游系统」的只读工具。

数据全部来自 `app.tools.mock_data`(**唯一真相源**,与两个 MCP Server 共用)。

⚠️ `query_logistics` **暂时**还在这里 —— T7 会把它删掉,由物流 MCP Server 接管。
"""

import json

from langchain.tools import tool

from app.tools.errors import ToolNotFound
from app.tools.mock_data import (
    ECHO_LIMIT,
    LOGISTICS_BY_STATUS,
    ORDER_STATUS,
    PRODUCT_NAMES,
    PRODUCT_SPECS,
    city_choices,
)
from app.tools.mock_data import logistics_record, order_record, require_order_no, rng
```

> 上面这串 import **照你实际搬过去的内容写** —— 用不到的不要引,
> 用到的一个都不要漏(本仓的 lint 会报未使用的 import)。

函数体照抄,只改名字。`query_logistics` 的体**缩减成一行**:

```python
@tool
async def query_logistics(order_id: str) -> str:
    """查询订单的物流状态、当前位置与轨迹。用户问"到哪了""发货没"时使用。"""
    return json.dumps(
        logistics_record(require_order_no(order_id)), ensure_ascii=False
    )
```

(原来的状态判空、时间戳推导已经全部搬进 `mock_data.logistics_record` —— T1 做的。)

文件末尾:

```python
def build(*, session, conversation_id, retriever):
    return [query_order, query_product, query_logistics]
```

- [ ] **Step 5: 建 `app/tools/builtin/knowledge.py`**

把 `business.py` 的 `make_query_faq` 与 `FAQ_LIMIT` 原样搬过来(函数体一字不改),
文件末尾:

```python
def build(*, session, conversation_id, retriever):
    return [make_query_faq(session, retriever)]
```

- [ ] **Step 6: 建 `app/tools/builtin/tickets.py`**

把 `business.py` 的 `make_create_ticket` 原样搬过来(**含那条关于列宽 `[:64]`
的注释** —— 它解释的是一个真实的截断决定,别在搬迁里丢掉),文件末尾:

```python
def build(*, session, conversation_id, retriever):
    return [make_create_ticket(session, conversation_id)]
```

**docstring 一字不动。** ⚠️ 原计划在这里让实现者把「重试白名单」那段注释改写成
「本章起由 `kind` 结构性推出」—— **那是 T4 的事**:此刻 `executor.py` 里
`RETRYABLE_TOOLS` 还在,注释会**描述一个不存在的机制**。注释与代码的改写必须在
**同一个提交**里落地,所以这条挪到 T4 Step 5(见那里)。

- [ ] **Step 7: 改写 `app/tools/registry.py`**

```python
"""工具注册表。

因 `query_faq` / `create_ticket` 需要每请求构造(见 `builtin/knowledge.py`
与 `builtin/tickets.py` 的说明),注册表不是纯模块级常量 ——
每个请求 `build_registry` 组装自己那份 `name → ToolSpec`。

**`ToolSpec` 三样齐备**:名 / 用途描述 / **原始 JSON Schema**(spec §3.1)。
"""

from langchain_core.tools import BaseTool

from app.config import get_settings
from app.retrieval.embedder import get_embedder
from app.retrieval.milvus import get_vector_store
from app.retrieval.reranker import get_reranker
from app.retrieval.search import KnowledgeRetriever
from app.tools import builtin
from app.tools.policy import kind_of
from app.tools.spec import ToolSpec


def build_retriever(session) -> KnowledgeRetriever:
    """按配置组装检索器。

    构造**不连 Milvus、不加载 BGE-M3**(两者都是懒加载),所以拿在请求
    路径上建它是安全的;真正的连接/加载发生在第一次 `search`。
    """
    settings = get_settings()
    return KnowledgeRetriever(
        session,
        get_vector_store(settings.milvus_uri, settings.milvus_collection),
        get_embedder(
            settings.embedding_model_path,
            settings.embedding_max_length,
            settings.embedding_batch_size,
        ),
        get_reranker(settings.reranker_model_path),
        top_k=settings.rerank_top_k,
        score_threshold=settings.retrieval_score_threshold,
    )


def _spec_from_tool(tool: BaseTool, *, source: str) -> ToolSpec:
    """`BaseTool` → 一条登记项。

    `kind` 由 `app/tools/policy.py` 给,**不由工具自己声明** ——
    一处真相源:外部 MCP 工具的用途声明是对方写的、不可信,
    「写」只认我们本地的表。
    """
    schema = tool.args_schema.model_json_schema() if tool.args_schema else {}
    return ToolSpec(
        name=tool.name,
        description=(tool.description or "").strip(),
        input_schema=schema,
        kind=kind_of(tool.name),
        source=source,
        tool=tool,
    )


def _dedupe(specs: list[ToolSpec]) -> dict[str, ToolSpec]:
    """建表,并处理重名 —— **两条规则,刻意不同**。

    - **内置 vs 内置 重名 ⇒ 响亮地抛。** 那是**我们自己的**接线 bug;
      静默去重会让「其中一个胜出」而谁胜出取决于排序 —— 表现是
      「这个工具偶尔返回另一种数据」,没人查得出来。
    - **任何涉及 MCP 的重名 ⇒ 丢掉外部那一个 + 一条响亮的 warn,内置留下。**

    ⚠️ 第二条是 T7 的实现者上报后定的:`specs` 里现在**混进了外部来源的清单**,
    而外部的**名字**和它们的**用途声明**一样不可信 ——
    **外部 Server 只要起一个叫 `query_order` 的工具,`_dedupe` 上抛就会把
    每一个聊天请求打成 500。** 那是验收 3 的反面(在 Server 侧加工具本该
    **不需要动客服系统**),而且外部能让我们的内置工具消失,方向完全错了。
    """

    def _warn(keep: ToolSpec, drop: ToolSpec) -> None:
        logger.warning(
            "工具重名,已丢弃外部来源的那一个:name=%s 保留=%s 丢弃=%s",
            keep.name, keep.source, drop.source,
        )

    out: dict[str, ToolSpec] = {}
    for spec in specs:
        existing = out.get(spec.name)
        if existing is None:
            out[spec.name] = spec
            continue
        if existing.source == "builtin" and spec.source == "builtin":
            raise ValueError(
                f"内置工具重名:{spec.name} —— 我们自己的接线 bug"
            )
        # **按 `source` 判胜负,不按顺序** —— 顺序是 `build_registry` 的实现细节,
        # 而这条规则要的是「内置永远赢」。
        if existing.source == "builtin":
            _warn(existing, spec)
            continue
        if spec.source == "builtin":
            _warn(spec, existing)
            out[spec.name] = spec
            continue
        # 两边都是外部的:先到先得。
        _warn(existing, spec)
    return out


def build_registry(*, session, conversation_id, settings=None) -> dict[str, ToolSpec]:
    """组装本请求的注册表:`name → ToolSpec`。

    `retriever` 在这里造好再喂进 `discover` —— 让 `builtin/knowledge.py`
    自己 import `registry` 会成环(registry → builtin → registry)。
    """
    retriever = build_retriever(session)
    specs = [
        _spec_from_tool(tool, source="builtin")
        for tool in builtin.discover(
            session=session, conversation_id=conversation_id, retriever=retriever
        )
    ]
    return _dedupe(specs)


def build_tools(*, session, conversation_id, settings=None) -> list[BaseTool]:
    """注册表的**投影**:只要绑给模型的那份工具列表。

    给评估脚本(`evals/run_tool_selection_eval.py`)用 —— 它不需要 schema
    也不需要权限,只要能把工具绑到模型上。
    """
    return [
        spec.tool
        for spec in build_registry(
            session=session, conversation_id=conversation_id, settings=settings
        ).values()
    ]


def registry_for(tools: list[BaseTool]) -> dict[str, BaseTool]:
    """建名字到工具的映射。**保留**(既有测试与调用方在用)。"""
    return {tool.name: tool for tool in tools}
```

- [ ] **Step 8: 删 `app/tools/business.py`,改全部 import**

```bash
git rm app/tools/business.py
```

改这几处(逐条,漏一条就是 ImportError):

| 文件 | 原 | 新 |
|---|---|---|
| `tests/test_tools_random.py:12` | `from app.tools.business import query_logistics, query_order, query_product` | `from app.tools.builtin.orders import query_logistics, query_order, query_product` |
| `tests/test_tools_random.py:94`(子进程字符串内) | `from app.tools.business import query_logistics;` | `from app.tools.builtin.orders import query_logistics;` |
| `tests/test_tools_db.py:11` | `from app.tools.business import make_create_ticket` | `from app.tools.builtin.tickets import make_create_ticket` |
| `tests/test_tools_query_faq.py:19` | `from app.tools.business import make_query_faq` | `from app.tools.builtin.knowledge import make_query_faq` |
| `app/agent/graph.py:5`(注释) | `app/tools/business.py` 的说明 | `app/tools/registry.py` 的说明 |
| `app/agent/nodes.py:3`(注释) | 同 `app/tools/business.py` 的既有做法 | `app/tools/builtin/` 的既有做法 |
| `app/refund/orders.py:3,39,43`(注释) | `app/tools/business.py` 的 `_order_record()` / `business.py:_require_order_no` | `app/tools/mock_data.py` 的 `order_record()` / `mock_data.py:require_order_no` |
| `tests/test_refund_orders.py:95`(注释) | `_require_order_no` 要 `isascii()` | `require_order_no` 要 `isascii()` |
| `scripts/acceptance.sh:236-249`(`shipped_order()`) | `from app.tools.business import query_order` + `LOGISTICS_BY_STATUS` | **改写成直接调 `app.tools.mock_data`**(见下方说明) |
| `scripts/acceptance.sh:257-272`(`expected_logistics_status()`) | `from app.tools.business import query_logistics` | **改写成直接调 `app.tools.mock_data.logistics_record`** |

**注释也要改** —— 这份代码库最贵的一类缺陷就是「注释还指着已经搬走的东西」。

> ⚠️ **上面这张表是扫描出来的,实测漏了三处**(T3 的实现者扫出来的,已修):
> - `tests/test_api_chat.py:987` —— **monkeypatch 的目标字符串**指向
>   `app.tools.business.make_query_faq`,改名后 patch 会 `AttributeError`。
>   **这是三处里唯一会响的**,另两处是散文注释。
> - `app/agent/refund_nodes.py:75`(散文注释)
> - `scripts/acceptance_ch05.sh:195`(散文注释)
>
> 教训与 T1 那次同源:**扫描要覆盖 `.py` / `.sh` / `.md`,并且
> monkeypatch 目标这类「字符串形式的引用」不在任何 import 图里**。
>
> 另:本任务 Step 9 的 `git add` 清单漏了 `scripts/` —— 而 Step 8 要求改它。

> **为什么 `scripts/acceptance.sh` 这两个 helper 要改成直查 `mock_data`**
> (而不是改成 `from app.tools.builtin.orders import …`):
> 那份脚本是 **ch01–ch04 的回归网**,不该被后面三个任务的搬迁连累 ——
> `query_order` 在 T3 换模块、`query_logistics` 在 **T7 整个搬进 MCP Server**,
> 跟着改一次就要再改一次。两个 helper 要的只是**确定性的订单状态**,
> 而脚本自己的注释已经写明了这个分工:
> 「*动态取值而非写死 —— 工具改了种子函数也不必改脚本;而『工具到底返回什么』
> 由 Tier 1 的跨进程确定性测试守护*」。改写后连 `asyncio` 与 `tool_call` 字典都不用了。
>
> **流水线口径不能变**:`shipped_order()` 的输出必须是**同一个**订单号,
> 所以它挑号码的规则(遍历 `1000..1039`、取第一个状态落在
> `LOGISTICS_BY_STATUS` 里的)要**一字不动**地保留。

- [ ] **Step 9: 跑全量测试**

Run: `.venv/Scripts/python.exe -m pytest`
Expected: 全绿。
⚠️ `tests/test_registry.py` 若断言的是旧的 `build_tools` 行为,加一条断言
`set(build_registry(...)) == set(registry_for(build_tools(...)))`,**不要**把旧断言删掉了事。

- [ ] **Step 10: 提交**

```bash
git add -A app/tools tests app/agent app/refund
git commit -m "refactor(ch08): 五个工具迁进 builtin/ 包 + 包内自动发现 + 注册表产出 ToolSpec"
```

---

### Task 4: 执行引擎 —— 权限闸 / 校验前置 / 六类分诊 / 重试规则

**Files:**
- Create: `app/tools/errors.py`(新增一个异常类)
- Modify: `app/tools/executor.py`(重写)
- Modify: `app/agent/nodes.py:337`(调用点)
- Modify: `app/agent/refund_nodes.py`(调用点 + `registry` 类型)
- Modify: `app/api/chat.py`(用 `build_registry`,传 `conversation_id`)
- Test: `tests/test_executor.py`(改)+ `tests/test_executor_gate.py`(新建)

**Interfaces:**
- Consumes: `spec.ToolSpec` / `spec.validate_args` / `spec.WRITE`(T2、T3)
- Produces:
  - `executor.ToolOutcome`(新增 `preview: dict | None`、`retry_count: int`、`source: str`)
  - `executor.execute_tool(*, tool_call, registry, settings, conversation_id="",
    write_decision="pending") -> ToolOutcome`
  - 常量 `ERROR_CONFIRMATION_REQUIRED` / `ERROR_PERMISSION_DENIED`,
    以及 `PENDING` / `APPROVED` / `DENIED`
  - `errors.TransientToolError` —— **暂时性故障的词汇**(T7 的 MCP 客户端用它)
  - `executor.render_tool_result(spec, raw) -> str`

**`registry` 的类型从 `dict[str, BaseTool]` 换成 `dict[str, ToolSpec]`** ——
这是一处**贯穿性改动**,上面四个调用点必须同任务改完。

- [ ] **Step 1: 写失败测试**

新建 `tests/test_executor_gate.py`:

```python
"""执行引擎的三道闸:权限、参数校验、重试规则。

⚠️ 本文件的**关键写法**:注入的是**未加工的输入**(裸 args、裸异常),
不是「已经被处理好的值」—— 本仓栽过三次的那类假绿就是
「测试把处理之后的形态喂给被测对象,于是处理那一步永远不被验」。

⚠️ **两处初稿缺陷,实现者实测后订正(T4 报告 §2/§3),照抄会红:**

1. **每条 `tool_call` dict 都必须带 `"type": "tool_call"`。** 初稿全部漏了 ——
   而 `BaseTool.ainvoke` 判「这是不是一次工具调用」**只看这个键**,缺键时它把
   整个 dict 当成**参数**去校验工具 schema,于是每次调用都返回一条「参数不合法」的
   **可恢复**失败。症状是「闸全对、工具一次没跑起来」,报错却指向 pydantic 的
   `Field required`,与真正的原因毫无相似之处(CLAUDE.md 的硬约束)。
   实测 **6 条用例**红在那里。**用下面这个 helper,不要内联 dict:**
   ```python
   def _tc(name: str, args: dict) -> dict:
       return {"name": name, "args": args, "id": "c1", "type": "tool_call"}
   ```
2. **`_spec` 不给 `schema` 时要**从 `tool` 派生**(与 `registry._spec_from_tool`
   同款:`tool.args_schema.model_json_schema()`)。初稿是个固定要求字段 `x` 的兜底
   schema —— 于是「工具的入参」与「校验用的 schema」说的是两件事:
   `test_invalid_args_*` 断的「文案点名到字段」会点在 `x` 上(而不是 `order_id`),
   而 `test_approved_write_is_executed_once` / `test_first_try_success_reports_zero_retries`
   这类会**先**被校验闸拦下,以「工具一次没被调用」的样子红 —— 那种红与它们真正要验的
   闸(权限、重试)**毫无关系**,又是「报错指向别处」。
"""

import asyncio

import pytest
from langchain_core.tools import tool

from app.tools import executor
from app.tools.errors import ToolInfrastructureError, TransientToolError
from app.tools.executor import (
    ERROR_CONFIRMATION_REQUIRED,
    ERROR_INVALID_ARGS,
    ERROR_PERMISSION_DENIED,
    ERROR_TIMEOUT,
    APPROVED,
    DENIED,
    PENDING,
    execute_tool,
)
from app.tools.spec import READ, WRITE, ToolSpec


class _Settings:
    # ⚠️ **超时用真实的小数字,绝不要 monkeypatch `asyncio.sleep`。**
    # `executor.asyncio` **就是** `asyncio` 模块本身 —— `monkeypatch.setattr(
    # executor.asyncio, "sleep", ...)` 会把它**全局**换掉,于是被测工具里那句
    # `await asyncio.sleep(10)` **立刻返回**、`wait_for` 根本不超时,测试会以
    # 一个看不懂的方式红。这是「替身把被测行为整个取消掉了」——
    # 本仓第 (g) 类假绿的反面:**不是假绿,是假红**。
    # (初稿正是这么写的,已订正。)
    tool_timeout_seconds = 0.01
    tool_retry_attempts = 2
    tool_retry_delay_seconds = 0.0
    tool_result_max_tokens = 1200


def _spec(*, name="demo", kind=READ, schema=None, tool=None) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="测试用",
        input_schema=schema
        or {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
        kind=kind,
        source="builtin",
        tool=tool,
    )


async def _collect(sink, kwargs):
    sink.append(kwargs)


# ---- 闸 1:权限 ---------------------------------------------------------


@pytest.mark.anyio
async def test_pending_write_is_not_executed():
    """未确认的写调用:**不执行**。用真工具来证 —— 计数器在函数体里,
    它必须**一次都没被调用**。"""
    calls = []

    @tool
    async def create_ticket(description: str) -> str:
        """建单。"""
        calls.append(description)
        return "ok"

    spec = _spec(name="create_ticket", kind=WRITE, tool=create_ticket)
    outcome = await execute_tool(
        tool_call={"name": "create_ticket", "id": "c1", "args": {"description": "坏了"}},
        registry={"create_ticket": spec},
        settings=_Settings(),
        conversation_id="c1",
        write_decision=PENDING,      # 显式写出默认值:Agent 那条路从不传别的
    )
    assert calls == []
    assert outcome.ok is False
    assert outcome.error_kind == ERROR_CONFIRMATION_REQUIRED


@pytest.mark.anyio
async def test_pending_write_preview_carries_the_args():
    """预览载荷 = 写操作的入参(前端要拿它渲染「工单类型 + 问题描述」)。"""
    spec = _spec(name="create_ticket", kind=WRITE)
    outcome = await execute_tool(
        tool_call={"name": "create_ticket", "id": "c1",
                   "args": {"description": "耳机坏了", "ticket_type": "售后"}},
        registry={"create_ticket": spec},
        settings=_Settings(),
        conversation_id="c1",
        write_decision=PENDING,
    )
    assert outcome.preview == {"description": "耳机坏了", "ticket_type": "售后"}


@pytest.mark.anyio
async def test_pending_write_is_not_audited(monkeypatch):
    """拦截**不是**拒绝 —— 那一刻没有任何人拒绝任何事。

    落了审计的话,验收 5 要的「这条 create_ticket 状态是权限拒绝」
    会变成「其中一行是」,断言从唯一事实退化成含糊。
    """
    seen: list[dict] = []
    spec = _spec(name="create_ticket", kind=WRITE)
    monkeypatch.setattr(executor, "record_audit", lambda **kw: _collect(seen, kw))
    await execute_tool(
        tool_call={"name": "create_ticket", "id": "c1", "args": {"x": "1"}},
        registry={"create_ticket": spec},
        settings=_Settings(),
        conversation_id="c1",
        write_decision=PENDING,
    )
    await asyncio.sleep(0)          # 给可能存在的 fire-and-forget 一点机会
    assert seen == []


@pytest.mark.anyio
async def test_denied_write_is_not_executed_but_is_audited(monkeypatch):
    """取消:不执行 + **落 `permission_denied`**(验收 5 断的就是它)。"""
    calls = []
    seen: list[dict] = []

    @tool
    async def create_ticket(description: str) -> str:
        """建单。"""
        calls.append(description)
        return "ok"

    spec = _spec(name="create_ticket", kind=WRITE, tool=create_ticket)
    monkeypatch.setattr(executor, "record_audit", lambda **kw: _collect(seen, kw))
    outcome = await execute_tool(
        tool_call={"name": "create_ticket", "id": "c1", "args": {"description": "坏了"}},
        registry={"create_ticket": spec},
        settings=_Settings(),
        conversation_id="c1",
        write_decision=DENIED,
    )
    assert calls == []
    assert outcome.error_kind == ERROR_PERMISSION_DENIED
    assert [s["status"] for s in seen] == ["permission_denied"]


@pytest.mark.anyio
async def test_approved_write_is_executed_once():
    calls = []

    @tool
    async def create_ticket(description: str) -> str:
        """建单。"""
        calls.append(description)
        return '{"ticket_no": "T-1"}'

    spec = _spec(name="create_ticket", kind=WRITE, tool=create_ticket)
    outcome = await execute_tool(
        tool_call={"name": "create_ticket", "id": "c1", "args": {"description": "坏了"}},
        registry={"create_ticket": spec},
        settings=_Settings(),
        conversation_id="c1",
        write_decision=APPROVED,
    )
    assert calls == ["坏了"]
    assert outcome.ok is True


# ---- 闸 2:参数校验 -----------------------------------------------------


@pytest.mark.anyio
async def test_invalid_args_are_caught_before_the_tool_runs(monkeypatch):
    """**注入的是裸 args,不是 ValidationError** —— 校验那一步必须真的发生。

    本仓栽过三次的假绿形态就是「注入已经被处理好的值」:那样
    「把校验错误翻成给模型看的话」这一步永远不被验。
    """
    calls = []
    seen: list[dict] = []

    @tool
    async def query_order(order_id: str) -> str:
        """查订单。"""
        calls.append(order_id)
        return "{}"

    spec = _spec(name="query_order", tool=query_order)
    monkeypatch.setattr(executor, "record_audit", lambda **kw: _collect(seen, kw))
    outcome = await execute_tool(
        tool_call={"name": "query_order", "id": "c1", "args": {}},   # 缺必填
        registry={"query_order": spec},
        settings=_Settings(),
        conversation_id="c1",
    )
    assert calls == [], "参数不合法却把工具跑起来了"
    assert outcome.error_kind == ERROR_INVALID_ARGS
    assert "order_id" in outcome.content, "回灌文案必须点名到字段"
    assert [s["status"] for s in seen] == ["invalid_args"]


@pytest.mark.anyio
async def test_invalid_args_is_not_retried():
    """重放同样的参数只会同样失败 —— 重试纯属浪费。**断的是 `retry_count`。**"""
    spec = _spec(name="query_order")
    outcome = await execute_tool(
        tool_call={"name": "query_order", "id": "c1", "args": {}},
        registry={"query_order": spec},
        settings=_Settings(),
        conversation_id="c1",
    )
    assert outcome.error_kind == ERROR_INVALID_ARGS
    assert outcome.retry_count == 0


# ---- 闸 3:重试规则 -----------------------------------------------------


@pytest.mark.anyio
async def test_read_tool_is_retried_on_timeout():
    calls = []

    @tool
    async def query_order(order_id: str) -> str:
        """查订单。"""
        calls.append(order_id)
        # 真睡 10 秒,由 `_Settings.tool_timeout_seconds = 0.01` 掐掉 ——
        # **不要**去 patch `asyncio.sleep`(见 `_Settings` 的说明)。
        await asyncio.sleep(10)
        return "{}"

    spec = _spec(
        name="query_order", kind=READ,
        schema={"type": "object", "properties": {"order_id": {"type": "string"}}},
        tool=query_order,
    )
    outcome = await execute_tool(
        tool_call={"name": "query_order", "id": "c1", "args": {"order_id": "1002"}},
        registry={"query_order": spec},
        settings=_Settings(),
        conversation_id="c1",
    )
    assert outcome.ok is False
    assert outcome.error_kind == ERROR_TIMEOUT
    # 1 次原始 + 2 次重试 = 3 次尝试(TOOL_RETRY_ATTEMPTS=2,spec §6.2)
    assert outcome.retry_count == 2


@pytest.mark.anyio
async def test_write_tool_never_retries_even_when_config_says_two():
    """**结构保证,不是配置恰好为 0。**

    `_Settings.tool_retry_attempts` 在上面就是 **2** —— 这条测试要是绿不了,
    就说明「不重试」是靠配置凑出来的,而不是靠 `kind == write` 推出来的
    (验收 6 后半条断的就是这个)。
    """
    calls = []

    @tool
    async def create_ticket(description: str) -> str:
        """建单。"""
        calls.append(description)
        await asyncio.sleep(10)
        return "{}"

    spec = _spec(name="create_ticket", kind=WRITE, tool=create_ticket)
    outcome = await execute_tool(
        tool_call={"name": "create_ticket", "id": "c1", "args": {"description": "x"}},
        registry={"create_ticket": spec},
        settings=_Settings(),
        conversation_id="c1",
        write_decision=APPROVED,
    )
    assert outcome.error_kind == ERROR_TIMEOUT
    assert outcome.retry_count == 0
    assert len(calls) == 1


@pytest.mark.anyio
async def test_first_try_success_reports_zero_retries(monkeypatch):
    """⚠️ **本章最容易写错的字段**(spec §7.4)。

    写成 `settings.tool_retry_attempts` 的实现会报 **2** ——
    而它看起来完全正常,没有任何别的断言会红。
    """
    seen: list[dict] = []

    @tool
    async def query_order(order_id: str) -> str:
        """查订单。"""
        return '{"ok": true}'

    spec = _spec(name="query_order", tool=query_order)
    monkeypatch.setattr(executor, "record_audit", lambda **kw: _collect(seen, kw))
    outcome = await execute_tool(
        tool_call={"name": "query_order", "id": "c1", "args": {"order_id": "1002"}},
        registry={"query_order": spec},
        settings=_Settings(),
        conversation_id="c1",
    )
    assert outcome.ok is True
    assert outcome.retry_count == 0
    assert seen[0]["retry_count"] == 0


@pytest.mark.anyio
async def test_transient_failure_succeeds_on_retry():
    """**暂时性故障是本任务新增的那一类可重试故障**(spec §6.3)。

    这一条双向都钉住:① 它**真的重试了**(`retry_count == 1`);
    ② 第二次成功就是成功,不会因为「它抖过」而把结果也丢掉。
    """
    calls = []

    @tool
    async def query_order(order_id: str) -> str:
        """查订单。"""
        calls.append(order_id)
        if len(calls) < 2:
            raise TransientToolError("connection refused")
        return '{"ok": true}'

    spec = _spec(name="query_order", tool=query_order)
    outcome = await execute_tool(
        tool_call={"name": "query_order", "id": "c1", "args": {"order_id": "1002"}},
        registry={"query_order": spec},
        settings=_Settings(),
        conversation_id="c1",
    )
    assert outcome.ok is True
    assert outcome.retry_count == 1
    assert len(calls) == 2


@pytest.mark.anyio
async def test_transient_exhausted_raises_infrastructure_error():
    """三次都不成 ⇒ **上抛 502**,不是回灌一句「工具暂时不可用」。

    重试用尽之后那个故障就不叫暂时性了;推一句软话给模型等于把基础设施故障
    伪装成一次普通的工具失败 —— 与「数据库挂了不许伪装成你的订单查不到」同一条规矩。
    """
    calls = []

    @tool
    async def query_order(order_id: str) -> str:
        """查订单。"""
        calls.append(order_id)
        raise TransientToolError("connection refused")

    spec = _spec(name="query_order", tool=query_order)
    with pytest.raises(ToolInfrastructureError):
        await execute_tool(
            tool_call={"name": "query_order", "id": "c1", "args": {"order_id": "1002"}},
            registry={"query_order": spec},
            settings=_Settings(),
            conversation_id="c1",
        )
    assert len(calls) == 3, "暂时性故障必须真的重试到用尽(1 次原始 + 2 次重试)"


@pytest.mark.anyio
async def test_unknown_tool_is_not_audited(monkeypatch):
    """接线 bug 不是一次调用 —— 它该响亮地暴露,不该混进审计流水。"""
    seen: list[dict] = []
    monkeypatch.setattr(executor, "record_audit", lambda **kw: _collect(seen, kw))
    outcome = await execute_tool(
        tool_call={"name": "nope", "id": "c1", "args": {}},
        registry={},
        settings=_Settings(),
        conversation_id="c1",
    )
    assert outcome.error_kind == "tool_missing"
    assert seen == []
```

- [ ] **Step 2: 跑测试,确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_executor_gate.py`
Expected: FAIL —— `ImportError: cannot import name 'ERROR_CONFIRMATION_REQUIRED'`

- [ ] **Step 3: 给 `app/tools/errors.py` 加一个异常类**

```python
class TransientToolError(Exception):
    """**暂时性**故障(网络抖动、连接被拒、传输中断)。

    与 `ToolInfrastructureError` 的区别是**要不要再试一次**:
    它属于「过一会儿可能就好了」,而「数据库连接串写错了」不是。
    执行器对它在重试白名单内重试;**重试用尽之后仍上抛
    `ToolInfrastructureError`** —— 三次都失败的故障不叫暂时性了,
    这时推 502 比推一句「工具暂时不可用」诚实。
    """
```

- [ ] **Step 4: 重写 `app/tools/executor.py`**

```python
"""工具执行:权限闸 → 参数校验 → 执行(超时/重试)→ 结果格式化 → 审计。

**顺序是有讲究的**,不是随手排的(spec §6):
1. 查注册表       —— 接线 bug,不审计、不上抛,回灌给模型
2. 权限闸         —— 写操作没确认就**不执行**;`pending` 不审计,`denied` 审计
3. JSON Schema 校验 —— 拦下之后**回灌给模型**,不抛异常
4. 执行           —— 超时 / 重试(只读可重试,**写操作结构性不重试**)
5. 结果格式化
6. 审计落库       —— 独立 session,失败只 `logger.error`,**不影响返回值**
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

from sqlalchemy.exc import SQLAlchemyError

from app.tools.errors import (
    ToolInfrastructureError,
    ToolNotFound,
    TransientToolError,
)
from app.tools.spec import WRITE, validate_args

logger = logging.getLogger(__name__)

#: tool_result 事件里给前端展示的摘要上限。
SUMMARY_MAX_CHARS = 200

# ---- 失败种类 ----------------------------------------------------------
#
# `ok=False` 只说明「没成功」,而「没成功」有六种来源,对调用方的含义完全不同
# —— 尤其是**能不能把这句话说给用户听**:只有 `NOT_FOUND` 是「工具明确说
# 没有这个东西」,其余都是**服务端或接线**的问题,把它们说成「你要的东西
# 不存在」就是拿服务端故障指责用户输入。
ERROR_NOT_FOUND = "not_found"                  # ToolNotFound:业务性未找到(可恢复)
ERROR_TIMEOUT = "timeout"                      # 超时(重试已用尽)
ERROR_INVALID_ARGS = "invalid_args"            # 参数不合 schema
ERROR_TOOL_MISSING = "tool_missing"            # 注册表里没有 = 接线 bug
ERROR_PERMISSION_DENIED = "permission_denied"  # 写操作被用户取消
ERROR_CONFIRMATION_REQUIRED = "confirmation_required"  # 写操作待确认

# ---- 写调用的三态决议(spec §5.3)----------------------------------------
#
# **布尔不够用**:「没问过」与「问过、用户说不」是两件不同的事。
# 用 `approved=False` 一个值表达两者的话,取消路径会**再拿到一次
# `confirmation_required`**,于是取消永远不会被记成「权限拒绝」。
PENDING = "pending"
APPROVED = "approved"
DENIED = "denied"


@dataclass(frozen=True)
class ToolOutcome:
    tool_call_id: str
    name: str
    ok: bool
    content: str    # 完整内容,回灌给模型
    summary: str    # 截断后的展示用摘要
    error_kind: str | None = None
    #: 写操作待确认时,把**入参**带出去给前端渲染预览卡片。
    preview: dict | None = None
    #: **真实发生过的**重试次数(= 实际尝试数 − 1),**不是配置值**。
    retry_count: int = 0
    source: str = ""


def _summarize(text: str) -> str:
    text = text.strip()
    if len(text) <= SUMMARY_MAX_CHARS:
        return text
    return text[:SUMMARY_MAX_CHARS] + "…"


def render_tool_result(spec, raw) -> str:
    """把工具的返回统一成**一个字符串**。

    - 内置工具返回的已经是手挑过字段的 JSON 字符串 ⇒ 原样透传
      (顺带保证中文不转义 —— 本仓全仓已用 `ensure_ascii=False`)。
    - MCP 工具返回的是 MCP 的内容块列表 ⇒ 压成一行紧凑 JSON,
      **不把整个响应体塞进上下文**。

    "只挑回答用得上的字段"与"内部枚举码翻人话"这两条**不在这里**:
    它们落在**工具自身**(内置那份就只有工具知道自己的枚举怎么翻)。
    """
    if isinstance(raw, str):
        return raw
    try:
        return json.dumps(raw, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(raw)


async def execute_tool(
    *,
    tool_call: dict,
    registry: dict,
    settings,
    conversation_id: str = "",
    write_decision: str = PENDING,
) -> ToolOutcome:
    """执行一次工具调用。

    可恢复的失败返回 `ok=False` 的 `ToolOutcome`(调用方回灌给模型);
    基础设施故障抛 `ToolInfrastructureError`(调用方推 error 帧终止流)。
    """
    name = tool_call.get("name", "")
    tool_call_id = tool_call.get("id", "")
    args = tool_call.get("args") or {}

    spec = registry.get(name)
    if spec is None:
        message = f"工具 {name} 不存在。可用工具:{', '.join(sorted(registry))}"
        # **不审计**:接线 bug 不是一次调用。混进流水会让验收 5/6 的
        # 「最近这几条」里混进与本次调用无关的行。
        return ToolOutcome(
            tool_call_id, name, False, message, _summarize(message),
            ERROR_TOOL_MISSING,
        )

    # ---- 闸 1:权限(只对写操作)------------------------------------
    if spec.kind == WRITE:
        if write_decision == PENDING:
            # 不执行、**不审计** —— 拦截那一刻没有任何人拒绝任何事。
            message = f"工具 {name} 是写操作,需要用户确认后才能执行。"
            return ToolOutcome(
                tool_call_id, name, False, message, _summarize(message),
                # ⚠️ `isinstance` 判定,不是 `dict(args)`:后者在 args 非 dict 时会抛,
                # 而这个位置**没有 handler 罩着** ⇒ 逃出 `execute_tool`,而不是变成这套
                # 分类学承诺的可恢复 `invalid_args`。宽容的 `validate_args` 在闸**之后**,
                # 所以「畸形 args 的写调用」恰好是唯一绕开它的地方。(T4 审查发现的。)
                ERROR_CONFIRMATION_REQUIRED,
                preview=args if isinstance(args, dict) else {"_raw": args},
                source=spec.source,
            )
        if write_decision == DENIED:
            message = f"用户取消了 {name} 的调用,未执行。"
            await record_audit(
                conversation_id=conversation_id, tool_call_id=tool_call_id,
                tool_name=name, source=spec.source, args=args,
                result_summary=_summarize(message),
                status=ERROR_PERMISSION_DENIED, retry_count=0, duration_ms=0,
            )
            return ToolOutcome(
                tool_call_id, name, False, message, _summarize(message),
                ERROR_PERMISSION_DENIED, source=spec.source,
            )
        if write_decision != APPROVED:
            # 认不出的决议 = **接线 bug**,不是用户动作。
            # ⚠️ **不许**把它当成「用户取消」:那会在审计表里写一条**谎报用户行为**的行,
            # 而验收 5 读的正是那张表(`permission_denied` 的语义是「**用户**点了取消」)。
            # **响亮地抛**(→502)且**不审计** —— 没有任何真实调用发生过。
            # 与 `tool_missing` 同族:接线 bug 要暴露,不要伪装成一次正常结果。
            # (T4 审查发现的 fail-open:初稿的 `else → 执行` 会让任何拼错的值
            #  无确认、无审计地跑完一次**不可逆**写操作。)
            raise ToolInfrastructureError(
                f"写操作的决议取值不合法:{write_decision!r}"
            )

    # ---- 闸 2:参数校验(**在 ainvoke 之前**)-------------------------
    problems = validate_args(spec, args)
    if problems:
        message = f"工具 {name} 的参数校验未通过:" + ";".join(problems)
        await record_audit(
            conversation_id=conversation_id, tool_call_id=tool_call_id,
            tool_name=name, source=spec.source, args=args,
            result_summary=_summarize(message),
            status=ERROR_INVALID_ARGS, error_detail=";".join(problems),
            retry_count=0, duration_ms=0,
        )
        return ToolOutcome(
            tool_call_id, name, False, message, _summarize(message),
            ERROR_INVALID_ARGS, source=spec.source,
        )

    # ---- 执行 ------------------------------------------------------
    # 重试次数由 **`kind` 推导**,不再是一张写死的白名单:
    # 新注册的只读工具自动可重试、写工具自动不可重试。
    # 写操作**永不重试**是结构保证 —— 超时未必没执行,重复执行比失败更糟。
    attempts = 1 + (settings.tool_retry_attempts if spec.kind != WRITE else 0)
    started = time.monotonic()
    last_message = ""
    last_kind: str | None = None
    # **真实发生过的**重试次数,不是配置值(spec §7.4)。写成配置值的话,
    # 一个第一次就成功的查询会被审计成「重试了 2 次」,而没有任何断言会红。
    retries = 0

    for attempt in range(attempts):
        if attempt:
            retries += 1
        try:
            message = await asyncio.wait_for(
                spec.tool.ainvoke(tool_call), timeout=settings.tool_timeout_seconds
            )
            content = render_tool_result(spec, message.content)
            await record_audit(
                conversation_id=conversation_id, tool_call_id=tool_call_id,
                tool_name=name, source=spec.source, args=args,
                result_summary=_summarize(content), status="success",
                retry_count=retries,
                duration_ms=int((time.monotonic() - started) * 1000),
            )
            return ToolOutcome(
                tool_call_id, name, True, content, _summarize(content),
                retry_count=retries, source=spec.source,
            )
        except TimeoutError:
            last_kind = ERROR_TIMEOUT
            last_message = (
                f"工具 {name} 执行超时(超过 {settings.tool_timeout_seconds} 秒)"
            )
            logger.warning("工具 %s 超时,第 %d/%d 次尝试", name, attempt + 1, attempts)
            if attempt + 1 < attempts:
                await asyncio.sleep(settings.tool_retry_delay_seconds)
        except TransientToolError as exc:
            # **暂时性**故障(网络抖动)—— 本章新增的那一类可重试故障(spec §6.3)。
            # 重试用尽后仍失败 ⇒ 三次都不成,不叫暂时性了,按基础设施故障上抛。
            # ⚠️ 这一支**要么重试、要么上抛**,永远不会走到循环底部 ⇒
            # 下面那行赋值是**不可达**的。实现者可以删掉它(更干净),
            # 也可以留着当防御 —— 两者都不算缺陷。
            last_kind = ERROR_TIMEOUT
            last_message = f"工具 {name} 暂时不可用:{exc}"
            logger.warning(
                "工具 %s 暂时性故障,第 %d/%d 次尝试:%s", name, attempt + 1, attempts, exc
            )
            if attempt + 1 < attempts:
                await asyncio.sleep(settings.tool_retry_delay_seconds)
            else:
                raise ToolInfrastructureError("工具暂时不可用") from exc
        except ValidationError as exc:
            # 第二道(第一道是上面的 `validate_args`)。走到这里说明工具的
            # `args_schema` 比它的 `input_schema` 更严 —— 那是我们的 bug。
            last_kind = ERROR_INVALID_ARGS
            last_message = f"工具 {name} 的参数不合法:{exc}"
            logger.warning("工具 %s 参数校验失败:%s", name, exc)
            break
        except ToolNotFound as exc:
            # 业务性未找到是**决定性**结果:重放同一个 tool_call 送的是同样的
            # 参数,只会同样落空。而且这是最常见的落空路径,重试白搭一次 DB
            # 往返加等待 —— 可恢复路径本该是最便宜的那条。
            last_kind = ERROR_NOT_FOUND
            last_message = str(exc)
            break
        except SQLAlchemyError as exc:
            logger.exception("工具 %s 命中数据库故障", name)
            raise ToolInfrastructureError("数据服务暂时不可用") from exc
        except Exception as exc:
            logger.exception("工具 %s 抛出未预期异常", name)
            raise ToolInfrastructureError("工具执行失败") from exc

    status = {
        ERROR_TIMEOUT: "timeout",
        ERROR_NOT_FOUND: "failed",
        ERROR_INVALID_ARGS: "invalid_args",
    }.get(last_kind or "", "failed")
    await record_audit(
        conversation_id=conversation_id, tool_call_id=tool_call_id,
        tool_name=name, source=spec.source, args=args,
        result_summary=_summarize(last_message), status=status,
        retry_count=retries,
        duration_ms=int((time.monotonic() - started) * 1000),
    )
    return ToolOutcome(
        tool_call_id, name, False, last_message, _summarize(last_message),
        last_kind, retry_count=retries, source=spec.source,
    )
```

**顶部还要补两行 import**(上面第 4 步的代码里没写全,照实际需要补):

```python
from app.tools.audit import record_audit       # ← T5 才有这个模块,见下面的说明
from pydantic import ValidationError
```

> ⚠️ **T5 之前 `app/tools/audit.py` 还不存在。** 本任务先写一个**最小占位**:

```python
# app/tools/audit.py(本任务临时版,T5 补全并加表)
import logging

logger = logging.getLogger(__name__)


async def record_audit(**kwargs) -> None:
    """占位:真正的落库在 T5。"""
    logger.debug("audit %s", kwargs)
```

- [ ] **Step 5: 改四个调用点**

`app/agent/nodes.py:337` 附近:

```python
                outcome = await execute_tool(
                    tool_call=call, registry=registry, settings=settings,
                    conversation_id=state["conversation_id"],
                )
```

`app/agent/refund_nodes.py`:同款 —— 加 `conversation_id=state["conversation_id"]`。

`app/api/chat.py`:把 `build_tools(...)` + `registry_for(...)` 换成

```python
registry = build_registry(
    session=session, conversation_id=session_id, settings=settings
)
```

并把传进 `build_graph` 的 `tools=` 改成从注册表投影:

```python
tools = [spec.tool for spec in registry.values()]
```

`evals/run_tool_selection_eval.py` **不动**(它用的是 `build_tools`,本任务保留了)。

**⚠️ 第五个调用点,初稿漏了:`app/api/chat.py` 的 `POST /api/ticket`(约 :555)。**
它是 ch05 投诉流程里「点按钮建工单」那条路,而它**是**一次 `execute_tool` 调用
—— 本节新增的权限闸一落地,它就**必然**被拦(拿到 `confirmation_required` → 502,
症状与「服务挂了」一模一样)。spec §5.4 初稿说它「不是工具调用、一行不动」是**错的**,
已订正。**改法:该调用点显式传 `write_decision=APPROVED`** ——
按钮点击**就是**用户确认,那一行只是把这个已有的语义告诉闸。

**Step 5b:把 `builtin/tickets.py` 那条重试注释改成新机制(T3 挪过来的)**

`make_create_ticket` 的 docstring 里,把

```
    非幂等写操作 —— executor 的重试白名单不含它,超时也绝不重试,
    否则会建出两张工单。
```

改成:

```
    **非幂等写操作** —— `app/tools/policy.py` 把它声明成写操作,
    执行器由 `kind == "write"` **结构性地**推出「永不重试」,
    否则超时重试会建出两张工单。新注册的写工具自动继承这条。
```

⚠️ **必须与本任务的执行器改动同一个提交**。T3 里提前改它会让注释描述一个
**还不存在**的机制 —— 与「注释指着已经搬走的东西」是同一类缺陷,只是方向相反。

- [ ] **Step 6: 跑测试**

Run: `.venv/Scripts/python.exe -m pytest`
Expected: 全绿。

⚠️ `tests/test_executor.py` 里若用的是旧的 `dict[str, BaseTool]` 注册表,
把它改成 `dict[str, ToolSpec]`(用 `registry._spec_from_tool` 或直接构造),
**不要**为了省事给 `execute_tool` 加一条「旧形状也认」的兼容分支 ——
那会让「注册表里到底装的是什么」有两个答案。

- [ ] **Step 7: 提交**

```bash
git add app/tools app/agent app/api tests
git commit -m "feat(ch08): 执行引擎加权限闸 + 校验前置 + 六类分诊;重试规则改由 kind 推导"
```

---

### Task 5: 审计留痕(表 + ORM + 唯一写口)

**Files:**
- Create: `db/ch08.sql`
- Modify: `app/db/models.py`(加 `ToolAuditLog`)
- Modify: `app/tools/audit.py`(把 T4 的占位换成真实现)
- Modify: `CLAUDE.md`(建库顺序那段)
- Test: `tests/test_tools_audit.py`(新建,不连库)+ `tests/test_tools_audit_db.py`(新建,`@pytest.mark.db`)

**Interfaces:**
- Consumes: `executor` 已经在调 `record_audit(...)`(T4,占位版)
- Produces:
  - `audit.record_audit(*, conversation_id, tool_call_id, tool_name, source, args,
    result_summary, status, error_detail="", retry_count=0, duration_ms=0) -> None`
  - `models.ToolAuditLog`

**两层测试的分工(刻意,不是偷懒)**:
- `tests/test_tools_audit.py`(**不连库**)验 `record_audit` 的**形状**:参数怎么排版、
  超长怎么截、失败怎么吞 —— 用替身 sessionmaker。
- `tests/test_tools_audit_db.py`(`@pytest.mark.db`)验它**真的落了库**、列值对不对。
- 执行器那侧(`test_executor_gate.py`)只 spy `record_audit`,**不重复验落库**。

> 这三层缺一层就会出现本仓记过的第 (g) 类假绿:**替身替被测对象完成了语义**。

- [ ] **Step 1: 写 `db/ch08.sql`**

照 spec §11.1 **逐字**写(整段复制过去,别改写);文件头是:

```sql
-- =============================================================
-- ch08 · 工具调用审计
-- 每次工具调用一行;被权限拒、被校验拦的同样要落。
-- =============================================================

-- 本机 locale 是 cp936,不钉这一行中文 COMMENT 会在客户端侧被重编码(本仓记过这条)。
SET NAMES utf8mb4;
```

**执行顺序**:`init_db.py`(create_all)之后再跑这份。全新建库时 `create_all`
会**顺带**建出这张表(因为 ORM 侧有同名模型),此时这份 DDL 会**响亮地报 1050** ——
**这是刻意的**,与 `db/ch06.sql` 的 `refund_requests` 同一个已知取舍。

- [ ] **Step 2: 给 `app/db/models.py` 加模型**

```python
class ToolAuditLog(Base):
    """工具调用审计。**只映射,不被任何业务读写。**

    **刻意不挂外键**(spec §7.1):审计是**旁路记录** —— 挂了外键的话,
    删会话/删工单会受约束,甚至反过来影响主流程。审计的职责是**只记不拦**。
    """

    __tablename__ = "tool_audit_logs"

    # `BigInteger` 与 DDL 的 `BIGINT` 对齐 —— 见下方形状对齐的说明。
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # 不是 ForeignKey —— 见类 docstring。
    conversation_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    tool_call_id: Mapped[str] = mapped_column(String(128), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(64), nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    args: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # ⚠️ **这两个列在 DDL 里带 `DEFAULT ''`,这里必须补 `server_default`。**
    # 只留 Python 侧的 `default` 会让 `create_all` 建的表与 `db/ch08.sql` 建的
    # 表**形状不同** —— 行为变成「看谁建的库」。这条是 ch07 两个锚点列
    # (`Conversation.summary_upto_msg_id` / `layer1_from_msg_id`)已经吃过一次的亏,
    # 那里的注释逐字写着「两侧默认值都要」。
    # (`args` 与 `status` 在 DDL 里**没有** DEFAULT,所以这里也不加 —— 照抄 DDL。)
    result_summary: Mapped[str] = mapped_column(
        String(500), nullable=False, default="", server_default=""
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    error_detail: Mapped[str] = mapped_column(
        String(500), nullable=False, default="", server_default=""
    )
    # 两侧默认值都要:与 Conversation 的两个锚点同款理由 ——
    # 只留 `default` 会让 create_all 建的表与 db/ch08.sql 建的表**形状不同**。
    retry_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    duration_ms: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), index=True
    )
```

> **形状对齐的界线,写清楚** —— 建库有**两条路**(`init_db.py` 的 `create_all`
> 与 `db/ch08.sql`),而它们必须建出**行为一致**的表。本章对齐这几样:
> 列类型(`BIGINT`)、列宽、**`server_default`**、`created_at` 索引。
>
> **刻意不对齐、已知且可接受的差异**:DDL 里的两个**索引名**(`idx_conv` /
> `idx_created` vs SQLAlchemy 自动生成的 `ix_tool_audit_logs_*`)与表的
> `COMMENT`。它们只影响 `SHOW CREATE TABLE` 的观感,不影响任何行为 ——
> 而**权威路径永远是 `db/ch08.sql`**(CLAUDE.md 明写:带 `db/chNN.sql` 的章
> 都必须在 `init_db` 之外再执行那份 DDL)。

- [ ] **Step 3: 写失败测试(不连库那层)**

新建 `tests/test_tools_audit.py`:

```python
"""审计写入的**形状**:参数怎么排版、超长怎么截、失败怎么吞。

⚠️ **不连库**。真落库那层在 `tests/test_tools_audit_db.py`。
分两层是刻意的:只留替身那层,「替身替被测对象完成了语义」——
端点/执行器删掉真写入照样绿(本仓记过的第 (g) 类假绿)。
"""

import json

import pytest

from app.tools import audit


class _FakeSession:
    def __init__(self, sink, *, boom=False):
        self.sink = sink
        self.boom = boom

    def add(self, row):
        if self.boom:
            raise RuntimeError("模拟库挂了")
        self.sink.append(row)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def commit(self):
        if self.boom:
            raise RuntimeError("模拟提交失败")


def _patch(monkeypatch, sink, *, boom=False):
    monkeypatch.setattr(
        audit, "get_sessionmaker",
        lambda: (lambda: _FakeSession(sink, boom=boom)),
    )


@pytest.mark.anyio
async def test_writes_one_row_with_the_right_columns(monkeypatch):
    rows: list = []
    _patch(monkeypatch, rows)
    await audit.record_audit(
        conversation_id="c1", tool_call_id="call_1", tool_name="query_order",
        source="builtin", args={"order_id": "1002"}, result_summary="ok",
        status="success", retry_count=0, duration_ms=12,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.conversation_id == "c1"
    assert row.tool_name == "query_order"
    assert row.source == "builtin"
    assert row.status == "success"
    assert row.retry_count == 0
    assert row.duration_ms == 12


@pytest.mark.anyio
async def test_args_are_stored_as_unescaped_json(monkeypatch):
    """中文不转义 —— 验收 5/6 是要**人眼**读这些行的。"""
    rows: list = []
    _patch(monkeypatch, rows)
    await audit.record_audit(
        conversation_id="c1", tool_call_id="c1", tool_name="create_ticket",
        source="builtin", args={"description": "耳机坏了"}, result_summary="",
        status="permission_denied",
    )
    assert "耳机坏了" in rows[0].args
    assert "\\u" not in rows[0].args
    assert json.loads(rows[0].args) == {"description": "耳机坏了"}


@pytest.mark.anyio
async def test_unserializable_args_do_not_raise(monkeypatch):
    """模型给的东西不受我们控制,`json.dumps` 失败**不能**反过来拦工具执行。"""
    rows: list = []
    _patch(monkeypatch, rows)
    await audit.record_audit(
        conversation_id="c1", tool_call_id="c1", tool_name="t", source="builtin",
        args={"weird": object()}, result_summary="", status="success",
    )
    assert len(rows) == 1


@pytest.mark.anyio
async def test_overlong_fields_are_truncated_not_raised(monkeypatch):
    """列宽是 `String(500)` —— MySQL 严格模式下超长会 `DataError`。

    `create_ticket` 当年就栽过同一件事(它把 `ticket_type` 夹到 64 而不是抛错):
    一个被模型撑爆的摘要字段不该让**整条审计行**丢掉。
    """
    rows: list = []
    _patch(monkeypatch, rows)
    await audit.record_audit(
        conversation_id="c1", tool_call_id="c1", tool_name="t", source="builtin",
        args={}, result_summary="啊" * 2000, status="failed",
        error_detail="唉" * 2000,
    )
    assert len(rows[0].result_summary) <= 500
    assert len(rows[0].error_detail) <= 500


@pytest.mark.anyio
async def test_write_failure_is_swallowed(monkeypatch, caplog):
    """**写审计失败不许反过来拦工具执行**(要求 5 明写)。

    所以这里是 `except` + 日志,**不是** raise。用 `raise` 的实现会让
    「库抖了一下」变成「工具调用失败」,方向正好反了。
    """
    _patch(monkeypatch, [], boom=True)
    await audit.record_audit(
        conversation_id="c1", tool_call_id="c1", tool_name="t", source="builtin",
        args={}, result_summary="", status="success",
    )   # 不抛即通过
```

- [ ] **Step 4: 跑测试,确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tools_audit.py`
Expected: FAIL —— `AttributeError: module 'app.tools.audit' has no attribute 'get_sessionmaker'`

- [ ] **Step 5: 写 `app/tools/audit.py`(真实现,替换 T4 的占位)**

```python
"""审计写入 —— 本章**唯一**的写口。

**两条硬约束**(要求 5 明写):

1. **写审计失败不许反过来拦工具执行。** 所以这里一律 `except` + 日志,
   绝不向上抛 —— 用 raise 的话,「库抖了一下」会变成「工具调用失败」,
   方向正好反了。
2. **自己开一个 session。** 不能复用工具那个:工具刚把 session 弄进
   待回滚状态时,拿它写审计会把两件事绑在一起(审计成了工具失败的陪葬)。

**为什么没有第二个调用点**:不变量要放在唯一写口上,不靠每个调用方自觉
(本仓的元教训之一)。执行器在固定的两处调它 —— 校验拦下、执行结束。
"""

import json
import logging

from app.db.base import get_sessionmaker
from app.db.models import ToolAuditLog

logger = logging.getLogger(__name__)

#: 与 `db/ch08.sql` 的列宽一致。超长必须**截断而不是抛** ——
#: 一个被模型撑爆的摘要字段不该让整条审计行丢掉
#: (`create_ticket` 当年就栽过同一件事,它把 `ticket_type` 夹到 64)。
_SUMMARY_MAX = 500      # ← 与 db/ch08.sql 的 VARCHAR(500) 逐列对齐
_DETAIL_MAX = 500
_NAME_MAX = 64
_SOURCE_MAX = 64
_CALL_ID_MAX = 128
#: `status` 今天的取值全部来自本模块自己的常量(最长 21 字符),
#: **但它是全表唯一一个「按列宽硬存、却没有截断」的串**。
#: 夹一下是一行的事,而漏夹的后果是 `DataError` 让**整条审计行**丢掉。
_STATUS_MAX = 32


def _clip(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _dump_args(args) -> str:
    """入参 → 存进表的 JSON 文本。

    `default=str` 是**必须的**:模型给的东西不受我们控制,
    一个不可序列化的值不该让整条审计行丢掉。
    """
    try:
        return json.dumps(args, ensure_ascii=False, default=str)
    except Exception:                                   # noqa: BLE001
        return repr(args)


async def record_audit(
    *,
    conversation_id: str,
    tool_call_id: str,
    tool_name: str,
    source: str,
    args,
    result_summary: str,
    status: str,
    error_detail: str = "",
    retry_count: int = 0,
    duration_ms: int = 0,
) -> None:
    """落一行。**永不抛。**"""
    try:
        async with get_sessionmaker()() as session:
            session.add(
                ToolAuditLog(
                    conversation_id=conversation_id[:32],
                    tool_call_id=_clip(tool_call_id, _CALL_ID_MAX),
                    tool_name=_clip(tool_name, _NAME_MAX),
                    source=_clip(source, _SOURCE_MAX),
                    args=_dump_args(args),
                    result_summary=_clip(result_summary, _SUMMARY_MAX),
                    status=_clip(status, _STATUS_MAX),
                    error_detail=_clip(error_detail, _DETAIL_MAX),
                    retry_count=retry_count,
                    duration_ms=duration_ms,
                )
            )
            await session.commit()
    except Exception:                                   # noqa: BLE001
        # 宽到 `Exception` 是**刻意的**:这条路径上任何失败都只该留下痕迹,
        # 不该影响工具执行。`BaseException`(CancelledError)不在内 —— 取消
        # 要照常向上传播。
        logger.exception(
            "写审计失败(不影响工具执行):tool=%s status=%s", tool_name, status
        )
```

- [ ] **Step 6: 写 db 层测试**

新建 `tests/test_tools_audit_db.py`:

```python
"""审计**真的落库了** —— 替身那层看不见这个。

读回时用**新 session**:SQLAlchemy 的身份映射持弱引用,同 session 重读
是否打到库取决于还有没有东西引用着那个 ORM 对象 —— 那会变成
「靠 refcount 走运」的断言(本仓记过)。
"""

import pytest
from sqlalchemy import select

from app.db.base import get_sessionmaker
from app.db.models import ToolAuditLog
from app.tools.audit import record_audit

pytestmark = pytest.mark.db


@pytest.mark.anyio
async def test_row_is_really_persisted():
    await record_audit(
        conversation_id="ch08-audit-db", tool_call_id="call_db_1",
        tool_name="query_order", source="builtin",
        args={"order_id": "1002"}, result_summary="ok", status="success",
        retry_count=0, duration_ms=7,
    )
    async with get_sessionmaker()() as session:      # ← 新 session
        row = (
            await session.execute(
                select(ToolAuditLog)
                .where(ToolAuditLog.conversation_id == "ch08-audit-db")
                .order_by(ToolAuditLog.id.desc())
                .limit(1)
            )
        ).scalars().one()
    assert row.tool_name == "query_order"
    assert row.status == "success"
    assert row.duration_ms == 7
    assert "1002" in row.args


@pytest.mark.anyio
async def test_retry_count_is_what_was_passed_not_a_default():
    """**判别力所在**:一个把列写死成 0 的实现会在这条变红。

    (执行器那侧另有一条「首次成功 ⇒ 0」,两条合起来才锁住 §7.4。)
    """
    await record_audit(
        conversation_id="ch08-audit-db", tool_call_id="call_db_2",
        tool_name="query_order", source="builtin", args={},
        result_summary="超时", status="timeout", retry_count=2, duration_ms=30000,
    )
    async with get_sessionmaker()() as session:
        row = (
            await session.execute(
                select(ToolAuditLog)
                .where(ToolAuditLog.tool_call_id == "call_db_2")
            )
        ).scalars().one()
    assert row.retry_count == 2
    assert row.duration_ms == 30000
```

- [ ] **Step 7: 建表并跑测试**

```bash
.venv/Scripts/python.exe scripts/init_db.py
.venv/Scripts/mysql  # 不走这个;用下面这条
```

建表(二选一,取决于库是否已经存在):

```bash
docker exec -i $(docker ps -qf "name=mysql") mysql -uroot -p"$MYSQL_ROOT_PASSWORD" mewhelp < db/ch08.sql
```

若报 `ERROR 1050 (42S01): Table 'tool_audit_logs' already exists` —— 那是
`init_db.py` 的 `create_all` 先建了(见本任务 Step 1 的说明),**不是故障**;
用 `SHOW CREATE TABLE tool_audit_logs\G` 核对列与这份 DDL 一致即可。

Run: `.venv/Scripts/python.exe -m pytest tests/test_tools_audit.py tests/test_tools_audit_db.py`
Expected: 全绿。

- [ ] **Step 8: `CLAUDE.md` 补建库顺序**

在「建库 / 升级」那段的两处清单里各加一条:

- 那段 `db/chNN.sql` 的列举里,`db/ch07.sql` 之后加
  **`db/ch08.sql`(新表 tool_audit_logs)**;
- 「全新 checkout 的顺序」那一行改成
  `init_db.py` → 依次 `db/ch03.sql` / `ch04` / `ch06` / `ch07` / **`ch08`**。

并补一句:**`db/ch08.sql` 在全新库上会报 1050**(create_all 已顺带建表),
这是刻意的,与 `db/ch06.sql` 的 `refund_requests` 同一个已知取舍。

- [ ] **Step 9: 跑全量测试并提交**

Run: `.venv/Scripts/python.exe -m pytest`
Expected: 全绿。

```bash
git add db/ch08.sql app/db/models.py app/tools/audit.py tests CLAUDE.md
git commit -m "feat(ch08): 审计表 tool_audit_logs + 唯一写口 record_audit(失败不拦执行)"
```

---

### Task 6: 两个业务 MCP Server

**Files:**
- Create: `mcp_servers/__init__.py`(空)
- Create: `mcp_servers/logistics.py`
- Create: `mcp_servers/aftersales.py`
- Test: `tests/test_mcp_servers.py`(新建,**进程内**,不联网)

**Interfaces:**
- Consumes: `app.tools.mock_data` 的 `order_record` / `logistics_record` / `require_order_no` / `rng`
- Produces:
  - `mcp_servers.logistics.mcp`(`FastMCP` 实例,含工具 `query_logistics`)
  - `mcp_servers.aftersales.mcp`(含工具 `query_warranty` / `query_return_progress`)

**为什么能在单测里不联网验**:mcp 1.30.0 的 `FastMCP` 有 `list_tools()` /
`call_tool()` 两个**纯内存方法**(源码里逐字核对过)—— 不起进程、不走 HTTP。
真进程只在端到端验收脚本里起。

- [ ] **Step 1: 写失败测试**

新建 `tests/test_mcp_servers.py`:

```python
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
    e = await aftersales.call_tool("query_return_progress", {"order_id": "1003"})
    assert json.dumps(c, ensure_ascii=False, default=str) != json.dumps(
        e, ensure_ascii=False, default=str
    )
```

- [ ] **Step 2: 跑测试,确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_mcp_servers.py`
Expected: FAIL —— `ModuleNotFoundError: No module named 'mcp_servers'`

- [ ] **Step 3: 建 `mcp_servers/__init__.py`**

```python
"""两个业务 MCP Server —— **各自独立进程**。

启动(各自一个终端):

    .venv/Scripts/python.exe -m mcp_servers.logistics     # 127.0.0.1:8101/mcp
    .venv/Scripts/python.exe -m mcp_servers.aftersales    # 127.0.0.1:8102/mcp

**不接真实系统、不建表** —— 数据全部来自 `app.tools.mock_data`
(与内置工具**共用**,详见该模块的 docstring 与 spec §8.2)。
"""
```

- [ ] **Step 4: 建 `mcp_servers/logistics.py`**

```python
"""物流 MCP Server。

**`query_logistics` 在本章从内置下线,由本 Server 接管**(spec §8.3)。
名字**保留不变** —— 评估集与提示词里的工具名口径不跟着漂。

启动:`.venv/Scripts/python.exe -m mcp_servers.logistics`
"""

import json

from mcp.server import FastMCP

from app.tools.mock_data import logistics_record, require_order_no

#: ⚠️ `FastMCP` 的传输参数是**直接关键字参数**,不是 `FastMCP(..., settings=Settings(...))`
#: —— 1.30.0 的 `__init__` 逐字核对过。同级那个也叫 `Settings` 的 pydantic 模型
#: 有若干**无默认值**的字段,照猜会踩进去。
#:
#: `stateless_http=True` 是**刻意的**:客户端每请求建连接,有状态模式会让
#: session 堆在 Server 侧(`max_sessions` 迟早成为一处没人会想到的故障点)。
#: `json_response=True`:这个 Server 只服务工具调用,不需要 SSE 流式响应。
mcp = FastMCP(
    "物流服务",
    host="127.0.0.1",
    port=8101,
    streamable_http_path="/mcp",
    stateless_http=True,
    json_response=True,
)


@mcp.tool()
async def query_logistics(order_id: str) -> str:
    """查询订单的物流状态、当前位置与轨迹。用户问"到哪了""发货没"时使用。"""
    return json.dumps(
        logistics_record(require_order_no(order_id)), ensure_ascii=False
    )


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 建 `mcp_servers/aftersales.py`**

```python
"""售后 MCP Server:查在保、查退货进度。

⚠️ **已知偏离(spec §8.6)**:本 Server「不接真实系统、不建表」,两个工具
返回的都是**伪随机 mock**。它与 ch06 的 `refund_requests` 表**无语义关联**
—— 用户刚在退款表单里提交的那条,来这里查进度会得到**另一套随机结果**。
演示时**不要**拿它当真实进度用。

要让它接真实数据,得让 Server 连 MySQL,那直接违反上面那条选型 —— 留作
将来单独一章的事。

启动:`.venv/Scripts/python.exe -m mcp_servers.aftersales`
"""

import json

from mcp.server import FastMCP

from app.tools.mock_data import order_record, require_order_no, rng

mcp = FastMCP(
    "售后服务",
    host="127.0.0.1",
    port=8102,
    streamable_http_path="/mcp",
    stateless_http=True,
    json_response=True,
)

_WARRANTY_STATES = ["保修中", "已过保", "延保中"]
_RETURN_STAGES = ["已受理", "待寄回", "已寄回", "质检中", "退款中", "已完成"]


@mcp.tool()
async def query_warranty(order_id: str) -> str:
    """查询某订单商品的保修状态与到期时间。用户问"还在保修吗""过保没"时使用。"""
    order_no = require_order_no(order_id)
    # 经 `order_record` 取商品 —— **不要另起一条随机流**去抽商品名,
    # 那正是 ch02 记过的自相矛盾源头(同一个订单在两处显示不同商品)。
    order = order_record(order_no)
    r = rng("warranty", order_no)
    state = r.choice(_WARRANTY_STATES)
    return json.dumps(
        {
            "order_id": order_no,
            "product": order["product"],
            "warranty_state": state,
            "expires_on": f"2027-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}",
        },
        ensure_ascii=False,
    )


@mcp.tool()
async def query_return_progress(order_id: str) -> str:
    """查询退货申请的处理进度。用户问"我的退货到哪一步了""退款什么时候到"时使用。

    ⚠️ 返回的是**伪随机 mock**,与 ch06 的 `refund_requests` 无语义关联(见模块 docstring)。
    """
    order_no = require_order_no(order_id)
    r = rng("return", order_no)
    stage = r.choice(_RETURN_STAGES)
    return json.dumps(
        {
            "order_id": order_no,
            "stage": stage,
            "updated_at": f"2026-{r.randint(1, 9):02d}-{r.randint(10, 28):02d}",
        },
        ensure_ascii=False,
    )


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 跑测试**

Run: `.venv/Scripts/python.exe -m pytest tests/test_mcp_servers.py`
Expected: PASS

⚠️ 若 `await logistics.list_tools()` 返回的对象上没有 `inputSchema` 而下划线命名
(`input_schema`),**以实际为准改测试**,并在本任务的报告里记一句 ——
这属于「版本对不上」的一类,别猜。

- [ ] **Step 7: 真机冒烟(起一次真进程)**

```bash
.venv/Scripts/python.exe -m mcp_servers.logistics &
sleep 3
curl -s -X POST http://127.0.0.1:8101/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
kill %1
```

Expected: 一串 JSON,里面有 `query_logistics`。
**这一步不能省** —— 本仓记过:「凡是对外部系统的调用,单测绿了还要真机冒烟一次」。

- [ ] **Step 8: 提交**

```bash
git add mcp_servers tests/test_mcp_servers.py
git commit -m "feat(ch08): 物流 / 售后两个业务 MCP Server(FastMCP + Streamable HTTP)"
```

---

### Task 7: MCP Client —— 每请求发现 + 单 Server 降级

**Files:**
- Create: `app/mcp/__init__.py`(空)
- Create: `app/mcp/client.py`
- Modify: `app/tools/registry.py`(`build_registry` 接 `extra`)
- Modify: `app/tools/builtin/orders.py`(**删掉 `query_logistics`**)
- Modify: `app/api/chat.py`(await 发现,喂给注册表)
- Modify: `tests/test_builtin_discovery.py`(删掉 `query_logistics` 那条,见 T3 的说明)
- Modify: `tests/test_tools_random.py`(子进程那条改指 `mcp_servers.logistics`)
- Test: `tests/test_mcp_client.py`(新建)

**Interfaces:**
- Consumes: `spec.ToolSpec`(T2)、每个 `ToolSpec` 由 `kind_of` 定 `kind`(T2)、
  `errors.TransientToolError`(T4)
- Produces:
  - `client.discover_mcp_specs(*, settings) -> list[ToolSpec]`(**async**)
  - `registry.build_registry(*, session, conversation_id, settings=None, extra=None)`

**⚠️ 本任务之后 `query_logistics` 只有一个提供者(物流 MCP Server)。**
删内置那份是**必须的**:重名的表现是「其中一个静默胜出」,
而谁胜出取决于排序 —— 没人查得出来(注册表的 `_dedupe` 会抛,但别依赖它兜底)。

- [ ] **Step 1: 写失败测试**

新建 `tests/test_mcp_client.py`:

```python
"""MCP 客户端:**每请求现问现拿** + **单 Server 降级** + **原始 schema 保真**。

⚠️ 不联网:整个 `MultiServerMCPClient` 被替身换掉。
"""

import pytest

from app.mcp import client as mcp_client
from app.tools.registry import build_registry


class _FakeMCPTool:
    def __init__(self, name, schema):
        self.name = name
        self.description = f"{name} 的用途"
        self.inputSchema = schema


class _FakeListed:
    def __init__(self, tools):
        self.tools = tools


class _FakeSession:
    def __init__(self, tools):
        self._tools = tools

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def list_tools(self):
        return _FakeListed(self._tools)


class _FakeClient:
    """`connections` → 每台服务器给什么工具。`raises` 里的服务器连接必失败。"""

    def __init__(self, connections, per_server, raises=()):
        self.connections = connections
        self._per_server = per_server
        self._raises = set(raises)

    def session(self, name, **_kw):
        if name in self._raises:
            raise ConnectionError(f"{name} 连不上")
        return _FakeSession(self._per_server[name])


@pytest.fixture
def patch_client(monkeypatch):
    def _install(per_server, raises=()):
        monkeypatch.setattr(
            mcp_client, "MultiServerMCPClient",
            lambda connections: _FakeClient(connections, per_server, raises),
        )
        monkeypatch.setattr(
            mcp_client, "convert_mcp_tool_to_langchain_tool",
            lambda session, tool, **kw: _FakeLC(tool.name),
        )
    return _install


class _FakeLC:
    def __init__(self, name):
        self.name = name
        self.description = f"{name} 的用途"
        self.args_schema = None


class _Settings:
    mcp_logistics_url = "http://127.0.0.1:8101/mcp"
    mcp_aftersales_url = "http://127.0.0.1:8102/mcp"
    mcp_discovery_timeout_seconds = 5.0


_SCHEMA = {
    "type": "object",
    "properties": {"order_id": {"type": "string", "minLength": 4}},
    "required": ["order_id"],
}


@pytest.mark.anyio
async def test_discovers_tools_from_both_servers(patch_client):
    patch_client(
        {
            "logistics": [_FakeMCPTool("query_logistics", _SCHEMA)],
            "aftersales": [
                _FakeMCPTool("query_warranty", _SCHEMA),
                _FakeMCPTool("query_return_progress", _SCHEMA),
            ],
        }
    )
    specs = await mcp_client.discover_mcp_specs(settings=_Settings())
    assert {s.name for s in specs} == {
        "query_logistics", "query_warranty", "query_return_progress"
    }


@pytest.mark.anyio
async def test_source_records_which_server(patch_client):
    """审计要记「来源是内置还是哪个 MCP Server」(要求 5)—— 就是这里给的。"""
    patch_client(
        {"logistics": [_FakeMCPTool("query_logistics", _SCHEMA)], "aftersales": []}
    )
    specs = await mcp_client.discover_mcp_specs(settings=_Settings())
    assert specs[0].source == "mcp:logistics"


@pytest.mark.anyio
async def test_raw_schema_survives_untouched(patch_client):
    """**本章最容易静默失效的一条**(spec §3.3)。

    走 adapters 的 pydantic 转换会把 `minLength` 这类约束削平,于是
    「统一按 JSON Schema 校验」退化成「只查必填和类型」,闸看起来在工作、
    实际漏掉一半。断言的是**原始约束还在**,不是「有个 schema 键」。
    """
    patch_client(
        {"logistics": [_FakeMCPTool("query_logistics", _SCHEMA)], "aftersales": []}
    )
    specs = await mcp_client.discover_mcp_specs(settings=_Settings())
    assert specs[0].input_schema == _SCHEMA
    assert specs[0].input_schema["properties"]["order_id"]["minLength"] == 4


@pytest.mark.anyio
async def test_one_dead_server_does_not_kill_the_other(patch_client):
    """降级(spec §8.5,用户 2026-09-22 拍板)。

    ⚠️ 断言的是**另一个 Server 的工具还在**,不是「没抛异常」——
    一个把所有 Server 都丢掉的实现同样「没抛异常」。
    """
    patch_client(
        {"logistics": [_FakeMCPTool("query_logistics", _SCHEMA)], "aftersales": []},
        raises=("logistics",),
    )
    specs = await mcp_client.discover_mcp_specs(settings=_Settings())
    assert "query_logistics" not in {s.name for s in specs}
    # aftersales 这边本来就没工具 —— 换一个真有工具的来证
    patch_client(
        {
            "logistics": [_FakeMCPTool("query_logistics", _SCHEMA)],
            "aftersales": [_FakeMCPTool("query_warranty", _SCHEMA)],
        },
        raises=("logistics",),
    )
    specs = await mcp_client.discover_mcp_specs(settings=_Settings())
    assert {s.name for s in specs} == {"query_warranty"}


@pytest.mark.anyio
async def test_both_dead_yields_empty_not_an_exception(patch_client):
    patch_client(
        {"logistics": [], "aftersales": []}, raises=("logistics", "aftersales")
    )
    assert await mcp_client.discover_mcp_specs(settings=_Settings()) == []


def test_mcp_tool_colliding_with_a_builtin_loses_without_raising():
    """⚠️ **这条是 T7 定稿后补的**(实现者上报的可用性风险)。

    外部 Server 的**名字**和它们的用途声明一样不可信。撞上内置名就上抛的话,
    **外部只要起一个叫 `query_order` 的工具,每一个聊天请求都会 500** ——
    那是验收 3 的反面(在 Server 侧加工具本该**不需要动客服系统**),
    而且方向错了:外部能让我们的内置工具消失。

    断两件事:**内置还在**,且**没有抛**。
    """
    from app.tools.spec import ToolSpec

    extra = [
        ToolSpec(name="query_order", description="外部的冒牌货",
                 input_schema=_SCHEMA, kind="read", source="mcp:logistics", tool=None)
    ]
    reg = build_registry(
        session=None, conversation_id="c1", settings=None, extra=extra
    )
    assert reg["query_order"].source == "builtin", "内置必须赢"


def test_two_mcp_servers_colliding_drops_the_later_one():
    from app.tools.spec import ToolSpec

    extra = [
        ToolSpec(name="dupe", description="a", input_schema=_SCHEMA,
                 kind="read", source="mcp:logistics", tool=None),
        ToolSpec(name="dupe", description="b", input_schema=_SCHEMA,
                 kind="read", source="mcp:aftersales", tool=None),
    ]
    reg = build_registry(
        session=None, conversation_id="c1", settings=None, extra=extra
    )
    assert reg["dupe"].source == "mcp:logistics", "先到先得"


def test_registry_merges_builtin_and_mcp():
    """`build_registry` 是**纯组装**:MCP 那半由调用方 await 之后喂进来。

    (所以它保持同步、可同步单测 —— 网络 I/O 全在 `app/mcp/client.py` 里。)
    """
    from app.tools.spec import ToolSpec

    extra = [
        ToolSpec(
            name="query_warranty", description="查在保",
            input_schema=_SCHEMA, kind="read", source="mcp:aftersales", tool=None,
        )
    ]
    reg = build_registry(
        session=None, conversation_id="c1", settings=None, extra=extra
    )
    assert "query_warranty" in reg
    assert reg["query_warranty"].source == "mcp:aftersales"
    assert "query_order" in reg          # 内置还在


def test_mcp_specs_come_after_builtin():
    """顺序稳定 ⇒ 工具定义块逐字节相同 ⇒ 前缀缓存命中(spec §3.4)。"""
    from app.tools.spec import ToolSpec

    extra = [
        ToolSpec(name="zz_mcp", description="d", input_schema={},
                 kind="read", source="mcp:zz", tool=None),
    ]
    reg = build_registry(session=None, conversation_id="c1", settings=None, extra=extra)
    names = list(reg)
    assert names.index("zz_mcp") == len(names) - 1
```

> ⚠️ `build_registry` **不要**加任何 `lru_cache` / 装饰器包装:
> 它每请求组装(`query_faq` / `create_ticket` 是每请求闭包),
> 缓存会让上一个会话的工具凭据被下一个会话用上。

- [ ] **Step 2: 跑测试,确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_mcp_client.py`
Expected: FAIL —— `ModuleNotFoundError: No module named 'app.mcp'`

- [ ] **Step 3: 建 `app/mcp/__init__.py` 与 `app/mcp/client.py`**

```python
# app/mcp/__init__.py
"""MCP 客户端接入(app/mcp/)。"""
```

```python
# app/mcp/client.py
"""MCP 客户端:**每请求现问现拿**两个业务 Server 的工具,单个挂了就降级。

**为什么不缓存**(spec §8.4):本地 `list_tools` 是毫秒级,而缓存会引入
「我刚加的工具为什么没生效」这类**只能靠猜**的故障。验收 3 要的正是现问现拿。

**三个会静默变坏的细节**(spec §2.3,逐字核对过 adapters 0.3.2 的签名):

1. `convert_mcp_tool_to_langchain_tool` 传 **`connection=` 而不是 `session=`**
   —— 传 session 的话,那个 session 一关,造出来的工具就废了。
2. `handle_tool_errors` **必须显式关**。默认 `True` 会把 MCP 的调用故障
   **包成一条正常的工具返回**,于是在执行器眼里「物流服务连不上」是**成功** ——
   直接违反本仓那条「基础设施故障绝不伪装成查不到」。
3. 注册表里的 `input_schema` 用**原始的 `inputSchema`**,不走 adapters 的
   pydantic 转换 —— 转换会削平 `minimum` / `enum` 这类约束。
"""

import logging
from datetime import timedelta

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import convert_mcp_tool_to_langchain_tool

from app.tools.policy import kind_of
from app.tools.spec import ToolSpec

logger = logging.getLogger(__name__)


def _connections(settings) -> dict:
    timeout = timedelta(seconds=settings.mcp_discovery_timeout_seconds)
    return {
        "logistics": {
            "transport": "streamable_http",
            "url": settings.mcp_logistics_url,
            "timeout": timeout,
        },
        "aftersales": {
            "transport": "streamable_http",
            "url": settings.mcp_aftersales_url,
            "timeout": timeout,
        },
    }


def _to_spec(*, server_name: str, tool: BaseTool, schema: dict) -> ToolSpec:
    return ToolSpec(
        name=tool.name,
        description=(tool.description or "").strip(),
        input_schema=schema,
        # 未声明的 MCP 工具一律**只读**(app/tools/policy.py,spec §5.2)——
        # 写权限只认我们本地的表,外部 Server 改自己的用途声明拿不到。
        kind=kind_of(tool.name),
        source=f"mcp:{server_name}",
        tool=tool,
    )


async def discover_mcp_specs(*, settings) -> list[ToolSpec]:
    """连上两个业务 Server,**现问现拿**它们的工具清单。

    单个 Server 连不上 ⇒ **跳过它、打一条响亮的 warn、其余照常**(spec §8.5)。
    两个都挂 ⇒ 返回空列表,聊天仍可用(只剩内置工具)。

    **为什么不选「上抛 502」**:工具清单是**能力**,不是**结果**。一个可选插件
    挂掉不该让整个客服不可用;而且缺失是**可见的** —— 模型看不到那个工具,
    会在回复里如实说没有,不会把「服务挂了」伪装成「你查的东西不存在」。
    """
    connections = _connections(settings)
    client = MultiServerMCPClient(connections)
    specs: list[ToolSpec] = []
    for name in sorted(connections):
        try:
            async with client.session(name) as session:
                listed = await session.list_tools()
                for mcp_tool in listed.tools:
                    lc_tool = convert_mcp_tool_to_langchain_tool(
                        None,
                        mcp_tool,
                        connection=connections[name],
                        server_name=name,
                        handle_tool_errors=False,   # 见模块 docstring 第 2 条
                    )
                    specs.append(
                        _to_spec(
                            server_name=name,
                            tool=lc_tool,
                            schema=mcp_tool.inputSchema,
                        )
                    )
        except Exception:                                # noqa: BLE001
            logger.warning(
                "mcp discovery failed server=%s,已跳过该 Server(其余照常)", name,
                exc_info=True,
            )
            continue
    return specs
```

> ⚠️ `mcp_tool.inputSchema` vs `input_schema`:以**跑出来的属性名**为准。
> mcp 1.30.0 的 `MCPTool` 是 pydantic 模型,通常是 camelCase 别名 + 允许
> 填充下划线。若 `call_tool`/`list_tools` 实测报 `AttributeError`,
> 改成 `mcp_tool.input_schema` 并在报告里记一句。

> ✅ **T6 已实测给出答案**:`mcp.types.Tool` 的字段**就是 camelCase `inputSchema`**
> (`input_schema` 不存在、也没有别名),与线上线格式一致。**照本条写即可。**

### ⚠️ 两条 T6 实测出来的、T7 必须知道的事实

1. **`FastMCP.call_tool()` 返回 2-tuple `(list[ContentBlock], dict)`**,
   而它的**返回注解写的是** `Sequence[ContentBlock] | dict[str, Any]` —— **注解与实测不符**。
   照注解写 `result[0].text` 会得到
   `TypeError: Object of type TextContent is not JSON serializable`。(T6 已订正。)
2. **`ToolNotFound` 经 HTTP 回来是 `isError: true` 的一次正常结果,
   不是 JSON-RPC 层的 error。**
   ⚠️ **但 `isError` 在 mcp 1.30.0 里是「通用」的,不能拿它当「业务性未找到」的同义词**:
   **工具名不存在**(`ToolManager.call_tool`)、**入参校验失败**(`lowlevel/server.py`)、
   **出参 schema 不匹配**(同上)全都汇进同一个 `_make_error_result`,
   **形状一模一样**。
   ⇒ **T7 必须靠文案(或先查 `spec is None`)来分辨**,不许只看 `isError`。
   判错的代价:**把「这一单查不到」变成 502** —— 正是本仓那条
   「不许拿服务端故障指责用户输入」的反面。

- [ ] **Step 4: 改 `app/tools/registry.py` 的 `build_registry`**

```python
def build_registry(
    *, session, conversation_id, settings=None, extra=None
) -> dict[str, ToolSpec]:
    """组装本请求的注册表:`name → ToolSpec`。

    `extra` 是 **MCP 那条路**拿回来的规格(由 `app/mcp/client.py` 的
    `discover_mcp_specs` 产出,那是异步的,**由调用方 await 之后喂进来**)。
    注册表本身是**纯组装**,不碰网络 —— 这样它保持可同步单测。

    `retriever` 在这里造好再喂进 `discover` —— 让 `builtin/knowledge.py`
    自己 import `registry` 会成环(registry → builtin → registry)。
    """
    retriever = build_retriever(session)
    specs = [
        _spec_from_tool(tool, source="builtin")
        for tool in builtin.discover(
            session=session, conversation_id=conversation_id, retriever=retriever
        )
    ]
    # 顺序稳定 = 工具定义块逐字节相同 = 前缀缓存命中(spec §3.4):
    # 内置在前,MCP 按 (server, name) 排。
    for spec in sorted(extra or [], key=lambda s: (s.source, s.name)):
        specs.append(spec)
    return _dedupe(specs)
```

- [ ] **Step 5: 删掉内置的 `query_logistics`,更新两处测试**

`app/tools/builtin/orders.py`:删掉 `query_logistics` 的定义与 `build()` 里的它,
以及只有它用得到的 import(`LOGISTICS_BY_STATUS` / `logistics_record` / `CITIES`)。

> ⚠️ **别顺手把 `mock_data.logistics_record` 也删了** —— 物流 MCP Server 还要用它。

`tests/test_builtin_discovery.py`:删掉 `test_query_logistics_is_still_builtin_before_t7`
与 `test_five_builtin_tools_are_registered` 里的 `query_logistics`,两条都改成**四个**工具。

`tests/test_tools_random.py`:两条与物流相关的测试
(`query_logistics` 的返回值、以及第 94 行那条**跨进程**测试)现在指向
`mcp_servers.logistics`。跨进程那条改成:

```python
    code = (
        "import json, sys;"
        "sys.path.insert(0, r'.');"
        "from app.tools.mock_data import logistics_record;"
        "sys.stdout.buffer.write("
        "json.dumps(logistics_record('1002'), ensure_ascii=False).encode('utf-8'))"
    )
```

> **为什么保留这条跨进程测试**:它钉的是「同一订单号在**三个进程**里永远
> 得到同样数据」—— 内置、物流 Server、售后 Server。搬家的只是调用点。

- [ ] **Step 6: 改 `app/api/chat.py`**

```python
from app.mcp.client import discover_mcp_specs
from app.tools.registry import build_registry, build_retriever

# ... 请求路径上,锁拿到之后、流开始之前:
mcp_specs = await discover_mcp_specs(settings=settings)
registry = build_registry(
    session=session, conversation_id=session_id, settings=settings, extra=mcp_specs
)
tools = [spec.tool for spec in registry.values()]
```

⚠️ **顺序**:这段必须在**拿到锁之后、构造 `EventSourceResponse` 之前** ——
与既有 `prepare_turn` 同一条规矩(SSE 一旦 yield 过第一帧,状态码再也改不了)。
给发现过程加一条**硬边界**:`settings.mcp_discovery_timeout_seconds` 已经在
连接配置里,不要再额外套 `asyncio.wait_for`(两处超时会让人分不清是哪条生效)。

- [ ] **Step 7: 跑测试**

Run: `.venv/Scripts/python.exe -m pytest`
Expected: 全绿。

- [ ] **Step 8: 提交**

```bash
git add app tests
git commit -m "feat(ch08): MCP 客户端每请求发现 + 单 Server 降级;query_logistics 下线内置"
```

---

### Task 8: 两个确认节点 + `ChatState` 的两个新通道

**Files:**
- Create: `app/agent/confirm_nodes.py`
- Modify: `app/agent/state.py`(加两个通道)
- Modify: `app/agent/nodes.py`(每轮清零那两行)
- Test: `tests/test_agent_confirm.py`(新建)

**Interfaces:**
- Consumes: `executor.execute_tool(..., write_decision=...)`(T4)、
  `executor.APPROVED` / `executor.DENIED`(T4)
- Produces:
  - `confirm_nodes.make_confirm_write_node()`
  - `confirm_nodes.make_apply_write_decision_node(*, registry, settings)`
  - `ChatState.pending_write: dict`(空 dict = 无)
  - `ChatState.write_decision: str`(空串 = 未决议)

**本任务的节点可以**直接调用单测**(不需要跑整个图)—— 它们就是
「state 进、dict 出」的纯函数,只有 `interrupt()` 那条要 monkeypatch。

- [ ] **Step 1: 写失败测试**

新建 `tests/test_agent_confirm.py`:

```python
"""确认流的两个节点。

⚠️ `confirm_write` 的三态判定是本文件的重点:`_decision` 必须**失败关闭** ——
认不出来的 resume 载荷一律不放行。默认放行的话,前端一个形状写错就**直接建出工单**,
而写操作是不可逆的。
"""

import pytest

from app.agent import confirm_nodes
from app.tools.executor import APPROVED, DENIED


class _Registry:
    def __init__(self, spec):
        self._spec = spec

    def get(self, name):
        return self._spec if self._spec and name == self._spec.name else None

    def __contains__(self, name):        # execute_tool 里 sorted(registry) 要用
        return self.get(name) is not None

    def __iter__(self):
        return iter([] if self._spec is None else [self._spec.name])

    def keys(self):
        return list(iter(self))


class _Settings:
    tool_timeout_seconds = 10.0
    tool_retry_attempts = 2
    tool_retry_delay_seconds = 0.0


def _state(**over):
    base = {
        "conversation_id": "c1",
        "pending_write": {
            "tool_call_id": "call_1",
            "name": "create_ticket",
            "args": {"description": "耳机坏了", "ticket_type": "售后"},
            "preview": {"description": "耳机坏了", "ticket_type": "售后"},
        },
        "write_decision": "",
        "turn_messages": [],
        "messages": [],
    }
    base.update(over)
    return base


# ---- confirm_write:只有 interrupt --------------------------------------


@pytest.mark.anyio
async def test_confirm_write_payload_carries_the_preview(monkeypatch):
    seen: list[dict] = []

    def fake_interrupt(payload):
        seen.append(payload)
        return {"approved": True}

    monkeypatch.setattr(confirm_nodes, "interrupt", fake_interrupt)
    node = confirm_nodes.make_confirm_write_node()
    await node(_state())
    assert seen[0]["frame"] == "ticket_confirm"
    assert seen[0]["preview"] == {"description": "耳机坏了", "ticket_type": "售后"}


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"approved": True}, APPROVED),
        ({"approved": False}, DENIED),
        ({"approved": "true"}, DENIED),      # 字符串不算批准
        ({"approved": 1}, DENIED),           # 真值不算,必须是 True
        ({}, DENIED),
        (None, DENIED),
        ("yes", DENIED),                     # 裸串不是约定形状
    ],
)
def test_decision_fails_closed(payload, expected):
    """**失败关闭**:认不出来的一律不放行。

    写操作不可逆 —— 默认 `APPROVED` 会让一个前端形状写错**直接建出工单**。
    """
    assert confirm_nodes._decision(payload) == expected


@pytest.mark.anyio
async def test_confirm_write_returns_the_decision(monkeypatch):
    monkeypatch.setattr(confirm_nodes, "interrupt", lambda payload: {"approved": True})
    node = confirm_nodes.make_confirm_write_node()
    out = await node(_state())
    assert out["write_decision"] == APPROVED


# ---- apply_write_decision:副作用恰一次 --------------------------------


class _Spec:
    name = "create_ticket"
    kind = "write"
    source = "builtin"
    input_schema = {
        "type": "object",
        "properties": {"description": {"type": "string"}, "ticket_type": {"type": "string"}},
        "required": ["description"],
    }


@pytest.mark.anyio
async def test_approved_writes_once_and_appends_the_tool_message(monkeypatch):
    calls: list = []

    class _Tool:
        async def ainvoke(self, call):
            calls.append(call)

            class _M:
                content = '{"ticket_no": "T-1"}'
            return _M()

    spec = _Spec()
    spec.tool = _Tool()
    monkeypatch.setattr(
        "app.agent.confirm_nodes.record_audit", lambda **kw: None, raising=False
    )
    node = confirm_nodes.make_apply_write_decision_node(
        registry={"create_ticket": spec}, settings=_Settings()
    )
    out = await node(_state(write_decision=APPROVED))
    assert len(calls) == 1
    assert [m.tool_call_id for m in out["turn_messages"]] == ["call_1"]
    assert out["pending_write"] == {}


@pytest.mark.anyio
async def test_denied_does_not_write(monkeypatch):
    calls: list = []

    class _Tool:
        async def ainvoke(self, call):
            calls.append(call)
            raise AssertionError("取消的调用**不许被执行**")

    spec = _Spec()
    spec.tool = _Tool()
    node = confirm_nodes.make_apply_write_decision_node(
        registry={"create_ticket": spec}, settings=_Settings()
    )
    out = await node(_state(write_decision=DENIED))
    assert calls == []
    assert out["pending_write"] == {}


@pytest.mark.anyio
async def test_turn_messages_are_appended_not_replaced(monkeypatch):
    """⚠️ **`turn_messages` 是覆写通道,不是追加通道。**

    它承载的是「本轮产生的**全部**消息」,由 `log_turn` 一次性落库。
    这里只返回 `[tool_msg]` 的话,**那条带 `tool_calls` 的 AIMessage 会被丢掉**
    —— 落库的历史里助手消息凭空少一条,而每一轮的回复看起来都正常。
    (这正是 ch07 记过的「累积 vs 覆写」那处坑的同款。)
    """
    from langchain_core.messages import AIMessage

    prior = AIMessage(content="", tool_calls=[
        {"name": "create_ticket", "args": {"description": "x"}, "id": "call_1"}
    ])

    class _Tool:
        async def ainvoke(self, call):
            class _M:
                content = "{}"
            return _M()

    spec = _Spec()
    spec.tool = _Tool()
    node = confirm_nodes.make_apply_write_decision_node(
        registry={"create_ticket": spec}, settings=_Settings()
    )
    out = await node(_state(write_decision=APPROVED, turn_messages=[prior]))
    assert len(out["turn_messages"]) == 2
    assert out["turn_messages"][0] is prior
```

- [ ] **Step 2: 跑测试,确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_confirm.py`
Expected: FAIL —— `ModuleNotFoundError: No module named 'app.agent.confirm_nodes'`

- [ ] **Step 3: 给 `ChatState` 加两个通道**

`app/agent/state.py`,放在 ch07 那几个通道之后:

```python
    # ---- 建工单确认流(ch08)----
    # ⚠️ **必须在 `ChatState` 里声明**:通道集合由 `StateGraph(ChatState)` 的注解决定,
    # 写没声明的通道 LangGraph **静默丢弃**(只 warning 不抛)—— ch06 的
    # `confidence` 就是这么丢的(T4 的 Critical)。
    #
    # 两个都**连同它们的每轮清零一起落地**(清零在 `nodes.make_resolve_references_node`,
    # ch05–ch07「通道与它的清零必须同处一地」的第四次应用)。漏了清零的后果是
    # **跨轮串味**:checkpointer 是进程级单例、thread_id = session_id,
    # 未写的通道保留上一轮的值 —— 于是**上一轮批准过的写操作,这一轮自动放行**。
    #
    # 注意:**不需要在端点播种**。它们都在**一轮的中途**被写(`agent` 写
    # `pending_write`、`confirm_write` 写 `write_decision`),而续跑路径不重跑
    # `resolve_references` —— 续跑续的是**同一轮**,清零不该发生。
    pending_write: dict          # 空 dict = 没有待确认的写操作
    write_decision: str          # "" = 未决议;APPROVED / DENIED
```

- [ ] **Step 4: 清零(改 `app/agent/nodes.py` 的 `make_resolve_references_node`)**

在已有的 `"order_no": "", "order_data": {}, "refund_decision": None,` 那一组之后加:

```python
            # ch08:建工单确认流的两个槽位。**同处一地**(见 state.py 的说明)。
            "pending_write": {},
            "write_decision": "",
```

- [ ] **Step 5: 写 `app/agent/confirm_nodes.py`**

```python
"""建工单确认流的两个节点。

**为什么拆成两个**(spec §9.1):ch06 实测过 —— **`resume` 时节点从头重跑**,
`interrupt()` **之前**的代码会再执行一遍。所以:

| 节点 | 做什么 | 有模型? | 有 `interrupt()`? |
|---|---|---|---|
| `confirm_write` | **只有 `interrupt()`** | 没有 | 有 |
| `apply_write_decision` | 执行或拒绝那次写调用 | 没有 | 没有 |

真正写 `tickets` 表的动作在 `apply_write_decision` 里,它在 resume **之后**
只跑一次 —— 两个节点合起来才保证「**副作用恰好一次**」。

**为什么不把 `interrupt()` 放进 `agent`**(最省事的那种写法):`agent` 里有模型
调用(`chat_temperature=0.7`),续跑会把模型再问一遍 —— 已推给前端的文本
**再推一遍**,而第二次的工具调用序列**可能与第一次不同** ⇒ 卡片上的预览与
真正落库的工单**对不上**。这是「看起来能跑、只在真实点击时错」的一类故障。
"""

from langchain_core.messages import ToolMessage
from langgraph.types import interrupt

from app.tools.executor import APPROVED, DENIED, execute_tool


def _decision(value) -> str:
    """resume 载荷 → 三态决议。**失败关闭。**

    只认 `{"approved": True}` 这一种形状是**刻意的**:前端一个形状写错
    (比如传了字符串 `"true"`)就会被判成取消,而不是**直接建出工单**。
    写操作不可逆 —— 认不出来的一律不放行。
    """
    if isinstance(value, dict):
        approved = value.get("approved")
    else:
        approved = getattr(value, "approved", None)
    return APPROVED if approved is True else DENIED


def make_confirm_write_node():
    """工单预览闸。**`interrupt()` 之外不干任何事。**

    载荷里的 `frame` 由**它自己**说,端点只做搬运 —— 所以本章
    **一行端点代码都不用改**(ch06 那处设计的直接回报)。
    """

    async def confirm_write(state) -> dict:
        pending = state.get("pending_write") or {}
        decision = interrupt(
            {
                "frame": "ticket_confirm",
                "preview": pending.get("preview") or {},
            }
        )
        return {
            "write_decision": _decision(decision),
            "trace": ["confirm_write"],
        }

    return confirm_write


def make_apply_write_decision_node(*, registry, settings):
    """决议落地:批准就执行一次,取消就落一条「权限拒绝」审计。

    **两条路都往本轮消息里追加一条 ToolMessage** —— 因为那条带 `tool_calls`
    的 AIMessage 已经在 `turn_messages` 里了,**少回灌一个 tool 结果就构成
    「有 tool_calls 没有对应 tool 消息」,上游直接 400**(CLAUDE.md 的硬约束)。
    """

    async def apply_write_decision(state) -> dict:
        pending = state.get("pending_write") or {}
        # ⚠️ **不要写 `or DENIED`。** 空决议说明 `confirm_write` 没跑、或它没写进通道,
        # 那是接线 bug;按 DENIED 处理会在审计表里**谎报一次用户取消** ——
        # 与执行器那条 `!= APPROVED` 闸上抛的理由完全相同,只是层数更高一层。
        # 空串会落进那一支,响亮地抛。
        decision = state.get("write_decision") or ""
        # ⚠️ **`"type": "tool_call"` 这个键必须在。** `BaseTool.ainvoke` 判
        # 「这是不是一次工具调用」**只看它** —— 缺键时它把整个 dict 当成**参数**去
        # 校验工具 schema,于是这次调用退化成一条「参数不合法」的**可恢复**失败:
        # **工单永远不会被建出来**,而调用方看起来一切正常。
        # (T4 的实现者在测试初稿上撞过同一件事,6 条用例红在 pydantic 的
        #  `Field required` 上 —— 那是**测试**;在这里它是**生产**。)
        call = {
            "name": pending.get("name", ""),
            "id": pending.get("tool_call_id", ""),
            "args": pending.get("args") or {},
            "type": "tool_call",
        }
        outcome = await execute_tool(
            tool_call=call,
            registry=registry,
            settings=settings,
            conversation_id=state["conversation_id"],
            write_decision=decision,
        )
        tool_msg = ToolMessage(content=outcome.content, tool_call_id=call["id"])
        # ⚠️ `turn_messages` 是**覆写**通道,承载「本轮产生的**全部**消息」。
        # 这里只返回 `[tool_msg]` 的话,那条带 `tool_calls` 的 AIMessage
        # 会被丢掉 —— 落库的历史里助手消息凭空少一条,而回复看起来完全正常。
        existing = list(state.get("turn_messages") or [])
        return {
            "messages": [tool_msg],
            "turn_messages": existing + [tool_msg],
            "pending_write": {},
            "trace": [
                "write:approved" if decision == APPROVED else "write:denied"
            ],
        }

    return apply_write_decision
```

- [ ] **Step 6: 跑测试**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_confirm.py`
Expected: PASS

- [ ] **Step 7: 提交**

```bash
git add app/agent tests/test_agent_confirm.py
git commit -m "feat(ch08): 建工单确认流的两个节点 + ChatState 两个新通道(连同每轮清零)"
```

---

### Task 9: `agent` 停循环 / 续跑 + 图接线

**Files:**
- Modify: `app/agent/nodes.py`(`agent` 节点的循环与返回)
- Modify: `app/agent/graph.py`(两个新节点 + 条件边)
- Test: `tests/test_agent_graph.py`(改)+ `tests/test_agent_write_flow.py`(新建)

**Interfaces:**
- Consumes: `confirm_nodes.make_confirm_write_node` /
  `make_apply_write_decision_node`(T8)、`ChatState.pending_write` / `write_decision`(T8)
- Produces:
  - `nodes.route_after_agent(state) -> str`(`"confirm_write"` | `"log_turn"`)
  - trace 标记 `agent:write_pending tool=<名>` / `agent:write_resumed`(spec §9.7)

**两条必须同时成立的约束**:

1. **`agent` 必须能「续跑」而不是「重跑」** —— 续跑判定复用**已有的**不变量:
   `turn_messages` 非空 = 这是续跑(ch07 已把 `turn_messages` 放进每轮重置清单)。
   **不新增判断通道。**
2. **少回灌一个 tool 结果就是上游 400** —— 同一轮里若还有别的工具调用,
   它们必须照常执行并回灌。

- [ ] **Step 1: 写失败测试**

新建 `tests/test_agent_write_flow.py`:

```python
"""`agent` 撞到未确认写调用时的「停」与「续」。

⚠️ 每条断言断的都是**只在目标行为发生时才出现的字符串**
(spec §9.7 的 trace 标记)—— **不要断 `agent_steps`**:
ch05 的验收 5 断 `agent_steps >= 2`,而那个名字读作「步数」、实际是
「绑工具轮次的序号且把收敛轮也算进去」,那条断言**零判别力**还漏得掉真回归。
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent import nodes as agent_nodes
from app.tools.executor import APPROVED, ERROR_CONFIRMATION_REQUIRED


class _Chunk:
    """最小 chunk 替身:`.text` + `.tool_calls` + 可相加。

    ⚠️ `tool_call` 条目必须带 `"type": "tool_call"` —— 本仓栽过:
    `BaseTool.ainvoke` 判「这是不是工具调用」**只看**这一个条件。
    """

    def __init__(self, text="", tool_calls=None):
        self.text = text
        self.tool_calls = tool_calls or []

    def __add__(self, other):
        return _Chunk(
            self.text + getattr(other, "text", ""),
            list(self.tool_calls) + list(getattr(other, "tool_calls", []) or []),
        )


class _Model:
    def __init__(self, rounds):
        self._rounds = list(rounds)
        self.calls = 0
        self.bound = None

    def bind_tools(self, tools):
        self.bound = tools
        return self

    async def astream(self, msgs):
        self.calls += 1
        yield self._rounds.pop(0)


class _Settings:
    brand_name = "本店"
    max_agent_steps = 3
    agent_token_budget = 10**9


@pytest.fixture
def patch_ctx(monkeypatch):
    monkeypatch.setattr(agent_nodes, "build_context_messages", lambda **kw: [])
    monkeypatch.setattr(agent_nodes, "count_tokens", lambda s: 1)
    monkeypatch.setattr(agent_nodes, "render_evidence", lambda e: "")
    monkeypatch.setattr(agent_nodes, "journal", type("J", (), {
        "model_ctx": staticmethod(lambda **kw: None)
    }))
    monkeypatch.setattr(
        agent_nodes, "layers", type("L", (), {
            "resplit": staticmethod(lambda h, **kw: None)
        })
    )


def _state(**over):
    base = {
        "conversation_id": "c1",
        "resolved_input": "帮我建个工单",
        "history": [],
        "summary_text": "",
        "evidence": [],
        "summary_upto_msg_id": 0,
        "layer1_from_msg_id": 0,
        "turn_messages": [],
        "pending_write": {},
        "write_decision": "",
    }
    base.update(over)
    return base


@pytest.mark.anyio
async def test_write_call_stops_the_loop_and_records_pending(monkeypatch, patch_ctx):
    """撞到未确认的写调用 ⇒ 停循环 + 记 `pending_write`,**且不发 tool 结果**。"""
    write_call = {
        "name": "create_ticket",
        "args": {"description": "耳机坏了", "ticket_type": "售后"},
        "id": "call_1",
        "type": "tool_call",
    }
    model = _Model([_Chunk("好的。", [write_call])])

    async def fake_execute(**kw):
        from app.tools.executor import ToolOutcome

        return ToolOutcome(
            kw["tool_call"]["id"], kw["tool_call"]["name"], False,
            "需要确认", "需要确认", ERROR_CONFIRMATION_REQUIRED,
            preview=kw["tool_call"]["args"],
        )

    monkeypatch.setattr(agent_nodes, "execute_tool", fake_execute)
    node = agent_nodes.make_agent_node(
        model=model, tools=[], registry={}, settings=_Settings(),
        emit=lambda f: None, context_budget=None,
    )
    out = await node(_state())

    assert out["pending_write"]["tool_call_id"] == "call_1"
    assert out["pending_write"]["preview"]["ticket_type"] == "售后"
    assert "agent:write_pending tool=create_ticket" in out["trace"]
    # **不许**给它回灌 tool 结果 —— 那次调用根本没发生,由
    # `apply_write_decision` 在决议之后补上。
    assert all(not isinstance(m, ToolMessage) for m in out["turn_messages"])
    assert model.calls == 1


@pytest.mark.anyio
async def test_other_calls_in_the_same_round_still_get_tool_messages(
    monkeypatch, patch_ctx
):
    """**同一轮里的只读调用照常执行并回灌。**

    少回灌一个 tool 结果就构成「有 tool_calls 没有对应 tool 消息」,
    上游直接 400 —— 这是 CLAUDE.md 里已有的硬约束。
    """
    read_call = {
        "name": "query_order", "args": {"order_id": "1002"},
        "id": "call_r", "type": "tool_call",
    }
    write_call = {
        "name": "create_ticket", "args": {"description": "x"},
        "id": "call_w", "type": "tool_call",
    }
    model = _Model([_Chunk("", [read_call, write_call])])

    async def fake_execute(**kw):
        from app.tools.executor import ToolOutcome

        name = kw["tool_call"]["name"]
        if name == "create_ticket":
            return ToolOutcome(
                "call_w", name, False, "需要确认", "需要确认",
                ERROR_CONFIRMATION_REQUIRED, preview=kw["tool_call"]["args"],
            )
        return ToolOutcome("call_r", name, True, "{}", "{}")

    monkeypatch.setattr(agent_nodes, "execute_tool", fake_execute)
    node = agent_nodes.make_agent_node(
        model=model, tools=[], registry={}, settings=_Settings(),
        emit=lambda f: None, context_budget=None,
    )
    out = await node(_state())
    ids = [m.tool_call_id for m in out["turn_messages"] if isinstance(m, ToolMessage)]
    assert ids == ["call_r"], "只读那条的回灌丢了"


@pytest.mark.anyio
async def test_continuation_round_is_unbound_and_keeps_turn_messages(
    monkeypatch, patch_ctx
):
    """续跑:**一轮不绑 tools**,并把这条新回复**追加**进 `turn_messages`。

    不绑 tools 是**结构保证** —— 那一轮模型在结构上不可能再触发第二次写。
    """
    prior = AIMessage(content="", tool_calls=[
        {"name": "create_ticket", "args": {"description": "x"}, "id": "call_1"}
    ])
    result_msg = ToolMessage(content='{"ticket_no": "T-1"}', tool_call_id="call_1")
    model = _Model([_Chunk("已为您建单:T-1")])

    node = agent_nodes.make_agent_node(
        model=model, tools=[], registry={}, settings=_Settings(),
        emit=lambda f: None, context_budget=None,
    )
    out = await node(
        _state(turn_messages=[prior, result_msg], pending_write={},
               write_decision=APPROVED)
    )
    assert model.calls == 1
    assert "T-1" in out["reply"]
    assert "agent:write_resumed" in out["trace"]
    # 追加而不是替换
    assert len(out["turn_messages"]) == 3
    assert model.bound is None, "续跑那一轮**不能**绑 tools"


@pytest.mark.anyio
async def test_continuation_does_not_reset_agent_steps(monkeypatch, patch_ctx):
    """⚠️ 续跑时 `steps` 局部变量是 0,直接返回会**把 `agent_steps` 归零**。"""
    prior = AIMessage(content="", tool_calls=[
        {"name": "create_ticket", "args": {"description": "x"}, "id": "call_1"}
    ])
    model = _Model([_Chunk("已建单")])
    node = agent_nodes.make_agent_node(
        model=model, tools=[], registry={}, settings=_Settings(),
        emit=lambda f: None, context_budget=None,
    )
    out = await node(
        _state(turn_messages=[prior], pending_write={}, write_decision=APPROVED,
               agent_steps=2)
    )
    assert out["agent_steps"] == 2
```

- [ ] **Step 2: 跑测试,确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_write_flow.py`
Expected: FAIL —— `TypeError: make_agent_node() got an unexpected keyword argument`
之类,或断言不成立(`pending_write` 不存在)。

- [ ] **Step 3: 改 `app/agent/nodes.py` 的 `make_agent_node`**

在 docstring 之后、`bound = model.bind_tools(list(tools))` 之前**不动**。
`agent_node` 函数体改成下面这样(把原来那一段整体替换,从 `parts: list[str] = []`
开始到 `return {...}` 结束):

```python
        parts: list[str] = []
        made: list[dict] = []
        trace: list[str] = []
        steps = 0
        usage_total = 0
        new_messages: list = []

        # ---- ch08:续跑判定(复用**已有的**不变量,不新增通道)----------
        # `turn_messages` 已被 ch07 放进 `resolve_references` 的每轮重置清单,
        # 所以「进场时非空」只可能是**一轮的中途**(`apply_write_decision` 刚
        # 追加了一条 tool 结果)。续跑续的是**同一轮**,不该重跑 ReAct 循环。
        existing_turn = list(state.get("turn_messages") or [])
        if existing_turn:
            msgs = msgs + existing_turn
            trace.append("agent:write_resumed")
            # **一轮不绑 tools**:结构上不可能再触发第二次写调用。
            # ⚠️ 用**未绑**的 `model`,不是 `bound`。
            final_acc, used = await _stream_round(model, msgs, parts)
            usage_total += used
            final_acc = (
                final_acc if final_acc is not None
                else AIMessage(content="".join(parts))
            )
            new_messages = existing_turn + [final_acc]
            return {
                "reply": "".join(parts),
                "messages": [final_acc],
                "turn_messages": new_messages,
                # ⚠️ **不要**写 `"agent_steps": steps` —— 续跑路径上 `steps`
                # 仍是初值 0,会把上一半算出来的步数**归零**。
                "agent_steps": state.get("agent_steps") or 0,
                "tool_calls_made": [],
                "usage": {"total_tokens": usage_total},
                "trace": trace,
            }

        needs_final = False
        pending: dict = {}

        for step in range(1, settings.max_agent_steps + 1):
            steps = step
            acc, used = await _stream_round(bound, msgs, parts)
            usage_total += used
            tool_calls = list(getattr(acc, "tool_calls", None) or [])

            if not tool_calls:
                needs_final = False
                msgs.append(acc)
                new_messages.append(acc)     # ← 这一轮的输出就是最终回复,收下
                break

            needs_final = True
            msgs.append(acc)
            new_messages.append(acc)         # ← 带 tool_calls 的 assistant
            for call in tool_calls:
                emit({"frame": "tool_call", "name": call["name"],
                      "args": call["args"], "tool_call_id": call["id"]})
                outcome = await execute_tool(
                    tool_call=call, registry=registry, settings=settings,
                    conversation_id=state["conversation_id"],
                )
                if outcome.error_kind == ERROR_CONFIRMATION_REQUIRED:
                    # 写操作待确认:那次调用**根本没发生** ⇒ 不回灌 tool 结果,
                    # 由 `apply_write_decision` 在决议之后补上。
                    if not pending:
                        pending = {
                            "tool_call_id": call["id"],
                            "name": call["name"],
                            "args": call["args"],
                            "preview": outcome.preview or dict(call["args"]),
                        }
                        trace.append(f"agent:write_pending tool={call['name']}")
                    else:
                        # 同一轮里的**第二个**待确认写调用:它不会有第二次
                        # confirm 机会,但**必须**补一条 tool 结果 ——
                        # 少回灌一个就构成「有 tool_calls 没有对应 tool 消息」,
                        # 上游直接 400(CLAUDE.md 的硬约束)。
                        stub = ToolMessage(
                            content="本轮已有一个写操作待用户确认,本次未执行。",
                            tool_call_id=call["id"],
                        )
                        msgs.append(stub)
                        new_messages.append(stub)
                    continue
                emit({"frame": "tool_result", "tool_call_id": outcome.tool_call_id,
                      "ok": outcome.ok, "summary": outcome.summary})
                tool_msg = ToolMessage(content=outcome.content, tool_call_id=call["id"])
                msgs.append(tool_msg)
                new_messages.append(tool_msg)      # ← 工具结果,层 2 要截的就是它
                made.append({"name": call["name"], "ok": outcome.ok})
                trace.append(f"agent:step{step} tool={call['name']}")

            if pending:
                # 停循环:交给 `confirm_write` → `apply_write_decision` → 回来续跑。
                # **不发收尾那一轮** —— 用户还没确认,现在就作答等于先把话说死。
                return {
                    "reply": "".join(parts),
                    "messages": new_messages,
                    "turn_messages": new_messages,
                    "agent_steps": steps,
                    "tool_calls_made": made,
                    "pending_write": pending,
                    "usage": {"total_tokens": usage_total},
                    "trace": trace,
                }

            if usage_total > settings.agent_token_budget:
                break

        if needs_final:
            final_acc, used = await _stream_round(model, msgs, parts)
            usage_total += used
            new_messages.append(
                final_acc if final_acc is not None else AIMessage(content="".join(parts))
            )

        trace.append("agent:converged")
        return {
            "reply": "".join(parts),
            "messages": new_messages,
            "turn_messages": new_messages,
            "agent_steps": steps,
            "tool_calls_made": made,
            "usage": {"total_tokens": usage_total},
            "trace": trace,
        }
```

**顶部 import 补两个**:

```python
from langchain_core.messages import AIMessage, ToolMessage
from app.tools.executor import ERROR_CONFIRMATION_REQUIRED, execute_tool
```

> ⚠️ `AIMessage` 原来可能没在该文件里直接 import(既有的收尾分支用的是
> `AIMessage(content=...)`)—— 以文件实际内容为准,缺什么补什么。

- [ ] **Step 4: 加路由函数与图接线**

`app/agent/nodes.py` 文件末尾(与 `make_*` 同级,纯函数):

```python
def route_after_agent(state) -> str:
    """`agent` 之后往哪走。

    **判据是 `pending_write` 非空** —— 那条路径上 `agent` 已经停循环、
    没有发收尾那一轮;其余一律照旧汇进 `log_turn`。
    """
    return "confirm_write" if state.get("pending_write") else "log_turn"
```

`app/agent/graph.py`:

```python
from app.agent.confirm_nodes import (
    make_apply_write_decision_node,
    make_confirm_write_node,
)
from app.agent.nodes import route_after_agent
```

**把 `"agent"` 从 `_OUTLETS` 里删掉**(它现在是条件出口):

```python
_OUTLETS = (
    "complaint_reply", "chitchat_reply", "fallback_reply",
    "refund_offer", "refund_explain",
)
```

并把那句注释改成:

```python
#: 出口节点 —— 它们统一汇进 log_turn 再结束。
#:
#: `agent` **不在此列**:ch08 起它是**条件出口** —— 撞到未确认的写调用时
#: 走 `confirm_write`(挂起)→ `apply_write_decision` → **回 agent 续跑** →
#: 才汇进 `log_turn`。把它也连上 `log_turn` 会让挂起那一轮**一半写库、一半没写**。
#:
#: `refund_pick_order` 同样不在此列:它可能停在 `interrupt()` 上。
```

`build_graph` 里加两个节点与三条边:

```python
    graph.add_node("confirm_write", make_confirm_write_node())
    graph.add_node(
        "apply_write_decision",
        make_apply_write_decision_node(registry=registry, settings=settings),
    )
```

```python
    # ch08:写操作确认流。`confirm_write` **只有 interrupt()**(resume 从头重跑,
    # 那个节点里不能有别的事);副作用在执行/拒绝那个节点里,resume 之后只跑一次。
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"confirm_write": "confirm_write", "log_turn": "log_turn"},
    )
    graph.add_edge("confirm_write", "apply_write_decision")
    graph.add_edge("apply_write_decision", "agent")
```

- [ ] **Step 5: 跑测试**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_write_flow.py tests/test_agent_graph.py`
Expected: PASS。

⚠️ `tests/test_agent_graph.py` 里若有「所有出口都连 `log_turn`」这类断言,
它现在**会正确地变红** —— 改成断言 `_OUTLETS` 的成员与
「`agent` 由 `route_after_agent` 导出」两件事,别把 `agent` 加回 `_OUTLETS` 了事。

- [ ] **Step 6: 跑全量测试**

Run: `.venv/Scripts/python.exe -m pytest`
Expected: 全绿。

- [ ] **Step 7: 提交**

```bash
git add app/agent tests
git commit -m "feat(ch08): agent 撞到未确认写调用时停循环,决议后续跑一轮不绑 tools"
```

---

### Task 10: 前端工单预览卡片(Vibe Coding,**不做 TDD**)

**Files:**
- Modify: `app/static/index.html`

**按项目规矩**:纯 UI 页面用 Vibe Coding 直做,不套 brainstorm / TDD / code review。
但**两条硬约束照旧**:① 不新增构建工具链(单文件、原生 JS);
② `resumeWith` 的载荷形状必须与 `confirm_nodes._decision` **逐字对上**
(它只认 `{"approved": true}`,别的形状一律判成取消)。

- [ ] **Step 1: 在 `handleFrame` 的 `switch` 里加一个分支**

紧挨着已有的 `case "order_choice":` 之后:

```javascript
      case "ticket_confirm":
        // ch08:写操作(建工单)待确认。与 order_choice 一样,本轮是**挂起**
        // 而不是答完 —— 不置这个标志的话,streamInto 的 finally 会往气泡里
        // 写一句「(没有返回内容)」,而 resume 复用同一个 ctx、token 走 `+=`,
        // 那句占位符会**永久粘在**卡片和后续回复前面。
        ctx.suspended = true;
        renderTicketPreview(ctx, payload.preview || {});
        break;
```

- [ ] **Step 2: 加渲染函数与样式**

放在 `renderOrderCards` 附近(同一节里),样式接在 `.order-card` 那组 CSS 之后:

```javascript
  // ── 工单预览卡(ch08 建工单确认流)────────────────────
  //
  // 后端在 `confirm_write` 节点里 `interrupt()` 时发来的一帧。
  // 点按钮发 resume,**回复流进同一个气泡** —— 与订单选择器同一条路。
  //
  // ⚠️ 载荷形状必须与后端 `app/agent/confirm_nodes.py::_decision` **逐字对上**:
  // 它只认 `{"approved": true}`,**其余一律判成取消**(失败关闭——
  // 认不出来的形状直接建出工单是不可接受的,写操作不可逆)。
  function renderTicketPreview(ctx, preview) {
    const box = document.createElement("div");
    box.className = "ticket-preview";

    const title = document.createElement("div");
    title.className = "ticket-title";
    title.textContent = "工单预览,请确认后提交:";
    box.appendChild(title);

    const rows = [
      ["工单类型", preview.ticket_type || "(未填)"],
      ["问题描述", preview.description || "(未填)"],
    ];
    rows.forEach(([label, value]) => {
      const row = document.createElement("div");
      row.className = "ticket-row";
      row.innerHTML =
        `<span class="ticket-label">${esc(label)}</span>` +
        `<span class="ticket-value">${esc(value)}</span>`;
      box.appendChild(row);
    });

    const bar = document.createElement("div");
    bar.className = "ticket-actions";
    const mk = (text, approved, cls) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "choice-btn " + cls;
      btn.textContent = text;
      btn.addEventListener("click", () => {
        // 一次性锁定:点过就不再让点第二个(与订单卡片同款)
        bar.querySelectorAll("button").forEach((b) => (b.disabled = true));
        btn.textContent = approved ? "已提交" : "已取消";
        resumeWith({ approved: approved }, ctx);
      });
      return btn;
    };
    bar.appendChild(mk("确认提交", true, "primary"));
    bar.appendChild(mk("取消", false, ""));
    box.appendChild(bar);

    ctx.bubble.appendChild(box);
    scrollToEnd();
  }
```

```css
  .ticket-preview {
    margin-top: 10px; padding: 12px 14px;
    border: 1px solid #d6e4f0; border-radius: 10px; background: #f7fbff;
  }
  .ticket-title { font-size: 13px; color: #33526b; margin-bottom: 8px; }
  .ticket-row { display: flex; gap: 10px; margin-bottom: 6px; font-size: 13px; }
  .ticket-label { flex: 0 0 64px; color: #7b8a99; }
  .ticket-value { flex: 1; color: #22303c; word-break: break-all; }
  .ticket-actions { display: flex; gap: 8px; margin-top: 10px; }
  .choice-btn.primary { background: #1a73e8; color: #fff; border-color: #1a73e8; }
  .choice-btn.primary:hover:not(:disabled) { background: #1668d0; }
```

- [ ] **Step 3: 手工验一遍(唯一能验的方式)**

起服务 → 聊天里说「帮我建个工单,我的耳机坏了」→ 应出现预览卡 →
点「确认提交」→ 回复续写进**同一个气泡**、带工单号。

**顺带手工确认一件事**:卡点过之后按钮禁用,且**没有**「(没有返回内容)」的占位符
粘在卡片前面(那正是 `ctx.suspended` 那个标志要挡的)。

- [ ] **Step 4: 提交**

```bash
git add app/static/index.html
git commit -m "feat(ch08): 前端工单预览卡片(确认提交 / 取消,复用同一个气泡续写)"
```

---

### Task 11: 配置项 + 端到端验收 1–6

**Files:**
- Modify: `app/config.py`
- Create: `scripts/acceptance_ch08.sh`
- Test: `tests/test_config.py`(改)

**Interfaces:**
- Consumes: 前面全部
- Produces:`settings.mcp_logistics_url` / `mcp_aftersales_url` /
  `mcp_discovery_timeout_seconds`;`tool_retry_attempts` 默认 **2**

- [ ] **Step 1: 改 `app/config.py`**

> ⚠️ **这三个字段 T7 已经加过了。** 我把任务顺序排错了:`app/mcp/client.py` 的
> `_connections` 读它们,而那时 T11 还没跑 ⇒ **每一个聊天请求都会 AttributeError**。
> T7 的实现者按本步的代码逐字补上并做了标记。
> **所以本步的正确动作是「确认它们已存在、值与下面逐字一致」,不是再定义一遍**
> —— 重复定义在 pydantic 里是**后一个覆盖前一个**,看不出错。
> 仍然归本任务的是后面那一条:`tool_retry_attempts` 的默认值 **1 → 2**。

```python
    # ---- ch08:MCP 接入 ----
    # 两个 URL 给本地演示的默认值(端口与 `mcp_servers/` 两张表一致)。
    # 发现超时**给界** —— 它挂在**请求路径上**(每请求现问现拿),
    # 写错会让每个请求都卡住。
    mcp_logistics_url: str = "http://127.0.0.1:8101/mcp"
    mcp_aftersales_url: str = "http://127.0.0.1:8102/mcp"
    mcp_discovery_timeout_seconds: float = Field(default=5.0, gt=0)
```

并把:

```python
    tool_retry_attempts: int = Field(default=1, ge=0)
```

改成(连同它上面那段注释一起改):

```python
    # ch08 把默认值从 1 提到 2(共 3 次尝试,最坏 20.3s → 30.3s)。
    # **这是跨章行为变更** —— 它是全局旋钮,ch03–ch07 的耗时一并变了。
    # 一句话回退:`.env` 里 `TOOL_RETRY_ATTEMPTS=1`。
    # ⚠️ 写操作**永不重试**是结构保证(由 kind 推出),不受这个数影响。
    tool_retry_attempts: int = Field(default=2, ge=0)
```

`tests/test_config.py` 里若有断言默认值的,同步改 —— 并且**再加一条**:

```python
def test_tool_retry_attempts_default_is_two():
    """ch08 拍板值。改它要连着 spec §6.2 一起改(跨章行为变更)。"""
    assert Settings(_env_file=None, **REQUIRED).tool_retry_attempts == 2


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_mcp_discovery_timeout_must_be_positive(bad):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, mcp_discovery_timeout_seconds=bad, **REQUIRED)
```

(`REQUIRED` 用该文件里已有的那组必填字段字典,**别另造一份**。)

- [ ] **Step 2: 写 `scripts/acceptance_ch08.sh`**

**照 `scripts/acceptance_ch07.sh` 的形状写**(那份 892 行,是本章的模板)。
必须沿用的五条:

1. **`KEEP` 与 `FAIL` 分离**,`fail_exit()` 先 `KEEP=1` **再**退出 ——
   失败路径**不许删自己的证据**(ch07 栽过)。
2. **`EXIT` 与 `INT TERM` 分开 trap**。
3. **中文 needle 用 `chr()` 从十六进制码点构造**,不写进脚本字面量。
4. **`wait_ready` 用墙钟 + `curl --max-time`**(不用固定 `sleep`)。
5. **断言前先 `join_tokens` 把 SSE 帧拼回**(逐 token 推送,`20240915` 会被切碎)。

验收 1–6 逐条:

| # | 题面 | 断言(逐字) |
|---|---|---|
| 1 | 往 `app/tools/builtin/` 写一个 `echo_note` 工具 → **重启客服服务** → 聊天里让它回显一句话 | 回复里有那句话;**且**审计表里该调用 `source='builtin'`、`tool_name='echo_note'` |
| 2 | 问「订单 1002 到哪了」(挑一个**真的发了货**的单号) | done 帧拼回后非空,**且**审计里该调用的 `source='mcp:logistics'` |
| 3 | 在 `mcp_servers/logistics.py` 里加一个工具 → **只重启该 Server** → 问对应问题 | 新工具可调,**且客服服务进程号没变** |
| 4 | 「帮我建个工单」但不说问题 → Agent 追问 → 补一句 → 出现卡片 → 点「确认提交」 | `tickets` 表多一行 + 回复里有 `ticket_no` + 审计里该 `create_ticket` 是 `success` |
| 5 | 同一条路径,这回点「取消」 | `tickets` **没**多行 + 审计里该 `create_ticket` 是 `permission_denied` |
| 6 | 用 `TOOL_TIMEOUT_SECONDS=0.001` 起服务,让一个只读工具超时;再让写操作超时 | 只读那条:`status='timeout'`、`retry_count=2`、`duration_ms>0`;写那条:**`retry_count=0`** |

**验收 3 的进程号怎么断**:

```bash
BEFORE=$(cat "$WORK/cs.pid")
# ... 重启 MCP server,再问一次 ...
AFTER=$(cat "$WORK/cs.pid")
[ "$BEFORE" = "$AFTER" ] || fail "客服服务被重启了(验收 3 要求它不动)"
```

> ⚠️ **验收 5/6 查审计表时,必须带上本次会话的 `conversation_id` 过滤。**
> T5 的实现者上报了一条跨任务影响:审计写口一落地,**整套测试**(6 个既有文件)
> 每轮会往真库写 ~150 行只追加的审计行。验收脚本若按「查最近这几条」去查,
> 撞上哪一条全看运气 —— 那正是本仓记过的第 (d) 类假绿
> (**被上次运行的数据污染**),而且它的表现是「偶尔红、偶尔绿」。
> 脚本自己知道本次的 `session_id`,过滤它是顺手的。

**验收 1/3 都要临时改文件** —— 用 `trap` 保证**无论成败都还原**,
并且**还原也走 `git checkout`**(不是手写删文件),这样「还原失败」自身会响亮报错。

**验收 6 的写操作超时**:走确认流点「确认提交」,但服务以
`TOOL_TIMEOUT_SECONDS=0.001` 启动 —— 那条 `create_ticket` 会超时。
**注意它仍然只执行一次**(写操作不重试),`tickets` 表**可能多一行也可能不多**
(超时未必没执行)—— 所以验收 6 **只断审计行**,**不要**断 `tickets` 的增减。

- [ ] **Step 3: 起服务,实跑整份验收**

```bash
# 先清端口(见到多个残留 uvicorn 全部清掉)
.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000 &
bash scripts/acceptance_ch08.sh
```

Expected: 逐条打印通过/失败;结束时给出 `N 通过 / M 失败`。

- [ ] **Step 4: 提交**

```bash
git add app/config.py scripts/acceptance_ch08.sh tests/test_config.py
git commit -m "test(ch08): 配置项 + 端到端验收 1–6"
```

---

### Task 12: 章级文档同步

**Files:**
- Modify: `CLAUDE.md`
- Modify: `AGENTS.md`(若它引用了工具清单)
- Modify: `dev-notes/ch08.md`(收尾段)
- Modify: `docs/superpowers/specs/2026-09-22-ecommerce-cs-ch08-tools-design.md`(补 §15)

- [ ] **Step 1: `CLAUDE.md` 的「项目」段加一条 ch08**

**照前七章那条的写法**:分支名、一句话定位、关键文件、以及与需求/选型的偏离。

- [ ] **Step 2: `CLAUDE.md` 的「架构」段更新**

- `app/tools/` 那一行的描述改成:`spec`(ToolSpec + 唯一校验器)/ `policy`(权限声明)
  / `audit`(唯一写口)/ `builtin/`(自动发现)/ `registry`(组装)/ `executor`(唯一执行点);
- 依赖方向图里加 `app/mcp/`(Client)与 `mcp_servers/`(两个独立进程的 Server);
- 补一句 **`app/tools/` 不依赖 LangChain 之外的东西、也不依赖 `app/mcp/`**
  —— 两者靠 `ToolSpec` 交接。

- [ ] **Step 3: `CLAUDE.md` 的「硬约束」段加本章的**

至少这几条(每条都要写清「报错指向别处」的那个形态):

- `mcp<2` 是 adapters 的硬依赖;1.x 的入口是 `FastMCP`,**Context7 整站已迁 v2**,
  这一处以锁定版本的轮子源码为准;
- `convert_mcp_tool_to_langchain_tool` 要传 **`connection=`**;
  **`handle_tool_errors=False` 必须显式关**,否则 MCP 故障被包成「正常返回」;
- 写操作决议是**三态**,布尔会让取消路径再拿到 `confirmation_required`(验收 5 落空);
- **`retry_count` 记真实重试次数,不是配置值**;
- **`turn_messages` 是覆写通道** —— 续跑与决议节点都必须**读旧值再追加**;
- **`agent` 不再是 `_OUTLETS` 成员**,它是条件出口;
- **`pending_write` / `write_decision` 必须连同每轮清零一起落地**。

并在「数据与产物」段给 `evals/tool_selection_cases.jsonl` 那条**追加限定**:

> **⚠️ 追加限定(2026-09-22,ch08)**:这个 13/15 是在**旧配置**下测得的。
> 本章起有两处变了:① 工具定义的顺序由手写的
> `[query_order, query_product, query_logistics, query_faq, create_ticket]`
> 变成 **(模块名, 工具名) 排序**(`builtin/` 的自动发现);
> ② **`query_logistics` 从内置下线、改由物流 MCP Server 提供**——
> 而 `evals/run_tool_selection_eval.py` 原先只用 `build_tools`(**内置那一半**),
> 于是那 3 条物流用例**在结构上不可能通过**。
> ⇒ T7 把该脚本改成**用生产同款的注册表**(含 MCP 发现,发现失败时同样降级),
> 否则它测的是一个**与生产不再对应**的工具集。
> **引用 13/15 时必须说明这一点,或重跑一次。**

- [ ] **Step 4: `CLAUDE.md` 的「高频命令」段加 ch08 两条**

```bash
# ch08(前置:两个 MCP Server 各自起进程;MySQL)
.venv/Scripts/python.exe -m mcp_servers.logistics     # 127.0.0.1:8101/mcp
.venv/Scripts/python.exe -m mcp_servers.aftersales    # 127.0.0.1:8102/mcp
bash scripts/acceptance_ch08.sh                       # ch08 验收 1–6
```

- [ ] **Step 5: 补 spec §15「实现订正」**

把整个实现过程中**代码与 spec 的偏离**逐条写进去(与 ch03–ch07 同规矩)。
至少要有:

- `FastMCP` 的传输参数是**直接关键字参数**(spec §2.1 初稿写的
  `settings=Settings(...)` 是错的,已订正)—— **这是写计划时逐字核对 `__init__`
  挖出来的**;
- `MCPTool.inputSchema` 的实际属性名(以实测为准);
- 执行器里 `ValidationError` 那道第二道校验的去留。

- [ ] **Step 6: `dev-notes/ch08.md` 收尾段**

记四样(用户关键原话 / 关键产出 / 被拒绝或纠偏了什么 / 翻车与返工),
并如实列出**未做项**。

- [ ] **Step 7: 提交**

```bash
git add CLAUDE.md AGENTS.md dev-notes/ch08.md docs/superpowers/specs
git commit -m "docs(ch08): 章级文档同步 —— CLAUDE.md 架构与硬约束 + spec §15 实现订正"
```

---

## 飞行前冲突扫描(执行前跑完)

> 这是 **subagent-driven-development 的 Setup 步骤**,不是可选项。
> 「扫描干净」这四个字之前必须有下面这些行。

### 逐任务自洽性

| 任务 | 它自己写的测试 vs 它自己写的代码 | 结论 |
|---|---|---|
| T1 | `test_mock_data.py` 断的 `require_order_no` / `order_record` / `rng` 都在 Step 3 的定义里 | 一致 |
| T2 | `test_tool_spec.py` 断的 `validate_args` 五种失败分支都在 `_readable` 里有对应分支 | 一致 |
| T3 | `test_builtin_discovery.py` 断的 `build_registry` 五元组 = Step 7 的实现 | 一致;⚠️ T7 会改其中两条(**已在 T7 Step 5 写明**) |
| T4 | `test_executor_gate.py` 断的六个常量都在 Step 4 顶部定义 | 一致 |
| T5 | 两层测试分工已在 Interfaces 里写死 | 一致 |
| T6 | 断的 `list_tools()` / `call_tool()` 是 mcp 1.30.0 的纯内存方法(已核对源码) | 一致;⚠️ 属性名以实测为准(**已写明**) |
| T7 | 断的降级/MCP 顺序都在 Step 4 的 `build_registry` 里 | 一致 |
| T8 | 断的三态失败关闭在 `_decision` 里 | 一致 |
| T9 | 断的四条 trace 标记都在 Step 3 的代码里 | 一致 |
| T12 | 无测试 | 一致 |

### 跨任务对(共享文件或接口的)

| 对 | 一个产出什么 / 另一个消费什么 | 结论 |
|---|---|---|
| T1→T3 | `mock_data` 的公开名 ← `builtin/orders.py` 的 import | 名称在本计划里逐字给出 |
| T1→T6 | `logistics_record` ← 物流 Server | **T7 不许删它**(已在 T7 Step 5 写明) |
| T2→T4 | `validate_args` / `WRITE` ← 执行器 | 签名一致 |
| T2→T7 | `ToolSpec` / `kind_of` ← MCP 客户端 | 一致 |
| T3→T4 | `build_registry` 产出 `dict[str, ToolSpec]` ← 执行器收同类型 | **`registry` 参数类型变了**,四个调用点同任务改完(**T4 Step 5**) |
| T3→T7 | `build_registry` 的 `extra=` 形参 ← MCP 发现的产出 | T3 **不写** `extra`,T7 才加 —— **两次任务都碰 `build_registry`,T7 的改动是纯增参** |
| T4→T5 | `record_audit` 的**占位** ← T5 换成真实现 | 签名在 T4 里就定死,两处一致 |
| T4→T8 | `APPROVED` / `DENIED` / `execute_tool(write_decision=)` ← 确认节点 | 一致 |
| T8→T9 | `make_*_node` 与两个通道 ← 图接线与 `agent` | 一致 |
| T9→T11 | `pending_write` 流 ← 验收 4/5 | 一致 |
| T3→T9 | `_OUTLETS` **少了 `agent`** ← 条件边接上 | **同一任务内完成**(T9 Step 4) |
| T6→T11 | 端口 8101/8102 ← `Settings` 默认 URL | 两处**都写死同一对端口**,T11 Step 1 已写明 |

### 计划明令、而评审可能当作缺陷的

| 计划里的规定 | 评审的可能反应 | 裁定 |
|---|---|---|
| T3 的 `test_query_logistics_is_still_builtin_before_t7` | 「T7 会删掉它,为什么不直接不写」 | **保留**。T3 与 T7 之间物流查询没有别的提供者,这条测试在那段时间里是**真守卫**;T7 删它是任务内的显式动作 |
| T2 的 `test_unexpected_extra_field_is_allowed` | 「为什么不管多余字段」 | 已在测试 docstring 里写明:**拦它只会白白浪费一轮对话**,而且 JSON Schema 的默认语义本就是「额外的键不校验」 |
| T5 的两层测试(替身 + db) | 「重复」 | 不重复。缺任一层就会出现第 (g) 类假绿(**替身替被测对象完成了语义**),已在 Interfaces 写明分工 |
| T10 不做 TDD | 「跳过了流程」 | **项目规矩明写**:纯 UI 页面用 Vibe Coding 直做 |

---

## 自查(spec 覆盖 / 占位符 / 类型一致)

**spec 覆盖**

| spec 节 | 落在哪个任务 |
|---|---|
| §1 目标 1(注册中心) | T2 + T3 + T7 |
| §1 目标 2(校验) | T2 + T4 |
| §1 目标 3(权限) | T2(`policy`)+ T4(闸)+ T8(决议) |
| §1 目标 4(执行引擎) | T4 |
| §1 目标 5(审计) | T5 |
| §1 目标 6(MCP) | T6 + T7 |
| §1 目标 7(确认流) | T8 + T9 + T10 |
| §2.1 版本冲突 | T1(依赖钉法)+ 本计划各处的「以实测为准」 |
| §2.3 adapters 两个签名细节 | T7 Step 3 |
| §2.5 已满足的两条 | T1(不返工)+ T4(`render_tool_result` 的透传契约) |
| §2.6 种子实现共用 | T1 + T6 |
| §3.2 热重载界线 | T3(自动发现) |
| §3.4 顺序稳定 | T3 Step 7 + T7 Step 1 |
| §6.2 重试 1→2 | T11 Step 1 |
| §6.3 暂时性故障可重试 | T4(`TransientToolError`)+ T7(客户端上抛) |
| §7.4 `retry_count` 语义 | T4 的两条测试 + T5 的 db 测试 |
| §8.5 单 Server 降级 | T7 Step 1 |
| §8.6 已知偏离 | T6 Step 5(模块 docstring)+ T12(spec §15) |
| §9.7 trace 标记 | T9 Step 1/3 |
| §11 DDL | T5 Step 1 |
| §12 配置项 | T11 Step 1 |
| §13.4 验收 1–6 | T11 Step 2 |
| §14 风险 1(跨章变更) | T11 Step 1 的注释 + T12 Step 3 |

**占位符扫描**:全文无 `TBD` / `TODO` / 「类似 Task N」/ 「加上适当的错误处理」。
三处**刻意留的「以实测为准」**(T6 的 `inputSchema` 属性名、T6 Step 6、
T7 Step 3 的同一个属性名)都写明了**改什么、记在哪** —— 那是版本核对的出口,
不是占位符。

**类型一致**:`ToolSpec` 的六个字段名在 T2 定义,在 T3/T7 被构造、T4/T8 被消费,
三处逐字一致;`write_decision` 的三个取值 `PENDING` / `APPROVED` / `DENIED`
在 T4 定义,T8/T9 消费;`pending_write` 的四个键
(`tool_call_id` / `name` / `args` / `preview`)在 T9 里写、T8 里读,**逐字一致**。

