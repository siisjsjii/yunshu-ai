# ch08 设计:把写死的五个工具升级成即插即用的工具系统

> 设计源。写法沿用前七章:**每节都写「为什么这么定」**,而不是只写「定成什么」
> —— 本章几乎每一处都对应一次真实的版本冲突、源码核对或一条实测约束。
>
> 本章把工具层从「`registry.py` 里硬编码的五个 `@tool`」升级成
> **注册中心 + 统一校验 + 权限门 + 单一执行引擎 + 审计留痕 + MCP 接入**,
> 并新增一条**建工单确认流**(LangGraph `interrupt()` + resume)。
>
> 不做 Skill 机制(只是生态概念,不落代码);不接仓储这类更多外部系统。

---

## §1 目标与非目标

### 目标

1. **工具注册中心**。内置与 MCP 来的工具**一视同仁**:一律带齐
   工具名 / 用途描述 / JSON Schema 参数定义三样,登记进同一份工具表。
   内置在服务启动时登记;MCP 工具连上 Server **动态发现、现问现拿**。
   **新工具注册进来就能被主力 Agent 用上,不改核心代码。**
2. **参数校验**。工具执行前统一按 JSON Schema 校验一遍;拦下之后**不抛异常了事**,
   而是把校验说明包成一条工具结果回灌给模型,让它追问用户或重新组织调用。
3. **权限控制**。工具分只读、写两类,写操作由**执行引擎**把门。外部 MCP 工具的
   用途声明是对方自己写的、不可信 —— 能不能调只认我们这侧的规则。
4. **执行引擎**。所有工具调用走同一处:超时、重试、错误分诊、结果格式化。
5. **审计留痕**。每次工具调用落一条,被权限拒、被校验拦的**同样要落**。
6. **MCP 接入**。自建两个业务 MCP Server(物流 / 售后),客服系统作为 MCP Client
   接入,拿回的工具与内置清单合成一份,主力 Agent 一视同仁地调。
7. **建工单确认流**(带前端配套)。客户明确要求建工单才走这条流;执行引擎**不直接执行**,
   用 `interrupt()` 把工单预览推给前端,点了「确认提交」才落 `tickets` 表。

### 非目标(明确不做)

- **Skill 机制**。只是生态概念,本章不落代码。
- **接更多外部系统**(仓储等)。两个业务 MCP Server 足够撑起本章。
- 工具的**热重载**(改完客服系统自己的代码不重启)。§3.2 会把界线划清楚。
- 多轮 Agent Loop、认证 —— 与前七章一致,不动。

---

## §2 设计依据

本章有五条依据,**全部来自本机源码核对或实测**,没有一条是「文档这么说的」。

### 2.1 ⚠️ 两个定死的选型之间存在**版本冲突**,照文档写必然返工

按项目规矩(涉及具体库先查 Context7),核对出的事实:

| 事实 | 证据 |
|---|---|
| `langchain-mcp-adapters` 最新 **0.3.2**,其 `Requires-Dist` 写死 **`mcp<2.0.0,>=1.24.0`** | 轮子 `METADATA` 逐字 |
| `mcp` 在 PyPI 上最新已是 **2.2.0** | `pip index versions mcp` |
| Context7 **整站在讲 v2**:v2 把 `FastMCP` 改名成 `mcp.server.mcpserver.MCPServer`,传输参数从构造函数挪到 `run()` | 站点 `/v2/...` 路径 |
| **连标着 v1 的 library id(`/websites/py_sdk_modelcontextprotocol_io`)返回的也已经是 v2 内容** | 实测两次查询 |

**结论:文档在这一处不可信,改以锁定版本的轮子源码为准。** 直接读 `mcp-1.30.0` 的
wheel 确认 v1 的真实形状:

| 事实 | 证据(wheel 内路径) |
|---|---|
| 1.x 的入口是 **`mcp.server.FastMCP`**(不是 `MCPServer`) | `mcp/server/__init__.py`:`from .fastmcp import FastMCP` |
| 传输参数在**构造函数**里(v1 语义),且是**直接关键字参数** | `FastMCP(name, *, host="127.0.0.1", port=8000, streamable_http_path="/mcp", json_response=False, stateless_http=False, …)` |
| ⚠️ 不是 `FastMCP(..., settings=Settings(...))` | 写实现计划时逐字核对 `__init__` 订正 —— 同级还有一个名字也叫 `Settings` 的 pydantic 模型(`debug` / `log_level` 等**无默认值**),照猜会踩进去 |
| `Settings` 含 `host` / `port` / `streamable_http_path` / `stateless_http` / `json_response` | `mcp/server/fastmcp/server.py` |
| `mcp.streamable_http_app() -> Starlette`(路由 `/mcp`) | 同上 |
| `mcp.run(transport="streamable-http")`(`Literal["stdio","sse","streamable-http"]`) | 同上 |

**依赖钉法:`mcp>=1.24,<2`(解析到 1.30.0)。不要装 2.x。**

### 2.2 `mcp.list_tools()` / `call_tool()` 是**纯内存**方法 ⇒ MCP Server 可进程内单测

同样是 wheel 源码里的一行。这对本项目特别值钱:
**「单测全程不联网」是硬规矩**,而它让「工具注册是否齐全」「入参校验是否生效」
可以在**不起服务、不走 HTTP** 的前提下验完。真起进程只留给端到端验收脚本。

### 2.3 adapters 的两个签名细节(都会静默变坏)

逐字核对 `langchain_mcp_adapters-0.3.2` 的 `tools.py` / `client.py`:

```python
convert_mcp_tool_to_langchain_tool(
    session: ClientSession | None, tool: MCPTool, *, connection: Connection | None = None,
    ..., server_name: str | None = None, tool_name_prefix: bool = False,
    handle_tool_errors: bool = True,
)
client.session(server_name, *, auto_initialize=True)   # asynccontextmanager
```

- **传 `connection=` 而不是 `session=`**:传 session 的话,那个 session 一关,
  造出来的工具就废了。传 connection 则**每次调用自建连接**。
- **`handle_tool_errors=True` 是默认值,且必须显式关掉**。它会把 MCP 的调用故障
  **包成一条正常的工具返回** —— 于是在执行器眼里「物流服务连不上」是**成功**,
  直接违反本仓那条「基础设施故障绝不伪装成查不到」。关掉后异常上抛,由执行器分诊。

### 2.4 `resume` 时节点**从头重跑** ⇒ `interrupt()` 不能待在含模型调用的节点里

ch06 实测(`dev-notes/ch06.md`、`app/agent/refund_nodes.py` 的注释逐字记着):
`resume` 时 `interrupt()` **之前**的代码会再执行一遍。

而 `agent` 节点里有模型调用(`app/agent/nodes.py::_stream_round`,走
`chat_temperature=0.7`)。**把 `interrupt()` 放进 `agent`,续跑会:**
① 模型被再问一遍,已推给前端的文本**再推一遍**;
② 第二次的工具调用序列**可能和第一次不同** ⇒ 卡片上的预览与真正落库的工单**对不上**。

**⇒ 确认节点必须是「只有 `interrupt()`」的独立节点,且 `agent` 要能「续跑」而不是「重跑」。**
这条是 §9 整个图拓扑的来源。

### 2.5 两条**已经满足**、不用返工的现状

- **`json.dumps` 全仓已带 `ensure_ascii=False`**(要求 4 的「中文不转义」那条)。
  本章只补一条守卫测试,不改实现。
- **`tickets` 表已有 `ticket_type` / `description`**(还有 `ticket_no` 主键、`status`、
  `created_at`)。工单预览卡片要的两个字段**不用加列**。

### 2.6 mock 数据必须沿用 ch02 的 `sha256` 种子实现

`app/tools/business.py::_rng` 用 `hashlib.sha256` 而非内置 `hash()`
(后者对 str 每进程随机化,会让「同一订单号永远返回同样数据」在重启后失效)。
**两个 MCP Server 是独立进程** —— 种子实现只要有一点不同,
「订单 1002 的物流」就会在 MCP 与内置之间对不上。见 §8.2 的搬迁方案。

---

## §3 工具注册中心

### 3.1 `ToolSpec` —— 一条登记项

```python
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str        # 用途描述,直接来自工具自身
    input_schema: dict      # **原始 JSON Schema**
    kind: str               # "read" | "write"
    source: str             # "builtin" | "mcp:logistics" | "mcp:aftersales"
    tool: BaseTool          # 绑给模型的那一份
```

注册表 = `dict[str, ToolSpec]`,由 `build_registry(*, session, conversation_id, settings)`
每请求组装(理由同 ch01–ch07:`query_faq` / `create_ticket` 是**每请求闭包**)。

`registry_for()` 保留,签名不变(`list[BaseTool] -> dict[str, BaseTool]`),
但它从 `ToolSpec` 派生 —— 既有调用方一个字不用改。

### 3.2 内置工具:包内**自动发现**,新工具 = 新增一个文件

```
app/tools/builtin/
    __init__.py      discover() —— pkgutil 走遍子模块,收集各自的 SPECS
    orders.py        query_order / query_product
    knowledge.py     query_faq
    tickets.py       create_ticket
```

**为什么不集中声明**:验收 1 要的是「新写一个简单工具,**只做注册动作、不动核心代码**」。
集中声明表意味着「核心代码里加一行」,那就不是「只做注册动作」了。
`pkgutil.iter_modules` 走一遍包,新增文件自动进表 —— 这是验收 1 的**直接依据**。

**热重载的界线 —— ⚠️ 本节初稿写错了,T11 实测订正后重写:**

初稿写的是「内置工具是 `import` 进来的,**新增内置工具需要重启客服服务**;
MCP 工具现问现拿,所以 Server 侧加工具不需要重启」。**前半句是错的。**

**实测(T11 的验收 1,在服务已经跑着的时候把一个新模块丢进 `app/tools/builtin/`)**:
**新工具立刻可调,不需要重启客服服务。** 机制是 `discover()` 每请求跑
`pkgutil.iter_modules(__path__)`(重新扫目录)+ `importlib.import_module`
(**新文件名不在 `sys.modules` 里 ⇒ 真的去 import 一次**)。

⇒ **两条通道在「加工具不用重启客服系统」这件事上是同一的**,
差别只在**工具从哪来**(本地文件 vs 远端 Server),不在热重载能力上。
初稿把「两条不同的通道」讲成了一个**能力差异**,而那个差异**不存在**。

**这个订正让验收 1 更强**:它本来就说「不动核心代码」,
现在连「重启」也不必了 —— 但它**仍然照旧重启一遍**(验收脚本保留那一步),
因为「重启之后仍然可用」也是要守的。

现有五个工具从 `business.py` **机械搬迁**进 `builtin/`,`business.py` 只留 mock 数据源
(mock 数据源本身还要再抽一层,见 §8.2)。

### 3.3 schema 从哪来(这一条决定校验闸是不是空话)

- **内置**:`tool.args_schema.model_json_schema()`。
- **MCP**:从 Server 的 **`inputSchema` 原样取**(经 `client.session(name)` →
  `session.list_tools()`),**不经过 adapters 的 pydantic 转换**。

**为什么 MCP 那条要绕开转换**:adapters 会把 `inputSchema` 转成 pydantic 模型,
而 `minimum` / `maxLength` / `enum` 这类约束**未必全部保留**。约束一丢,
「统一按 JSON Schema 校验」就退化成「只查必填和类型」——
闸看起来在工作,实际漏掉一半。

> 🔴 **上面这条理由已被证伪,而决定另有承重理由**(T7 的实现者主动撤回 + 终审界复审逐行核过)。
>
> **撤回的是什么**:`langchain_mcp_adapters-0.3.2` 的 `tools.py` 里**就是**
> `args_schema=tool.inputSchema`,而 `langchain_core/tools/base.py` 对 **dict** 类型的
> `args_schema` **原样返回** ⇒ `lc_tool.args_schema` **就是** `mcp_tool.inputSchema`,
> **同一个对象**。⇒ 「adapters 的 pydantic 转换会削平约束」**在这条栈上不描述任何代码路径**。
>
> **真正的承重理由(复审替我们找到的)**:`app/tools/registry.py::_spec_from_tool` 走的是
> `tool.args_schema.model_json_schema()` —— 而 MCP 工具的 `args_schema` **是个 dict**,
> 那句会直接 **`AttributeError`**。⇒ 把 MCP 塞进 `_spec_from_tool` 不是「有损」,
> **是根本跑不通**。**决定因此更硬:必须走原始 `inputSchema`,而且今天就是承重的。**
>
> ⚠️ **下面那条「未验证」的说明仍然成立**(它讲的是约束是否真被削平),两份一起看。

> ⚠️ **原论证在本章的实测条件下「未被验证」**(T7 的实现者主动撤回):
> **FastMCP 给这两个 Server 生成的 `inputSchema` 本身就不含 `minLength` / `enum`**,
> 而 adapters 那条路实测给出的 `args_schema` 是个**普通 dict**(`.model_json_schema()`
> 不可达)。⇒ **今天两条路取到的 schema 逐字节相同**,
> `test_raw_schema_survives_untouched` 因此是一条**前瞻性守卫,不是当前的承重断言**。
>
> **决定不变**(取原始 `inputSchema` 今天免费、明天是保险),但**引用这条理由时
> 必须带上「在 FastMCP 生成的 schema 上未观察到差异」这句** ——
> 否则它就是一个「听起来很对、其实没被验证过」的论证,而本仓对这类说法的规矩是
> **要么给可复现证据,要么显式标注「未验证」**。

### 3.4 顺序必须稳定(前缀缓存)

ch07 已把「system + 红线 + 工具定义」固定在消息最前面以保住前缀缓存。
工具清单的顺序**不能随发现顺序漂移**,否则每次请求都重算前缀。规则:

1. 内置工具按 `(模块名, 工具名)` 排序;
2. MCP 工具按 `(server 名, 工具名)` 排序;
3. 内置整体排在 MCP 之前。

在 MCP Server 侧加工具**理应**改变工具定义块(那正是要的效果),不在本条的约束范围。

---

## §4 参数校验

### 4.1 只有一个校验器

```python
def validate_args(spec: ToolSpec, args: dict) -> list[str]:
    """返回给**模型看**的问题列表;空列表 = 通过。"""
```

对 `spec.input_schema`(**原始 JSON Schema**)跑 `jsonschema`,内置与 MCP 走同一条路。

**要新增依赖 `jsonschema`**(用户 2026-09-22 批准)。理由:
「按 JSON Schema 校验」最直接的落法是它;退路(内置用 pydantic、MCP 用 adapters
转出来的 pydantic 模型)会让**两条路的校验强度不一样**;而更硬的一条是
**MCP 那条路根本走不通** —— `registry._spec_from_tool` 走
`tool.args_schema.model_json_schema()`,而 MCP 工具的 `args_schema` **是个 dict**,
那句会 `AttributeError`(§3.3 已订正)。

### 4.2 文案是给模型看的,不是给人看的

现在的实现把 `ValidationError` 直接 `f"工具 {name} 的参数不合法:{exc}"` —— 那是
pydantic 的堆栈式原文。本章改成**逐条、点名到字段**的说明,例如:

> 参数校验未通过:`order_id` 必填但未提供;`quantity` 必须是整数,收到 "三"

配合「回灌给模型」这条要求,模型据此**追问用户**或**重新组织调用**。

### 4.3 校验发生在 `tool.ainvoke` **之前**

不能靠 `BaseTool.ainvoke` 内部的 pydantic 校验当唯一闸:那一步的失败形态是
`ValidationError` 原文,而且 MCP 工具**压根没有** pydantic 模型可依赖
(`args_schema` 是 dict,§3.3 已订正)。**统一校验前置**,
`ainvoke` 内部那次校验退化成第二道(冗余但无害)。

---

## §5 权限模型

### 5.1 本地声明表是唯一真相源

`app/tools/policy.py` 里一份显式的 **工具名 → `read` | `write`** 声明。

- **不看 Server 的用途声明**。外部 MCP 工具的 description 是对方自己写的,不可信。
- **不让模型临场判断**。模型只负责「调不调」,不负责「能不能调」。
- 今天 `write` 只有一条:**`create_ticket`**。

### 5.2 未知 MCP 工具默认**只读**(用户 2026-09-22 拍板)

我们这侧**没有声明过**的 MCP 工具,一律按**只读**放行 ——
能查数据,**永远进不了写操作白名单**。

- 这是验收 3 成立的前提:在 Server 侧加工具、只重启该 Server,客服系统这边
  **代码不动、服务不重启**就能用上。默认拒绝的话,新工具还要回来加一行声明,
  直接与需求 1 / 验收 3 冲突。
- 同时它也是安全的:外部 Server **无法靠改自己的声明**拿到写权限 ——
  写只认我们本地的表。

### 5.3 未确认的写调用:引擎拒绝,且**不审计**

执行引擎遇到 `kind == write` 且**决议是 `pending`**(还没问过用户)的调用,
**不执行**,返回 `confirmation_required`。

**写调用的决议是三态,不是布尔。** 签名是
`execute_tool(..., write_decision: str = "pending")`,取值 `"pending" | "approved" | "denied"`:

| 决议 | 引擎做什么 | `error_kind` | 落审计? |
|---|---|---|---|
| `pending` | 不执行(Agent 那条正常路径从不传) | `confirmation_required` | 否 |
| `approved` | 真执行 | —— | 是 |
| `denied` | **不执行** | `permission_denied` | 是 |

**为什么布尔不够用** —— 这条是自审时抓到的一处真矛盾:
「**没问过**」与「**问过、用户说不**」是两件不同的事。
用 `approved=False` 一个值表达两者的话,§9.4 的**取消路径会再拿到一次
`confirmation_required`** —— 于是取消永远不会被记成「权限拒绝」,
**验收 5 直接落空**,而实现看起来完全正常(`confirmation_required` 是个合法取值)。

**`pending` 这一刻不落审计。** 理由:拦截那一刻**没有任何人拒绝任何事** ——
它是一次「待确认」,不是一次「拒绝」。而验收 5 要的是
「这条 `create_ticket` 状态是权限拒绝」;若拦截也落一行,同一逻辑链条会出现**两行**,
那条断言就从「唯一事实」退化成「其中一行」。
只有**用户点了取消**,才由 §9 的决议节点落 `permission_denied`。

### 5.4 与 ch05 投诉按钮那条路并存

`POST /api/ticket`(投诉流程里前端按钮建单)**走的是同一个执行引擎** ——
它在 `app/api/chat.py` 里调 `execute_tool`。

⚠️ **本节初稿写的是「它不是工具调用,是端点直接落库,一行不动」,那是错的**
(T4 的实现者按代码核实并纠正,见 §15)。它一直是工具调用,只是**没有经过本节的闸**
而已 —— 本节新增的闸一落地,它就**必然**被拦。

**点按钮本身就是用户确认** ⇒ 该调用点显式传 `write_decision=APPROVED`。
那**一行**是这个闸的必要改动,不是新行为:闸只认「有没有确认」,
而按钮点击就是确认。不加的话,投诉按钮会拿到 `confirmation_required` → 502,
**症状与「服务挂了」一模一样** —— 正是本仓最贵的那类故障。

---

## §6 执行引擎

`app/tools/executor.py::execute_tool` 是**所有工具调用的唯一去处**,新增六步流水线。
**顺序是有讲究的**,不是随手排的:

| 步 | 动作 | 失败时 | 落审计? |
|---|---|---|---|
| 1 | 查注册表 | `tool_missing`(不重试) | **否** —— 接线 bug,不是一次调用 |
| 2 | **权限闸**(三态,§5.3):`write_decision == "pending"` | `confirmation_required` | **否**(§5.3) |
| 2b | **权限闸**:`write_decision == "denied"` | `permission_denied` | **是** |
| 3 | **JSON Schema 校验**(§4) | `invalid_args`(不重试) | **是**,`invalid_args` |
| 4 | 执行(超时 / 重试) | `timeout` / `not_found` / 上抛 `ToolInfrastructureError` | **是** |
| 5 | 结果格式化 | —— | —— |
| 6 | 审计落库 | 只 `logger.error`,**不影响返回值** | —— |

### 6.1 重试白名单从写死的集合改成**由 `kind` 推导**

现状是 `RETRYABLE_TOOLS = frozenset({"query_order", ...})` —— 一张**写死的名单**。
本章改成:**`spec.kind == read` ⇒ 可重试;`kind == write` ⇒ 永不重试。**

- 新注册的只读工具**自动**可重试、新写工具**自动**不可重试 ——
  不用改核心代码,这正是「即插即用」的一半。
- 与旧语义等价:旧的四个白名单工具恰好都是只读。
- **写操作不重试是结构保证,不是配置恰好为 0**。超时未必没执行,
  重复执行比失败更糟。验收 6 后半条断的就是这个 —— 把配置调大也照样 0。

### 6.2 重试次数 **1 → 2**(用户 2026-09-22 拍板;**跨章行为变更**)

| 项 | 原值 | 现值 |
|---|---|---|
| `tool_retry_attempts` | 1(共 2 次尝试) | **2(共 3 次尝试)** |

**如实记账**:改的是**全局**旋钮,所以它**同时改变了 ch03–ch07 的运行行为**
(最坏耗时 20.3s → 30.3s)。**一句话回退**:`.env` 里 `TOOL_RETRY_ATTEMPTS=1` 即恢复。

重试是**顺序**的,这段时间**卡在用户的 SSE 流里**。

### 6.3 新增一类**可重试的故障**:MCP 传输类故障

现在的重试循环里,**唯一真的会重试的故障是「超时」** ——
`ToolNotFound` 与 `ValidationError` 立即 `break`,`SQLAlchemyError` 直接上抛。

于是要求里那句「重试只给网络抖动这类暂时性故障」**是句空话**:
连接被拒 / 传输中断走的是通用 `except Exception` → 直接 `ToolInfrastructureError` → 502,
**一次都不会重试**。

本章把 **MCP 的传输类故障**(连接失败、HTTP 层错误)纳入可重试 —— 它才是真正会抖的那种。
**数据库故障仍然不重试**:重试只是把 502 推迟 10 秒,而那 10 秒用户是在等的。

### 6.4 错误分诊扩到六类

`ToolOutcome.error_kind` 由四类扩到六类(只加取值,不改既有语义):

| 取值 | 含义 | 回灌给模型? |
|---|---|---|
| `not_found` | 业务性未找到(可恢复) | 是 |
| `timeout` | 超时(重试可能已用尽) | 是 |
| `invalid_args` | 参数不合 schema | 是(§4.2 的文案) |
| `permission_denied` | 用户点了取消 | 是 |
| `confirmation_required` | 写操作待确认(§5.3) | 否 —— 由 §9 的图路由接住 |
| `tool_missing` | 注册表里没有这个名字 = 接线 bug | 否,上抛 |

**`ToolInfrastructureError` 必须向上抛,不能回灌** —— 这条不变。

### 6.5 结果格式化

要求 4 的三件事,**落在三个不同的地方** —— 如实写清楚,免得后面有人以为
「引擎里有一个函数全干了」:

| 子要求 | 落在哪 | 为什么 |
|---|---|---|
| 只挑回答用得上的字段 | **内置**:工具自身(它们返回的就是手挑过的 JSON)。**MCP**:引擎统一过 `render_tool_result(spec, raw) -> str` | MCP 的返回体是**对方给的、不可控**,不能原样塞进上下文 |
| 内部枚举码翻人话 | **工具自身** | 枚举是我们定义的(`_ORDER_STATUS` 之类),只有工具知道怎么翻;引擎不认识任何业务枚举 |
| JSON 中文不转义 | 已满足(§2.5) | 只补守卫测试 |

`render_tool_result` 的契约是**窄**的:永远返回 `str`;内置那份已经是 `str` ⇒
原样透传(顺带保证 `ensure_ascii=False`),MCP 那份把 MCP 的内容块压成紧凑 JSON。

---

## §7 审计留痕

### 7.1 表结构(§11 有完整 DDL)

一次工具调用一行。**不挂外键**(要求明写):审计是**旁路记录**,
外键会让删会话时审计行被约束住,甚至反过来影响主流程。

**写审计失败不许反过来拦工具执行** —— `record_audit` 用**独立 session**,
失败只 `logger.error`,绝不向上抛。

### 7.2 单一写口

`app/tools/audit.py::record_audit(...)` 是**唯一**写审计的地方,
由执行引擎在固定的两处调用(§6 表格的步 3、步 6)。
**没有第二个调用点** —— 本章的规矩是「不变量放在唯一写口上,不靠每个调用方自觉」。

### 7.3 状态取值与映射

| 审计 `status` | 何时 |
|---|---|
| `success` | 执行成功 |
| `failed` | 业务性落空 / 未预期失败 |
| `timeout` | 超时(含重试用尽) |
| `invalid_args` | 被参数校验拦下 |
| `permission_denied` | 写操作被用户取消 |

### 7.4 ⚠️ `retry_count` 记的是**真实发生过的重试次数**,不是配置值

**`retry_count = 实际尝试次数 − 1`**,由执行器的循环自己计数。

**为什么单列一条**:写成 `settings.tool_retry_attempts` 的话,
一个**第一次就查成功**的查询会被审计成「重试了 2 次」—— 而它看起来完全正常,
没有任何断言会变红。这正是本仓记过的第 (f) 类假绿形状
(**记一个字段之前,先去读它是怎么被赋值的**)。

计划里必须配一条用例:**首次成功 ⇒ `retry_count == 0`。**

### 7.5 哪些**不**落审计

- `tool_missing`(接线 bug,不是一次调用)
- `confirmation_required`(§5.3:没人拒绝任何事)

两张清单都是**刻意**的,不是漏了。

---

## §8 MCP 接入

### 8.1 两个 Server

| 模块 | Server 名 | 工具 | 默认端口 |
|---|---|---|---|
| `mcp_servers/logistics.py` | 物流服务 | `query_logistics(order_id)` | 8101 |
| `mcp_servers/aftersales.py` | 售后服务 | `query_warranty`(查在保)、`query_return_progress`(查退货进度) | 8102 |

- 用 **`FastMCP`(mcp 1.x)** + **Streamable HTTP**,各自独立进程:
  `python -m mcp_servers.logistics`。
- **`stateless_http=True` 是刻意的**:我们每请求建连接,有状态模式会让 session
  堆在 Server 侧(`max_sessions` 迟早成为一处没人会想到的故障点)。
- **`json_response=True`**:这个 Server 只服务工具调用,不需要 SSE 流式响应。

### 8.2 mock 数据源抽成 `app/tools/mock_data.py`,三处共用

`_rng` / `_order_record` / 订单状态表等从 `business.py` 抽到
`app/tools/mock_data.py`;**内置工具与两个 MCP Server 都 import 它**。

**为什么必须共用**:ch02 已确立「`_order_record()` 是订单的**唯一真相源**」。
`query_logistics` 从内置搬进 MCP Server 之后,如果 Server 自带一套随机数,
那么**同一个订单号**在内置 `query_order` 与 MCP `query_logistics` 之间会**对不上** ——
用户问「订单 1002 到哪了」,订单说是「已发货」,物流却报「待付款」。这属于
「系统在说两套话」,演示时一眼就穿帮。

### 8.3 `query_logistics` 从内置**下线**

物流查询由物流 MCP Server 接管。**名字保留 `query_logistics`**
(避免评估集与提示词里工具名口径漂移),但**只能有一处提供它** ——
内置那份必须删掉,否则注册表里出现重名,而重名的表现是「其中一个静默胜出」。

### 8.4 Client:每请求发现,**不缓存**

```python
MultiServerMCPClient(connections)   # 每请求新建
```

- 连接配置从 `Settings` 来(§12)。
- **发现**:`async with client.session(name) as s: resp = await s.list_tools()`,
  取原始 `inputSchema`(§3.3);每个工具用
  `convert_mcp_tool_to_langchain_tool(None, t, connection=conn, server_name=name,
  handle_tool_errors=False)` 造 LangChain 工具 —— **传 `connection` 而非 `session`**(§2.3)。
- **不缓存**:本地 `list_tools` 是毫秒级,而缓存会引入「我刚加的工具为什么没生效」
  这类**只能靠猜**的故障。验收 3 要的正是「现问现拿」。
- `source` 字段记 `f"mcp:{name}"`,进审计。

### 8.5 单个 Server 连不上 ⇒ **降级,不报错**(用户 2026-09-22 拍板)

连接 / 列举失败时:`logger.warning("mcp discovery failed server=%s", …)` + **跳过该 Server**,
其余照常。两个都挂 ⇒ 只剩内置,聊天仍可用。

**为什么不选「上抛 502」**:工具清单是**能力**,不是**结果**。一个可选插件挂掉
不该让整个客服不可用;而且缺失是**可见的** —— 模型看不到那个工具,会在回复里
如实说没有,不会把「服务挂了」伪装成「你查的东西不存在」。

### 8.6 ⚠️ 已知偏离:售后 MCP 与 ch06 的 `refund_requests` **无语义关联**

按定死的选型(「Server 内部照第 2 章的做法随机生成 mock 数据返回,
**不接真实系统、不建表**」),`query_return_progress` 返回的是**伪随机 mock**。

**与 ch06 对不上的地方,如实写在这里**:ch06 的退款表单会往 `refund_requests`
表里落**真数据**,而本 Server 不读那张表 ——
**刚提交的退款,去查进度会得到另一套随机结果**。

演示时**不要**拿它当真实进度用。要让它接真实数据,就得让 Server 连 MySQL,
那直接违反上面那条选型 —— 留作将来单独一章的事。

---

## §9 建工单确认流

### 9.1 图拓扑(方案 A,用户 2026-09-22 拍板)

```
agent ──┬─(pending_write 非空)─► confirm_write ─► apply_write_decision ─► agent(续跑) ─► log_turn
        └─(空)─────────────────────────────────────────────────────────────────────► log_turn
```

| 节点 | 做什么 | 有模型调用? | 有 `interrupt()`? |
|---|---|---|---|
| `agent` | 停在半路的 ReAct;续跑时跑一轮不绑 tools 的收尾 | 有 | **没有** |
| `confirm_write` | **只有 `interrupt()`** | 没有 | 有 |
| `apply_write_decision` | 执行或拒绝那次写调用 + 落审计 | 没有 | 没有 |

**`confirm_write` 里除了 `interrupt()` 什么都不干** —— 照 ch06 那条实测约束
(§2.4:resume 时节点从头重跑)。这也**同时**保证了副作用恰一次:
真正写 `tickets` 表的动作在 `apply_write_decision` 里,它在 resume **之后**只跑一次。
计划里配一条「`create_ticket` 恰好执行一次」的用例,计数器放在
**tool 的 `ainvoke` 边界**上(ch06 的既有做法)。

### 9.2 `agent` 撞到未确认的写调用时**停循环**

`execute_tool` 返回 `confirmation_required` 时,`agent`:

- **不**把这条结果作为 ToolMessage 追加(那次调用根本没发生);
- 把预览载荷写进 `pending_write`,**结束本轮**,交给条件边。

`pending_write` 的载荷:`{"tool_call_id": …, "name": "create_ticket", "args": {…},
"preview": {"ticket_type": …, "description": …}}`。

### 9.3 `confirm_write` 的载荷与帧名

```python
interrupt({"frame": "ticket_confirm", "preview": {...}})
```

**端点一行都不用改**:ch06 的端点已经是
`yield _frame(value.get("frame", "interrupt"), {k: v for k, v in value.items() if k != "frame"})`
—— 帧名由载荷自己说,端点只做搬运,不认字面量。这是 ch06 那处设计的直接回报。

### 9.4 续跑怎么判:用**已有的** `turn_messages` 不变量

`turn_messages` 已经被 ch07 放进 `resolve_references` 的**每轮重置清单**。
所以:**`agent` 进场时 `turn_messages` 非空 ⇒ 这是续跑**。

- 续跑 ⇒ `msgs = build_context_messages(...) + turn_messages`,跑**一轮不绑 tools**。
  **结构上不可能**再触发第二次写调用(那一轮没有工具)。
- 开轮 ⇒ 现状不变。

**不新增判断通道**;复用一个已经存在、已经有测试守着的不变量。

### 9.5 两个新通道,**连同每轮清零一起落地**

`pending_write: dict` 与 `write_decision: str`(`confirm_write` 从 resume 值写它,
`apply_write_decision` 读它,§5.3 的三态),都必须:

1. **在 `ChatState` 里声明** —— 未声明通道的写入被 **LangGraph 静默丢弃**
   (只 warning 不抛)。ch06 因此丢过一整个交付物(`confidence`)。
2. **进 `resolve_references` 的每轮清零** —— checkpointer 是进程级单例、
   `thread_id = session_id`,未写的通道**保留上一轮的值**。漏掉清零 =
   上一轮批准过的写操作,这一轮**自动放行**。

### 9.6 被否的方案,以及否它的证据

> **把 `interrupt()` 放进 `agent` 节点内部**(最省事的一种)。

**否**。依据 §2.4:resume 时节点从头重跑,而 `agent` 里有模型调用 ⇒
文本重复推送 + 工具调用序列可能变化 ⇒ **卡片预览与落库工单对不上**。
这是「看起来能跑、只在真实点击时错」的一类故障,不在本章接受范围内。

### 9.7 trace 标记(验收断的是它们,不是 `agent_steps`)

| 标记 | 何时追加 |
|---|---|
| `agent:write_pending tool=create_ticket` | `agent` 决定停下等确认 |
| `agent:write_resumed` | `agent` 续跑那一轮 |

**为什么单列一条**:ch05 的验收 5 断 `agent_steps >= 2`,而那个名字读作「步数」、
实际是「绑工具轮次的序号且把收敛轮也算进去」—— 那条断言**零判别力**还漏得掉真回归。
本章的验收一律断**只在目标行为发生时才出现的字符串**。

---

## §10 接口契约与前端

### 10.1 `POST /api/chat/stream`(复用 ch06 的 resume,**不新增端点**)

工单确认沿用订单选择器那条路:

```json
{"session_id": "…", "resume": {"approved": true}}
```

- `resume` 分支**不推导预算、不组装上下文**(从 checkpoint 还原)—— ch07 已如此。
- 客户端点「取消」时载荷 `{"approved": false}`。

**取消之后这一轮仍然走完**(`apply_write_decision` 追加一条「用户取消了建单」的
ToolMessage → `agent` 续跑 → 模型自然地说一句)→ 所以**取消也发 done 帧**,
与「挂起」是两回事。

### 10.2 新 SSE 帧:`ticket_confirm`

```json
{"frame": "ticket_confirm", "preview": {"ticket_type": "…", "description": "…"}}
```

### 10.3 前端(纯 UI,按项目规矩用 Vibe Coding 直做)

- 收到 `ticket_confirm` → 渲染**工单预览卡片**:工单类型 + 问题描述 + 两个按钮。
- 「确认提交」/「取消」→ 复用**已有的** `resumeWith({approved: …}, ctx)`,
  回复**续写进同一个气泡**(与订单卡片一致)。
- 侧栏/会话逻辑不动。

### 10.4 `POST /api/ticket` **一行不动**

ch05 投诉流程里「点按钮本身就是用户确认」那条路保持原样(§5.4)。

---

## §11 数据

### 11.1 `db/ch08.sql` —— 新表 `tool_audit_logs`

```sql
-- =============================================================
-- ch08 · 工具调用审计
-- 每次工具调用一行;被权限拒、被校验拦的同样要落。
-- =============================================================

-- 本机 locale 是 cp936,不钉这一行中文 COMMENT 会在客户端侧被重编码(本仓记过这条)。
SET NAMES utf8mb4;

-- 不带 IF NOT EXISTS(与 db/ch03.sql / ch04.sql / ch06.sql / ch07.sql 同规矩):
-- 重复执行要**响亮地失败**,否则「表已存在但形状不对」会被静默咽掉。
--
-- **刻意不挂外键**(要求明写):审计是旁路记录。挂了外键的话,
-- 删会话/删工单会受约束,甚至反过来影响主流程 —— 而审计的职责是**只记不拦**。
CREATE TABLE tool_audit_logs (
  id              BIGINT       NOT NULL AUTO_INCREMENT,
  conversation_id VARCHAR(32)  NOT NULL COMMENT '所属会话',
  tool_call_id    VARCHAR(128) NOT NULL COMMENT '本次调用的 id(模型给的)',
  tool_name       VARCHAR(64)  NOT NULL,
  source          VARCHAR(64)  NOT NULL COMMENT 'builtin | mcp:logistics | mcp:aftersales',
  args            TEXT         NOT NULL COMMENT '调用参数(JSON)',
  result_summary  VARCHAR(500) NOT NULL DEFAULT '' COMMENT '结果摘要',
  status          VARCHAR(32)  NOT NULL COMMENT 'success|failed|timeout|invalid_args|permission_denied',
  error_detail    VARCHAR(500) NOT NULL DEFAULT '',
  retry_count     INT          NOT NULL DEFAULT 0 COMMENT '真实发生过的重试次数(不是配置值)',
  duration_ms     INT          NOT NULL DEFAULT 0,
  created_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_conv (conversation_id),
  KEY idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='工具调用审计(ch08)';
```

`created_at` 上有索引:验收 5/6 都是「查最近这几条」。

**执行顺序**:`init_db.py`(create_all)之后再跑这份 —— 全新建库时
`create_all` 也会**顺带**建出这张表(ORM 侧有同名模型),此时这份 DDL 会**响亮地报 1050**。
**这是刻意的**(与 ch06 的 `refund_requests` 同一个已知取舍),但要在 CLAUDE.md
和 dev-notes 里说清楚,免得被当成脏库。

### 11.2 ORM 侧

`app/db/models.py` 加同名 `ToolAuditLog`(**只做映射**,验收脚本查它)。
**不加外键**,与 DDL 一致。

---

## §12 配置项

### 12.1 新增

```python
# ch08 MCP。两个 URL 给本地演示的默认值(端口与 §8.1 的两张表一致);
# 发现超时给界 —— 它挂在**请求路径上**,写错会让每个请求都卡住。
mcp_logistics_url: str = "http://127.0.0.1:8101/mcp"
mcp_aftersales_url: str = "http://127.0.0.1:8102/mcp"
mcp_discovery_timeout_seconds: float = Field(default=5.0, gt=0)
```

### 12.2 改值(**跨章行为变更**,见 §6.2)

```python
tool_retry_attempts: int = Field(default=2, ge=0)   # 原 1
```

**一句话回退**:`.env` 里 `TOOL_RETRY_ATTEMPTS=1`。

---

## §13 测试与验收口径

### 13.1 单测(全程不联网)

| 主题 | 关键口径 |
|---|---|
| 注册中心 | 自动发现能收到新模块的 `SPECS`;新增一个 `builtin/` 模块**不改核心代码**即可进表(验收 1 的单测版) |
| 顺序稳定 | 同一批工具两次组装,顺序一致(前缀缓存的守卫) |
| 校验闸 | 必填缺失 / 类型不对 / 取值越界**各一条**;断言的是**回灌文案点名到字段**,不是「抛了异常」 |
| MCP schema 保真 | Server 声明的 `enum` / `minimum` **真的出现在注册表里**(否则「统一校验」是空话) |
| 权限 | 未确认的写调用 ⇒ `confirmation_required` 且**不落审计**;取消 ⇒ `permission_denied` 且**落审计** |
| 重试规则 | 写操作把配置改成 2 **仍为 0 次**(结构保证);只读工具可重试 |
| `retry_count` | **首次成功 ⇒ 0**(§7.4);超时用尽 ⇒ 等于真实重试次数 |
| 降级 | 一个 Server 挂 ⇒ **另一个 Server 的工具还在**(不是只断言「没抛异常」) |
| 图 | `create_ticket` **恰好执行一次**(计数器在 `ainvoke` 边界);取消路径**不写 tickets** |
| 新通道 | 两个通道未清零时**跨轮串味**必须能被测出来 |

MCP **Server** 那侧用 `mcp.list_tools()` / `call_tool()` **进程内**验(§2.2),
不起进程、不走 HTTP。真进程只在验收脚本里起。

### 13.2 假绿防线(本章最容易长出来的四条)

1. **`retry_count`**:断言首次成功为 0 —— 否则「等于配置值」的实现照样绿。
2. **写操作不重试**:把 `tool_retry_attempts` **调到 2** 再断言 0 ——
   否则「配置恰好是 0」的实现照样绿。
3. **MCP schema 保真**:断言原始约束**出现在注册表里** ——
   只断言「有 input_schema 这个键」是恒真的。
4. **降级**:断言的是**另一个 Server 的工具仍在** ——
   只断言「没抛异常」的话,一个把所有 Server 都丢掉的实现照样绿。

### 13.3 明确不写测试的三处(如实记账)

- **两个 MCP Server 的 mock 数值本身**(它们复用 ch02 已测的 `mock_data`)。
- **`stateless_http` / `json_response` 这两个开关的实际效果** —— 属「库在情况 Z 下
  表现 Y」类断言,没有可复现证据就不写。
- **前端卡片**(按项目规矩,Vibe Coding 直做,不做 TDD)。

### 13.4 端到端验收:新增 `scripts/acceptance_ch08.sh`

| # | 验收 | 断什么 |
|---|---|---|
| 1 | 新写一个简单工具,只做注册动作,Agent 就能用上 | 新增一个 `builtin/` 模块 → 重启服务 → 对话里调到它 |
| 2 | 问物流轨迹,Agent 通过 MCP 查到 | done 帧有内容 **且** 审计里该调用的 `source = mcp:logistics` |
| 3 | Server 侧加工具、只重启该 Server,客服系统不动 | 新工具可调,**且客服服务进程号没变** |
| 4 | 建工单:先追问 → 卡片 → 确认 → 落库 + 回复带工单号 | `tickets` 多一行 + 回复里有 `ticket_no` |
| 5 | 同一路径点「取消」 | `tickets` 没多行 + 审计里该 `create_ticket` 是 `permission_denied` |
| 6 | 人为超时 | 审计里 `status=timeout`、`retry_count`、`duration_ms` 齐;写操作超时 ⇒ `retry_count=0` |

**验收 6 用演示配置把超时压短**(如 `TOOL_TIMEOUT_SECONDS=0.001`),
否则光等超时就要 30 秒 ×2。

脚本沿用 ch07 的形状:`KEEP` 与 `FAIL` 分离、`fail_exit()` 不删证据、
`EXIT` 与 `INT TERM` 分开 trap、中文 needle 用码点构造、`wait_ready` 用墙钟。

### 13.5 老回归网

`scripts/acceptance.sh`(ch01–ch04)与 `acceptance_ch05/06/07.sh` 的既有口径不动。
**注意 §6.2 的重试次数变更会影响它们的耗时**,但不应影响断言的通过与否。

---

## §14 风险与已知取舍

| # | 风险 | 处理 |
|---|---|---|
| 1 | **重试次数 1→2 是全局变更**,波及前几章 | 已记账 + 一句话回退;验收脚本用演示配置压短超时 |
| 2 | MCP 每请求发现给**请求路径**加了往返 | **快路径已实测**:两 Server 都在 ⇒ 3 个工具 / **0.43s**(0.38–0.43 复现)。**慢路径的 ~4.8s 是「本机」的性质,不是这条链路的性质 —— 引用任何具体秒数都必须带「本机实测」四个字。** 根因(T7 用**与本项目无关的对照端口**定位的):**本机对一个「已关闭的回环端口」调裸 `socket.connect()` 要 ~2.05s** 才拿到拒绝(端口 8101 / 8102 / **9** / **54321** 全是这个数;而一个**在监听**的端口 0.016s 就连上,Milvus 19530)。两个 Server ≈ 2×2.05 + adapters 0.35×2 ≈ 4.83s,**与实测吻合**。⇒ **不是 5s 超时、不是 adapters 重试、也不是「有端点没在拒绝」**(复审查过一遍重试路径,实现者又用对照端口证伪了它)。**可移植的上界只有一个:`mcp_discovery_timeout_seconds` × 2 = 10s —— 但这个上界本身`未实测`**(它对应的是「Server 只吞 SYN 不回」那条路径,而实测走的是「立刻拒绝」那条;且一轮发现里未必只有一次请求,所以它**未必是紧的**)。缓解手段本章不做(验收 3 的「现问现拿」排除了缓存) |
| 3 | 售后 MCP 与 `refund_requests` **无语义关联** | §8.6 明写;演示不拿它当真实进度 |
| 4 | `db/ch08.sql` 在全新库上会报 1050 | §11.1 说明;与 ch06 同一个已知取舍 |
| 5 | 未知 MCP 工具默认只读 = **默认放行** | 这是验收 3 的前提;写权限靠本地表收口,外部 Server 拿不到 |
| 6 | `apply_write_decision` 里写 `tickets` 是**副作用** | 它在 resume 之后只跑一次;配「恰好一次」用例(计数器在 `ainvoke` 边界) |

---

## §15 实现订正

> 本章实现过程中发现的「代码与本文档的偏离」逐条记在这里,与 ch03–ch07 同规矩。
> 收尾时补齐。

*(待填)*
