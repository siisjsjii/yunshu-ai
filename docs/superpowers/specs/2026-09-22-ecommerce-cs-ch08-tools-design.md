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

**追加订正(T11 审查轮 1,把上面这句的范围收窄 —— 它一度被写过头了):**

- **新增**(新文件名)当场生效,见上;
- **修改**一个**已有**的内置模块**仍然必须重启** —— `importlib.import_module`
  对已导入的名字直接返回 `sys.modules` 的**缓存项**,磁盘上的改动不会重读。
  两条都实测过(改 `orders.py` 让 `query_order` 无条件抛,没重启的服务照样
  返回 `success` + 真数据;新增 `zz_echo_note.py` 则没重启就被模型调到了)。

所以准确的说法是:**「改已有的」这一格,内置要重启、MCP 不用** ——
热重载能力**在这一点上确实是不对等的**;「加新的」那一格才是同一的。
上面「不在热重载能力上」只对「新增」成立,别把它读成全称。

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
> **每条的格式:设计说 / 代码做 / 为什么。**
>
> ⚠️ 本章的特殊之处:**偏离的数量远多于前几章,而且其中好几条是「本文档先写下一个说法,
> 后来被实测证伪」**。有一条被证伪的说法在**六处**有回声(§3.3、§4.1、§4.3、`app/tools/spec.py`
> 的 docstring、计划两处),花了整整一轮才清干净。**教训写在最后。**

---

### 15.1 依赖与版本(§2.1)

**① `mcp` 的版本区间是被上游 `Requires-Dist` 钉死的,不是我们选的。**

- **设计说**:「依赖钉法:`mcp>=1.24,<2`(解析到 1.30.0)。不要装 2.x。」
- **代码做**:`requirements.txt` 逐字如此;实测 `langchain-mcp-adapters 0.3.2` 的
  `Requires-Dist` 是 `mcp<2.0.0,>=1.24.0`,已装的 `mcp` 是 **1.30.0**。
- **为什么**:两边**没有可谈的空间** —— adapters 是定死的选型(用户点名),
  它怎么写我们就只能怎么装。**1.x / 2.x 的 API 形状完全不同**(见下一条)。

**② `FastMCP` 的传输参数是直接关键字参数,不是 `settings=Settings(...)`。**

- **设计说**:§2.1 初稿曾把入口写成 `FastMCP(..., settings=Settings(...))`。
- **代码做**:`mcp_servers/logistics.py` / `aftersales.py` 都是
  `FastMCP("物流服务", host="127.0.0.1", port=8101, streamable_http_path="/mcp", stateless_http=True, json_response=True)`。
  实测 `FastMCP.__init__` 的签名里 `host` / `port` / `streamable_http_path` /
  `stateless_http` / `json_response` **就是**关键字参数。
- **为什么**:**这是写实现计划时逐字核对 `__init__` 挖出来的**,不是跑出来的。
  同一个模块里还有**另一个**名字也叫 `Settings` 的 pydantic 模型,它的
  `debug` / `log_level` / `host` / `port` / … **全部没有默认值**
  (`mcp/server/fastmcp/server.py`,实测逐字段 `is_required() == True`)——
  照「文档里那个名字」猜进去,会在**构造 Server 的那一刻**报一堆「缺字段」,
  而报错指向 `Settings`,读起来像「我们少配了什么」。
- **代价(若判错)**:无 —— 这条是**实测的结论**。

**③ Context7 在这一处不可信,以锁定版本的轮子源码为准。**

- **设计说**:项目规矩是「涉及具体库/框架/API 先用 Context7 查最新官方文档」。
- **实际做**:本章在 MCP 这一处**故意不那么做** —— §2.1 记了两条实测:
  Context7 **整站在讲 v2**(v2 把 `FastMCP` 改名成 `mcp.server.mcpserver.MCPServer`,
  传输参数从构造函数挪到 `run()`);而且**连标着 v1 的 library id
  (`/websites/py_sdk_modelcontextprotocol_io`)返回的也已经是 v2 内容**。
- **为什么**:文档说得没错,**说的是另一个大版本**。本栈被 adapters 钉在 1.x
  ⇒ **以 `mcp-1.30.0` 的 wheel 源码为准**。这不是「不查文档」,是**查完之后
  发现文档与锁定版本不是同一件事**,并按「哪个是运行时的真相」做了选择。

---

### 15.2 MCP 接入(§2.3 / §3.3 / §8.4 / §8.5)

**④ 一个被证伪的论证在六处有回声 —— 决定保留,理由换掉。**

- **设计说**(§3.3 初稿):「MCP 那条要绕开 adapters 的 pydantic 转换,
  因为那个转换会把 `minimum` / `maxLength` / `enum` 这类约束**削平**。」
- **代码做**:仍然是取**原始的 `inputSchema`**(`app/mcp/client.py::_to_spec`)。
- **为什么**:**上面那条理由在这条栈上不描述任何代码路径。**
  `langchain_mcp_adapters/tools.py` 里**就是** `args_schema=tool.inputSchema`,
  而 `langchain_core/tools/base.py` 对 **dict** 形状的 `args_schema` **原样返回**
  —— 实机比对过:两台 Server 的 `lc_tool.args_schema` **逐字节等于** `mcp_tool.inputSchema`。
  **真理由比原来那条硬得多**:`registry._spec_from_tool` 走的是
  `tool.args_schema.model_json_schema()`,而 MCP 工具的 `args_schema` **是 `dict`**
  ⇒ 那句直接 `AttributeError: 'dict' object has no attribute 'model_json_schema'`(实测)。
  ⇒ 走那条路**不是「有损」,是根本跑不通**;取原始 `inputSchema` **今天就是承重的**。
- **回声清单(全部已清)**:§3.3、§4.1、§4.3、`app/tools/spec.py` 的 docstring、
  计划的 T2 / T7 三处。**同一句话在六个地方读起来一样可信。**
- **代价(若判错)**:无 —— 决定没变,换的是理由。

**⑤ 「被削平的约束」本身未被验证 —— 已就地标注。**

- **设计说**(§3.3):约束一丢,「统一按 JSON Schema 校验」就退化成「只查必填和类型」。
- **实测**:FastMCP 给这两个 Server 生成的 `inputSchema` **本身就不含**
  `minLength` / `enum` ⇒ **今天两条路取到的 schema 逐字节相同**。
  那条守 `test_raw_schema_survives_untouched` 因此是**前瞻性守卫,不是当前的承重断言**。
- **为什么记**:本仓对「库 X 在情况 Z 下表现 Y」这类断言的规矩是**要么给可复现证据、
  要么显式标注未验证**。这一条**是这一章里主动撤回自己论证的唯一一处**
  (实现者本可以拿那条「有判别力」的守卫邀功,它选择说清它今天守不住任何东西)。

**⑥ `convert_mcp_tool_to_langchain_tool` 传 `connection=`,不是 `session=`。**

- **设计说**:§2.3 逐字核对 adapters 0.3.2 的签名后写下的。
- **代码做**:`app/mcp/client.py` 里传 `connection=connections[name]`,
  `session=` 那个位置显式传 `None`。
- **为什么**:传 session 的话,**那个 session 一关,造出来的工具就废了**。
  传 connection 则每次调用自建连接 —— 而本章的客户端是
  `async with client.session(name) as session:` **用完就退出的**,所以这条不是理论问题。

**⑦ `handle_tool_errors=False` 必须显式关(默认是 `True`)。**

- **设计说**:§2.3 写明「它会把 MCP 的调用故障**包成一条正常的工具返回**」。
- **代码做**:`app/mcp/client.py` 显式传 `handle_tool_errors=False`(带注释指向本条)。
- **为什么**:不关的话,在执行器眼里「**物流服务连不上**」是一次**成功** ——
  直接违反本仓那条「基础设施故障绝不伪装成查不到」。关掉后异常上抛,
  由执行器分诊成 `TransientToolError` → 重试 → 用尽仍失败则 502。
  **这条是「结论反了都不报错」的类型:关不关,成功路径的行为一模一样。**

**⑧ `isError` 是通用的,不能当「业务性未找到」的同义词。**

- **设计说**:计划里一度写「`ToolNotFound` 经 HTTP 回来是 `isError: true`」,
  并暗示可以据此分类。
- **代码做**:T7 **不靠 `isError` 分辨**,靠**文案**(以及先查 `spec is None`)。
- **为什么**:读 `mcp/server/lowlevel/server.py` 的 `CallToolRequest` handler 得出:
  **工具名不存在**(经 `ToolManager.call_tool` 抛 `ToolError` → 行 589 的兜底
  `except Exception`)、**入参校验失败**(行 538)、**出参 schema 不匹配**(行 568 / 575)、
  **返回类型不认识**(行 563)**全都**汇进同一个 `_make_error_result`
  ⇒ 形状一模一样(单条 `TextContent` + `isError=True`)。
  判错的代价:**把「这一单查不到」变成 502** —— 正是本仓那条
  「不许拿服务端故障指责用户输入」的反面。

**⑨ 单 Server 连不上 ⇒ 降级,不报错(§8.5,实测数已订正)。**

- **设计说**:§8.5 拍板降级;§14 风险表一度写「两个 Server 都没起时每请求白等 **4.79s**」。
- **代码做**:`discover_mcp_specs` 每个 Server 各自 `try/except`,
  失败**跳过它 + 一条响亮的 `logger.warning`**,其余照常;两个都挂 ⇒ 返回空列表,
  聊天仍可用(只剩内置工具)。
- **为什么(那个 4.79s 的三次订正,值得逐字读)**:第一次写成「≈ 一次 5s 超时」,
  复审顺着内部矛盾查到底、结论「至少有一个端点当时不是拒绝态」;
  **实现者用与本项目无关的对照端口把它整个推翻** ——
  **本机对一个已关闭的回环端口调裸 `socket.connect()` 要 ~2.05s 才拿到拒绝**
  (8101 / 8102,以及与本项目无关的 **1 / 9 / 65500 / 54321** 全是这个数),
  而**在监听**的端口(Milvus 19530)**毫秒级**就连上。
  两个 Server ≈ 2×2.05 + adapters 的 0.35×2 ≈ **4.83s**,与实测吻合。
  ⇒ **那是「本机」的性质,不是这条链路的性质。**
  **T12 复测(2026-09-23)**:端口 1 / 9 / 65500 / 54321 / 8101 / 8102 =
  2.036 / 2.055 / 2.050 / 2.055 / 2.039 / 2.055 秒;19530 = 0.4ms、3307 = 22ms。
  **可移植的上界只有一个:`mcp_discovery_timeout_seconds × 2 = 10s` —— 而那个上界
  本身**未实测**(它对应「只吞 SYN 不回」那条路径,实测走的是「立刻拒绝」那条,
  且一轮发现里未必只有一次请求,所以它**未必是紧的**)。**
- **引用规矩**:**任何具体秒数都必须带「本机实测」四个字。**

**⑩ `_dedupe` 从「重名就抛」拆成两条规则(计划外改动)。**

- **设计说**:计划里 `_dedupe` 只有一句「重名 ⇒ 响亮地抛」。
- **代码做**:内置 vs 内置仍然抛(**我们自己的接线 bug**);
  **任何涉及 MCP 的重名 ⇒ 丢掉外部那一个 + 响亮 warn,内置留下**,
  且**按 `source` 判胜负、不按顺序**(顺序是 `build_registry` 的实现细节)。
- **为什么**:`specs` 里这一章起**混进了外部来源的清单**,而外部的**名字**和它的
  用途声明一样**不可信** —— **外部 Server 只要起一个叫 `query_order` 的工具,
  每一个聊天请求都会 500**;更糟的是方向:外部还能让**我们的**内置工具消失。
- **代价(若判错)**:一个撞名的外部工具静默不可用 —— 但有 warn,且内置照常。

**⑪ `query_logistics` 下线内置之后,评估脚本必须跟着改(§8.3 的连带)。**

- **设计说**:计划只打算在 `CLAUDE.md` 上加一句「评估口径变了」。
- **代码做**:`evals/run_tool_selection_eval.py` 改成与 `app/api/chat.py`
  **同款的两步**(`await discover_mcp_specs` → `build_registry(extra=…)`,
  发现失败时同样降级)。**用例集一字未动。**
- **为什么**:那个脚本原先只用 `build_tools`,而 `build_tools` 只投影**内置那一半**
  ⇒ **那 3 条物流用例在结构上不可能通过**。「口径变了」这句话
  **描述不了「有 3 条根本跑不了」**。
- **⚠️ 未验证**:这次改动之后**脚本没有被重跑**(要真实 key + MySQL + 两个 Server)
  ⇒ `evals/tool_selection_cases.jsonl` 的 13/15 现在的准确状态是
  **「一个描述旧配置、且其测量脚本已改而从未执行」的数**。见 `CLAUDE.md` 的追加限定。

---

### 15.3 注册中心与内置发现(§3.2 / §3.4)

**⑫ 工具定义的顺序:手写列表 → `(模块名, 工具名)` 排序。**

- **设计说**:§3.4「内置工具按 `(模块名, 工具名)` 排序」。
- **代码做**:`builtin.discover()` 里 `found.sort(key=lambda pair: (pair[0], pair[1].name))`。
- **为什么**:前缀缓存要求工具定义块**逐字节相同**(ch07 已把 system + 红线 + 工具定义
  固定在最前面)。**代价**:发给模型的工具定义块与 ch02–ch07 **逐字节不同**
  ⇒ 13/15 那个评估数字**描述的是旧顺序**(已记账)。

**⑬ 「新增内置工具要重启」是本文档写错的事实,而第一次订正又过了头。**
(§3.2 已就地重写,这里留一条索引)

- **设计说**(§3.2 初稿):「内置工具是 `import` 进来的,新增内置工具**需要重启**;
  MCP 现问现拿所以不用重启」,并把这件事讲成「**两条不同的通道**」。
- **实测**:服务**已经跑着**的时候把一个新模块丢进 `app/tools/builtin/`,
  **新工具立刻可调,不需要重启**。机制:`discover()` **每请求**跑
  `pkgutil.iter_modules(__path__)`(重扫目录)+ `importlib.import_module`
  (**新文件名不在 `sys.modules` ⇒ 真的 import 一次**;FileFinder 的目录缓存按 mtime 失效)。
- **第一次订正过了头**:改成「两条通道在热重载上是同一的」。**不对** ——
  `importlib.import_module` 对**已导入的名字**直接返回 `sys.modules` 的**缓存项**
  ⇒ **修改**一个**已有**模块**仍然必须重启**。
- **最终措辞**:「**新增**」那一格两边同一(都不用重启);
  「**改已有的**」那一格**内置要重启、MCP 不用** —— 热重载能力**在这一点上确实不对等**。
  **别把上面的「同一」读成全称。**
- **两条都实测过**:改 `orders.py` 让 `query_order` 无条件抛 ⇒ 没重启的服务**照样**
  返回 `success` + 真数据;新增 `zz_echo_note.py` ⇒ 没重启就被模型调到了。
- **这条订正让验收 1 更强**(连重启也不必),但**脚本仍保留重启那一步** ——
  「重启之后仍然可用」也是要守的。

**⑭ 顺序稳定测试:断的东西一度是错的。**

- **设计说**(计划 T3):`assert names == sorted(names)`。
- **代码做**:打乱 `pkgutil.iter_modules` 的枚举顺序(正序/逆序/交换)后结果不变。
- **为什么**:那条断言**按原文不可能通过** —— 注册表按 `(模块名, 工具名)` 排序,
  而 `create_ticket` 来自 `tickets` 模块 ⇒ 它落在**最后**,整体**不是**按工具名字母序。
  实现者拒绝把它改成硬编码五名清单,理由是「**每加一个工具就烂一次 —— 而那正是
  验收 1 授权的动作**」,改成了断言那句话真正买到的东西。**这是更好的写法。**

---

### 15.4 执行引擎与权限(§5.3 / §5.4 / §6.1 / §6.4 / §6.5 / §7.4)

**⑮ `POST /api/ticket` 「一行不动」是本文档写错的事实(§5.4 已就地订正)。**

- **设计说**(§5.4 初稿):「`POST /api/ticket` **不是工具调用**,是端点直接落库,一行不动。」
- **代码做**:它一直是 `execute_tool` 调用;本章给它补了 `write_decision=APPROVED`。
- **为什么**:新的权限闸只认「有没有确认」,而**点按钮本身就是确认** ⇒ 不补这一行,
  投诉按钮会拿到 `confirmation_required` → 502,**症状与「服务挂了」一模一样**。
  那一行不是新行为,是**这个闸的必要改动** —— 初稿错的是**事实描述**,不是意图。

**⑯ 认不出的决议值:响亮地抛 + 不审计(否决了审查员的建议)。**

- **设计说**(§5.3):闸写成「`== PENDING` 拦、`== DENIED` 拦」。
- **代码做**:`if write_decision != APPROVED: raise ToolInfrastructureError(...)`,**且不审计**。
- **为什么**:初稿的形状是**失败开放**的 —— `"Approved"` / `None` 这类拼错或半接线的取值
  会**无确认、无审计地执行一次不可逆的写**。审查员建议「`!= APPROVED` 就当拒绝处理」,
  **没有被采纳**:那会在 `tool_audit_logs` 里写一条「**用户**点了取消」的行,
  而那张表**正是验收 5 读的表** —— 用一个 bug 去谎报用户行为,仍然是在污染唯一的事实来源。
  ⇒ 认不出的决议是**接线 bug**,与 `tool_missing` 同族:暴露、不伪装。
- **代价(若判错)**:接线时把 `"approved"` 写成小写会 **502 而不是静默拒绝** ——
  **那正是要的**:它会在第一次手工点确认时立刻暴露,而不是等到有人翻审计表。

**⑰ `retry_count` 记真实发生过的重试次数,不是配置值(§7.4)。**

- **设计说**:§7.4 已经写明。
- **代码做**:`retries = 0`,只在 `for attempt in range(attempts)` 里 `if attempt: retries += 1`
  —— 即 `attempts_made − 1`,写进成功、失败、审计三条路径。
- **为什么**:写成配置值的话,一个**第一次就成功**的查询会被审计成「重试了 2 次」,
  而**没有任何断言会红**。⚠️ **验收 6 那条 `timeout|2` 区分不了两者**
  (全超时的一轮里「真实重试」与「配置值」**都是 2**),真正守住它的是
  `tests/test_executor_gate.py`。**引用验收 6 时不许说它证了这条。**

**⑱ `preview=args if isinstance(args, dict) else {"_raw": args}`。**

- **设计说**(计划 T4):`preview=dict(args)`。
- **代码做**:上面那样。
- **为什么**:`args` 不是 dict 时 `dict(args)` 当场抛 `ValueError`/`TypeError`,
  而那个位置**没有任何 handler 罩着** ⇒ 异常**逃出** `execute_tool`,
  于是这套分类学承诺的可恢复 `invalid_args` 变成一次 **500**。偏偏
  「畸形 args 的写调用」**恰好是唯一绕开 `validate_args` 那条宽容路径的地方**
  (它在闸**之后**)—— 实测出来的正是这个组合。

**⑲ 错误分诊从四类扩到六类,重试白名单改由 `kind` 推导(§6.1 / §6.4)。**

- **设计说**:§6.1 / §6.4 已经是这个设计。
- **代码做**:六个常量 `not_found` / `timeout` / `invalid_args` / `tool_missing` /
  `permission_denied` / `confirmation_required`;`attempts = 1 + (retry_attempts if kind != WRITE else 0)`。
- **为什么**:重试白名单从「一张写死的集合」改成**由 `kind` 推导**,新注册的只读工具
  自动可重试、写工具自动不可重试。**写操作永不重试是结构保证**,不是配置约定。

**⑳ `render_tool_result(spec, raw)` 的 `spec` 一开始没人读。**

- **设计说**:计划钉的签名收了一个 `spec`。
- **代码做**:保留签名,`spec` 今天**不参与**渲染(内置工具返回的已是手挑过字段的
  JSON 字符串 ⇒ 原样透传;MCP 返回内容块列表 ⇒ `json.dumps`)。
- **为什么**:「只挑用得上的字段」与「内部枚举码翻人话」两条**落在工具自身**
  (内置那份只有工具知道自己的枚举怎么翻)。签名保留是因为它**同时**服务两条分支,
  删掉会逼下一次改动再改一遍全部调用点。

---

### 15.5 确认流与图(§9)

**㉑ `agent` 从 `_OUTLETS` 里去掉,改成条件出口(§3 的代码块与 §9.1 已同步)。**

- **设计说**:§9.1 方案 A。
- **代码做**:`_OUTLETS = ("complaint_reply", "chitchat_reply", "fallback_reply",
  "refund_offer", "refund_explain")` —— **`agent` 不在其中**;`agent` 后面接
  `add_conditional_edges("agent", route_after_agent, {"confirm_write": …, "log_turn": …})`。
- **为什么**:把 `agent` **无条件**接回 `log_turn`,会让**挂起的那一轮一半写库、一半没写**
  —— 而这在**单测里看不出来**(单测要显式 resume 才走得到那儿)。
  (同款:`refund_pick_order` 也**不在** `_OUTLETS` 里,理由与它一样。)

**㉒ 两个新通道必须连同每轮清零一起落地(§9.5)。**

- **设计说**:§9.5 明写。
- **代码做**:通道在 `app/agent/state.py`;清零在
  `nodes.make_resolve_references_node` 的返回值里(`"pending_write": {}`、
  `"write_decision": ""`),与 `turn_messages` / `order_no` / `order_data` /
  `refund_decision` 同一份清单。
- **为什么**:checkpointer 是**进程级单例**、`thread_id = session_id`,
  **未写的通道保留上一轮的值** ⇒ 漏了清零的后果是
  **上一轮批准过的写操作,这一轮自动放行**。
  这是 ch05–ch07「通道与它的清零必须同处一地」的**第四次应用**。

**㉓ `apply_write_decision` 的空 `pending_write` 守卫:节点自己也要拒(审查建议,已采纳并扩大)。**

- **设计说**(计划):守卫**只放在路由上**(`route_after_agent`)。
- **代码做**:路由侧一条 + **节点自己再拒一条**(空 ⇒ 上抛,那是接线 bug)。
- **为什么**:「那等于把不变量**寄存在调用方的记忆里**」—— 正是本仓元教训警告的形态。
  空 `pending_write` 往下走会造出 `tool_call_id=""` 的 `ToolMessage` ⇒ **上游 400**。

**㉔ 挂起路径上那条永远结算不了的徽标(§10.3 的连带,由 T9 复审发现)。**

- **设计说**:§9.3 只写了帧名与载荷。
- **代码做**:`make_apply_write_decision_node` 加 `emit`,**执行后补发同款 `tool_result`**。
- **为什么**:`agent` 的循环是**先发 `tool_call` 帧再执行**、撞到待确认就 `continue`
  ⇒ 那个 `tool_call` 发出去了、`tool_result` **永远不来**,前端徽标一直转。
  **更省事的修法「挂起前先把徽标关掉」被否掉** —— 那会让徽标的语义变成
  「已经跑完了」,而它**确实还在等用户**;**让它转着才是诚实的**。

**㉕ `turn_messages` 是覆写通道,续跑路径必须「读旧值再追加」(§9.4 的不变量)。**

- **设计说**:§9.4「用**已有的** `turn_messages` 不变量」。
- **代码做**:`apply_write_decision` 与 `agent` 的续跑分支都先
  `existing = list(state.get("turn_messages") or [])`,再 `existing + [new…]`。
- **为什么**:`log_turn` 只拿 `turn_messages` **落库** ⇒ 只返回新那一条会
  **丢掉带 `tool_calls` 的 AIMessage**,而那一轮**看起来一切正常**
  (回复正常、落库正常),少的是历史里的结构。

**㉖ `emit` 是无默认值的关键字参数(实现者的选择,已批准)。**

- 漏传 = **硬错**,不会静默退化。前端消费的三个字段在生产↔消费两侧逐字段相同。

---

### 15.6 审计与建库(§7 / §11)

**㉗ `db/ch08.sql` 与 ORM 建出的表形状仍有三处不同;而 CLAUDE.md 一度写成「只剩两处」。**

- **设计说**:§11.1 给 DDL;§11.2 只映射 ORM。
- **实测(编译 `CreateTable(ToolAuditLog.__table__)` 与 DDL 逐列对)**:
  行为差异(T5 修复轮)**已全部对齐** —— `id` 两条路都是 `BIGINT`;
  带 `DEFAULT ''` 的字符串列**恰好两个**(`result_summary` / `error_detail`);
  `args` 与 `status` 两条路**都没有** `DEFAULT`;`created_at` 两条路**都有索引**。
  **剩下的三处全是文本差异**:
  ① **两个索引名**(`idx_conv` / `idx_created` vs SQLAlchemy 自动生成的
  `ix_tool_audit_logs_conversation_id` / `ix_tool_audit_logs_created_at`);
  ② **表的 `COMMENT`**(DDL 有,ORM 侧为 None);
  ③ **七个列级 `COMMENT`**(DDL 里 `conversation_id` / `tool_call_id` / `source` /
  `args` / `result_summary` / `status` / `retry_count` 各带 `COMMENT '…'`,
  ORM 侧**全文没有任何 `comment=`**)。
- **为什么值得记**:③ **一度被写小成「六个」**(审查员列清单时**漏了 `tool_call_id`**),
  而 `CLAUDE.md` 当时只写了 ①②。**「只剩两处」是个源不支持的绝对断言**,
  而它还被归因给一个与结果矛盾的方法(「逐列比对…得出」)。
  ⇒ **数一遍再引用。** T12 已把 `CLAUDE.md` 改成「三处」并逐个列名。

**㉘ `status` 列宽:`_STATUS_MAX = 32` 的哨兵性质与两处口径错误。**

- **设计说**(计划):只给几个自由文本列加 `_clip`。
- **代码做**:`_STATUS_MAX = 32` 也加上;`app/tools/audit.py` 的注释写明
  `record_audit` 全仓只有 **4 个调用点**(都在 `executor.py`),能传进来的取值共 **5 个**
  (`success` / `failed` / `timeout` / `invalid_args` / `permission_denied`),
  最长的 `permission_denied` **17** 字符 ⇒ **这条今天触发不到**,它是哨兵。
- **为什么**:初稿的注释说「**8 个**状态值、最长 **21** 字符」—— **两半都错**。
  21 字符的 `confirmation_required` **永远不会到达 `record_audit`**
  (待确认那条路在写审计**之前**就返回了,且刻意不审计)。
  仍要守的理由:漏夹 ⇒ `DataError` ⇒ 被 `except` 吞掉 ⇒ **整行审计静默消失**,
  而那时它记的可能已经是一次**不可逆的写操作**。

**㉙ 审计的「只记不拦」与独立 session 都是硬约束(§7.2 已写)。**

- **代码做**:`record_audit` **永不抛**、自己开 session、`except Exception`(不含
  `BaseException` —— 取消要照常传播)。
- 与 `app/api/ticket` 那条 502 路径**方向相反**:审计失败只留痕迹,不影响工具执行。

---

### 15.7 数据、配置与验收(§8.6 / §12 / §13)

**㉚ 售后 MCP 与 ch06 的 `refund_requests` 无语义关联(§8.6,保持不变)。**

- **设计说**:§8.6 明写;`query_return_progress` 返回**伪随机 mock**。
- **代码做**:照做。**演示时不许拿它当真实进度用。**
- 若要接真实数据,得让 Server 连 MySQL —— 那**直接违反**用户定死的选型
  (「不接真实系统、不建表」),留作将来单独一章的事。

**㉛ `tool_retry_attempts` 默认 1 → 2 是跨章行为变更(§6.2 / §12.2,用户拍板)。**

- **代码做**:`Field(default=2, ge=0)`;`.env.example` 已同步(**`.env` 未改**,仍是默认值)。
- **一句话回退**:`.env` 里 `TOOL_RETRY_ATTEMPTS=1`。
- ⚠️ **写操作永不重试不受影响**(由 `kind` 推出)。
- ⚠️ **连带**:`scripts/acceptance_ch08.sh` 的 A6 **必须显式钉 `TOOL_RETRY_ATTEMPTS=2`**,
  否则照「一句话回退」做的人会把那一格弄红 —— 而那不是缺陷。**同理钉
  `TOOL_RETRY_DELAY_SECONDS=0.3`**(`0` 是**合法值**,`Field(ge=0)`;
  设成 0 会让三次尝试之间不再等待,`duration_ms ≥ 500` 那一格掉到 ~10ms 变红)。

**㉜ 三个 `mcp_*` 配置项的任务归属排错了(已补救)。**

- **设计说**:计划把它们排在 T11。
- **代码做**:`app/mcp/client.py` 在 **T7** 就要读它们,于是实现者按 T11 的 brief
  **提前落地**(`app/config.py` 的三个 `mcp_*` 字段),T11 那一步改成「**确认已存在**、值逐字一致」。
- **为什么**:不补的话**每个聊天请求 AttributeError**。而「确认已存在」也是必要的
  —— **重复定义在 pydantic 里是静默覆盖**,看不出来。

**㉝ 验收 4 的题面:「Agent 先追问补齐」那半不可达(本章最重的未达成项)。**

- **设计说**:§13.4 的验收 4 含「缺必填项时 Agent 先追问补齐」。
- **实测**:脚本那条题面**用单轮提示直奔卡片**,那半**从未跑过**。
  实现者随后探了**九种说法**,结论是**结构性**的:
  没有业务落点的建单请求全部走 **其他→兜底** 或 **投诉→固定话术出口**
  (那个出口的按钮走 `POST /api/ticket`,**不是**确认流);
  而**能**走到 Agent 的说法都自带业务落点,模型会**从 `query_order` /
  `query_logistics` 的返回里合成一个描述**直接调 `create_ticket` —— **它不追问**。
- ⇒ 用户要求里的「缺必填项就**主动追问**、**不许瞎编**」这一条**没有实现**;
  「卡 → 确认 → 落库 → 带工单号」三段**都是通的**。
  **「不许瞎编」那一半今天是靠「模型没瞎编」侥幸成立的,不是被守住的性质。**
- **明确禁止的处置**:为了让它通过而**改断言**。**需要用户拍板。**

**㉞ 一条 pre-existing 的泄漏,被本章的 A6 照出来。**

- **现象**:验收 6 的写路径(`TOOL_TIMEOUT_SECONDS=0.001` + `create_ticket`)那次续跑
  会推一条 `error` 帧,文案是
  `This Session's transaction has been rolled back due to a previous exception
  during flush. To begin a new transaction with your Session, first issue
  Session.rollback(). Original exception was: …` —— **裸的 SQLAlchemy 内部文本**
  直接进了**用户可见的**帧。
- **两个成因,缺一不成**:① **ch01 起的通道** —— `app/api/chat.py` 的
  `except Exception as exc:` 直接 `redact_api_key(str(exc))`(它只抹**密钥**,
  不抹**内部实现细节**);② `create_ticket` **没有 ch03 那种取消路径的 `rollback()`**
  —— 超时把协程取消在 SQL 中间,session 停在**待回滚**状态。
- **⚠️ 不是本章引入的回归**:两个成因都在 ch08 之前就在。
  A6 **只断审计行**,所以它既没判过也没判红 —— 方向上与本仓「所有出站错误文本
  必须过 sanitize」「基础设施故障一律固定文案」那两条**不一致**
  (泄漏的不是密钥,是内部实现细节)。
- **两条候选修法(未实施)**:
  1. **给 API 层兜底** —— 那个 `except Exception` 不再直接 `str(exc)`,
     改固定文案 + 原文 `logger.error`(与 502 那条路径同款)。改一处,覆盖面最大。
  2. **给写工具的取消路径补 `rollback()`** —— 照 ch03 `retrieval/search.py`
     在 `except BaseException` 里先 `rollback()` 再抛的样子。
     **治因**,但只治这一条路径。

**㉟ 验收脚本自身的四处「装置故障」(值得记形状,不值得记结论)。**

T11 的验收**跑了六轮才绿**,前五轮红里**三条是实现者的装置**:
① 探针 `PYTHONPATH`(按**路径**跑脚本时 Python 把**脚本所在目录**塞进 `sys.path`,
而不是当前目录 ⇒ `ModuleNotFoundError: No module named 'app'`);
② Windows 端口归还竞态(`kill -9` 之后监听套接字**不是立刻归还**,
新进程以 `[Errno 10048]` 退出,而这条报错出现在**新进程**的控制台里,
读起来像「新起的服务坏了」);
③ 对以 `ticket_no` 为主键的表写 `ORDER BY id` ⇒ `DBQ-ERROR … Unknown column 'id'`,
而那个串**非空**,于是断言以「回复里没有工单号」的样子红。
**一条是真实的**:同一个建单题面,**分类器 run-to-run 会在「物流」与「投诉」之间摇摆**
⇒ 在**「追问」那一步**加了重试(**断言一字未改**)。

**㊱ ⚠️ `TransientToolError` **全仓没有任何生产抛出点** —— §6.3 的「MCP 传输类故障可重试」没有落地。**

- **设计说**(§6.3):本章「新增一类**可重试的故障**:MCP 传输类故障」,
  由 MCP 客户端抛 `TransientToolError`,执行器在重试规则内重试它,
  用尽仍失败则上抛 `ToolInfrastructureError`。`app/tools/errors.py` 的
  `TransientToolError` docstring 也明写「**谁抛它**:MCP 客户端(T7)」。
- **代码实际**:`TransientToolError` **只在三处出现** —— `errors.py` 的定义、
  `executor.py:241` 的 `except`、`tests/test_executor_gate.py` 的两处**测试注入**。
  **`app/` 里没有任何一行 `raise TransientToolError`**;`app/mcp/client.py` 的
  `except Exception` 是**发现期**的降级,不覆盖**调用期**。
- **实测**(T12,单测内注入 `ToolException` + `tool_retry_attempts=2`):
  工具**只被调用了一次**,执行器直接抛 `ToolInfrastructureError("工具执行失败")`
  ⇒ **MCP 传输故障今天不重试,第一次就 502。**
  路径是 `ToolException` 落进最后的 `except Exception`(它不是
  `ValidationError` / `ToolNotFound` / `SQLAlchemyError`),而那一条**直接上抛**。
- ⇒ **那条重试分支在生产上是死代码**;`errors.py` 的 docstring 是**错的**;
  §6.3 只交付了「异常上抛、不伪装成成功」那一半,**可重试那一半没有**。
- **未修**(T12 按「不改代码行为」的边界交回控制者裁定)。两条候选修法:
  1. **在 `app/mcp/client.py` 包一层**:把 MCP 的调用期异常翻译成
     `TransientToolError`(那正是 `errors.py` 已经写明的分工)。改动小、语义正。
  2. **在执行器里加一条 `except ToolException`**,把它归到暂时性那一支。
     **不推荐** —— 那会让「参数不合法」这类由 adapters 抛出的 `ToolException`
     也被重试三次,而重放同样的参数只会同样失败。
- **顺带**:`tests/test_executor_gate.py` 的两条用例断的是**执行器**的行为,
  它们**测不到**「生产里没有人抛它」—— 这正是本仓「替身替被测对象完成了语义」
  的又一形态:**替身抛了一个生产永远不抛的异常**。

---

**㊲ 一条恒真断言在验收脚本里活到了 T11 审查(本仓第 (d) 类的新形态)。**

- **A3 的「客服服务进程号没变」比的是同一个文件读两次**:`BEFORE_PID` 与 `AFTER_PID`
  都是 `$WORK/cs.pid`,中间**没有任何东西写它** ⇒ `before == after` **无条件成立**,
  而 PASS 文案在宣称一个它证明不了的结论。**它甚至抓不住它要抓的失败**
  (客服被外部重启或崩掉,pid **文件**里还是旧值,照样 PASS)。
- **修法**:改成读 **OS 级监听进程**(`netstat -ano`)+ `kill -0`。
  注意 `kill -0` 在 MSYS 下**够不着 Windows pid**,实现者换成 MSYS pid 是**保住了原意**
  (外部重启/崩溃仍可检出),不是绕开。

---

### 15.8 两条元教训(本章复盘)

**① 一句被证伪的话不会自己消失 —— 它会以「解释」的形式散落,而每一处读起来一样可信。**

「adapters 的 pydantic 转换会削平约束」这句话在**六处**有回声(§3.3、§4.1、§4.3、
`app/tools/spec.py` 的 docstring、计划 T2 / T7)。它被撤回之后,**那些回声一处都不会自己红**
—— 它们全是散文。**只有有人拿着「这句还成立吗」去逐处读,才会发现它们还活着。**
⇒ 订正一条论证时,**要做的是 grep 它、而不是改写下它的那一处**。

**② 本章的多数缺陷,落点不在代码而在计划文本。**

T4 的终审定界复审在范围外捞到三条计划缺陷(权威代码块仍留着 fail-open 的闸、
`apply_write_decision` 漏 `"type": "tool_call"`、`decision = … or DENIED` 会谎报用户取消),
**三条全在计划里**,而其中两条会在 **T8 变成真 bug**。
T11 的 A3 恒真断言同样是 **plan-mandated**。
⇒ 与 ch01–ch07 的结论一致:**没人负责的地方,只在有人真的去读那份计划时才暴露。**

---

### 15.9 未验证清单(不许读成「已验」)

1. **`mcp_discovery_timeout_seconds × 2 = 10s` 这个上界** —— 见 ⑨,未实测。
2. **「adapters 会削平约束」** —— 见 ⑤,今天观察不到差异,已标注。
3. **`run_tool_selection_eval.py` 改动后从未执行** —— 见 ⑪。
4. **`db/ch08.sql` 与 ORM 的七列 COMMENT 差异只影响 `SHOW CREATE TABLE` 的观感**,
   权威路径永远是 `db/ch08.sql`。
5. **`evals/tool_selection_cases.jsonl` 的 13/15** 描述的是旧配置 + 旧顺序(见 ⑫ / ⑪)。
6. **前端工单卡片的真实渲染 / 手工点击验收** —— 本环境**没有浏览器**,
   如实入账,**不许任何人说成「已验」**。
7. **Milvus / 检索链路在本章验收里零覆盖**(六项验收没有一轮走到检索)。
8. **`TransientToolError` 那条重试分支在生产上是死代码** —— 见 ㊱。
   「MCP 传输类故障可重试」是**设计里有、代码里没有**的一条,而它的两条测试
   (`tests/test_executor_gate.py`)断的是**执行器**的行为,**测不到「没人抛它」**。
