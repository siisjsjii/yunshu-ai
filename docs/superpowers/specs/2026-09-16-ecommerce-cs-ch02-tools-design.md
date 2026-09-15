# 电商智能客服系统 · 第 02 章:Function Calling 查数据能力 — 设计文档

- 日期:2026-09-16
- 状态:待用户评审
- 范围:后端单轮工具调用 + 建库基建 + 聊天页改造。本章**不做**多轮 Agent Loop、向量检索/RAG。
- 前序:ch01(纯对话)已合并到 `main`,见 `docs/superpowers/specs/2026-09-15-ecommerce-cs-ch01-design.md`

## 1. 目标与验收

给现有客服对话装上「查数据」能力:模型自己判断该调哪个工具,后端执行后回灌,模型据结果组织回答。**只做单轮** —— 模型调一次工具就收敛。

验收标准(用户给定,逐字):

1. 浏览器打开聊天页,问「订单 1001 的物流到哪了」,能看到模型选中工具(气泡带工具徽章)并按返回结果作答
2. 问「退货政策是什么」,`query_faq` 查得到并作答
3. 换个说法问「邮费是多少」,确认关键词查表查不出来 —— 这个漏召回是**预期结果**,记下来留给下一步升级

## 2. 技术栈与版本

| 组件 | 版本 | 核实方式 |
|---|---|---|
| Python | 3.13.14 | 已存在的 `.venv` |
| FastAPI | 0.141.1 | ch01 已核实 |
| LangChain | 1.4.0 | ch01 已核实;**本章新增用其 `@tool` 与 `bind_tools`** |
| langchain-core | 1.6.3 | `langchain.__version__` 实测 |
| SQLAlchemy | 2.0.53 | `pip index versions` |
| asyncmy | 0.2.14 | `pip download --only-binary=:all:` 确认有 `cp313-cp313-win_amd64` 原生轮子,**无需本地编译** |
| cryptography | 50.0.1 | **实测必需,原设计遗漏。** MySQL 8.0 默认认证插件是 `caching_sha2_password`,asyncmy 走该认证需要此包,否则连接直接抛 `RuntimeError: 'cryptography' package is required` |
| MySQL | **8.0.46** | 实测:`SELECT VERSION()`。由用户自行以 Docker 提供,见 §7.4 |
| 上游模型 | `deepseek-flash` | ch01 已核实;本章新增的 tool calling 能力见 §9 |

## 3. 已完成的实测(本章设计的地基)

本章设计建立在以下**真机实测**之上,非凭记忆或文档推断。

| # | 核实项 | 方法 | 结论 |
|---|---|---|---|
| 1 | **本端点是否支持 tool calling** | 真机 `bind_tools` + `ainvoke` | **支持**。返回 `tool_calls=[{'name':'query_logistics','args':{'order_id':'1001'},...}]` |
| 2 | ch01 那个 `function_calling` 400 的真实边界 | 对比 `with_structured_output` 的调用形状 | **挂掉的是强制 `tool_choice`(thinking 模式不支持),不是 tool calling 本身。** 不强制即正常 —— 这条决定了本章可行性,也是 ch01 spec §9 那行记录的必要补充 |
| 3 | 第一轮流式的 chunk 形状 | `bind_tools` + `astream` 实测 | 调工具时:**0 个文本 chunk + 12 个 tool_call_chunk**(name/id 在首块,args 分片);不调工具时:**53 个文本 chunk + 0 个 tool_call_chunk**。两者零重叠 |
| 4 | 第二轮仍绑 tools 会怎样 | `ainvoke` 实测 | 本次模型未再调(`tool_calls=[]`),但**这是模型行为,不是保证** —— 故不采用它作为"单轮"的依据 |
| 5 | 第二轮不绑 tools | `astream` 实测 | 强制收敛为文本,可正常流式 |
| 6 | `@tool` 导入路径 | 装好的 1.4.0 | `from langchain.tools import tool` ✓ |
| 7 | `ToolMessage` 导入路径 | 装好的 1.4.0 | `from langchain.messages import ToolMessage` ✓ |
| 8 | `@tool` 自动生成参数 schema | 装好的 1.4.0 | `demo.args_schema` 存在(pydantic 模型),**参数校验无需手写** |
| 9 | 异步工具支持 | 装好的 1.4.0 | `ainvoke` / `invoke` 均存在 ✓ |
| 10 | Docker | `docker --version` / `docker info` | CLI 29.6.2 + Compose v5.3.1 已装;**daemon 未启动** |

## 4. 架构

依赖方向延续 ch01 的单向约束,不打破:`api → services → {tools, db, memory, prompts, llm} → config`。

```
app/
  config.py         # 扩展:新增 DATABASE_URL 与工具相关配置项
  llm.py            # 不变
  prompts.py        # 扩展:System Prompt 增加工具使用指引
  schemas.py        # 扩展:Message 增加 tool_calls / tool_call_id
  sanitize.py       # 不变
  db/               # 新增
    base.py         #   DeclarativeBase + async engine + async_sessionmaker
    models.py       #   四张表的 ORM 模型
    session.py      #   FastAPI Depends 用的 get_session
  tools/            # 新增
    business.py     #   五个 @tool 定义
    registry.py     #   注册表 + 未知工具名校验
    executor.py     #   超时 / 重试 / 错误分类 / ToolMessage 构造
  memory/
    store.py        # 瘦身为锁注册表(见 §6.4)
    trim.py         # 改:分轮规则改为 user 边界(见 §6.3)
  services/
    chat.py         # 重构:单轮编排
    extract.py      # 不变
    history.py      # 新增:从 messages 表读历史并组装
  api/
    chat.py         # 扩展:新增两个 SSE 事件
    extract.py      # 不变
  static/           # 新增:聊天页(单页静态文件)
```

### 4.1 与 ch01 的既定决策的冲突与调和

ch01 spec §3 写着「分层薄封装,**不使用 LCEL,不使用 LangGraph**」。本章**继续遵守** —— 单轮编排是手写的,不引入 LangGraph(它只是 langchain 1.x 的传递依赖,不直接使用)。理由见 §5.2。

ch01 的「会话存进程内 dict」在本章被**部分推翻**:历史迁至 MySQL(用户裁决),进程内只保留锁。这是有意的架构演进,不是对 ch01 决策的否定 —— ch01 当时的范围是纯对话,没有持久化需求。

## 5. 接口契约

### 5.1 `POST /api/chat/stream`(扩展)

请求体新增可选字段:

```json
{"session_id": "可选", "message": "订单 1001 的物流到哪了", "user_id": "可选"}
```

`user_id` 缺省时落 `"demo-user"`。本章**没有认证**,该字段不做任何鉴权,仅作为 `conversations.user` 的取值来源。

`user_id` 只在**新建会话时**写入。会话已存在(同 `session_id`)时,**忽略本次传入的 `user_id`**,以创建时记录为准 —— 否则任何客户端都能改掉一条会话的归属。新建会话的 `status` 初始为 `active`。

### 5.2 SSE 事件协议

ch01 的四个事件保持不变,新增两个:

| event | data | 时机 |
|---|---|---|
| `meta` | `{"session_id": "...", "model": "..."}` | 首帧(不变) |
| `token` | `{"text": "..."}` | 每个 token 一帧(不变) |
| **`tool_call`** | `{"name": "...", "args": {...}, "tool_call_id": "..."}` | 模型决定调用时,**执行前** |
| **`tool_result`** | `{"tool_call_id": "...", "ok": true, "summary": "..."}` | 执行完成时 |

`tool_result.summary` 的定义:**供前端展示的简短摘要,固定截断至 200 字符**,不是完整结果。完整结果只经 `ToolMessage` 回灌给模型、并落 `messages.content`。失败时 `summary` 为错误原因的一句话(同样脱敏后),`ok=false`。
| `done` | `{"finish_reason": "...", "usage": {...}}` | 正常结束(不变) |
| `error` | `{"message": "..."}` | 出错(不变) |

拆成 `tool_call` / `tool_result` 两个事件而非一个带 `status` 字段的事件,是为了让前端徽章能显示「正在调用 → 已返回」两态,而不是等结果出来才出现。

`ok=false` 表示工具执行失败但**已回灌给模型**(可恢复错误,见 §6.2),流仍会正常 `done`。这与 `error` 事件语义不同:`error` 表示流被终止。

### 5.3 `POST /api/extract`(不变)

ch01 交付的抽取接口本章不动。

## 6. 关键设计决策

### 6.1 会话历史的归属:MySQL 为真相,锁留进程内

> **用户裁决**:「MySQL 作历史真相,锁留进程内」

每轮请求的流程:取锁 → `SELECT` 该会话历史 → 裁剪 → 组装 → 模型 → 写回 `messages`。

- `conversations.id` **直接复用 ch01 的 `session_id`**(uuid4 hex,32 字符),不做映射表 —— 两个体系共用同一个 id。
- 好处:重启能续聊;`conversations.status` 才有实际意义;表即真相。
- 代价:每轮多一次 DB 往返。
- 被否决的方案:进程内仍是真相 + MySQL 只做流水(重启丢历史,字段成死列);两者并存 + 降级回源(两套真相,复杂度最高)。

### 6.2 单轮工具编排:手写 + 第一轮流式探测

> **用户裁决**:方案 A

```
第一轮:model.bind_tools(TOOLS).astream(msgs)
         ├ 文本 chunk     → 直接推 SSE token 帧(真流式)
         └ tool_call chunk → 累积,不外推
     └ 结束后:
          有 tool_calls → SSE tool_call 帧 → 执行 → SSE tool_result 帧
                        → 回灌 ToolMessage → 第二轮
          无 tool_calls → 已经流完,直接 done(单次 API 调用)
第二轮:model.astream(msgs)   ← **不绑 tools**
         └ 逐 token 推 → done
```

**「只做单轮」是结构保证,不是提示词约定。** 第二轮不绑 tools,模型在结构上**没有能力**再调工具。实测第 4 条表明"第二轮仍绑 tools 时模型这次没再调",但那是模型行为、不是保证 —— 不采用它。

被否决的方案:LangGraph `create_agent` + `recursion_limit=2`(`recursion_limit` 是软约束;与「不做 Agent Loop」冲突;其 event 流需手工翻译成 ch01 的 SSE 协议);先 `ainvoke` 再 `astream`(不调工具的提问也走非流式,实测「你好呀」本可获得 53 chunk 的正常流式体验,退化可惜)。

**边缘情况的行为定义**:若已向客户端推出文本 token 后又收到 `tool_call`,照常执行工具并继续第二轮。用户会看到一段前言 + 最终回答。这是可接受的,不视为错误 —— 实测未出现此情况。

### 6.3 历史裁剪必须按「user 边界」切

**这是本章最容易翻车的地方。**

ch01 的 `_to_rounds` 规则是「遇到 `assistant` 就收一轮」。引入 `tool` 角色后,该规则会把 `tool` 消息与它的 `assistant` 父亲切到不同的轮里。而 OpenAI 兼容 API 有硬校验:

> `tool` 消息前面必须紧跟着带对应 `tool_call_id` 的 `assistant` 消息,否则 400

一旦裁剪把这对切开,表现是**偶发 400**,且只在历史长到触发裁剪时复现。

因此一轮的定义改为:**从一条 `user` 消息开始,到(不含)下一条 `user` 消息为止**。`assistant` 与其 `tool` 消息天然同轮,不可能被切开。

相应调整:进程内的 `Message` 模型新增两个可选字段 `tool_calls` / `tool_call_id`;`trim.py` 的 token 计数仍只看 `content`。

### 6.4 `SessionStore` 瘦身为锁注册表

历史外迁后,它只剩「每会话锁 + 孤儿锁清扫」。`_sessions`、`_enforce_capacity` 的 LRU、TTL 淘汰历史全部退役 —— 淘汰的理由(内存中的历史)已不存在。

锁本身**仍然必要**:同会话并发请求会各自读到同一份历史、各自追加,后写覆盖先写。

**测试影响**:ch01 的 `tests/test_store.py` 中,LRU 淘汰、TTL 过期、容量上限等测试**失去被测对象,应删除**,而不是保留成空壳。锁相关测试保留并改写。ch01 复盘明确反对「为了绿而绿」的测试,此处照此办理。

注:ch01 spec §9 记录的「锁获取取消竞态」在本章**依然存在**,本章不处理(见 §9)。

**订正(写计划时发现,经用户裁决)**:本节原写「「`_locks`/`_touched` 无硬数量上限」在本章依然存在,本章不处理」。但历史外迁后,**`MAX_SESSIONS` 这个配置项失去了它唯一的用途** —— 它当初就是为「限制进程内历史条数」而存在的,`_sessions` 一退役,它就成了死配置(读 `.env.example` 的人会以为它在管事)。死配置是 ch01 最反对的那类东西。

> **用户裁决**:把 `max_sessions` 改用于限制 `_locks` / `_touched`。

于是 `SessionStore(ttl_seconds, max_sessions)` 的容量上限从「限制历史」变为「限制锁表」,**ch01 那条「锁表无硬数量上限」的风险由此关闭**。

淘汰规则沿用 ch01 Task 4 的教训:**遇到被持锁的条目整个停下,不跳过** —— 跳过会删掉比它更新的条目,把 LRU 语义弄反。同时 `lock_for` 在刷新时间戳时要把条目 `move_to_end`,否则淘汰的就不是 LRU 而是插入序,刚建的锁会被优先选中,破坏「同一 session 两次 `lock_for` 返回同一把锁」的幂等性(ch01 Task 4 已栽过一次)。

淘汰一个**未被持有**的锁是安全的:没有持锁者,就不存在被破坏的互斥;后续请求会拿到一把全新的、未锁定的锁。

### 6.5 五个工具

```python
query_order(order_id: str)      # 随机数据,以 order_id 为种子
query_product(keyword: str)     # 随机数据,以 keyword 为种子
query_logistics(order_id: str)  # 随机数据,以 order_id 为种子
query_faq(keyword: str)         # SQL LIKE 查 faq 表
create_ticket(description: str, ticket_type: str)  # 写 tickets 表
```

`query_product` 用 `keyword` 而非 `product_id`:因不建商品表,而用户问「无线耳机多少钱」时手里没有 id,`keyword` 能让模型直接传词。

**三个「假装有上游」的工具用种子化确定性伪随机。**

> **用户裁决**:「以订单号为种子的确定性伪随机」

**种子必须用 `hashlib.sha256` 等稳定哈希,不能用内置 `hash()`。** CPython 对 str 的 `hash()` 每进程随机化(`PYTHONHASHSEED`),用它会导致「同一订单号永远返回同样数据」这条承诺在进程重启后失效,且**同进程内的测试测不出来**。

**工具返回显式 `json.dumps(..., ensure_ascii=False)` 的字符串**,不返回 dict:`ToolMessage.content` 本就应为字符串,且 `ensure_ascii=True` 会把中文转成 `\uXXXX` 白烧 token。

**`create_ticket` 的 `conversation_id` 不进模型的参数 schema** —— 让模型自己填会编造 id。

**实现机制经实测订正。** 原设计写「用 `InjectedToolArg` 注入,由 executor 绑定」,实测发现该机制在本版本上不完整:

| 实测项 | 结果 |
|---|---|
| `create_ticket.args_schema`(原始) | **含** `conversation_id` |
| `create_ticket.tool_call_schema`(真正发给模型的) | **不含** ✓ —— 对模型确实隐藏了 |
| 直接用 `tool_call` 调用 | **`ValidationError: conversation_id Field required`** —— 值必须另行注入,而该注入机制在 langchain-core 1.6.3 上无现成文档 |

**改用闭包工厂**(已验证可用):

```python
def make_create_ticket(conversation_id: str):
    @tool
    async def create_ticket(description: str, ticket_type: str) -> str:
        """创建人工工单,会话由系统自动关联。"""
        ...
    return create_ticket
```

`conversation_id` **根本不在签名里**,值从闭包来,模型既看不见也传不错,且不需要依赖任何注入机制。代价:`create_ticket` 需**每请求构造**,故工具集不是纯模块级常量 —— 见 §6.6 的注册表设计。

### 6.6 工具基础设施四件套

| 能力 | 落法 |
|---|---|
| 注册管理 | `build_tools(conversation_id) -> list[BaseTool]` 组装本请求的工具集;**`registry_for(tools) -> dict[str, BaseTool]`** 由该列表建映射(因 `create_ticket` 每请求构造,注册表不再是纯模块级常量);模型返回未知工具名 → 可恢复错误,回灌「工具不存在」 |
| 参数 Schema 校验 | `@tool` 已从类型注解自动生成 pydantic `args_schema`(实测第 8 条),无需手写 |
| 执行错误处理 | 见 §6.7 |
| 超时重试 | `asyncio.wait_for`,默认 10s;**重试白名单制** |

**重试用白名单,不用黑名单。** 只重试三个 `query_*`;**`create_ticket` 永不重试** —— 它是写操作,超时后重试会建出两张工单,而"超时"恰恰意味着我们不知道第一次到底成没成。默认不重试、显式声明可重试,比反过来安全。

### 6.7 错误分类

> **用户裁决**:区分可恢复/不可恢复

| 情况 | 分类 | 行为 |
|---|---|---|
| 订单/商品不存在 | 可恢复 | `ToolMessage` 回灌,模型自然语言兜住;SSE `tool_result(ok=false)` |
| 参数校验失败 | 可恢复 | 同上 |
| 未知工具名 | 可恢复 | 同上 |
| 工具超时(重试后仍超时) | 可恢复 | 同上 |
| DB 连接失败等基础设施故障 | **不可恢复** | SSE `error` 帧 + 终止流 |
| 未预期异常 | **不可恢复** | SSE `error` 帧 + 终止流 |

**单轮的必然代价(用户已确认接受)**:参数校验失败时,模型**没有第二次机会改参数** —— 第二轮不绑 tools,它只能在最终回答里解释「没能查到,请核对单号」。给它改参数的机会就等于放开第二轮工具调用,即 Agent Loop。

### 6.8 `conversations.status` 的实际使用

取值 `active` / `pending_human`。**`create_ticket` 成功时把该会话置为 `pending_human`** —— 否则该字段是死列。聊天页页脚文案「涉及具体订单会为你转接人工核实」与它呼应。

### 6.9 消息落库时机

**沿用 ch01 的「流完整走完才写」**:整轮成功后一次性写入该轮全部消息。

- 调了工具:`user` / `assistant`(带 `tool_calls`,`content` 为空串) / `tool`(带 `tool_call_id`) / `assistant`(最终回复)
- 未调工具:`user` / `assistant`(最终回复)

好处:不会留下「有问无答」的孤儿行,与 ch01「半截回复不污染历史」的语义一致。

代价:**工具执行完但第二轮流断了,这轮什么都不落**,包括工具调用记录。

## 7. 数据层

### 7.1 四张表

| 表 | 列 | 说明 |
|---|---|---|
| `faq` | id(PK), question, answer, category | `category` 建索引 |
| `conversations` | id(PK, CHAR(32)), user, status, created_at | id 复用 `session_id` |
| `messages` | id(PK), conversation_id(FK), role, content, tool_calls, tool_call_id, created_at | `tool_calls` 用 **JSON 列** —— assistant 的「工具调用申请」是数组 |
| `tickets` | ticket_no(PK), conversation_id(FK), description, ticket_type, status, created_at | 按用户要求用**工单号做主键**,不用自增 id |

**字符集必须 `utf8mb4`** —— 中文 + emoji(实测模型回复中出现过 😊)在 `utf8` 下会失败。

`messages.role` 取值 `user` / `assistant` / `tool`(不含 `system`)。

### 7.2 建表方式

> **用户裁决**:同意用 `metadata.create_all()` + 显式 init 脚本,**不上 Alembic**

- `scripts/init_db.py` —— 建库建表
- `scripts/seed_db.py` —— 灌种子数据

理由:本章是首次建表、无迁移历史、schema 仍会变动;Alembic 会让每次列变更都产生迁移文件。**这是有意的取舍,不是遗漏** —— 若后续章节需要迁移历史,届时引入。

### 7.3 种子数据

- **`faq`**:灌若干条,分类覆盖退换货 / 发票 / 物流 / 商品等。
  **刻意不含「邮费」「运费」相关条目** —— 验收 3 的漏召回是预期结果,种子数据必须保证它确实查不到,否则验收 3 会**假通过**。
- 其余三张表:灌一组样例(1 个会话 + 2 条消息 + 1 张工单),让表非空、便于肉眼验证 schema。运行时数据照常写入。

### 7.4 数据库实例(实现订正)

**本节原写「`docker-compose.yml` 起 MySQL 8.4,端口 3306」,与实际不符,订正如下。**

数据库实例**由用户自行提供**,不在本仓库的管理范围内:

| 项 | 实际值 | 如何核实 |
|---|---|---|
| 镜像 | `mysql:8.0`(服务器版本 **8.0.46**) | `docker ps` + `SELECT VERSION()` |
| 主机端口 | **3307** → 容器 3306 | `docker ps` |
| 库名 | `mewhelp` | 用户已建,初始为空 |
| 字符集 / 排序规则 | `utf8mb4` / **`utf8mb4_0900_ai_ci`** | `SELECT @@character_set_database, @@collation_database` |
| 连接串 | `mysql+asyncmy://root:***@127.0.0.1:3307/mewhelp?charset=utf8mb4` | 已配在 `.env`,非跟踪文件 |

仓库**不提交 `docker-compose.yml`** —— 实例已存在,再提交一份会与用户的运行中容器冲突(同名服务、同端口),制造"起不来"的困惑。复现环境所需的全部信息是上表,写在这里即可。

**`DATABASE_URL` 已由用户配进 `.env`**,但 **`.env.example`(入库模板)里缺这一项**,需补 —— 否则新克隆的仓库不知道要配它。

**字符集风险已实测排除**:`utf8mb4_0900_ai_ci` 下中文 `LIKE` 子串匹配正常(正反例均验证),**无需改用 `utf8mb4_unicode_ci`**。§9 中原列的那条风险据此关闭。

## 8. 测试与验收

### 8.1 分三层

ch01 的约定是「单测全程不联网」。本章有真实 DB,而用 SQLite 顶替 MySQL 会漏掉 MySQL 特有的行为(`utf8mb4`、JSON 列、LIKE 的中文 collation)—— 那正是 ch01 反复栽的「假绿」。故分三层:

| 层 | 跑法 | 覆盖 |
|---|---|---|
| **Tier 1 纯逻辑** | `pytest -q`,不联网、不依赖 Docker | 编排、错误分类、重试白名单、裁剪分轮、注册表、种子确定性 |
| **Tier 2 DB 集成** | `pytest -q -m db`,需 MySQL 在跑 | schema 真建得出来、中文往返不炸、JSON 列读写、LIKE 查中文 |
| **Tier 3 真实模型** | 评估集 + 验收脚本,需真实 key | 工具选择、端到端 |

Tier 1 靠**注入假 repository** 把 DB 摘掉。Tier 2 用 `@pytest.mark.db` 标记,`pytest.ini` 默认跑,无 Docker 时用 `-m "not db"` 跳过。

### 8.2 Tier 1 里必须能「区分正确与错误实现」的断言

ch01 复盘的头号结论是「测试要能区分正确与错误实现」。以下四条按此设计,每条都注明**改错了必须挂**:

| 断言 | 错误实现下应失败 |
|---|---|
| 裁剪不切开 tool 配对 —— 构造「历史刚好长到触发裁剪」的输入,断言留下的序列里每条 `tool` 消息前面都有它的 `assistant` 父亲 | 把 §6.3 的规则改回 ch01 的旧写法,必须红 |
| `create_ticket` 不重试 —— 假工具记下调用次数,超时场景断言**恰好 1 次** | 去掉重试白名单,必须红 |
| 种子跨进程确定性 —— `subprocess` 起两个独立进程跑同一入参,断言输出**逐字节相同** | 把 `sha256` 换成内置 `hash()`,必须红(同进程内测不出来) |
| 错误分类分界 —— DB 故障断言出 `error` 帧;订单不存在断言出 `tool_result(ok=false)` + 正常 `done` | 「一律回灌」或「一律报错」都必须红 |

### 8.3 工具选择评估集(替代 TDD 的那一步)

按用户规则 1 —— 「非可单测的产出用评估集验证」。本章的非可单测产出是**模型选哪个工具**。

`evals/tool_selection_cases.jsonl`:约 15 条问句 → 期望工具名(含 `null` 表示**不该调工具**)。覆盖:

- 四类明确查询(订单 / 商品 / 物流 / FAQ)
- 转人工(应 `create_ticket`)
- **不该调工具的**:闲聊「你好」、不涉及数据的「你们几点下班」
- **边界诱饵**:「订单 1001 的物流到哪了」同时涉及 order 与 logistics

评分口径:**闭式精确匹配**(工具名是枚举)。这与 ch01 的 `expected_solution` 形成对比 —— 那里是自由文本、关键词口径最终被证明是样本拟合。本章口径**不会重蹈该覆辙**,可信度与 `order_id` / `request_type` 同级。

「不调工具」的用例尤其重要 —— 防的是「模型见谁都调工具」。

### 8.4 端到端验收

扩展 `scripts/acceptance.sh`,继续遵守该文件头部已记录的**两个平台陷阱规避**(含中文的请求体走 stdin;断言比对「拼回后的文本」而非原始 SSE 流):

1. 「订单 1001 的物流到哪了」→ 断言出现 `tool_call` 帧且 `name == query_logistics`;最终回复含确定性数据中的**具体词**
2. 「退货政策是什么」→ 断言 `tool_call` 是 `query_faq`;回复非空且含中文
3. 「邮费是多少」→ **反向断言**:模型**没有**成功从 `query_faq` 拿到结果,且回复未编造价目表

第 3 条是**预期失败**,脚本把它断言成「确实失败了」,并记入 dev-notes —— 对应验收标准 3 的「记下来留给下一步升级」。

**脚本的能力边界**:验收 1 里「浏览器看到气泡上的工具徽章」这一半**脚本验不到** —— 脚本只能验到 SSE 层的 `tool_call` 帧确实推了出来。徽章渲染是否正确,属于 §8.5 的人工浏览器确认。两者相加才覆盖验收标准 1 的全文。

### 8.5 前端

按用户规则 1 的例外条款,聊天页走 **Vibe Coding**,不进 TDD / 评估集 / code review。

单页静态文件由 FastAPI `StaticFiles` 挂载,对齐 `asserts/img.png` 的布局(顶栏 + 气泡 + 工具徽章 + 输入栏),**底色改浅蓝、顶栏与气泡配色跟着调整**。排在最后做。

## 9. 待实测项与风险

| 项 | 说明 | 处置 |
|---|---|---|
| ~~MySQL 版本与实例~~ | **已实测,已解决。** 实际为 `mysql:8.0`,服务器 **8.0.46**,主机端口 **3307**;实例由用户提供,仓库不管理 | 已回写 §7.4。原「8.4 + 端口 3306」的设想作废 |
| ~~`utf8mb4` 下的中文 LIKE~~ | **已实测,已解决。** `utf8mb4_0900_ai_ci` 下中文子串匹配正常,正反例均验证通过 | 无需改用 `utf8mb4_unicode_ci`。Tier 2 仍保留中文往返测试作为回归防护 |
| `.env.example` 缺 `DATABASE_URL` | 入库模板里没有这一项,新克隆的仓库不知道要配 | 实现时补上(已列入计划) |
| 第二轮不绑 tools 是否影响回复质量 | 实测第 5 条只验证了"能收敛成文本",未评估质量 | 评估集与验收观察;若质量下降,退回"绑 tools 但限制轮数" |
| **参数校验失败无第二次机会** | §6.7 已记录;单轮的必然代价 | 用户已确认接受;评估集可观察到发生频率 |
| 「文本先出后又调工具」 | §6.2 已定义行为 | 实测未出现;若频繁出现需重新评估第一轮策略 |
| 工具超时/重试具体阈值 | 10s / 1 次重试均为估计值 | 先用默认;拿到真实分布后再调 |
| ~~ch01 遗留:`_locks`/`_touched` 无硬数量上限~~ | **本章关闭。** `MAX_SESSIONS` 改用于限制锁表,见 §6.4 订正 | 已解决 |
| ch01 遗留:锁获取的取消竞态 | 见 ch01 spec §9;owner-tracking 约 10 行可关 | **本章不处理**,留待需要时 |
| 多进程 / 多 worker | 锁在进程内,历史在 MySQL | 仍不在本章范围;若引入多 worker,锁需改为分布式 |

## 10. 本章不做

- 多轮自动循环的 Agent Loop(单轮是结构保证)
- 向量检索、RAG
- 认证与鉴权(`user_id` 不做任何校验)
- Alembic 迁移
- 真实电商 / 物流系统对接(三个工具内部生成数据)
- 商品表、订单表、物流表(不建,只有四张表)
- 前端构建工具链(单页静态文件)

## 11. 配置项(.env 新增)

```bash
# 必填
DATABASE_URL=mysql+asyncmy://root:pass@127.0.0.1:3306/mewhelp?charset=utf8mb4

# 可选,均有默认值
TOOL_TIMEOUT_SECONDS=10
TOOL_RETRY_ATTEMPTS=1
TOOL_RETRY_DELAY_SECONDS=0.3
```

`TOOL_TIMEOUT_SECONDS` 是**单次尝试**的超时,不是整轮工具调用的总超时。

`TOOL_RETRY_ATTEMPTS` 是**重试次数,不含首次** —— 默认 `1` 表示「首次失败后额外尝试 1 次,共 2 次尝试」。设为 `0` 表示不重试。

ch01 已有的配置项不变。`DATABASE_URL` 设为必填(与 `OPENAI_MODEL` 同理:默认值会拿一个可能不对的连接串去连,报错指向"连不上"而非"没配")。
