# ch07 设计:把简单裁剪升级成正经的上下文管理

> 设计源。写法沿用前六章:每节都写「为什么这么定」,
> 而不是只写「定成什么」—— 本章的每一处都对应一次真实故障或一条实测约束。
>
> 本章**只做当前会话**:三层滑窗 + 后台摘要 + 多会话前端。
> 不做跨会话长期记忆、不做用户画像、不做语义检索捞历史、不做主题重要度。

---

## §1 目标与非目标

### 目标

把 ch01 那条「按整轮裁到 token 预算」的单层裁剪(`app/memory/trim.py::select_history`,
**该函数已在 T10 删除** —— 它与三层结构在语义上不可叠加,见 §12.7;同一节里
`to_rounds` 仍在、并接管了「按 `user` 边界切轮」那条不变量),
升级成**三层结构 + 后台异步摘要**的上下文管理:

1. 历史按离当前轮的远近切三段:最近**原文**、中间**截短**、最远**梗概**;
   两个锚点 id 划边界,降级只挪 id、不搬数据。
2. 中间层攒到超预算时,后台异步压成一段梗概追加进摘要表,**不阻塞用户这一轮**。
3. 调模型的上下文按**固定顺序**拼装,`system` 只占第一条,保住前缀缓存。
4. token 预算**从模型窗口倒推**,不写死常量。
5. 会话上下文随 LangGraph 的 `State` 贯穿(`add_messages` reducer + checkpoint)。
6. 上下文可观测:每轮把实际发出的上下文原样落盘。
7. 前端多会话:侧栏列表 + 切换回载。

### 非目标(明确不做)

- **跨会话长期记忆 / 用户画像**。`conversation_summaries` 按会话隔离,**永不跨会话读**。
- **语义检索捞历史**、**主题重要度留关键事实**。理论篇的三种选法里,本章只做
  「滑窗打底 + 摘要」这两层,另外两层按需再上。
- 认证。侧栏列的是固定 `user='demo-user'` 的全部会话(见 §2.4)。
- 摘要的淘汰与清理。表只追加,不删除、不重写。
- 多轮 Agent Loop(ch05 起就没做,本章不动)。

---

## §2 设计依据

本章有五条依据,**四条是本机实测/源码核对得出的**,一条是产品口径。
每一条都改变了做法,不是背景介绍。

### 2.1 `add_messages` 是 append-only,无 id 的消息会被当场赋 uuid

Context7 查 `langgraph.graph.message.add_messages` 的源码原文:

> "By default, this ensures the state is 'append-only', unless the new message
> has the same ID as an existing message."
>
> ```python
> for m in right:
>     if m.id is None:
>         m.id = str(uuid.uuid4())     # ← 没有 id 就现赋一个
> ```

**后果**:如果每轮都「从 MySQL 读全量历史 → 塞进 `state.messages`」,那重新构造的
消息**没有 id**、拿到的是**全新 uuid**、**一个都匹配不上** ⇒ 整段历史被**再追加一遍**。
第三轮的时候历史就是三份,而**每一轮的回复看起来都完全正常**。

这是本章最容易溜过去的一处:它与 ch05 那条「未声明通道写入被静默丢弃」是同一族
(静默、不报错、单测绿),只是这次的证据来自官方源码而不是本机实测。

**做法**:播种**只在 `state["messages"]` 为空时发生**(见 §7.4),而不是每轮无条件播种。

### 2.2 `InMemorySaver` 是纯内存字典,「落盘」不成立

```python
class InMemorySaver:
    """An in-memory checkpoint saver.

    Note:
        Only use `InMemorySaver` for debugging or testing purposes.
    """
    def __init__(self, ...):
        self.storage = factory(lambda: defaultdict(dict))
```

且环境里 **只装了 `langgraph.checkpoint.{base, memory, serde}`** —— 没有任何持久化
saver(sqlite / redis / postgres 一个都没装)。

**这直接推翻了需求里「State 里的完整历史靠 checkpoint **落盘**留着」这句**。
在本仓库,checkpoint 是进程内的热副本,服务一停就没了。要真落盘得装新包,
而那与「技术栈无新增组件」冲突。

**做法**:接受内存 checkpoint,**MySQL 仍是跨会话/跨进程的权威源**。
本条作为**与原始需求的偏离**记录在此,实现订正段(§12)沿用。

### 2.3 `trim_messages` 的签名(逐字核对)

```python
trim_messages(
  messages, *, max_tokens: int,
  token_counter: Callable[[list[BaseMessage]], int] | BaseLanguageModel | Literal['approximate'],
  strategy: Literal['first','last'] = 'last',
  allow_partial: bool = False,
  end_on=None, start_on=None, include_system: bool = False, text_splitter=None,
) -> list[BaseMessage]
```

两条关键:
- **`start_on` 只对 `strategy='last'` 生效**(文档原文:"only for strategy='last'")。
  层 1 正好是「保尾」,所以能用。
- **`token_counter` 收的是 `list[BaseMessage]`**,而本仓的 `count_tokens` 收 `str` ——
  中间要一个适配器(`prompts.py` 里,见 §7.3)。

**用法**:层 1 = `trim_messages(msgs, max_tokens=层1预算, token_counter=适配器,
strategy="last", start_on="human", include_system=False, allow_partial=False)`。
`start_on="human"` 保证不从一轮中间切开,从而**保住 `tool` 消息与它 `tool_call_id` 父亲的配对**
(切开就是上游 400,且只在历史长到触发裁剪时复现 —— ch01 的 `_to_rounds` 就是为它写的)。

**层 2 没有现成 API**:「用户原话不动、客服答复截短、工具结果一行化」是自定义规则,
自己写(§3.3)。

### 2.4 三条现状偏差(都是「读代码才知道」的)

**(a) `log/app.log` 不存在。** 全仓没有 `basicConfig` / `FileHandler` / `dictConfig`,
`logger.info(...)` 全靠 uvicorn 默认 handler 打到控制台。§6.3 要新增日志落盘配置。

**(b) `init_db.py` 不会加列。** 它跑 `Base.metadata.create_all`,只建**不存在的表**。
给 `conversations` 加两个锚点列**必须手写 DDL**,`create_all` 补不上 ——
与 ch06 的 `db/ch06.sql` 同模式。

**(c) 工具结果从来不落表。** `app/agent/nodes.py::make_agent_node` 在**轮内**用局部变量
`msgs` 攒 ReAct 往返,只有最后一句回复经 `append_turn` 写成 `user` + `assistant` 两行。
`grep -rn 'role="tool"' app/` **在 production 里零命中**(只有 `tests/test_history.py`
为了验往返写过)。所以 `messages` 表的 `role` / `tool_calls` / `tool_call_id` 三列
**一直是备而未用**;`trim.py::_to_rounds` 那段「按 user 边界切轮以免切开 tool 配对」
的注释,描述的是一个**当时还不存在的形态**。

**本章起工具结果落表**(用户 2026-09-22 拍板),上面这些终于进入真链路。

**产品口径**:`GET /api/conversations` 固定按 `user='demo-user'` 过滤(无认证,
前端从来不传 `user_id`,服务端因此一直回退到该默认值)。

---

## §3 上下文结构

### 3.1 三层与两个锚点

```
messages 表(按 id 升序)
├───────────────┬──────────────────────┬─────────────────────────┤
│  已被梗概覆盖  │        层 2           │         层 1            │
│  (不再逐条读)  │   中间,截短,预算 30%  │  最近,原文,预算 70%    │
└───────────────┴──────────────────────┴─────────────────────────┘
                ▲                      ▲
     summary_upto_msg_id      layer1_from_msg_id
```

两个锚点存在 `conversations` 两列上(`BIGINT NOT NULL DEFAULT 0`):

| 列 | 含义 | `0` 的语义 |
|---|---|---|
| `summary_upto_msg_id` | 梗概已覆盖到哪条(**含**) | 尚无任何梗概 |
| `layer1_from_msg_id` | 层 1 从哪条(**含**)起 | 层 1 起于最早,层 2 为空 |

**不变量**:`0 ≤ summary_upto_msg_id ≤ layer1_from_msg_id`。
两者都是 `messages.id`(自增),大小关系即先后关系。

**「降级只挪两个 id、不搬数据」**:两个锚点都是纯指针,挪动不涉及任何行级读写。
这条是本章结构上的核心卖点,也是 §10 里最好测的一组断言(纯函数 + 两个整数)。

### 3.2 三层各自的 token **按哪一版数**(这条决定截短有没有用)

| 层 | 数的是 | 为什么 |
|---|---|---|
| 层 1 | **原文** | 它就是原文,没有别的版本 |
| 层 2 | **截短后的渲染结果** | ← **这条是层 2 存在的全部意义** |
| 梗概 | 梗概正文 | 注入时就那么长 |

**层 2 必须按截短后的版本计数。** 按原文数的话,§3.4 的截短就退化成纯粹的渲染装饰 ——
层 2 该什么时候触发摘要还是什么时候触发,截短**对级联零影响**。
而按截短后数,截短就成了一道**免费的有损压缩**:一段 3000 token 的原文截完只剩
三四百,层 2 能多装**好几倍**的轮次,才轮到「摘要」这个不可逆动作上场。

这就是三档压缩**强度递增、代价也递增**的账:

```
段原文 3000 token
  ├ 层 1:3000 token,一字不动                      ← 代价 0
  ├ 层 2:~400 token,截短                          ← 有损但**可逆**(原文仍在 MySQL)
  └ 梗概:~150 token,提炼                          ← **不可逆**,原文从此不再进上下文
```

**先用便宜的,不够了才用贵的** —— 层 2 把摘要往后推,而摘要一旦发生就回不去了。
`per_round_steady`(§9.1)量的因此是**原文**的每轮稳态占用(它对应层 1,
即「想原样留住几轮」);层 2 的每轮占用由截短规则算出,不是配置项。

### 3.3 两个动作

**降级(层 1 的原文 token 超预算)**:把 `layer1_from_msg_id` **往后挪**,直到层 1 装得下。
挪过去的那几轮**自动落进层 2** —— 不需要额外写任何东西,而它们**下一轮就以截短形态出现**。

**摘要(层 2 的截短后 token 超预算)**:把区间 `(summary_upto, layer1_from]`
的**原文**压成一段梗概,**追加**进 `conversation_summaries`;
成功后把 `summary_upto_msg_id` **推到** `layer1_from_msg_id`。层 2 因此清空,循环重新开始。

**注意两个动作用的版本不同**:降级看**层 1 原文**,摘要看**层 2 截短后**,
而摘要**读的是原文**(截短只服务于「装进上下文」,不服务于「提炼梗概」——
拿截短的文本去提炼,等于把截断损失焊进梗概)。

**降级要循环到收敛**:挪一次 `layer1_from` 之后层 1 变小、层 2 变大,
所以「挪 → 重算 → 还超就再挪」是一个循环,不是一次判断。

```
新消息不断到来 → 层1 涨 → 超预算 → 降级(挪 layer1_from)→ 层2 涨
                                      → 超预算 → 摘要(挪 summary_upto)→ 层2 清空
```

**旧梗概只作背景给模型看,不参与合并** —— 每次摘要的输入**只有原文区间**,
不喂上一段梗概。理由是需求里的那条:同一事实被反复有损压缩会逐次失真,
而「压完不回头重写」保证每段梗概只被压**一次**。

### 3.4 层 2 的截短规则

逐条处理,**保持消息的角色与结构**:

| 角色 | 处理 |
|---|---|
| `user` | **原样,一个字不动** |
| `assistant`(纯文本) | `content` 截到前 **50** 字,加 `…` |
| `assistant`(带 `tool_calls`) | `content` **按同一规则截到 50 字**(通常本来就是空串,截了也没差);**`tool_calls` 原样保留** |
| `tool` | `content` 换成一行标识:`[工具结果] <前 60 字>…` |

**`content` 一律按同一条规则截**,不为「带 `tool_calls` 的 assistant」开小灶 ——
需求 1 的原话是「**客服答复**只留开头几十个字」,而工具调用前那句开场白**就是**客服答复。
原先那句「`content` 保留」是在**陈述常见情形**(带工具调用的 assistant,turn content
通常是空串),不是一条豁免规则;写成表格里的一行容易被读成豁免,故订正措辞。

**为什么 `tool_calls` 必须原样保留**:上游要求 `tool` 消息前面紧跟带对应
`tool_call_id` 的 `assistant`。截断 `tool_calls` 就等于把这对拆开 ⇒ 400,
且只在历史长到触发分层时复现。**只截 `tool` 的 `content`,不动配对的结构字段** ——
这样「工具结果一行化」与「配对完整」两个要求同时成立。

**不把层 2 拍平成一段文本**:拍平更省 token,但会同时丢掉角色与配对,
让「谁说的哪句」变成模型自己猜。本章不省这个。

#### 截短到底省了多少(拿一轮带工具调用的真实形状算)

一轮「订单 1002 能退吗」经 ReAct 走完,落表 4 行:

| 行 | 原文 | 截短后 |
|---|---|---|
| `user` 原话 | `订单 1002 能退吗` / ~15 tok | **不动** / ~15 tok |
| `assistant`(带 `tool_calls`) | `""` + tool_calls / ~45 tok | 保留结构 / ~45 tok |
| `tool` 结果 | `{"order_no":"1002","status":"已取消",…}` / **~300 tok** | `[工具结果] {"order_no":"1002","stat…` / **~70 tok** |
| `assistant` 回复 | `您的订单 1002 当前状态为已取消…` / ~120 tok | 前 50 字 + `…` / **~55 tok** |
| | **合计 ~480 tok** | **合计 ~185 tok** |

**压缩比约 2.6×,而这一档是可逆的**(原文一行没删,切回旧会话照样看得见全文)。
层 2 预算 2082 因此能装下 **约 11 轮**这样的对话,才轮到摘要上场;
按原文数的话只装得下 **4 轮**,摘要会**提前三次**发生 ——
而摘要不可逆。**这就是「先用便宜的」在数字上的样子。**

工具结果越大,这个比值越夸张(单条封顶 1200 tok,截完 70 tok,接近 17×)。

---

## §4 Prompt(非可单测产出 → 用标注样例验证)

本章只新增**一个** Prompt:摘要。其余(`COMPLAINT_REPLY` 等)不动。

### 4.1 摘要 Prompt

**要提炼什么**(四样,少一样就丢上下文):
1. 问过的商品 / 款式;
2. 报过的订单号、手机号等标识;
3. 明确的诉求;
4. **还没解决的问题**。

**硬约束**:
- **对话里没出现的内容一个字不许编** —— 摘要会被当成背景注入后续每一轮,
  编出来的事实会被模型当成真的,而且**无法追溯**(原文已经不喂了)。
- 寒暄、客套、对话状态**不留**。
- 长度 **几十到一两百字**。

**为什么这条要按 ch03 的教训写**:ch03 的挖知识 prompt 首版没写「什么不算知识」,
把客服「抱歉查不到运费」这种**非答案**挖成了知识,直接把验收的正确答案挤下 top-1。
**模型的失败被挖进知识库,再教它下次继续失败。** 摘要有同一个形状的失效模式:
把寒暄与自我否定压进梗概,然后每轮都注入一遍。

**验证方式**:非可单测产出 → 按工作要求 1,用**标注样例**跑一遍
(`evals/summary_cases.jsonl`,见 §10.4),不写逐字断言。

---

## §5 接口契约

### 5.1 `GET /api/conversations`(新增,只读)

```
→ 200 {"items": [
    {"id": "…32位…", "created_at": "2026-09-22T10:00:00",
     "preview": "这个能退吗", "summarized": true}
  ]}
```

- `WHERE user = 'demo-user' ORDER BY created_at DESC`(新在前)
- `preview` = 该会话**第一条 `role='user'` 消息**的前 30 字;没有则空串
- `summarized` = 该会话 `summary_upto_msg_id > 0`
- 不分页(演示规模,与会话数的量级匹配);不是分页接口,别按分页写前端

### 5.2 `GET /api/conversations/{id}/messages`(新增,只读)

```
→ 200 {"items": [{"role": "user", "content": "…", "created_at": "…"}]}
→ 404 {"detail": "会话不存在"}      ← 会话 id 不存在时
```

按 `id` 升序返回**原文**(不截短、不含梗概)—— 侧栏切回来要看的是当初聊了什么。

**并且过滤掉两类「用户没见过」的行:**
- `role='tool'` 的行;
- **`content` 为空串的行** —— 工具调用那条 assistant 消息的 `content` 是空串(它只带 `tool_calls`,
  见 `app/agent/nodes.py` 的 `content=m.content or ""`),不过滤的话**每个「模型调了工具」的轮次
  都会在侧栏画出一个空气泡**。用 `content != ''` 而不是「assistant 且不带 tool_calls」:
  它是**更窄、更保守**的那个条件,**不是**「assistant 且不带 `tool_calls`」的等价物
  —— 2026-09-22 订正:`_lc_to_records` 会把 `content` 与 `tool_calls` **一起**写,
  而 `_stream_round` 边累积 chunk 边把文字发 token 帧,所以
  **一条既带 `tool_calls`、又带开场白的 assistant 行是结构上可达的**;
  对那种行,「`content != ''`」**返回**它,而「不带 `tool_calls`」会**丢掉**它。
  **丢掉它是错的** —— 那段开场白**是用户看见过的**(它以 token 帧流出去过)。
  所以取 `content != ''`:它**只**滤掉用户没见过的空气泡,不误伤任何见过的东西。
  (写作要求:实现侧的 docstring 要写**这条更窄的理由**,不要再写成「等价」。)

2026-09-22 裁定 —— 这是本章新引入的一个缝:
ch07 起**工具结果开始落库**(§2.4c),而本端点返回每一行 ⇒ 若不筛,
侧栏回载会把 `{"order_no":"1002","status":"已取消"}` 这种**原始工具载荷**
当成一条消息气泡画出来。**用户要求 7 的「回载历史原文」指的是「他看见过的对话」**,
而工具往返是**模型与工具之间的**,不是对话的一部分。
(同理 `preview` 也只取 `role='user'` 的行 —— 它本来就只取 user。) 

### 5.3 `POST /api/chat/stream`(改造)

请求体形状**不变**。新增两条校验/行为:

- `max_user_input_tokens` 超限 → **400**(与既有 `ContextOverflowError` 同一族,
  必须是普通 JSON 而非 SSE —— 一旦 yield 过首帧就改不了状态码)。
- 历史预算连**一轮**都装不下 → **400**,文案固定。

`done` 帧形状不变。**`usage` 仍是死值 `None`** —— 本章不接它
(接它必须连「每轮重置里清 usage」一起做,否则非 Agent 轮会报上一轮的数;
见 `app/api/chat.py` 那段注释)。本章**不做**这件事,如实记账。

---

## §6 数据

### 6.1 `db/ch07.sql` — 新表 `conversation_summaries`

**⚠️ 约束必须在 ORM 与 DDL 两侧都声明**,且 DDL **不加 `IF NOT EXISTS`**:
建库有两条路径(`scripts/init_db.py` 的 `create_all` 与手工执行 `db/ch07.sql`),
只在一侧声明唯一键,两条路径建出来的表**形状不同** —— 行为变成「看谁建的库」,
而没有任何东西报错。加 `IF NOT EXISTS` 更糟:`create_all` 先跑时这一句**静默跳过**,
唯一键永远不存在。(同类教训:`RefundRequest.status` 的 `default`/`server_default` 两处都要。)

**⚠️ 文件内的顺序是刻意的:`ALTER TABLE conversations` 在前、`CREATE TABLE` 在后。**
mysql 客户端**遇到第一个错误就中止整个脚本**。反过来排的话,在「梗概表已存在、
但两个锚点列还没加」的库上(先跑过 `scripts/init_db.py` 的 `create_all` 就是这情形),
`CREATE` 报 1050 中止 ⇒ **`ALTER` 永远不执行** ⇒ 库缺两列,
而报错读起来像「表已存在 ⇒ 已经装好了」。

```sql
ALTER TABLE conversations
  ADD COLUMN summary_upto_msg_id BIGINT NOT NULL DEFAULT 0,
  ADD COLUMN layer1_from_msg_id  BIGINT NOT NULL DEFAULT 0;

CREATE TABLE conversation_summaries (
  id             BIGINT       NOT NULL AUTO_INCREMENT,
  conversation_id VARCHAR(32) NOT NULL,
  seq            INT          NOT NULL COMMENT '第 N 段,从 1 起,只增不改',
  upto_msg_id    BIGINT       NOT NULL COMMENT '这一段覆盖到哪条 messages.id(含)',
  content        TEXT         NOT NULL COMMENT '梗概正文,几十到一两百字',
  created_at     DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_conv_seq (conversation_id, seq),
  KEY idx_conv (conversation_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

`seq` 由摘要任务在**成功提交前**取 `SELECT COALESCE(MAX(seq), 0) + 1` 算得,
只在同一个事务里与「推进 `summary_upto_msg_id`」一起提交 ——
**两者要么都成、要么都不成**:梗概追加了但边界没推,下次会把同一段原文**再压一遍**
(重复梗概,而每一段单独看都正常);边界推了但梗概没落,那段历史就**永久消失**
(区间已经不在层 2 的读取范围里,而摘要表里没有对应的替换物)。
这是本章**唯一**一处必须原子提交的两步写。

`UNIQUE (conversation_id, seq)` 是**并发保护的第二道**:同一会话两个摘要任务同时
跑到提交步时,后者撞唯一键 ⇒ 失败 ⇒ 边界不推进 ⇒ 下次重来。比只靠内存锁可靠
(内存锁挡不住多进程)。

### 6.2 `db/ch07.sql` — `conversations` 加两个锚点列

```sql
ALTER TABLE conversations
  ADD COLUMN summary_upto_msg_id BIGINT NOT NULL DEFAULT 0,
  ADD COLUMN layer1_from_msg_id  BIGINT NOT NULL DEFAULT 0;
```

**必须手写**(§2.4b):`scripts/init_db.py` 跑的是 `Base.metadata.create_all`,
只建表、**不加列**。ORM 侧 `app/db/models.py::Conversation` 同步加两个字段,
`server_default="0"` —— 两侧默认值都要(`default` 管 ORM 插入,`server_default`
管裸 SQL),理由与 `RefundRequest.status` 那处相同。

### 6.3 日志落盘(新增配置)

`log/app.log` 当前**不存在**(§2.4a)。新增 `app/logging_setup.py`:

- `RotatingFileHandler("log/app.log", maxBytes=5MB, backupCount=3, encoding="utf-8")`
- **`encoding="utf-8"` 是硬要求,不是讲究**:本机 locale 是 **cp936**。
  不给 `encoding` 时 Python 用 `locale.getpreferredencoding()`,中文日志行会**直接抛
  `UnicodeEncodeError`**。这条在 ch02 以别的形态发作过三次。
- 目录不存在时 `mkdir(parents=True, exist_ok=True)`
- 在 `app/main.py` 的 `lifespan` 里调用

---

## §7 模块布局

| 新/改 | 文件 | 职责 | 依赖 |
|---|---|---|---|
| 新 | `app/memory/budget.py` | 窗口→历史预算→层1/层2 推导;**启动自检** | 无(纯) |
| 新 | `app/memory/layers.py` | 按锚点切三层 + 层 2 截短渲染 | 无(纯,只碰 `schemas.Message`) |
| 新 | `app/memory/summarize.py` | 摘要 prompt 组装 + 触发判定(纯)+ 落库 + 推进锚点 | db、llm |
| 新 | `app/memory/tasks.py` | 后台摘要执行体 | 复刻 `app/kb/orchestrate.py` |
| 新 | `app/memory/journal.py` | `model_ctx` / `history_ctx` 组装与原样落盘 | 无 |
| 新 | `app/logging_setup.py` | 日志落盘配置 | 无 |
| 新 | `app/api/conversations.py` | 两个只读端点 | db |
| 新 | `db/ch07.sql` | 新表 + 两个锚点列 | — |
| 改 | `app/prompts.py` | 新增 `build_context_messages`(**LangChain 唯一面**) | LC |
| 改 | `app/services/history.py` | 写工具行、读梗概、推进两个锚点 | db |
| 改 | `app/agent/state.py` | `messages`(add_messages)+ 两个锚点通道 | — |
| 改 | `app/agent/nodes.py` | agent 节点改读 `state["messages"]`;`log_turn` 写工具往返 | — |
| 改 | `app/api/chat.py` | 播种、起后台任务、用户输入上限校验 | — |
| 改 | `app/config.py` | §9 的配置项 | — |
| 改 | `app/db/models.py` | `ConversationSummary` + `Conversation` 两列 | — |
| 改 | `app/main.py` | `setup_logging` + 启动自检 | — |

### 7.1 为什么 LangChain 只出现在 `prompts.py`

CLAUDE.md 有一条贯穿性约定:**`memory/` 与 `services/history.py` 不依赖 LangChain**,
只碰 `app.schemas.Message` 纯数据类;转 `BaseMessage` 是 `prompts.py::to_lc_messages`
**一处**的职责。

而本章技术栈定死了 `trim_messages`,它**只吃 LangChain 消息**。两条约束正面撞上。

**做法**:把 LangChain 面**全部收在 `prompts.py`** —— 它本来就是「消息组装的唯一出口」。
`build_context_messages` 在那里调 `trim_messages` 选层 1 并定序组装;
`memory/` 新增的四个模块**全部保持纯 `schemas.Message`**。**约定因此不破**,
而不是「本章例外」。

### 7.2 `memory/budget.py`(纯函数)

```
固定开销 = 系统提示 tokens + 工具定义 tokens + 检索证据 tokens
         + 注入梗概 tokens + max_output_tokens + safety_margin_tokens
单轮峰值 = max_agent_steps × tool_result_max_tokens + max_user_input_tokens

历史预算 = min( keep_rounds × per_round_steady ,                       ← 「想留住的轮数 × 每轮稳态」
                model_context_window − 固定开销 − 单轮峰值 )            ← 「窗口实际匀得出来的」
层1 = floor(0.7 × 历史预算)
层2 = 历史预算 − 层1
```

**两个数取小的那个**是需求 4 的原话,而且两条都真的会赢:轮数那一支在
「窗口很大但只想留 20 轮」时赢,窗口那一支在「想留 100 轮但窗口不够」时赢。
本章的演示配置下**窗口那一支赢**(见 §9)。

**启动自检**:`历史预算 < per_round_steady`(连一轮都装不下)⇒
**`logger.error` 报警 + 一条醒目的 startup 日志,但不拒绝启动**。
需求 4 的原文是「报警」不是「拒启动」;写成拒启动会让一个纯算术问题变成服务起不来。
两者都只差一行,**这是个可以一句话翻的选择**,如实记在此处备查。

### 7.3 `prompts.py::build_context_messages`

组装顺序(**固定的,每轮逐字节相同的前缀在前的**):

```
[0] SystemMessage(人设 + 红线)          ← 每轮完全一致,前缀缓存命中区
[1..a] 层 2 截短后的消息                  ← 内容稳定(已冻结的旧消息)
[a+1..b] 层 1 最近几轮原文                ← 只有新增轮次会变
[b+1] HumanMessage(用户原话 + 梗概 + 检索证据)
```

**`system` 只有一条,而且必须是第 0 条。** 需求 3 的理由值得逐字保留:
上游模板会把**所有** `system` 消息上提合并渲染,一旦有第二条 `system`,
工具定义就被挤到可变内容**之后**,前缀缓存整段作废。

**梗概与证据并进用户那条 `HumanMessage` 的内容、附在原话之后**(用户 2026-09-22 拍板),
**不另起一条消息** —— 与现有 `build_messages` 同形状,只是把证据的位置从原话**前**
改到原话**后**:

```python
# 现在(ch01–ch06)
text = user_input if not evidence else f"{render_evidence(evidence)}\n\n用户问题:{user_input}"
# 本章
text = f"{user_input}\n\n{render_summary_and_evidence(summary, evidence)}"
```

签名(**读 `history` 而非 `messages`** —— 传进来的已经是精简版,见 §7.5):

```python
def build_context_messages(
    *,
    brand_name: str,
    history: Sequence[Message],        # 精简版 = 层2 截短 + 层1 原文
    user_input: str,
    summary: str,                      # 梗概全文;空串 = 还没有梗概
    evidence: list[dict] | None = None,
) -> list[BaseMessage]:
```

`build_messages` **保留**(`agent` 之外没有别的调用点,但删它要连 4 条
`tests/test_prompts.py` 用例一起改,而本章不需要删),由 `build_context_messages`
在内部复用它的 evidence 渲染段。

**token_counter 适配器**:`trim_messages` 要 `Callable[[list[BaseMessage]], int]`,
本仓的 `trim.count_tokens` 收 `str`。适配器写在这里:

```python
def _lc_token_counter(messages: list[BaseMessage]) -> int:
    return sum(trim.count_tokens(m.content if isinstance(m.content, str)
                                 else str(m.content)) for m in messages)
```

`tool_calls` **不计入**(与 `trim.select_history` 一致的既有口径:结构性元数据不占预算)。

### 7.4 播种规则(§2.1 的落点)

```
state["messages"] 为空  → 从 MySQL 读全量 → to_lc_messages → 播种
state["messages"] 非空  → 不播种          ← 这条是防重复追加的唯一防线
```

播种发生在 `app/api/chat.py` 构造 `stream_input` 时(与现有的 `history` 通道并列),
判据是 `graph.aget_state(...)` 快照里的 `messages` 是否为空 —— 端点**已经**在用
`aget_state` 做待续检查(ch06),复用同一次调用,不额外加一次。

**精简版从 MySQL 的 `load_history` 派生,不从 `state["messages"]` 派生。**

这条是 **2026-09-22 订正的**。原先这里写「用快照里的 `messages` 派生精简版」,
**它做不到**:`layers.split` 是按 **`messages.id`(MySQL 主键)** 切的,
而快照里的 `messages` 是 LangChain 消息 —— **只有播种进来那一批带稳定 id**
(`to_lc_messages` 用 `str(row.id)`),**此后每一轮新增的消息拿到的是 `add_messages`
现赋的 uuid4**,与两个锚点**不可比**。要在 state 上分层,就得把新消息的 MySQL 主键
写回 state —— 而 `append_turn` 的返回值(T3 加)至今**没有写入者**,
那条路今天不存在。

所以:**端点用它在 `prepare_turn` 时已经读过的 `load_history` 结果做分层**
(那一份**本来就带 id**),`.sql` 与 `layers` 都吃 `schemas.Message`。
`messages` 通道的读边是**播种判据本身**(`if not seeded`),加上需求 5 要的
「完整历史随 State 贯穿、靠 checkpoint 留着」——
**它不是一个内容消费者,这一点如实记在这里**,免得日后有人以为有个读者而去找。

> 记账:这条错误让 T5 的实现者白写了一段「从快照派生」的说明性文字,
> 并在 T10 之前被发现。**发现方式是审查者标了一条 ⚠️**
> (「`append_turn` 的 id 返回值仍无写入者」)—— 即**一条被搁置的观察牵出了
> 一处设计矛盾**。

**重启自愈**:`InMemorySaver` 进程内,**服务重启后 checkpoint 全空** ⇒
下一次请求 `messages` 为空 ⇒ 自动从 MySQL 重新播种。这正是 §2.2「MySQL 是权威源」
在代码上的体现。

**已知缺口(如实记)**:重启后**本轮之前那一轮的 ReAct 往返**(assistant/tool_calls +
tool 行)虽然在 MySQL 里,但**不在 checkpoint 里**,而播种读的是 MySQL 全量 ⇒
**反而补得回来**。真正补不回的是 `trace` 等非消息通道,那些本来就是逐轮重置的,无影响。

### 7.5 `state.py` 的通道

```python
# ---- 完整历史(跨轮,add_messages)----
messages: Annotated[list[AnyMessage], add_messages]

# ---- 精简版(逐轮覆写,从 messages 派生)----
history: list[Message]        # ← 通道已存在,本章**改语义**,见下

# ---- 两个锚点(逐轮覆写,只读快照)----
summary_upto_msg_id: int
layer1_from_msg_id: int
```

#### `messages`:谁写、谁读

**写**:各节点只管**吐新消息**,由 reducer 按顺序并入 ——
`agent` 节点返回本轮 ReAct 的 `AIMessage(tool_calls)` / `ToolMessage` / 最终回复,
`log_turn` 再把它们写进 MySQL。这就是需求 5 的「框架自动按顺序并入」。

**读**:端点在组装本轮上下文时,从它**已经**要取的那份快照里读(§7.4),
派生出错精简版。**这条读边必须存在** —— 否则 `messages` 就是一个有写无读的死通道,
而本仓已经吃过这个形态的亏(`ChatState.usage` 与 `choices` 至今是有写无读,
done 帧的 usage 被写死 `None`)。**加通道时连带说清读边在哪,是本项目的硬规矩。**

#### `history`:语义变更(重要)

ch05–ch06 的 `history` 是「`select_history` 裁过的**单层**历史」;
本章起它变成「**精简版**」= 层 2 截短段 + 层 1 原文段,由端点**每轮重新派生并播种**。

**为什么精简版要进 state(订正前一版口径)**:需求 6 的 `history_ctx` 就是它,
节点必须看得见。而它**不会**变成「过期派生值」或「忘记重置的通道」,原因是
**覆写语义 + 每轮重新播种**:

- 每轮 `stream_input` 都带 `history` ⇒ 平凡的覆盖通道,没有累积;
- **唯独 `resume` 路径不带**(`stream_input` 是 `Command`)⇒ `history` 保留挂起
  那一轮的值。这**正是想要的**:续跑续的是同一轮,上下文该还原;
- 所以它**不进** `resolve_references` 的逐轮重置清单 —— 那份清单是给
  「没有播种者、只能靠重置」的通道用的。

#### 两个锚点:谁写

端点(预算校验那一段,§7.2 的降级动作)**在流开始前**算出并随 `stream_input` 播种。
同样每轮覆写。`resume` 路径读到的是 checkpoint 里上一轮的值 ——
它们**只供日志**,不参与任何判断,所以这个差异没有后果。

#### 重置清单**不**增加

`resolve_references` 里那份「逐轮通道归零」清单(`gate_passed` / `agent_steps` /
`reply` / `choices` / `citations` / `evidence` / `tool_calls_made` / `order_no` /
`order_data` / `refund_decision`)是 ch05–ch06 用**真实故障**换来的(漏一个就静默串轮)。

本章新增的四个通道**全部是跨轮或每轮覆写语义,故意一个都不进清单**。
**这一点必须在代码注释里写死** —— 否则下一个人会「顺手补全」,
而把 `messages` 加进重置清单等于**每轮清空完整历史**,是本章最严重的一种改坏方式
(且它在单轮测试里完全看不出来)。

---

### 7.6 可观测:两个上下文日志 + 摘要任务生命周期

`app/memory/journal.py`,写入 §6.3 新配的 `log/app.log`。

**`model_ctx`** —— 主力 Agent **每轮一行**(在 ReAct 循环**之前**、组装完上下文时):

> **订正(2026-09-22,终审 Important 5)**:这里原写「**每次调模型前**一行」,与实现
> 不符 —— `app/agent/nodes.py` 在 `agent_node` 里**进循环之前打一次**,循环里的
> 每一步模型调用**不再各打一行**。**实现是对的**(初始上下文才是要观测的那个:
> 它由端点按两个锚点切出来,而循环后续几轮的增量是当轮 ReAct 自己产生的),
> 所以订正的是这句 prose,不是代码。
> **已知局限,一并写死**:一个三步的轮次里,第 2、3 步的模型调用**没有对应的
> `model_ctx`**,而那一行里的 `tokens` 也只反映**第一次调用**时的窗口。
> 要按步观测得另加一行(本轮不做)——**不要**把这一行读成「本轮所有模型调用
> 的上下文用量」。

| 字段 | 内容 |
|---|---|
| `summary` | **梗概全文**(没有则空;不是条数) |
| `sliding` | **滑窗逐条消息**(角色 + 内容,截短后的实际形态) |
| `rounds` | 窗口条数 |
| `tokens` | **分段估算**:层1 / 层2 /**注入梗概** / 证据 / 总计,以及各自的预算 |
| `bounds` | `summary_upto_msg_id` / `layer1_from_msg_id` |

**`history_ctx`** —— 指代消解 / 意图识别共用的那份,**每轮必打**:

| 字段 | 内容 |
|---|---|
| `summary` | **摘要行**(一行一条:第 N 段 + 覆盖到哪条 id) |
| `sliding` | 滑窗 |
| `tokens` | 同上(它更小,预算是另一套) |

**「每轮必打」是硬要求,包括不进 Agent 的那几轮**(闲聊 / 投诉 / 兜底 / 退款子流程)。

**`resume`(续跑)不算新的一轮,因此不打。** 2026-09-22 裁定:续跑是**同一轮**的继续
(它的 `history` 根本没重新组装过 —— 端点那条路不读历史、不跑预算),打一行
描述「本轮发了什么上下文」的日志会**与事实不符**。这与「挂起的那一轮不落库」
是同一个理由:**续跑不是新轮次**。

**`history_ctx` 的 `sliding` 必须是分层后的精简版。** 2026-09-22 记账:
T7 先接上这条线时,它传的还是 `prepare_turn` 交给 `trim.select_history` 的输出 ——
而那个函数**只整轮丢弃、从不标注内容**,所以 `…` 与 `[工具结果] ` **不可能出现**,
**验收 4b 指定的那条线在结构上承载不了 4b**。分层(T10)接上之后才成立。
**T13 的验收脚本必须断在 `history_ctx` 这一行上,不能 grep 整个日志文件** ——
否则会命中 `model_ctx.sliding` 而**假通过**。
理由:那几轮恰恰是最容易「看起来正常、其实上下文是错的」的地方 ——
ch06 的 T4 就是这么丢的(节点写了 `confidence`、通道不存在、**单测全绿而生产恒为 `None`**),
而它当时**没有任何观测面**。

**日志里必须能看见分段的 token 数,而不只是总数** —— 这是 §10.5 验收 4b 的观测面:
「层 2 按截短后计数」如果只打一个总计,截短失效和截短生效**长得一模一样**。

**摘要任务生命周期** —— 五个节点各一行,都带 `conversation_id`:

| 事件 | 附带 |
|---|---|
| `summary trigger` | 层 2 的**截短后** token 数与预算(触发那条断言就靠它) |
| `summary start` | 起止 id(即当时的两个锚点) |
| `summary done` | 第 N 段、覆盖 `(a, b]`、**耗时** |
| `summary skip` | 原因(已有任务在跑 / 区间为空 / 边界已变) |
| `summary fail` | 异常摘要(**过 `redact_api_key`**)+ 边界**未推进** |

**不记 prompt / response 原文**:那既是密钥泄漏面,也是日志膨胀源(§8)。

## §8 错误语义(沿用既有边界,不新立规矩)

| 情形 | 处置 |
|---|---|
| 用户输入超 `max_user_input_tokens` | **400** + 固定文案(流开始前) |
| 历史预算装不下一轮 | **400** + 固定文案(流开始前) |
| 会话 id 不存在(`GET .../messages`) | **404** |
| 摘要任务的 LLM 调用失败 | **留日志,不重试,边界不动** —— 失败等于什么都没发生,下次再触发 |
| 摘要任务写库失败 | 同上;`(conversation_id, seq)` 唯一键冲突同样落到这里 |
| 检索/Milvus 故障 | **不变**(`ToolInfrastructureError` → 502,绝不降级成「没搜到」) |

**出站文本一律过 `app/sanitize.py::redact_api_key`**(既有硬规矩,本章不新增例外)。
**摘要任务的日志里不得出现模型返回的 raw 文本以外的上游响应体** ——
它只记边界 id 与耗时,不记 prompt/response 原文(那会同时是密钥泄漏面与日志膨胀源)。

---

## §9 配置项

### 9.1 新增

| 键 | 默认 | 边界 | 说明 |
|---|---|---|---|
| `model_context_window` | 18000 | `ge=1024` | 取代 `context_budget_tokens` |
| `max_output_tokens` | 2000 | `ge=1` | 取代 `reserved_output_tokens` |
| `max_user_input_tokens` | 2000 | `ge=1` | 本轮用户输入上限 |
| `tool_result_max_tokens` | 1200 | `ge=1` | 单个工具结果上限;也是单轮峰值的一项 |
| `rerank_top_k` | 5 | `ge=1` | **取代 `retrieval_top_k`**(同一把旋钮;**两个读点**:`app/tools/registry.py:40` 与 `evals/run_retrieval_eval.py:193` —— 后者不在 `testpaths` 里,漏改不会被单测发现) |
| `keep_rounds` | 20 | `ge=1` | 想留住的轮数 |
| `per_round_steady` | **600** | `ge=1` | 每轮稳态占用(**未实测值**,见 9.3) |
| `layer2_assistant_chars` | 50 | `ge=1` | 层 2 客服答复保留字数 |
| `layer2_tool_chars` | 60 | `ge=1` | 层 2 工具结果保留字数 |
| `summary_max_chars` | 200 | `ge=1` | 梗概长度上限(也是「注入梗概」开销的来源) |
| `evidence_block_tokens` | 250 | `ge=1` | 单块检索证据的估算开销 |
| `tool_def_tokens` | 800 | `ge=1` | 五个工具定义渲染后的估算开销 |

### 9.2 替换掉的两个

`context_budget_tokens`(8192)与 `reserved_output_tokens`(1024)**直接删除**,
不保留兼容名。读点只有 `app/services/chat.py` 一处 + `tests/test_trim.py` /
`tests/test_config.py`。本章**已经吃过**「改了没反应的配置项」的亏
(`reranker_use_fp16` 在 GPU 分支落地后无人读,已删) —— 留着两个含义重叠的旋钮
就是同一个坑的第二次。

`safety_margin_tokens`(512)**保留不动**,继续参与固定开销。

### 9.3 ⚠️ 未实测值(引用时必须带上这句)

- **`per_round_steady = 600`** 是**估算,不是实测**。它决定了「想留住的轮数」
  那一支会不会赢。用户 2026-09-22 拍板把初版估值 200 上调(200 会让该支在演示配置下
  胜出,滑窗只有 4000,层 1 会**每轮都降级**)。
  **校准方法**:跑一轮真实对话,从 `model_ctx` 日志里读每轮实际占用,取中位数回填。
  与 `dedupe_threshold = 0.95` 同族 —— 都是「写下来但还没验证过」的数,**不要当成已验证的**。
- **`evidence_block_tokens` / `tool_def_tokens`** 同理,是估算。启动自检不依赖它们的
  精确性(它们只影响固定开销的大小,进而影响预算),但预算数字会随它们平移。

### 9.4 演示配置与它算出来的数

验收 2 要求按这组显式配置再聊一遍:

```
MODEL_CONTEXT_WINDOW=18000  MAX_OUTPUT_TOKENS=2000  MAX_USER_INPUT_TOKENS=2000
MAX_AGENT_STEPS=3  TOOL_RESULT_MAX_TOKENS=1200  RERANK_TOP_K=5
```

**用户 2026-09-22 拍板:这三个数是「软」的 —— 公式算出多少就是多少,
验收 2 断言的是实际值,不是某个预先给定的数。**

按 §9.1 的默认值代入:

```
固定开销 = 系统提示(705) + 工具定义(800) + 证据(5×250=1250)
         + 梗概(200) + 输出预留(2000) + 安全余量(512) = 5467
单轮峰值 = 3 × 1200 + 2000 = 5600
历史预算 = min(20×600=12000, 18000−5467−5600=6933) = 6933     ← 窗口那一支赢
层1 = 4853      层2 = 2080
```

**这些数会随 §9.3 那几个估算值一起平移,验收脚本必须自己算、不能硬编码。**

> **已按实际值回填(2026-09-22,终审 Important/Minor 14)。** 原版那四个数
> (`5462 / 6938 / 4856 / 2082`)是**动笔时的估算**,而本节自己写了「实现完成后
> 以实际日志为准回填」—— 那一步当时**没做**。回填后的数由**生产代码现算**
> (`budget.derive`,系统提示取 `render_system_prompt(brand_name)` 的真实
> `count_tokens` = 705,不是「~700」),并与验收 2 那次实跑的日志**逐字对上**
> (`layer1=4853 / layer2=2080 / history=6933 / fixed=5467 / peak=5600`,
> 见 `scripts/acceptance_ch07.sh` 的输出与 `.superpowers/.../task-13-report.md`)。
> **为什么要专门回填**:这些数会被人拿去核对日志,而**差 3~5 的旧数字长得
> 完全像真的** —— 本仓已经为「看起来验过、其实描述的是另一套配置的数」付过账
> (ch07 终审还抓到同一族的 0.389)。默认配置一侧同理:`layer1=3173 /
> layer2=1360 / history=4533 / peak=8000`(验收 3 跑的就是它)。

---

## §10 测试与验收口径

### 10.1 流程口径(用户 2026-09-22 拍板:方案 A)

**流程照走**(逐任务实现 + 规格审查 + 质量审查 + 修复循环),**单测只写关键路径** ——
砍掉边界分支与「为了断言而断言」的用例;评估集与端到端验收脚本照常。

### 10.2 单测(关键路径)

| 目标 | 为什么它在关键路径上 |
|---|---|
| `budget.derive` 的级联数值 + 启动自检 | 本章全部数值行为的地基 |
| `layers` 切三段的**不重不漏** | 边界 off-by-one 会静默丢消息或重复 |
| 两个锚点的**降级/推进**(纯 id 移动,数据行不变) | 本章的结构核心 |
| **播种幂等**:`messages` 非空时不再追加 | §2.1,Context7 查出来的坑,不钉就会每轮翻倍 |
| 层 2 截短**保住 tool 配对**(`tool_calls` 原样) | 切开 ⇒ 上游 400,且只在长历史下复现 |
| **层 2 的计数确实按截短后的版本**(截短前后计数**不同**,且截短后显著更小) | §3.2 的那条洞:数错版本 ⇒ 截短对级联零影响,而**所有输出看起来都正常** |
| 摘要任务的**并发与失败**:同会话只跑一个;失败/撞唯一键时**边界不动** | 边界动错了就是永久静默丢历史 |
| `max_user_input_tokens` 超限 → 400(普通 JSON,非 SSE) | 流开始后改不了状态码 |
| 两个只读端点的形状 + 404 | 前端契约 |

**不写**:层 2 每种角色组合的排列、日志文案逐字、`journal` 的输出格式(见 10.3)。
理由是用户点名的「少测试」,而**这些恰好是本章最不容易出真缺陷的地方**。

### 10.3 明确不写测试的三处(如实记账,不是忘了)

1. **`journal` 的日志格式** —— 它是给人看的,断言文案等于把 `dev-notes` 抄成测试。
2. **日志落盘本身**(`log/app.log` 真的出现、中文不炸)—— 由验收脚本以
   「文件存在 + grep 到中文」间接覆盖。**这是 Windows cp936 的高危点**,
   单测里 `Settings(_env_file=None)` 那套替身盖不住文件句柄的真实编码行为。
3. **前端** —— 按既有惯例走 Vibe Coding,验收 5 用浏览器人工验。

### 10.4 评估集(非可单测产出)

**新增 `evals/summary_cases.jsonl`** —— 标注样例,验 §4.1 的四样提炼物与三条硬约束:

- 正例:含订单号 / 商品 / 未解决问题的多轮对话 → 梗概**必须**含这三样;
- 负例:纯寒暄几轮 → 梗概**应当为空或极短**(不能硬凑);
- **幻觉探针**:对话里**没有**订单号的,梗概里**不得出现**任何 `\d{4,32}` 形态的数字。
  （写法参照 `tests/test_refund_categories.py` 那类闭式口径;`\d{4,32}` 这条
  要与 T1 那条被实现者用变异运行抓出来的**同义反复断言**区分开 —— 探针的数字形态
  必须是**真能出现**的形态,而不是被正则长度排除在外的形态。)

### 10.5 端到端验收:新增 `scripts/acceptance_ch07.sh`

对需求里的五条验收:

| # | 断言 | 确定性 |
|---|---|---|
| 1 | 连聊二十轮(token 不爆、不崩) | **确定性**:看 `done` 帧是否齐全、无 error 帧 |
| 2 | 演示配置下的**完整级联** | 看日志里的 `layer1 降级 X→Y` 与 `summary trigger/done 第N段` |
| 3 | **默认窗口下纯聊天二十轮不触发任何降级与摘要** | **确定性**,且它是「装得下就不压」的反向断言 |
| 4 | 摘要**不阻塞**该轮回复 + `grep model_ctx / history_ctx` 看得见 | 确定性:回复先于 `summary done` |
| 4b | `history_ctx` 里**看得见截短后的形态**:客服答复带 `…`、工具结果是一行 `[工具结果] …` | **确定性**(对日志文本做形态匹配,不做逐字断言)—— 没有这条,截短是否真的生效**没有任何观测面** |
| 5 | 侧栏多会话、切回旧会话、接着聊 | **浏览器人工** |

**验收 3 的口径在本章被收窄了**(用户 2026-09-22 拍板):原文是「聊二十轮」,
但工具结果落表之后,一轮带工具调用的对话约 2400 token,二十轮 ≈ 48000,
**超过任何合理的默认窗口** ⇒ 字面口径必然不成立。收窄为**纯聊天二十轮**
(一轮约 100 token,二十轮 2000,占层 1 预算约四成),带工具的轮次归验收 1 的
「不爆不崩」。**这不是放宽,是把一条原本恒假的断言改成可判定的。**

**验收 2 的落点**(需求原话「这时再问『最开始那个订单后来怎么说』,要能靠梗概里
存下的订单号和诉求答对」)是本脚本**唯一依赖模型判定**的一条 ⇒ 用 ch06 已有的
`warn()` 档:**显式计数并打印,不当作通过**。理由 ch06 已经写死过:
「必须显式计数并打印,否则它就退化成一条悄悄跳过的检查」。

窗口参数按 §9.4 显式覆盖,断言只读日志里**实际算出来的**数。

### 10.6 老回归网

`scripts/acceptance.sh`(ch01–ch06)**当前是红的**,8 条因 ch06 改路由而失败
(题面走了退款子流程并挂起)、1 条 KB 漂移。**本章不改它** ——
它是 ch06 留下的独立欠账,与上下文管理无关,**不在本章范围内**。
如实记入 §11,不假装绿。

---

## §11 风险与已知取舍

1. **`per_round_steady` 是估算**(§9.3)。若真实每轮占用远大于 600,
   「想留住的轮数」那一支会赢,层 1 会比预期小。**这是本章最可能返工的一个数**,
   校准方法已写明。
2. **`InMemorySaver` 无淘汰、无 TTL**(§2.2)。见过的每个 thread 常驻进程,
   而本章起 `state.messages` 装**完整历史** ⇒ 内存占用随会话数与轮数单调涨。
   演示规模可接受;**正式版要换成有界或持久化的 checkpointer**,与 ch05 记的是同一笔账。
3. **工具结果落表之后 `messages` 表增速显著加快**(单条上限 1200 token)。
   本章不加清理策略。**这会放大 §10.6 那条 KB 漂移同类的问题** ——
   记账,不在本章修。
4. **老回归网 8 条口径待重切**(ch06 欠账,**本章不做**,见 §10.6)。
5. **`usage` 仍是死值**(§5.3)。本章不接,因为接它必须连每轮重置一起做。
6. **摘要质量只能靠标注样例 + 人工抽查**,没有自动化真值。这是 LLM 产出的固有代价。

---

## §12 实现订正

_(实现过程中与本章设计的偏离、原因、以及可回退性,在此累积。收尾时汇总。)_

### 12.1 T4 分层:计划里的示例代码有三处缺陷(2026-09-22)

**本节订正的是「计划文本」,不是本 spec** —— §3.1 的区间语义(层 1 **含**起点、
层 2 **不含**两端)与 §3.2 的计数口径(层 2 按截短后)**本来就是对的**;
错的是计划里照着写的那份代码与此前给的一条断言。三处都由实现者**用变异运行**
发现并验证。

**① 区间语义写反 ⇒ 新会话上下文静默翻倍。**
计划给的 `_after(history, lo=, hi=)` 把 `hi == 0` 当作「到末尾」、且区间**含**
`layer1_from` 本身。后果:**两个锚点都是 0 时(每一个新会话),同一条消息
同时落在层 2 与层 1 里** —— 而 T6 的组装是 `layer2 + layer1`,
于是**上下文整段重复**,且没有任何东西报错。
订正为两个函数:`_tail`(含起点,`0` ⇒ 全部)与 `_middle`(两端都不含,
`0` ⇒ 空),即 `[.., summary_upto] / (summary_upto, layer1_from) / [layer1_from, ..]`。

**② `degrade` 可能死循环。**
计划把边界设成「被丢弃那一轮的**最后一个** id」。当某一轮以 `assistant` 开头时
(截断或异常历史都可能造出来),那个 id 会自成一「轮」,`while True` 永不前进 ——
实现者实测 `timeout 30` → `exit=124`。
订正为:边界推到**下一轮的起点**,并加了一条进度守卫。

**③ 守护本章最要命性质的那条断言,本身不可能满足。**
计划的核心测试第三条写的是
`raw.layer2_tokens < sum(count_tokens(m.content) for m in raw.layer2)`,
而 `raw.layer2` **已经是截短过的** —— 这是 `x < x`,**任何实现都过不了**。
讽刺的是它要守护的正是 §3.2 那条「层 2 按截短后计数」——**全章最容易静默失效的一处**。
订正为对着**未截短的原文**比;在「按原文计数」的变异下两者是 `60 == 60` ⇒ 变红。

**可回退性**:三条都是纯粹的缺陷订正,没有设计取舍;回退任意一条都会
重新引入静默错误(①上下文翻倍 / ②死循环 / ③断言不可满足),**不建议回退**。

### 12.2 T8 摘要:一处反向依赖边 + 一处被实现者补上的歧义(2026-09-22)

**① 反向依赖边(记账,不重构):`app/memory/summarize.py` → `app/services/history.py`。**

本项目的依赖方向是 `api → services → {tools, db, memory, prompts, llm}`,
而 `summarize.py` 为了调 `append_summary_and_advance` 反着 import 了 `services`。
**裁定:接受并记账** —— 与 ch03 的 `retrieval → tools.errors` 同一处理
(那条也记在 ch03 spec §12)。理由:今天**无环**
(`services/history.py` 只 import `db.models` 与 `schemas`,不回头引 `memory`),
且原子落库那一步**必须与锚点推进同处一地**,把那对操作拆到两个模块只会
重新引入「只成一半」的风险 —— 而那正是本任务要消灭的东西。
**代价(若判断错)**:日后若 `services/history.py` 反过来 import `memory`,
就会成环;届时把 `append_summary_and_advance` 换成一个传入的回调即可,
改动局限在 `summarize_range` 的签名。

**① 的补记 —— 同一族的第二条反向边:`app/memory/tasks.py` → {`app/services/history.py`, `app.db.models`}(终审补记)。**

本节原先只记了 `summarize.py` 那一条,**T9 的后台执行体又加了两条**,而它们
**没进 §12** —— `tasks.py` 既 `from app.services.history import load_history`
(与 ① 同款:那段读历史的 I/O 只有一份实现),又 `from app.db.models import
Conversation`(重读两个锚点用)。两处都在 `app/memory/tasks.py` 的模块 docstring
里记着,但**记账写在代码里不等于记在 spec 里**。
**裁定与 ① 相同:接受并记账,不重构。**
**图仍无环**:`app/services/history.py` 只 import `db.models` 与 `schemas`,
不回头引 `memory`;`db.models` 更不引任何业务模块。
**可回退性**:与 ① 同 —— 真要解环就把 `load_history` 与 `Conversation` 的读取
收成注入进来的可调用对象,改动局限在 `_reload_state` 的签名。
**连带**(终审 Important 1):`CLAUDE.md` 那句「依赖方向严格单向」是**后来人被告知
要去信的那一句**,它必须把这两条边一并说出来 —— 已补。

> **计数说明**:这段是 **① 的补记,不是第 ③ 条订正** —— §12.2 仍是 2 条条目
> (`①` / `②`),§12.11 两张汇总表的数字**因此不变**。写在这里是为了让
> 「依赖方向」这一类**一处就能看全**,不是为了让它们变得更严重。

**② 空输出不写库、不推锚点(实现者补的歧义)。**
模型返回空串或纯空白时,`summarize_range` **不落空梗概、不推进锚点**,
返回 `None`。这条**必须在 spec 里写死**,理由是它的反面极其隐蔽:
写一条空梗概**再**推进锚点,等于**把那一段历史静默删除** ——
层 2 不再读它(区间已被覆盖),而摘要表里那一段是空的,没有任何东西报错。

**连带**:`None` 因此在 T9 的 `summary skip` 日志里**有两个含义**
(「区间为空」与「模型输出为空」)。T9 必须把两者**分开记**,
否则运维看到一串 skip 会以为是「没东西可压」,而真实原因可能是**模型一直返回空**。

### 12.3 T3 数据层:两条「建库路径形状不同」的静默洞(2026-09-22)

**① 唯一键只在 DDL 声明、ORM 一个字段都没有。** 后果是一条**静默链**:全新库按
「`init_db.py`(create_all)→ `db/ch07.sql`」建库时,`create_all` 先按 ORM 建出
**没有 `uk_conv_seq`** 的表,接着 `CREATE TABLE IF NOT EXISTS` **静默跳过** ——
摘要并发保护的第二道**永远不存在,且不报错**。与 `RefundRequest.status` 那处同族。
**修法**:ORM 补 `__table_args__` 的唯一约束;`db/ch07.sql` 去掉 `IF NOT EXISTS`
改裸语句(该文件本来就不幂等 —— 后面的 `ALTER` 再跑也会报错,留着它只让**这一句**跳过)。
**并补一条纯单测读 ORM 元数据**断言 `uk_conv_seq` 在 `__table__.constraints` 里 ——
原来那条「撞唯一键 ⇒ IntegrityError」抓不到这个洞(它跑在**已由 DDL 建好、带键**的库上)。

**② 去掉 `IF NOT EXISTS` 引出的回归:`ALTER` 必须在 `CREATE` 之前(文件内顺序)。**
mysql 客户端**遇到第一个错误就中止整个脚本**。原来的顺序(先 CREATE)在
「梗概表已存在、两个锚点列还没加」的库上会 1050 中止 ⇒ `ALTER` 永不执行 ⇒
库缺两列,而报错读起来像「表已存在 ⇒ 已经装好了」。
**两条修法都不可回退**(回退即重新引入静默缺失),**顺序那条尤其**:它只在
「老形状的库」上才暴露,而那正是升级路径本身。

**遗留(纯外观,未修)**:`create_all` 把普通索引命名为
`ix_conversation_summaries_conversation_id`,DDL 叫 `idx_conv` —— 同列同语义、**名字不同**。

### 12.4 T5 / T6:播种的读边 + 一处与贯穿性约束打架的计划文字(2026-09-22)

**① 用户消息从不进 `messages` ⇒ §7.4 承诺的那条读边兑现不了。**

两轮之后 `state["messages"]` 里**只有 assistant 回复**,没有用户消息 ——
唯一写用户消息的时刻是「快照为空时」那一次播种。于是 `messages` 与 MySQL **从第二轮起分叉**,
**§7.4 的读边(用快照派生精简版)兑现不了**,而每轮回复正常、落库正常、现有断言全绿。
**修法**:`resolve_references` 返回 `"messages": [HumanMessage(user_input)]`。
**只放那里**的依据是需求 5 的原话「**各节点只管吐新消息**」;不放端点的依据是
**`resume` 不重跑 `resolve_references`**,所以续跑不会多出一条。

**② §7.4 的派生来源订正(连带)。** 精简版**从 MySQL 的 `load_history` 派生**(端点做预算校验时已读过,
且本来就带 `id`),不从快照的 `messages` 派生 —— 后者只有播种进来那批带稳定 id,
此后每轮新增的拿到的是 `add_messages` 现赋的 uuid4,**与两个锚点不可比**。
(订正过程见 §7.4 正文:一条被搁置的观察牵出了这处矛盾。)

**可回退性**:两条都是缺陷订正;回退①那条读边**没有任何消费者会发现**
(本章内),但 T10 的分层会拿到一份缺掉每轮提问的精简版 —— 静默、且看起来正常。

**③ T6:一处与**贯穿性约束**打架的计划文字(`select_layer1` 为什么在 `prompts.py`)。**

计划 T6/T10 的文字说「层 1 的选择(`trim_messages`)在 T10 **由端点**接入」,
而 Global Constraints 写死「**`prompts.py` 是 LangChain 的唯一面**」——
**照字面写就违反那条约束**。
**裁定(以约束为准)**:新增 `prompts.select_layer1(history, *, max_tokens)`,
把 `trim_messages` 包在里面,端点调它。T10 已按此落地
(`trim_messages` 全仓只有一个调用点)。
**可回退性**:不可回退 —— 回退等于让 LangChain 从端点直接进来,`memory/` 与端点的
「不依赖 LangChain」约定同时破,**而这正是本章 §7.1 花力气保住的东西**。
(同任务另有一条**组合契约未被钉住**:`_render_tail` 里「梗概在前、证据在后」的顺序
与中间的 `"\n\n"` 当时没有任何用例覆盖 —— 两段对调的变异能通过全部六条测试。
已带进 T10 覆盖。)

### 12.5 T7/T10:`history_ctx` 那条线承载不了验收 4b(2026-09-22)

**① `history_ctx` 那条线承载不了验收 4b。**

T7 接上 `history_ctx` 时,传进去的是 `trim.select_history` 的输出 ——
而那个函数**只整轮丢弃、从不标注内容**,所以 `…` 与 `[工具结果] ` **不可能出现**。
**验收 4b 指定的那条线在结构上承载不了 4b**;而 T7 报告里那句「真正观测的是
`model_ctx` 那一行」**与 §10.5 相反**。
**裁定**:**明令禁止把 4b 挪到 `model_ctx` 上** —— 那是改验收标准去迎合实现。
T10 把 `history=` 换成分层后的 `trimmed` 之后 4b 才成立,并由
`tests/test_api_chat.py::test_history_ctx_carries_the_truncated_layer2_forms` 钉住。
**别名警告**:`model_ctx.sliding` 也带同一批截短形态,**T13 的验收脚本必须断在
`history_ctx` 这一行上**,整文件 grep 会假通过。

**② `model_ctx` 的锚点默认值会让日志说谎**(「通往错误的路」,不是当下的错)。:`0` 是个**有含义的值**,读不出「未知」;
  T10 的调用点当时根本没有锚点来源 ⇒ 它会打 `bounds {0,0}` 而切分用的是真锚点。
**修法**:`Layers` 带锚点、`model_ctx` 从 `layers` 读、**去掉默认值**。
**③ `setup_logging()` 必须在 pytest 下跳过**:不跳的话整套测试会往仓库根写一个
  108KB 的 `log/app.log`,而 **4b 就是 `grep log/app.log`** ⇒ 单测留下的**陈旧行让那条断言恒真**。

### 12.6 T9 后台摘要:两个「同一事实两处算」的口子(2026-09-22)

**① `summary done` 里没有「第 N 段」,而理由是「不想为日志好看把单测拖去打 DB」。**
**取舍对,结论要反过来**:段号是 `append_summary_and_advance` **自己算的**,
丢掉再让调用方查一遍 = **同一个事实算两次**,而那两次**可以不一致**(并发另一段在中间提交)。
⇒ 让那个函数把 `seq` **返回出来**(`services/history.py` 的签名 `-> None` 改 `-> int`,
`summarize_range` 返回 `(seq, text)` 元组)。**这是单一来源问题,不是日志美观问题。**
**禁止的替代写法**:「只断日志里有 `summary done`」(形态 ①);要断**非 1 的段号**。

**② `summary trigger` 在 T9 那个接缝上发不出来**(它拿不到 `layer2_tokens`/`layer2_budget`),
于是新增 `log_trigger(...)` 由**触发方**打(T10 的端点调),否则**验收 2 无物可命中**。

**可回退性**:两条都不建议回退(回退①让日志与落库可漂移;回退②让验收 2 落空)。

### 12.7 T10 端点接线:spec §10.5 的一处「不可达验收」(2026-09-22)

**① 这是本章唯一一处「验收标准本身不可达」的洞,而它在任务之间的缝上。**

§10.5 的验收 2 要求日志里有 `layer1 降级 X→Y`。而 T10 审查者全仓 grep 的结论是:
**那个字符串只出现在 spec 那一行里** —— `layers.degrade` 不打、`advance_anchors` 不打、
端点也不打。`degrade` 返回新值、`advance_anchors` 收新值、端点写库,**每个实现者都做了
自己那份,而没有一个人有义务打那行日志,因为计划从没说过谁打**。
⇒ T13 当时只能去 grep 一个**没有生产者的字符串**(红,但不是代码错)。
**修法**:在端点那个接缝补一行(只有它同时拿着旧值与新值),格式与 `journal`/`tasks` 同款,
并带上 `layer1_tokens`/`layer1_budget` —— **只报「挪了」而不报「为什么挪」,读的人还得自己推**。

**同一任务另外四条**:
**② `prepare_turn` 变纯校验、`trim.select_history` 退役**。它不是「清理」:
  它与三层方案**在语义上不可叠加** —— 叠在最前面会把层 2 的整轮消息先丢掉,
  `layer2_tokens` 随之少算 ⇒ `should_summarize` **永不触发**,而级联第二环
  **在日志里平静地缺席**。**留着一个只能踩的陷阱比删掉更危险。**
  测试**不是删是搬**:「按 `user` 边界切轮、不切开 tool 配对」那条不变量从
  `select_history` 的用例**搬到 `to_rounds`** —— 那才是它现在真正的作用点。
**③ `ContextOverflowError` 拆成两个子类**:它原本承载两种不同故障(输入超限 / 预算装不下一轮),
  一个模板说不对两件事。
**④ 一条测试被「连坐」删掉**:`test_context_overflow_error_carries_numbers` 既不测
  `select_history`、也不测任何分层的东西,却随它退役被删 ⇒
  `ContextOverflowError.used`/`.budget` 与其文本里的两个数字**无处断言**,
  且 `ContextBudgetUnavailable` 只作为父类被验到(**子类退化能让整套测试保持绿**)。已恢复并重定向。
**⑤ 三处「缺 message 会在 tiktoken 里炸成 500」的反事实注释**订正:重构后那条路不可能再抛,
  422 validator 仍对,**错的只是它陈述的机制** —— 而它把下一个调试的人**指到错的地方**。

**已知残余(够 Minor,但形状很熟,如实记)**:`layers.degrade` 的**第二个出口**
(只剩一轮仍超预算)会让日志报出**高于自己预算的层 1 数**,而 `select_layer1` 接着
把那一整轮丢掉 ⇒ **层 1 实际发出去的是空的,而日志报告了它的内容**。
需要「单轮就超掉整个层 1 预算」才可达;**但它正是本章一直在追的「日志里的切分 ≠ 发出去的切分」。**

### 12.8 T11 两个只读端点:一条规则与它背后的原则(2026-09-22)

**① 我 spec 里写的是一条具体规则(「过滤 `role='tool'` 的行」),而不是它背后的原则
(「回载的是**用户看见过的对话**」)。规则写完就以为完事了,原则才会自己长出下一半。**
于是 T11 第 1 轮滤掉工具行之后,审查者**越界观察**发现**空助手气泡还在** ——
工具调用那条 assistant 的 `content` 是空串,不过滤的话**每个「模型调了工具」的轮次
都会在侧栏画出一个空气泡**。§5.2 已按原则补齐两类过滤。

**② 我给实现者的「写作要求」本身是一条假命题。** 我断言 `content != ''` 与
「assistant 且不带 `tool_calls`」**等价**,并要求写进 docstring。审查者证明它在**一个方向上不成立**:
`_lc_to_records` 把 `content` 与 `tool_calls` **一起**写,而 `_stream_round` 边累积 chunk
边把文字发 token 帧 ⇒ **一条既带 `tool_calls`、又带开场白的 assistant 行结构上可达**;
对那种行 `content != ''` **返回**它、「不带 tool_calls」**丢掉**它 —— **丢掉是错的**,
因为那段开场白**以 token 帧流出去过,用户见过**。
⇒ 它是**更窄、更保守**的条件,不是等价物。§5.2 已订正,实现侧 docstring 写的是**这条更窄的理由**。
**这是本章的新形态:错的不是判断,是「教别人去写的话」。**

**③ 「替身替被测对象完成了语义」(第 7 种假绿形态)。** `FakeSession` 自己 `sorted(...)`
⇒ **端点删掉 `order_by` 照样绿**;user 过滤退到 Python、`summarized: true` 真实库无实例。
**判据:被测对象是否把这件事委托给了替身。**
**修法**:新增 `tests/test_api_conversations_db.py`,在**真实库**上覆盖三件替身验不了的事
(SQL 的排序、SQL 的 user 过滤、`summarized` 的真值);**user 过滤回到 SQL,扩替身而不给实现加兜底**。

### 12.9 T12 前端会话侧栏:一处竞态与它的修法(2026-09-22)

**① 代际令牌收进唯一写口。** 这一条**不属于与设计的偏离**(§1 的目标 7 本来就要求多会话前端),但记一条**修法**:
「切换会话」与「正在进行的 SSE 流」之间的竞态,表现为
`renderHistory` 的 `replaceChildren()` **销毁正在流式的气泡**,而 `setSessionId` 悄悄移走当前会话。
**数据层没错,是视图层失效。** 修法:**把代际令牌的递增收进 `setSessionId`
(当前会话的唯一写口)**,一次关掉三种顺序(切侧栏 → 新对话、回载在路上 → 新对话、回载在路上 → 再切)。
—— **推论:不变量要放在唯一写口上,不要靠调用点自觉。**

### 12.10 T13 收尾:验收与评估跑出来的两件事(2026-09-22)

**① 验收脚本必须自己起服务、自己截断 `log/app.log`。** 两个理由都不是讲究:
验收 2 要的是一组**显式覆盖的窗口参数**(§9.4),而 `.env` 里是默认值 ——
让操作者手工带 env 起服务等于把「这一轮验的是哪套配置」交给记忆;而**不截断日志**
会让断言命中上一次运行(或单测)留下的旧行,**于是无论被测代码对不对它们都是真的**。
**连带**:验收 3(反向断言)必须跑在**默认配置**的服务上(层 1 预算 3173 < 演示配置的 4853,
那才是更严的一侧);在演示配置下跑等于放宽口径。

**②(不是本章的订正 —— 记账)**环境里有一处与本章无关的冷启动欠账,被验收抓了出来,**不在本章修**:
`app/main.py` 的预热只覆盖**嵌入模型**(BGE-M3),而 `bge-reranker-v2-m3` 是首次
`rerank()` 才加载的 —— **实测 21.0 秒**(独立进程直测),真机上第一次 `query_faq`
实测 **24 秒**。这是 ch03 记过的那条冷启动陷阱,只是重排器那一半没人补(ch04 增补的欠账)。

**⚠️ 但它不是「超时 502」—— 这条一开头写错过,按实测订正。**
`execute_tool` 的 `asyncio.wait_for(..., timeout=10)` **拦不住**它:那次重排是
**同步调用、阻塞事件循环**的,而 `wait_for` 的定时器回调在循环被占住时**根本没机会跑**。
**实测(python 3.13.14,本机)**:同样 3 秒的阻塞,`await wait_for(to_thread(blocking), 1)`
→ `TimeoutError(1.0s)`;而 `await wait_for(直接阻塞循环的协程, 1)` → **不超时(3.0s)**。
⇒ 用户看到的是**一次 24 秒的静默停顿**,不是 502 —— **代价换了个样子,仍然要付**,
而且 `tool_timeout_seconds` 这个旋钮在**这条路径上不生效**(它只对真正让出控制权的
异步调用生效)。**验收脚本因此先打一发检索类问题把重排器逼出来**,并把「检索真的通了」
(命中 ≥ 1 或 `query_faq` 返回 `ok=true`)当作**环境预检**:三次都拿不到就退出,
免得二十轮的断言红在一个与上下文无关的地方。

**③(不是订正 —— 一条读数)**摘要评估集跑出来的一件必须记的事:负例(纯寒暄)上模型**不返回空串**,
而是返回约 40 字的**元叙述**(「本次对话未涉及任何商品、款式、订单号、手机号等标识,
无明确诉求,无未解决问题」)。它**没有编事实**,但那句正是 prompt 点名的
「**对话状态一律不留**」。⇒ 两条推论:① `SUMMARY_SYSTEM_PROMPT` 的这条约束**遵从度不满**;
② T8 的「空输出退路」(`SKIP_BLANK_MODEL_OUTPUT`)在真实模型上很难触发 ——
纯寒暄它也会吐 40 字。**都在评估集里有据可查,不修 prompt**(本章代码已审查冻结;
真要收紧,加一句「没有任何可提炼内容时,输出空」即可,一句话可回退)。

### 12.11 汇总(收尾)

> **本节的每一个数字都是从 §12.1–§12.10 的条目里逐条数出来的**(带编号的那些:
> `①`/`②`/… 与 bullet),不是估的 —— 2026-09-22 起先写错过一版(正文说 12、清单加出 25、
> 表加出 19),**一份写得不准的汇总比不写更坏:它会让后来的人以为那个数是核过的。**

**本章共 25 处实现订正**,逐条写在 §12.1–§12.10 里:

| 任务 | 节 | 条数 | 是哪几条 |
|---|---|---|---|
| T3 数据层 | §12.3 | 2 | ① ORM 侧漏了唯一键(⇒ 静默永不建)② `ALTER` 被 `CREATE` 的 1050 挡掉 |
| T4 分层与截短 | §12.1 | 3 | ① 区间语义写反 ② `degrade` 死循环 ③ 那条不可能满足的断言 |
| T5 播种 | §12.4 | 2 | ① 用户消息从不进 `messages` ② 精简版的派生来源订正 |
| T6 定序组装 | §12.4 | 1 | ③ `select_layer1` 收进 `prompts.py`(计划文字与硬约束打架) |
| T7 日志与可观测 | §12.5 | 3 | ① `history_ctx` 承载不了 4b ② 锚点默认值让日志说谎 ③ `setup_logging` 污染 4b |
| T8 摘要 | §12.2 | 2 | ① 反向依赖边(接受并记账)② 空输出不落库不推锚点 |
| T9 后台摘要执行体 | §12.6 | 2 | ① `seq` 必须从落库那一步返回 ② `log_trigger` 必须有人调 |
| T10 端点接线 | §12.7 | 5 | ① `layer1 降级` 没有生产者 ② `prepare_turn` 纯校验 / 退役 `select_history` ③ 异常子类 ④ 连坐删掉的用例 ⑤ 三处反事实注释 |
| T11 两个只读端点 | §12.8 | 3 | ① 空助手气泡(规则 vs 原则)② `content != ''` 不是等价物 ③ 替身替被测对象干活 |
| T12 前端会话侧栏 | §12.9 | 1 | ① 代际令牌收进唯一写口 |
| T13 收尾 | §12.10 | 1 | ① 验收脚本必须自己起服务 + 自己截断日志 |
| **合计** | | **25** | |

**同一批 25 处的另一种切法**(按性质;与上表**互不相加**,它就是对上表那 25 条的重新归类):

| 性质 | 条数 | 是哪些 |
|---|---|---|
| **静默失效**(不报错、输出看起来正常) | 7 | T3① T3② T4① T5① T5② T8② T10② |
| **不可达 / 无生产者**(验收或观测面结构上够不着) | 3 | T7① T9① T10① |
| **假绿断言**(实现改错了它照样绿) | 7 | T4③ T7② T7③ T9② T10④ T11③ T13① |
| **措辞 / 机制说错**(代码对、话错,把下一个人指到错的地方) | 2 | T10⑤ T11② |
| **显式故障**(死循环 / 视图失效) | 2 | T4② T12① |
| **依赖方向 / 结构约定** | 1 | T6① |
| **用户可见缺陷 / 错误模型不精确** | 2 | T10③ T11① |
| **接受并记账的偏离** | 1 | T8① |
| **合计** | **25** | |

**三件不属于「订正」的事,单列(不要与上面的 25 相加)**:

- **已知残余 3 条**(纯外观,或需要极端配置才可达):`create_all` 与 DDL 的索引名不一致 /
  `degrade` 第二出口会让日志报出高于自己预算的层 1 数 / 摘要 prompt 对「不写对话状态」的遵从不严;
- **环境欠账 1 条**(§12.10 ②):重排器冷加载 21.0 秒没人预热 —— **ch04 增补的欠账,不在本章修**;
- **评估读数 1 条**(§12.10 ③):负例上模型不返空、而吐 40 字元叙述。

**一条贯穿全章的结论**:上表 **7 条假绿断言里,没有一条是实现者写的** ——
它们全部出自主控的 spec / 计划文本 / 简报,由实现者用**变异运行**或审查者用**独立复算**抓出来。
**「写断言前先问『实现改错了,这条的输出会不会不同』」这条判据写在 CLAUDE.md 里,
而本章连跳了七次** —— 就是上表「假绿断言」那一行的七条,**逐条对上**:
**T4③ / T7② / T7③ / T9② / T10④ / T11③ / T13①**。
(旧版这里写成 `T4/T5/T6/T7/T9/T11/T13 各一次`,**与它自己上面那张表打架**:
列了表里没有的 T5/T6、漏了表里有的 T10、还把 T7 的两条并成一条 ——
条数 7 碰巧对,名单全错。终审 Minor。**一份数对了、名单错了的汇总比不写更坏**:
读的人会以为那个名单也是逐条核过的。)
变异运行是这条判据的**机械化形式**,
本章的全部证据都指向:**它必须是默认动作,不是补做动作。**

**可回退性(统一口径)**:上表 25 条订正**没有一条是设计取舍**,全部是缺陷订正;
**回退任意一条都会重新引入一条静默错误**(或一条不可达的验收),故不建议回退。
