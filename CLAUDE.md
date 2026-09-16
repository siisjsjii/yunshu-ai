# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目

电商智能客服系统。

- **ch01(纯对话)** 已合并到 `main`:SSE 流式对话 + 结构化抽取。
- **ch02(Function Calling 查数据)** 交付:模型单轮选工具 → 后端执行 → 回灌 → 作答。含四张 MySQL 表、五个 `@tool`、工具执行器、评估集、端到端验收、聊天页。

**本章不做**:多轮 Agent Loop、向量检索/RAG、认证。

文档即设计源:`docs/superpowers/specs/` 下的 spec 是权威设计文档(内有「实现订正」小节,记录代码与最初设计的偏离及原因);`dev-notes/chNN.md` 是按阶段实时记录的开发留痕。改行为前先读 spec 对应章节。

## 高频命令

```bash
.venv/Scripts/python.exe -m pytest                             # 全部测试(含 db,需 MySQL)
.venv/Scripts/python.exe -m pytest -m "not db"                 # 只跑不需要数据库的
.venv/Scripts/python.exe -m pytest tests/test_trim.py          # 单个文件
.venv/Scripts/python.exe -m pytest tests/test_trim.py::test_select_history_never_returns_half_a_round

.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000    # 起服务;浏览器开 http://localhost:8000
.venv/Scripts/python.exe evals/run_tool_selection_eval.py       # 工具选择评估集,需真实 key + MySQL
bash scripts/acceptance.sh                                      # 端到端验收,需服务已启动 + 真实 key
```

`pytest.ini` 已设 `testpaths = tests`、`addopts = -q`。异步测试用 `@pytest.mark.anyio`(backend 由 `tests/conftest.py` 固定为 asyncio),不用 pytest-asyncio。

**不要再往命令行加 `-q`。** `addopts` 里已有一个,叠加成 `-qq` 后 pytest 在 `verbosity < -1` 时**整行不打印 `N passed`** —— 失败仍会报,但通过数没了,等于把「我验过」的证据抹掉。要过滤用 `-m "not db"`。

## 架构

依赖方向严格单向:`api → services → {tools, db, memory, prompts, llm}`。

```
app/config.py     pydantic-settings 读 .env;四个必填字段(三个 OPENAI_* + DATABASE_URL),数值项带下界
app/llm.py        ChatOpenAI 工厂,_build 收口全部硬约束
app/prompts.py    System/抽取 Prompt + 消息组装;Message -> BaseMessage 转换的**唯一**出口
app/schemas.py    纯数据模型,唯一被到处引用的类型源
app/db/           base(引擎/会话工厂)、models(四张表)、session(FastAPI 依赖)
app/tools/        business(五个 @tool)、registry(每请求组装)、executor(超时/重试/错误分类)
app/memory/       store.py(锁注册表)、trim.py(token 预算与按整轮裁剪)—— **不依赖 LangChain**
app/services/     chat.py(单轮编排)、extract.py(抽取)、history.py(会话历史读写)
app/api/          chat.py、extract.py
app/static/       聊天页(单页,无构建工具链)
```

两条贯穿性的结构约定:

- **`memory/` 与 `services/history.py` 不依赖 LangChain**,只碰 `app.schemas.Message` 纯数据类。转 `BaseMessage` 是 `prompts.py:to_lc_messages` 一处的职责。
- **`services/` 的函数接收 llm 实例作为参数**,不在模块层建全局单例;FastAPI 侧靠 `Depends` 注入,测试用 `dependency_overrides` 替换。

### 两条链路

**对话**:`api/chat.py` → `store.lock_for()` 取锁 → `ensure_conversation` + `load_history`(MySQL) → `prepare_turn` 校验预算 → 显式构造 `EventSourceResponse` → `stream_turn` 单轮编排 → **流完整走完后**才 `append_turn` 落库 → `finally` 释放锁。

**抽取**:`/api/extract` 独立于对话,本章未改动。

### 单轮工具编排(本章核心)

```
第一轮:model.bind_tools(TOOLS).astream(msgs)
         ├ 文本 chunk     → 推 SSE token 帧
         └ tool_call chunk → 累积,不外推
     └ 有 tool_calls → tool_call 帧 → 执行 → tool_result 帧 → 回灌 ToolMessage → 第二轮
        无 tool_calls → 已经流完,单次调用即 done
第二轮:model.astream(msgs)   ← **不绑 tools**
```

**「只做单轮」是结构保证,不是提示词约定。** 第二轮不绑 tools,模型在结构上无法再调。实测过「第二轮仍绑 tools 时模型这次没再调」—— 但那是**模型行为,不是保证**,不采用。

注意:**「单轮」不等于「单工具」**。模型可以在同一轮里并发发多个 `tool_call`,执行器**全部执行**并逐个回灌 —— 这是**必须的**,少回灌一个就构成「有 `tool_calls` 没有对应 tool 消息」,上游直接 400。

SSE 事件协议:`meta` → `token` / `tool_call` → `tool_result` → `done` / `error`。

## 若干不读多文件就会踩的硬约束

每一条都对应一个「报错指向别处、极难定位」的故障。

**`use_responses_api=False`**(`app/llm.py`)。LangChain 1.x 的 OpenAI provider 默认走 Responses API,DeepSeek 等兼容网关只实现 Chat Completions。不关掉会调用失败,而报错指向「模型不存在」。

**`tool_call` 条目必须带 `"type": "tool_call"` 键**(`app/tools/`、所有测试替身)。`BaseTool.ainvoke` 判「这是不是工具调用」**只看** `x.get("type") == "tool_call"` 这一个条件;缺键时它把整个 dict 当成**参数**去校验工具 schema,于是每次调用都返回「参数不合法」的可恢复失败。真实链路上模型流出的 chunk 经 LangChain 解析后**是带这个键的**。测试里的 `FakeChunk` 必须补上。

**抽取只能用 `method="json_mode"`**(`app/services/extract.py`)。本项目端点上 `function_calling` 与 `json_schema` 均返回 400,换模型也一样。连带:提示词里**必须出现字面 `JSON` 字样**,且 `EXTRACT_SYSTEM_PROMPT` 描述结构时**不得使用裸花括号**(`ChatPromptTemplate` 按 f-string 解析)。

**流式取文本用 `chunk.text`,不是 `chunk.content`**。1.x 里后者是 content block 列表。

**`create_ticket` 的 `conversation_id` 用闭包工厂,不用 `InjectedToolArg`**。后者对模型隐藏了字段,但值必须另行注入,而该机制在 langchain-core 1.6.3 上无文档 —— 直接用 `tool_call` 调用会抛 `ValidationError`。闭包让参数**根本不在签名里**。

**工具伪随机必须用 `hashlib.sha256` 种子,不能用内置 `hash()`**。`hash()` 对 str 每进程随机化,会让「同一订单号永远返回同样数据」在重启后失效,而**同进程内的测试完全测不出来**(`test_seed_tools_random.py` 里有一条跨进程测试专钉这个)。

**重试用白名单,不是黑名单**。只重试幂等的 `query_*`;**`create_ticket` 永不重试**(超时后重试会建出两张工单)。`ToolNotFound` 与 `ValidationError` 也都不重试 —— 重放同样的参数只会同样失败。

**错误语义边界**:`422` 只表示「模型输出无法解析为约定结构」;上游故障(401/超时/限流/连接失败)一律 `502` + 固定文案。因为 openai SDK 的 `str(exc)` 是**上游响应体原文**,原样回显会泄漏密钥。所有出站错误文本(SSE `error` 帧、`tool_result` 的失败 `summary`、422/502 的 detail)必须过 `app/sanitize.py:redact_api_key`。

**`ToolInfrastructureError` 必须向上抛,不能回灌给模型** —— 数据库故障绝不能被伪装成「你的订单号查不到」。

**锁必须在所有非流式退出路径上释放**,包括 `except BaseException`(`CancelledError` 是 `BaseException`)。**每一条路径都要有具名测试** —— 漏掉的话持锁会话**永久** 409(持锁会话不被 TTL 也不被 LRU 淘汰)。

**预算校验必须在流开始前完成**。SSE 一旦 yield 过第一帧,响应头就发出了,状态码再也改不了 —— 所以端点是普通 `async def` 手工构造 `EventSourceResponse`。

**裁剪按 `user` 边界切轮,不按 `assistant`**(`memory/trim.py`)。OpenAI 兼容 API 要求 `tool` 消息前面紧跟着带对应 `tool_call_id` 的 `assistant` 消息;按 `assistant` 收轮会把这对切开,而那**只在历史长到触发裁剪时偶发 400**。

**`mount("/")` 必须在 `include_router` 之后**(`app/main.py`),否则静态目录会抢走 `/api/*`。

## 写测试的规矩(本项目血的教训)

**本项目的头号风险是「假绿测试」,不是错误代码。**

ch01 抓到 4 类;ch02 又抓到 **7 条「在它本该禁止的实现下依然通过」的断言**,其中 4 条出自主控写的计划文本、3 条是实现者**先测了一下**才发现的。**两次全章复盘,没有一处缺陷是实现者引入的。**

写断言前先反问:**如果实现改错了,这条断言的输出会不会不同?** 具体到本仓库:

- **`Settings(...)` 构造必须传 `_env_file=None`**(`tests/test_*` 里统一如此)。仓库根有真实 `.env`,`pydantic-settings` 会自动读它 —— 不传的话,「缺字段应报错」的测试会因为 `.env` 把值补上而**静默通过**。
- **db 测试读真实 `.env`,不加 `_env_file=None`**;它们用 `@pytest.mark.db` 标记。
- **别为自由文本写字符串断言**。`deepseek-flash` 在 `temperature=0` 下依然非确定。
- **计数类断言必须实测能区分**:`@tool` 的参数校验发生在**函数体之外**,所以「这个工具被调用了几次」的计数器**必须放在 `ainvoke` 边界**,写在函数体里在参数非法时恒为 0、区分不出任何实现。
- **读回数据库的值要用新 session**:SQLAlchemy 身份映射持**弱引用**,同 session 重读是否打到库取决于还有没有东西引用着那个 ORM 对象 —— 会变成「靠 refcount 走运」的断言。
- **复述类断言要对着真实来源验**:让替身**真的把密钥写进异常文本**,否则「响应里没有密钥」是恒真的。
- 单测**全程不联网**;评估集与验收脚本才允许打真实网络。

## 数据与产物

- `evals/tool_selection_cases.jsonl` —— 15 条工具选用例,`expected` 是工具名(闭式精确匹配)或 `null`(不该调工具)。
- **工具选择准确率 13/15 = 86.7% 可引用**(闭式枚举,与 ch01 那个被样本拟合的 `expected_solution` 关键词口径不同)。但引用时须一并说明:**该数字是在没有生产 system prompt 的条件下测得的**,且**用例集偏弱**(非 null 的 13 条里 11 条从不失误,信息量主要来自 2 条诱饵)。
- `evals/extract_cases.jsonl`(ch01)的 `expected_solution` 分数**不可引用** —— 关键词是看到输出措辞后才放宽的。
- `evals/results/` 被 gitignore,是历史运行产物。

## 平台陷阱(Windows + Git Bash)

本机 locale 是 **cp936**,这个陷阱在 ch02 咬过**三次**,属**复发型**:

- **含中文的请求体不能走 `curl` 的 argv**。MSYS2 会按 CP936 重编码,服务端只回 `error parsing the body`。一律走 stdin heredoc。
- **子进程输出要显式钉编码**。跨进程测试给子进程加 `-X utf8`,否则管道上的 stdout 按 GBK 编码而父进程按 UTF-8 解码,报错表现为 `proc.stdout is None`。
- **脚本打印非 ASCII 要钉输出边界**,用 `sys.stdout.buffer.write(...encode("utf-8"))`,不要依赖控制台 codec(`✓`/`✗` 不在 GBK 里,`print` 会直接崩)。
- **验收断言不能直接 grep 原始 SSE 流**。回复逐 token 推送,`20240915` 会被切成三个独立帧。用 `join_tokens` 拼回后再比对。
- **不要用 `grep '[一-龥]'` 检查中文完好性**:C locale 下 bracket expression 退化成字节区间,对真实 UTF-8 和 mojibake 全部匹配,是个恒真的假断言。脚本里的 `has_cjk` 按 Python 码点判断。
- **起服务前先查端口**:8000 上残留的僵尸进程会让你 curl 到旧代码,从而得出「新代码坏了」的**假红**。ch02 的最终验证就差点栽在这上面。

## 工作方式要求

用户对本项目开发有固定要求(见 memory,勿自行放宽):

1. **全程走 Superpowers 流程**,技能自动触发不跳步。
2. **非可单测产出(Prompt 模板、数据类)把 TDD 换成评估集/标注样例验证**,其余步骤照走;纯 UI 页面例外,用 Vibe Coding 直做。
3. **涉及具体库/框架/API 的用法,先用 Context7 MCP 查最新官方文档再动手**,不许凭记忆写。
4. **用户点名的技术选型是定死的** —— 走不通就停下来问,不许自行换方案。spec §9 中预先标注为「待实测」的参数,按实测结果改动属设计授权,但仍需记账并告知用户可一句话回退。
5. **过程实时留痕到 `dev-notes/chNN.md`**,每完成一个阶段就补一段,记四样:用户关键原话、关键产出、被拒绝/被纠偏了什么、翻车与返工。**明确不允许收尾时一次性补记。**
