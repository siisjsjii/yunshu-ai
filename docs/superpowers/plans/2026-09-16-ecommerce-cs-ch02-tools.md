# ch02 Function Calling 查数据能力 · 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给现有客服对话装上单轮工具调用能力 —— 模型自己决定调哪个工具,后端执行后回灌,模型据结果作答;聊天记录与工具调用落 MySQL。

**Architecture:** 手写单轮编排(不引入 LangGraph):第一轮 `bind_tools` + `astream` 边流边分拣,文本直接推、tool_call 累积;执行工具、回灌 `ToolMessage`;第二轮 `astream` 且**不绑 tools**,从结构上保证单轮收敛。会话历史以 MySQL 为真相,进程内只保留每会话锁。

**Tech Stack:** FastAPI 0.141.1 / SQLAlchemy 2.0.53 (asyncmy) / MySQL 8.0.46 / LangChain 1.4.0 / Python 3.13.14

**Spec:** `docs/superpowers/specs/2026-09-16-ecommerce-cs-ch02-tools-design.md`

## Global Constraints

- **Python 3.13.14**,解释器一律用 `.venv/Scripts/python.exe`(Windows + Git Bash)。
- **新增依赖(版本锁死)**:`sqlalchemy==2.0.53`、`asyncmy==0.2.14`、`cryptography==50.0.1`。
  `cryptography` **不是可选项** —— MySQL 8.0 默认认证插件 `caching_sha2_password`,asyncmy 缺它会直接抛 `RuntimeError: 'cryptography' package is required`。
- **数据库**:`mysql+asyncmy://root:***@127.0.0.1:3307/mewhelp?charset=utf8mb4`,字符集 `utf8mb4` / 排序规则 `utf8mb4_0900_ai_ci`。实例由用户以 Docker 提供,**仓库不提交 `docker-compose.yml`**。
- **不使用 LangGraph、不使用 LCEL**(沿用 ch01 spec §3)。
- **工具返回** 一律 `json.dumps(..., ensure_ascii=False)` 的字符串,不返回 dict。
- **确定性伪随机的种子必须用 `hashlib.sha256`**,禁止用内置 `hash()`(对 str 每进程随机化)。
- **`create_ticket` 永不重试**(写操作);重试走**白名单**而非黑名单。
- **`ToolMessage` / `AIMessage` 从 `langchain.messages` 导入**;**`tool` 从 `langchain.tools` 导入**。
- **单测不联网**:除 `@pytest.mark.db`(需 MySQL)与评估集/验收脚本外,一律不打网络。
- **测试中的 `Settings(...)` 必须传 `_env_file=None`** —— 否则真实 `.env` 会补上字段,「缺字段应报错」的测试会**静默通过**(ch01 教训)。
- **提交信息用中文**,格式 `type: 描述`(沿用 ch01 风格)。
- **跑测试时不要再加 CLI `-q`。** `pytest.ini` 已有 `addopts = -q`,再加一个 `-q` 会变成
  `-qq`,而 pytest 在 `verbosity < -1` 时**整行不打印 `N passed`**(`_pytest/terminal.py`,
  `MoreQuietAction` 是逐次递减的 action)。失败仍会报,所以不会造成假绿,但**看不到通过数** ——
  等于把"我验过"这句话的证据抹掉了。用 `.venv/Scripts/python.exe -m pytest`,
  需要过滤时用 `.venv/Scripts/python.exe -m pytest -m "not db"`。

### 已实测确认的 API 事实(不要再猜)

| 事实 | 值 |
|---|---|
| 工具错误如何传播 | **一律抛出**,不包成 ToolMessage(`handle_tool_error` 默认 `False`)。`ValidationError` / `SQLAlchemyError` / 其他异常**可区分** |
| 第一轮分片汇聚 | `acc = chunk if acc is None else acc + chunk`,结束后的 `acc.tool_calls` 是**已解析**的列表 |
| 第一轮 chunk 分离 | 调工具时 **0 文本 chunk**;不调工具时 **0 tool_call chunk**。两者零重叠 |
| 从 DB 重建的历史 | `HumanMessage` / `AIMessage(content="", tool_calls=[...])` / `ToolMessage(content=..., tool_call_id=...)` 序列**被模型正常接受**并正确读出结果 |
| `create_ticket` 的 `conversation_id` | **用闭包工厂**,不用 `InjectedToolArg`(后者对模型隐藏了字段,但调用时值必须另行注入,机制在本版本无文档) |
| 工具集构造 | `query_faq` 与 `create_ticket` 需 DB 会话 → **每请求工厂构造**;其余三个是模块级常量 |

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `app/config.py` | **改**:新增 `database_url`(必填)与三个工具配置项 |
| `app/db/base.py` | **新**:`Base` / `get_engine` / `get_sessionmaker` / `get_session` |
| `app/db/models.py` | **新**:四张表的 ORM 模型 |
| `app/db/session.py` | **新**:`get_session` 的 FastAPI 依赖(转出 `base.py` 以便测试覆盖) |
| `app/tools/business.py` | **新**:三个随机工具(模块级)+ `make_query_faq` / `make_create_ticket`(工厂) |
| `app/tools/registry.py` | **新**:`build_tools` / `registry_for` |
| `app/tools/executor.py` | **新**:超时 / 重试白名单 / 错误分类 / `ToolOutcome` |
| `app/memory/trim.py` | **改**:分轮规则改为 **user 边界** |
| `app/memory/store.py` | **改**:瘦身为锁注册表 |
| `app/schemas.py` | **改**:`Message` 增加 `tool_calls` / `tool_call_id` |
| `app/services/history.py` | **新**:读历史 → LangChain 消息;落库 |
| `app/services/chat.py` | **改**:单轮编排 |
| `app/prompts.py` | **改**:System Prompt 增加工具使用指引 |
| `app/api/chat.py` | **改**:新增 `tool_call` / `tool_result` 事件 + `user_id` |
| `app/main.py` | **改**:挂载静态目录 |
| `app/static/index.html` | **新**:聊天页(Vibe Coding,**不走 TDD**) |
| `scripts/init_db.py` / `scripts/seed_db.py` | **新**:建表 / 灌种子 |
| `evals/tool_selection_cases.jsonl` / `evals/run_tool_selection_eval.py` | **新**:工具选择评估集 |
| `scripts/acceptance.sh` | **改**:补三条验收 |

---

## Task 1: 依赖、配置与测试标记

**Files:**
- Modify: `requirements.txt`
- Modify: `.env.example`
- Modify: `pytest.ini`
- Modify: `app/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: 无
- Produces: `Settings.database_url: str`(必填)、`Settings.tool_timeout_seconds: float = 10.0`、`Settings.tool_retry_attempts: int = 1`、`Settings.tool_retry_delay_seconds: float = 0.3`

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_config.py`(该文件已有 `REQUIRED` 字典与 `_env_file=None` 的用法,照抄其风格):

```python
def test_database_url_is_required():
    """DATABASE_URL 必填,不给默认值 —— 默认值会拿一个可能不对的连接串去连。

    注意不能直接写 Settings(_env_file=None, **REQUIRED):REQUIRED 是"一份
    完整的合法载荷"(其它测试都靠它),而本测试要的恰恰是"缺一个字段"。
    两者矛盾 —— 必须显式把该字段剔出去。
    """
    missing = {k: v for k, v in REQUIRED.items() if k != "database_url"}
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **missing)


def test_database_url_is_read_from_settings():
    settings = Settings(_env_file=None, **REQUIRED)
    assert settings.database_url == "mysql+asyncmy://u:p@h:3306/db"


def test_tool_defaults():
    settings = Settings(_env_file=None, **REQUIRED)
    assert settings.tool_timeout_seconds == 10.0
    assert settings.tool_retry_attempts == 1
    assert settings.tool_retry_delay_seconds == 0.3
```

- [ ] **Step 1b: 修复三个因本次改动而"为错误原因通过"的 ch01 测试**

`database_url` 变成必填后,`tests/test_config.py` 里 ch01 的三条拒绝测试**同时缺了它**,
于是它们无论各自命名的那个字段是否仍必填都会抛错、都会通过:

```python
def test_missing_model_is_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, openai_base_url="x", openai_api_key="y")
        # ← 这里也缺 database_url,所以即使 openai_model 有了默认值,本测试照样通过
```

也就是说:**把 `openai_model` 改成有默认值,这条测试不会报警** —— 它还在,但不再钉住
它名字声称的东西。这正是 ch01 复盘的头号问题。

把这三条(`test_missing_model_is_rejected` / `test_missing_base_url_is_rejected` /
`test_missing_api_key_is_rejected`)改成"除命名字段外全部给齐 + 断言错误里出现该字段名":

```python
def test_missing_model_is_rejected():
    """OPENAI_MODEL 必填:不给默认值,避免换模型时静默用错模型名。"""
    with pytest.raises(ValidationError) as exc:
        Settings(
            _env_file=None,
            openai_base_url="x",
            openai_api_key="y",
            database_url="mysql+asyncmy://u:p@h:3306/db",
        )
    assert "openai_model" in str(exc.value)
```

另两条同形:`test_missing_base_url_is_rejected` 给齐 `openai_api_key` / `openai_model` /
`database_url` 并断言 `"openai_base_url" in str(exc.value)`;`test_missing_api_key_is_rejected`
给齐其余三项并断言 `"openai_api_key" in str(exc.value)`。

**并附「断言能区分」的证据**:临时给 `openai_model` 加一个默认值,证明
`test_missing_model_is_rejected` **会失败**,再改回来证明通过。两次输出都写进报告。

**同时,必须在下面全部 5 个测试文件的 `REQUIRED` 字典里加 `database_url`** ——
把 `database_url` 设为必填会打断**每一个**构造 `Settings(...)` 的测试,漏掉一个
整套就红:

```
tests/test_config.py         ← 本任务要改的
tests/test_llm.py
tests/test_api_chat.py
tests/test_chat_service.py
tests/test_api_extract.py    ← 本章不动 /api/extract,但它的测试同样构造 Settings
```

在每个文件的 `REQUIRED` 字典里加同一行(**不要改动该文件里其它内容**):

```python
    "database_url": "mysql+asyncmy://u:p@h:3306/db",
```

`tests/test_config.py` 的 `REQUIRED` 加完后应形如:

```python
REQUIRED = {
    "openai_base_url": "https://api.deepseek.com/v1",
    "openai_api_key": "sk-test",
    "openai_model": "deepseek-chat",
    "database_url": "mysql+asyncmy://u:p@h:3306/db",
}
```

`tests/test_api_chat.py` 与 `tests/test_chat_service.py` 会在 Task 11 / Task 10 被
**重写**,但本任务同样要改它们 —— 否则从此刻到那两个任务之间,测试套件一直是红的。

- [ ] **Step 2: 运行测试,确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config.py`
Expected: FAIL —— `test_database_url_is_required` 报 `DID NOT RAISE`,另外两个报 `Settings` 无该属性

- [ ] **Step 3: 实现**

`app/config.py` 在「必填」组加入:

```python
    # 必填:无默认值
    openai_base_url: str
    openai_api_key: str
    openai_model: str
    database_url: str
```

在「可选」组加入:

```python
    # 工具执行。加界是**故意的**:负的 tool_retry_attempts 会让重试循环
    # 一次都不执行,last_message 停在空串,执行器返回一个 content 为空的
    # 结果给模型 —— 静默失败。本项目配置层的既定立场是让配错**响亮地早失败**
    #(注意:不是"进程启动即报" —— get_settings 是 lru_cache 的、经 FastAPI Depends
    #  解析,所以实际在**第一次解析它的请求**上抛错。但那次失败早于任何建锁,目标达成)
    # (OPENAI_MODEL / DATABASE_URL 都是必填而非给默认值)。
    tool_timeout_seconds: float = Field(default=10.0, gt=0)
    tool_retry_attempts: int = Field(default=1, ge=0)
    tool_retry_delay_seconds: float = Field(default=0.3, ge=0)
```

注意 pydantic **不校验 `default` 字面量本身**:`Field(default=-1, ge=0)` 仍能实例化出
`-1`。所以这些界拦住的是 **env / 初始化入参**(即运维配错 `.env` 这条路径),而非代码里
的字面量笔误 —— 后者由 `test_tool_defaults` 的断言兜住。

`requirements.txt` 追加:

```
sqlalchemy==2.0.53
asyncmy==0.2.14
cryptography==50.0.1
```

`.env.example` 的「必填」段追加(注意放在必填段,与 `.env` 实际一致):

```
# 数据库。实例由 Docker 提供,仓库不管理容器。
# MySQL 8.0 默认认证插件 caching_sha2_password 需要 cryptography 包。
DATABASE_URL=mysql+asyncmy://root:yourpassword@127.0.0.1:3307/mewhelp?charset=utf8mb4
```

`.env.example` 的「可选」段追加:

```
TOOL_TIMEOUT_SECONDS=10
TOOL_RETRY_ATTEMPTS=1
TOOL_RETRY_DELAY_SECONDS=0.3
```

`pytest.ini` 改为:

```ini
[pytest]
testpaths = tests
addopts = -q
markers =
    db: 需要真实 MySQL(无 Docker 时用 -m "not db" 跳过)
```

- [ ] **Step 4: 运行测试,确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config.py`
Expected: PASS

- [ ] **Step 5: 运行全量,确认没有打破既有测试**

Run: `.venv/Scripts/python.exe -m pytest -m "not db"`
Expected: PASS(此时还没有 db 标记的测试,行为应与 ch01 的 96 passed 一致)

- [ ] **Step 6: 提交**

```bash
git add requirements.txt .env.example pytest.ini app/config.py tests/test_config.py
git commit -m "feat: 配置层加入 DATABASE_URL 与工具执行参数"
```

---

## Task 2: ORM 模型与引擎

**Files:**
- Create: `app/db/__init__.py`, `app/db/base.py`, `app/db/models.py`, `app/db/session.py`
- Create: `scripts/init_db.py`
- Test: `tests/test_db_models.py`

**Interfaces:**
- Consumes: `Settings.database_url`(Task 1)
- Produces:
  - `app.db.base.Base`(DeclarativeBase)、`app.db.base.get_engine() -> AsyncEngine`、`app.db.base.get_sessionmaker() -> async_sessionmaker[AsyncSession]`
  - `app.db.session.get_session() -> AsyncIterator[AsyncSession]`(FastAPI 依赖,测试用 `dependency_overrides` 替换)
  - `app.db.models.Faq` / `Conversation` / `MessageRecord` / `Ticket`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_db_models.py`:

```python
"""DB 集成测试。需要 MySQL 在跑:见 spec §7.4。"""

import pytest
from sqlalchemy import select, text

from app.db.base import get_engine, get_sessionmaker
from app.db.models import Conversation, Faq, MessageRecord, Ticket

pytestmark = pytest.mark.db

SCRATCH_CONVERSATION = "test0000000000000000000000000000"


@pytest.mark.anyio
async def test_tables_exist_and_chinese_roundtrips():
    """四张表建得出来,且中文与 JSON 列往返不炸。"""
    engine = get_engine()
    async with get_sessionmaker()() as session:
        # 清理上次残留
        await session.execute(
            text("DELETE FROM messages WHERE conversation_id = :c"),
            {"c": SCRATCH_CONVERSATION},
        )
        await session.execute(
            text("DELETE FROM tickets WHERE conversation_id = :c"),
            {"c": SCRATCH_CONVERSATION},
        )
        await session.execute(
            text("DELETE FROM conversations WHERE id = :c"),
            {"c": SCRATCH_CONVERSATION},
        )

        session.add(
            Conversation(id=SCRATCH_CONVERSATION, user="tester", status="active")
        )
        session.add(
            MessageRecord(
                conversation_id=SCRATCH_CONVERSATION,
                role="assistant",
                content="",
                tool_calls=[
                    {"name": "query_logistics", "args": {"order_id": "1001"}, "id": "call_1"}
                ],
            )
        )
        session.add(
            Ticket(
                ticket_no="T-TEST-1",
                conversation_id=SCRATCH_CONVERSATION,
                description="鞋码不对想换",
                ticket_type="换货",
                status="open",
            )
        )
        await session.commit()

    async with get_sessionmaker()() as session:
        row = (
            await session.execute(
                select(MessageRecord).where(
                    MessageRecord.conversation_id == SCRATCH_CONVERSATION
                )
            )
        ).scalars().one()
        assert row.tool_calls[0]["name"] == "query_logistics"   # JSON 列往返
        assert row.tool_call_id is None

        conv = (
            await session.execute(
                select(Conversation).where(Conversation.id == SCRATCH_CONVERSATION)
            )
        ).scalars().one()
        assert conv.user == "tester"

    # 收尾清理,不留垃圾数据
    async with get_sessionmaker()() as session:
        await session.execute(
            text("DELETE FROM messages WHERE conversation_id = :c"),
            {"c": SCRATCH_CONVERSATION},
        )
        await session.execute(
            text("DELETE FROM tickets WHERE conversation_id = :c"),
            {"c": SCRATCH_CONVERSATION},
        )
        await session.execute(
            text("DELETE FROM conversations WHERE id = :c"),
            {"c": SCRATCH_CONVERSATION},
        )
        await session.commit()

    await engine.dispose()
```

再在同一文件加一条**中文 LIKE** 的回归测试(把 spec §7.4 的实测钉住):

```python
@pytest.mark.anyio
async def test_faq_like_matches_chinese_substring():
    """中文子串能命中。实测已确认 utf8mb4_0900_ai_ci 正常,此测试防回归。"""
    async with get_sessionmaker()() as session:
        session.add(
            Faq(question="退货政策是什么", answer="七天无理由退货", category="退换货")
        )
        await session.commit()
        hits = (
            await session.execute(
                select(Faq).where(Faq.question.like("%退货%"))
            )
        ).scalars().all()
        assert len(hits) >= 1
        await session.execute(
            text("DELETE FROM faq WHERE question = :q"),
            {"q": "退货政策是什么"},
        )
        await session.commit()
```

- [ ] **Step 2: 运行测试,确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_db_models.py`
Expected: FAIL —— `ModuleNotFoundError: No module named 'app.db'`

- [ ] **Step 3: 实现 `app/db/base.py`**

```python
from collections.abc import AsyncIterator
from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings


class Base(DeclarativeBase):
    """全部 ORM 模型的基类。"""


@lru_cache
def get_engine() -> AsyncEngine:
    """进程内单例。pool_pre_ping 让空闲连接被 MySQL 掐断后能自愈。"""
    return create_async_engine(get_settings().database_url, pool_pre_ping=True)


@lru_cache
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    # expire_on_commit=False:提交后仍可读属性,否则异步下访问会触发隐式 IO 报错。
    return async_sessionmaker(get_engine(), expire_on_commit=False)
```

`app/db/session.py`:

```python
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_sessionmaker


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI 依赖。测试用 app.dependency_overrides 替换。"""
    async with get_sessionmaker()() as session:
        yield session
```

`app/db/__init__.py` 留空。

- [ ] **Step 4: 实现 `app/db/models.py`**

```python
from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Faq(Base):
    __tablename__ = "faq"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(String(255), nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(64), nullable=False, index=True)


class Conversation(Base):
    """会话壳。id 复用 ch01 的 session_id(uuid4().hex,32 字符)。"""

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    user: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class MessageRecord(Base):
    """消息流水。类名不叫 Message —— ch01 的 app.schemas.Message 已占用该名字。"""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("conversations.id"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # assistant 的「工具调用申请」是**数组**(可能一次申请多个),故用 JSON 列。
    tool_calls: Mapped[list | None] = mapped_column(JSON, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class Ticket(Base):
    """工单。按用户要求用业务工单号做主键,不用自增 id。"""

    __tablename__ = "tickets"

    ticket_no: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("conversations.id"), nullable=False, index=True
    )
    description: Mapped[str] = mapped_column(Text, nullable=False)
    ticket_type: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="open")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
```

- [ ] **Step 5: 实现 `scripts/init_db.py`**

```python
"""建表。用法:.venv/Scripts/python.exe scripts/init_db.py

幂等 —— create_all 只建不存在的表,重复跑不会破坏已有数据。
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import models  # noqa: F401  导入以确保模型注册到 Base.metadata
from app.db.base import Base, get_engine


async def main() -> None:
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()
    print("建表完成:", ", ".join(sorted(Base.metadata.tables)))


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 6: 建表并运行测试**

Run:
```bash
.venv/Scripts/python.exe scripts/init_db.py
.venv/Scripts/python.exe -m pytest tests/test_db_models.py
```
Expected: 建表输出四个表名;测试 PASS

- [ ] **Step 7: 提交**

```bash
git add app/db scripts/init_db.py tests/test_db_models.py
git commit -m "feat: 数据层 —— 四张表的 ORM 模型与异步引擎"
```

---

## Task 3: 种子数据

**Files:**
- Create: `scripts/seed_db.py`
- Test: `tests/test_seed_db.py`

**Interfaces:**
- Consumes: `app.db.models` 的四个模型、`get_sessionmaker`
- Produces: 可重复执行的 `seed()` 函数;`faq` 表中**不含**「邮费」「运费」条目

- [ ] **Step 1: 写失败测试**

创建 `tests/test_seed_db.py`:

```python
"""种子数据测试。需要 MySQL。"""

import pytest
from sqlalchemy import func, select

from app.db.base import get_sessionmaker
from app.db.models import Faq
from scripts.seed_db import FAQ_ROWS, seed

pytestmark = pytest.mark.db


@pytest.mark.anyio
async def test_seed_is_idempotent_and_loads_faq():
    await seed()
    await seed()  # 再跑一次,不应产生重复
    async with get_sessionmaker()() as session:
        count = (await session.execute(select(func.count()).select_from(Faq))).scalar()
    assert count == len(FAQ_ROWS)


def test_seed_contains_no_shipping_fee_entry():
    """验收 3 依赖「邮费」查不到。种子数据里出现该词,验收 3 就会假通过。"""
    blob = " ".join(f"{r['question']} {r['answer']} {r['category']}" for r in FAQ_ROWS)
    assert "邮费" not in blob
    assert "运费" not in blob
```

- [ ] **Step 2: 运行测试,确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_seed_db.py`
Expected: FAIL —— `ModuleNotFoundError: No module named 'scripts.seed_db'`

- [ ] **Step 3: 实现**

创建 `scripts/__init__.py`(空文件,让 `scripts` 可被 import)与 `scripts/seed_db.py`:

```python
"""灌种子数据。用法:.venv/Scripts/python.exe scripts/seed_db.py

幂等:faq 按 question 去重;样例会话与工单用固定主键,重复跑不会堆积。
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select

from app.db.base import get_sessionmaker
from app.db.models import Conversation, Faq, MessageRecord, Ticket

# 注意:本表刻意不含「邮费」「运费」相关条目 —— 验收 3 的漏召回是预期结果,
# 若此处出现该词,验收 3 会假通过。见 spec §7.3。测试 test_seed_contains_no_shipping_fee_entry 守护此约束。
FAQ_ROWS = [
    {"question": "退货政策是什么", "answer": "支持七天无理由退货,商品需保持完好、吊牌齐全。", "category": "退换货"},
    {"question": "怎么申请退货", "answer": "在订单详情页点击「申请退款」,选择退货原因并提交,审核通过后会给出寄回地址。", "category": "退换货"},
    {"question": "可以换货吗", "answer": "可以。签收后七天内,商品无使用痕迹即可申请换货,请提供订单号与原/目标规格。", "category": "退换货"},
    {"question": "退款多久到账", "answer": "退货入库验收通过后 1-3 个工作日退回原支付渠道,具体到账时间以银行为准。", "category": "退换货"},
    {"question": "发票怎么开", "answer": "下单时可在备注中填写抬头与税号;已完成的订单可联系客服补开电子发票。", "category": "发票问题"},
    {"question": "发票可以开专票吗", "answer": "可以开具增值税专用发票,请提供公司名称、税号、地址电话与开户行账号。", "category": "发票问题"},
    {"question": "物流一直没更新", "answer": "物流信息可能存在延迟,一般 24 小时内会更新。超过 48 小时未更新请联系客服为您催件。", "category": "物流异常"},
    {"question": "快递显示签收但我没收到", "answer": "请先与快递员或代收点核实。确认未收到的,联系客服,我们会向快递公司发起核查。", "category": "物流异常"},
    {"question": "发什么快递", "answer": "默认发中通/圆通,偏远地区可能改发邮政。下单后无法指定快递公司。", "category": "物流异常"},
    {"question": "商品有货吗", "answer": "商品页显示库存实时同步。若显示缺货,可点击「到货通知」,补货后会短信提醒。", "category": "商品咨询"},
    {"question": "支持哪些支付方式", "answer": "支持微信、支付宝、银行卡以及花呗分期。", "category": "商品咨询"},
    {"question": "怎么联系人工客服", "answer": "在对话框中直接说明需要人工,或拨打客服热线 400-000-0000(9:00-21:00)。", "category": "其他"},
]

SAMPLE_CONVERSATION = "seed0000000000000000000000000000"
SAMPLE_TICKET_NO = "T-SEED-0001"


async def seed() -> None:
    async with get_sessionmaker()() as session:
        existing = set(
            (await session.execute(select(Faq.question))).scalars().all()
        )
        for row in FAQ_ROWS:
            if row["question"] not in existing:
                session.add(Faq(**row))

        if not (
            await session.execute(
                select(Conversation).where(Conversation.id == SAMPLE_CONVERSATION)
            )
        ).scalars().first():
            session.add(
                Conversation(
                    id=SAMPLE_CONVERSATION, user="demo-user", status="active"
                )
            )
            session.add(
                MessageRecord(
                    conversation_id=SAMPLE_CONVERSATION,
                    role="user",
                    content="订单 20240915 的鞋码不对,我想换大一码",
                )
            )
            session.add(
                MessageRecord(
                    conversation_id=SAMPLE_CONVERSATION,
                    role="assistant",
                    content="好的,请提供原规格与目标规格,我为您登记换货。",
                )
            )
            session.add(
                Ticket(
                    ticket_no=SAMPLE_TICKET_NO,
                    conversation_id=SAMPLE_CONVERSATION,
                    description="鞋码偏小,想换成大一码",
                    ticket_type="换货",
                    status="open",
                )
            )

        await session.commit()


if __name__ == "__main__":
    asyncio.run(seed())
    print(f"种子完成:faq {len(FAQ_ROWS)} 条 + 1 组样例会话/消息/工单")
```

- [ ] **Step 4: 运行测试,确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_seed_db.py`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add scripts/__init__.py scripts/seed_db.py tests/test_seed_db.py
git commit -m "feat: 种子数据,faq 刻意不含邮费条目"
```

---

## Task 4: 工具异常类型与三个确定性伪随机工具

**Files:**
- Create: `app/tools/__init__.py`(空), `app/tools/errors.py`, `app/tools/business.py`
- Test: `tests/test_tools_random.py`

**Interfaces:**
- Consumes: 无(纯函数,不碰 DB)
- Produces:
  - `app.tools.errors.ToolNotFound`(可恢复)、`app.tools.errors.ToolInfrastructureError`(不可恢复)
  - `app.tools.business.query_order` / `query_product` / `query_logistics` —— 模块级 `BaseTool`
  - `app.tools.business._rng(*parts: str) -> random.Random`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_tools_random.py`:

```python
"""三个确定性伪随机工具的测试。不联网、不碰 DB。"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.tools.business import query_logistics, query_order, query_product
from app.tools.errors import ToolNotFound

REPO_ROOT = Path(__file__).resolve().parents[1]


def _call(tool, args: dict) -> str:
    tool_call = {"name": tool.name, "args": args, "id": "call_1", "type": "tool_call"}
    return asyncio.run(tool.ainvoke(tool_call)).content


def test_same_order_id_gives_same_result():
    """同一订单号永远返回同样数据。"""
    assert _call(query_logistics, {"order_id": "1001"}) == _call(
        query_logistics, {"order_id": "1001"}
    )


def test_different_order_ids_differ():
    """不同订单号应有不同数据,否则工具等于常量。"""
    assert _call(query_logistics, {"order_id": "1001"}) != _call(
        query_logistics, {"order_id": "1002"}
    )


def test_result_is_json_with_chinese_not_escaped():
    """返回 JSON 字符串,且中文不被转义成 \\uXXXX(白烧 token)。"""
    raw = _call(query_logistics, {"order_id": "1001"})
    payload = json.loads(raw)
    assert set(payload) >= {"order_id", "status", "location"}
    assert "\\u" not in raw
    assert any("一" <= ch <= "鿿" for ch in raw)


def test_seed_is_stable_across_processes():
    """跨进程确定性。

    这条是本章最容易写错的断言 —— 内置 hash() 对 str 每进程随机化
    (PYTHONHASHSEED),用它会让同一订单号在重启后返回不同数据,
    而同进程内的任何测试都测不出来。故必须另起两个进程比对。
    """
    code = (
        "import asyncio, sys; sys.path.insert(0, '.'); "
        "from app.tools.business import query_logistics; "
        "print(asyncio.run(query_logistics.ainvoke("
        "{'name': 'query_logistics', 'args': {'order_id': '1001'}, "
        "'id': 'c', 'type': 'tool_call'})).content)"
    )
    outs = []
    for _ in range(2):
        # 子进程的输出编码必须显式钉成 UTF-8。本机 locale 是 cp936,Python 会把
        # 管道上的 stdout 按 GBK 编码,而下面按 UTF-8 解码 —— 不钉的话子进程输的
        # 中文会在解码时炸掉,表现为 proc.stdout 为 None(随后 .strip() 报
        # AttributeError)。这与工具逻辑无关,纯属平台差异:同一份代码在
        # PYTHONUTF8=1 下 6 passed,不加就红。见 ch01 记录的同类 MSYS2 CP936 陷阱。
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
        _call(query_logistics, {"order_id": "abc"})


def test_query_product_uses_keyword():
    a = _call(query_product, {"keyword": "无线耳机"})
    b = _call(query_product, {"keyword": "保温杯"})
    assert a != b
    assert "无线耳机" in a
```

- [ ] **Step 2: 运行测试,确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tools_random.py`
Expected: FAIL —— `ModuleNotFoundError: No module named 'app.tools'`

- [ ] **Step 3: 实现 `app/tools/errors.py`**

```python
class ToolNotFound(Exception):
    """业务性未找到(订单/商品不存在)。

    可恢复:executor 会把它转成 ok=False 的 ToolOutcome,回灌给模型,
    由模型用自然语言兜住。流不会中断。
    """


class ToolInfrastructureError(Exception):
    """基础设施故障(数据库连不上等)。

    不可恢复:executor 向上抛,API 层推 error 帧并终止流 —— 不能让
    "数据库挂了"被伪装成"你的订单号查不到"。
    """
```

`app/tools/__init__.py` 留空。

- [ ] **Step 4: 实现 `app/tools/business.py` 的三个随机工具**

```python
"""五个业务工具。

三个「假装有上游系统」的工具(query_order / query_product / query_logistics)
在本模块内用确定性伪随机生成数据 —— 不接真实接口、不建表。同一入参
永远得到同样结果,故验收可以写会失败的断言。

另两个工具需要数据库会话,故用工厂函数**每请求构造**,见 make_query_faq /
make_create_ticket。
"""

import hashlib
import json
import random

from langchain.tools import tool

from app.tools.errors import ToolNotFound


def _rng(*parts: str) -> random.Random:
    """由入参派生稳定种子。

    **绝不能用内置 hash()** —— 它对 str 每进程随机化(PYTHONHASHSEED),
    会让"同一订单号永远返回同样数据"在进程重启后失效,而同进程内的
    测试完全测不出来。sha256 跨进程、跨平台稳定。
    """
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest, "big"))


#: 回显给模型的入参最多截这么长 —— 模型给的输入不受我们控制,原样回灌
#: 等于让它自己决定往上下文里塞多少 token。
_ECHO_LIMIT = 32


def _require_order_no(order_id: str) -> str:
    """订单号须为 4-32 位 ASCII 数字。不符合视为查无此单,而不是编一个结果。

    必须是 `isascii() and isdigit()` 两个条件:单独一个 `isdigit()` 是
    Unicode 感知的,`"١٢٣٤".isdigit()`(阿拉伯-印度数字)与 `"²²²²".isdigit()`
    (上标)都为 True —— 这类输入会**通过**校验并拿到一张凭空编造的订单,
    而不是 ToolNotFound。
    """
    cleaned = order_id.strip()
    if not (cleaned.isascii() and cleaned.isdigit()) or not (4 <= len(cleaned) <= 32):
        raise ToolNotFound(f"未找到订单 {cleaned[:_ECHO_LIMIT]},请核对订单号后重试")
    return cleaned


_ORDER_STATUS = ["待付款", "已付款", "已发货", "已完成", "已取消"]
_PRODUCT_NAMES = ["无线耳机", "运动鞋", "双肩包", "保温杯", "机械键盘"]


@tool
async def query_order(order_id: str) -> str:
    """查询订单详情:状态、商品、金额、下单时间。仅当用户给出订单号时使用。"""
    order_no = _require_order_no(order_id)
    r = _rng("order", order_no)
    return json.dumps(
        {
            "order_id": order_no,
            "status": r.choice(_ORDER_STATUS),
            "product": r.choice(_PRODUCT_NAMES),
            "amount": f"{r.randint(49, 999)}.{r.randint(0, 99):02d}",
            "created_at": (
                f"2026-{r.randint(1, 9):02d}-{r.randint(10, 28):02d} "
                f"{r.randint(9, 21):02d}:{r.randint(0, 59):02d}"
            ),
        },
        ensure_ascii=False,
    )


_PRODUCT_SPECS = ["标准版", "Pro 版", "家用款", "经典款"]


@tool
async def query_product(keyword: str) -> str:
    """按关键词查询商品信息:名称、价格、库存、规格。用户问商品价格、有没有货时使用。"""
    cleaned = keyword.strip()
    if not cleaned:
        raise ToolNotFound("请提供商品名称或关键词")
    r = _rng("product", cleaned)
    # 只抽一次。抽两次的话 name 里的规格与 spec 字段相互独立,四次里只有一次
    # 对得上 —— 工具会把自相矛盾的数据喂给模型,而本章验收全靠模型如实转述
    # 工具结果,喂矛盾数据等于从源头破坏它。
    spec = r.choice(_PRODUCT_SPECS)
    return json.dumps(
        {
            "keyword": cleaned,
            "name": f"{cleaned}({spec})",
            "price": f"{r.randint(29, 1299)}.{r.randint(0, 99):02d}",
            "stock": r.randint(0, 200),
            "spec": spec,
        },
        ensure_ascii=False,
    )


_LOGISTICS_STATUS = ["已揽件", "运输中", "派送中", "已签收"]
_CITIES = ["广州分拨中心", "上海分拨中心", "北京分拨中心", "成都分拨中心"]


@tool
async def query_logistics(order_id: str) -> str:
    """查询订单的物流状态、当前位置与轨迹。用户问"到哪了""发货没"时使用。"""
    order_no = _require_order_no(order_id)
    r = _rng("logistics", order_no)
    status = r.choice(_LOGISTICS_STATUS)
    city = r.choice(_CITIES)
    day = r.randint(1, 15)
    return json.dumps(
        {
            "order_id": order_no,
            "status": status,
            "location": city,
            "traces": [
                {
                    "time": f"2026-09-{day:02d} {r.randint(9, 21):02d}:{r.randint(0, 59):02d}",
                    "desc": f"{city} 已发出",
                },
                {"time": "当前", "desc": f"当前状态:{status}"},
            ],
        },
        ensure_ascii=False,
    )
```

- [ ] **Step 5: 运行测试,确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tools_random.py`
Expected: PASS(6 passed)

- [ ] **Step 6: 验证「跨进程」那条断言真的能区分错误实现**

把 `_rng` 里的 `hashlib.sha256(...)` 临时换成 `hash(...)`,重跑:

Run: `.venv/Scripts/python.exe -m pytest tests/test_tools_random.py::test_seed_is_stable_across_processes`
Expected: **FAIL**

确认失败后**改回 sha256**,再跑一次确认 PASS。把两次输出都记进任务报告 —— 这是「断言能区分正确与错误实现」的证据。

- [ ] **Step 7: 提交**

```bash
git add app/tools/__init__.py app/tools/errors.py app/tools/business.py tests/test_tools_random.py
git commit -m "feat: 三个确定性伪随机工具,种子用 sha256 保证跨进程稳定"
```

---

## Task 5: 两个 DB 工具、注册表

**Files:**
- Modify: `app/tools/business.py`(追加两个工厂)
- Create: `app/tools/registry.py`
- Test: `tests/test_tools_db.py`, `tests/test_registry.py`

**Interfaces:**
- Consumes: `ToolNotFound`(Task 4)、`app.db.models`(Task 2)、`get_sessionmaker`(Task 2)
- Produces:
  - `app.tools.business.make_query_faq(session) -> BaseTool`
  - `app.tools.business.make_create_ticket(session, conversation_id) -> BaseTool`
  - `app.tools.registry.build_tools(*, session, conversation_id) -> list[BaseTool]`
  - `app.tools.registry.registry_for(tools) -> dict[str, BaseTool]`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_registry.py`(纯逻辑,不需要 DB):

```python
"""注册表测试。用替身工具,不碰 DB。"""

import pytest
from langchain.tools import tool

from app.tools.registry import build_tools, registry_for


@tool
async def fake_a(x: str) -> str:
    """替身 A。"""
    return x


@tool
async def fake_b(x: str) -> str:
    """替身 B。"""
    return x


def test_registry_for_maps_name_to_tool():
    reg = registry_for([fake_a, fake_b])
    assert set(reg) == {"fake_a", "fake_b"}
    assert reg["fake_a"] is fake_a


def test_build_tools_includes_all_five_names():
    """五个工具的名字必须齐全 —— 少一个,模型就永远调不到它。"""
    tools = build_tools(session=None, conversation_id="s1")
    assert {t.name for t in tools} == {
        "query_order",
        "query_product",
        "query_logistics",
        "query_faq",
        "create_ticket",
    }
```

创建 `tests/test_tools_db.py`:

```python
"""两个 DB 工具的测试。需要 MySQL。"""

import asyncio
import json

import pytest
from sqlalchemy import select, text

from app.db.base import get_sessionmaker
from app.db.models import Conversation, Faq, Ticket
from app.tools.business import make_create_ticket, make_query_faq
from app.tools.errors import ToolNotFound

pytestmark = pytest.mark.db

SCRATCH_CONVERSATION = "tooltest000000000000000000000000"


def _call(tool, args: dict) -> str:
    tool_call = {"name": tool.name, "args": args, "id": "call_1", "type": "tool_call"}
    return asyncio.run(tool.ainvoke(tool_call)).content


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    async def _drop():
        async with get_sessionmaker()() as s:
            await s.execute(
                text("DELETE FROM tickets WHERE conversation_id = :c"),
                {"c": SCRATCH_CONVERSATION},
            )
            await s.execute(
                text("DELETE FROM conversations WHERE id = :c"),
                {"c": SCRATCH_CONVERSATION},
            )
            await s.commit()
    asyncio.run(_drop())


def test_query_faq_finds_seeded_row():
    from scripts.seed_db import seed

    asyncio.run(seed())

    async def run():
        async with get_sessionmaker()() as session:
            tool = make_query_faq(session)
            return await tool.ainvoke(
                {"name": "query_faq", "args": {"keyword": "退货"}, "id": "c", "type": "tool_call"}
            )

    hits = json.loads(asyncio.run(run()).content)
    assert hits["count"] >= 1
    assert any("退货" in item["question"] for item in hits["items"])


def test_query_faq_raises_not_found_for_unmatched_keyword():
    """验收 3 的漏召回路径:查不到要走 ToolNotFound,不是返回空列表假装成功。"""
    async def run():
        async with get_sessionmaker()() as session:
            tool = make_query_faq(session)
            return await tool.ainvoke(
                {"name": "query_faq", "args": {"keyword": "邮费"}, "id": "c", "type": "tool_call"}
            )

    with pytest.raises(ToolNotFound):
        asyncio.run(run())


def test_create_ticket_does_not_expose_conversation_id_to_model():
    """conversation_id 必须对模型不可见 —— 让模型填会编造 id。"""
    tool = make_create_ticket(session=None, conversation_id=SCRATCH_CONVERSATION)
    assert set(tool.args_schema.model_json_schema()["properties"]) == {
        "description",
        "ticket_type",
    }


def test_create_ticket_writes_row():
    async def run():
        async with get_sessionmaker()() as session:
            session.add(
                Conversation(id=SCRATCH_CONVERSATION, user="tester", status="active")
            )
            await session.commit()
            tool = make_create_ticket(
                session=session, conversation_id=SCRATCH_CONVERSATION
            )
            return await tool.ainvoke(
                {
                    "name": "create_ticket",
                    "args": {"description": "鞋码不对", "ticket_type": "换货"},
                    "id": "c",
                    "type": "tool_call",
                }
            )

    payload = json.loads(asyncio.run(run()).content)
    assert payload["ticket_no"]
    assert payload["conversation_id"] == SCRATCH_CONVERSATION

    async def check():
        async with get_sessionmaker()() as session:
            row = (
                await session.execute(
                    select(Ticket).where(Ticket.conversation_id == SCRATCH_CONVERSATION)
                )
            ).scalars().one()
            conv = (
                await session.execute(
                    select(Conversation).where(Conversation.id == SCRATCH_CONVERSATION)
                )
            ).scalars().one()
            return row.ticket_type, conv.status

    ticket_type, conv_status = asyncio.run(check())
    assert ticket_type == "换货"
    assert conv_status == "pending_human"   # 建单即转人工
```

- [ ] **Step 2: 运行测试,确认失败**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_registry.py
.venv/Scripts/python.exe -m pytest tests/test_tools_db.py
```
Expected: FAIL —— `make_query_faq` 不存在、`app.tools.registry` 不存在

- [ ] **Step 3: 在 `app/tools/business.py` 追加两个工厂**

```python
# ---- 以下两个工具需要数据库会话,故每请求构造 ----
#
# 为什么用闭包工厂而不是 InjectedToolArg:实测发现后者虽然能把参数从
# 发给模型的 schema 里隐藏(tool_call_schema 确实不含它),但调用时该值
# **必须另行注入**,而注入机制在 langchain-core 1.6.3 上没有现成文档,
# 直接调用会抛 ValidationError。闭包让参数**根本不在签名里**,模型既看
# 不见也传不错,且不依赖任何注入机制。

FAQ_LIMIT = 3


def make_query_faq(session):
    """构造 FAQ 查询工具。会话绑在闭包里,模型看不到。"""

    @tool
    async def query_faq(keyword: str) -> str:
        """查询常见问题库:退货政策、发票、物流规则等。用户问政策或规则类问题时使用。"""
        from sqlalchemy import or_, select

        from app.db.models import Faq

        cleaned = keyword.strip()
        if not cleaned:
            raise ToolNotFound("请提供要查询的关键词")

        # 关键词里的 % 与 _ 必须按字面匹配,否则 LIKE 会把它们当通配符:
        # "%" 能命中表里任意一行,于是这条查询**永远查得到**,返回 ok=true
        # 加三条与用户问题无关的答案,模型会照着它们自信作答 —— 漏召回这条
        # 防线(见下面的 ToolNotFound)就被从另一头绕过了。关键词由模型从
        # 用户原话里摘("100% 纯棉"这类),% 与 _ 会原样传进来。
        #
        # 反斜杠必须**第一个**替换:放后面会把它自己刚加进去的转义符再翻一倍。
        # 不用 contains(autoescape=True):它在 MySQL 上的渲染没实测过,而显式
        # 写法的语义毫无歧义。
        escaped = (
            cleaned.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        )
        pattern = f"%{escaped}%"
        rows = (
            (
                await session.execute(
                    select(Faq)
                    .where(
                        or_(
                            Faq.question.like(pattern, escape="\\"),
                            Faq.answer.like(pattern, escape="\\"),
                        )
                    )
                    .limit(FAQ_LIMIT)
                )
            )
            .scalars()
            .all()
        )
        if not rows:
            # 回显同样截断:这段文本会回灌进模型上下文(可恢复路径),而关键词
            # 是模型给的。理由与 _require_order_no 那处一致。
            raise ToolNotFound(
                f"常见问题库里没有与「{cleaned[:_ECHO_LIMIT]}」相关的内容,"
                f"请如实告知用户暂未收录,不要自行编造答案"
            )
        return json.dumps(
            {
                "keyword": cleaned,
                "count": len(rows),
                "items": [
                    {"question": r.question, "answer": r.answer, "category": r.category}
                    for r in rows
                ],
            },
            ensure_ascii=False,
        )

    return query_faq


def make_create_ticket(session, conversation_id: str):
    """构造建工单工具。conversation_id 绑在闭包里,模型看不到。

    非幂等写操作 —— executor 的重试白名单不含它,超时也绝不重试,
    否则会建出两张工单。
    """
    import secrets
    from datetime import datetime

    from sqlalchemy import select

    from app.db.models import Conversation, Ticket

    @tool
    async def create_ticket(description: str, ticket_type: str) -> str:
        """创建人工工单转交人工处理。用户明确要求人工介入、投诉或需人工核实时使用。"""
        cleaned = description.strip()
        if not cleaned:
            raise ToolNotFound("请描述需要人工处理的问题")

        ticket_no = f"T-{datetime.now():%Y%m%d%H%M%S}-{secrets.token_hex(2).upper()}"
        session.add(
            Ticket(
                ticket_no=ticket_no,
                conversation_id=conversation_id,
                description=cleaned,
                # 夹到列宽(String(64))而不是抛错:这是**写**路径,目的是把
                # 用户的问题留下来。超长在 MySQL 严格模式下抛 DataError,
                # T6 归类为不可恢复 → 502 且整单丢失 —— 一个被模型撑爆的
                # 标签字段不该毁掉 description 里真正的问题描述。
                ticket_type=ticket_type.strip()[:64] or "其他",
                status="open",
            )
        )
        # 建单即转人工 —— 否则 conversations.status 是死列。
        conversation = (
            await session.execute(
                select(Conversation).where(Conversation.id == conversation_id)
            )
        ).scalars().one_or_none()
        if conversation is not None:
            conversation.status = "pending_human"
        await session.commit()

        return json.dumps(
            {"ticket_no": ticket_no, "conversation_id": conversation_id, "status": "open"},
            ensure_ascii=False,
        )

    return create_ticket
```

- [ ] **Step 4: 实现 `app/tools/registry.py`**

```python
"""工具注册表。

因 create_ticket / query_faq 需要每请求构造(见 business.py 的说明),
注册表不是纯模块级常量 —— 每个请求用 build_tools 组装自己的工具集,
再由 registry_for 建名字到工具的映射。
"""

from langchain_core.tools import BaseTool

from app.tools.business import (
    make_create_ticket,
    make_query_faq,
    query_logistics,
    query_order,
    query_product,
)


def build_tools(*, session, conversation_id: str) -> list[BaseTool]:
    """组装本请求可用的五个工具。"""
    return [
        query_order,
        query_product,
        query_logistics,
        make_query_faq(session),
        make_create_ticket(session, conversation_id),
    ]


def registry_for(tools: list[BaseTool]) -> dict[str, BaseTool]:
    """建名字到工具的映射,供 executor 按模型给的名字查找。"""
    return {tool.name: tool for tool in tools}
```

- [ ] **Step 5: 运行测试,确认通过**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_registry.py
.venv/Scripts/python.exe -m pytest tests/test_tools_db.py
```
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add app/tools/business.py app/tools/registry.py tests/test_registry.py tests/test_tools_db.py
git commit -m "feat: query_faq 与 create_ticket 闭包工厂,工具注册表"
```

---

## Task 6: 工具执行器(超时 / 重试白名单 / 错误分类)

**Files:**
- Create: `app/tools/executor.py`
- Test: `tests/test_executor.py`

**Interfaces:**
- Consumes: `ToolNotFound` / `ToolInfrastructureError`(Task 4)、`Settings` 的三个工具配置项(Task 1)
- Produces: `app.tools.executor.ToolOutcome`(dataclass)、`app.tools.executor.execute_tool(*, tool_call, registry, settings) -> ToolOutcome`、`app.tools.executor.RETRYABLE_TOOLS`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_executor.py`:

```python
"""执行器测试。全部用替身工具,不联网、不碰 DB。"""

import asyncio

import pytest
from langchain.tools import tool
from pydantic import ValidationError
from sqlalchemy.exc import OperationalError

from app.config import Settings
from app.tools.errors import ToolInfrastructureError, ToolNotFound
from app.tools.executor import RETRYABLE_TOOLS, execute_tool

REQUIRED = {
    "openai_base_url": "https://example.invalid/v1",
    "openai_api_key": "sk-test",
    "openai_model": "test-model",
    "database_url": "mysql+asyncmy://u:p@h:3306/db",
}


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **REQUIRED, **overrides)


def _tc(name: str, args: dict) -> dict:
    return {"name": name, "args": args, "id": "call_1", "type": "tool_call"}


class _CountingTool:
    """计次壳,只透传 ainvoke,用来数"执行器尝试了几次"。

    计次必须在 ainvoke 这一层,**不能数工具体调用**:langchain 的参数校验由
    pydantic validate_arguments 包在工具体外层(StructuredTool.from_function
    → create_schema_from_function),参数不合法时工具体根本不进入 ——
    按工具体计数恒为 0,重试与否就区分不出来了。
    """

    def __init__(self, inner, counter: dict):
        self._inner = inner
        self._counter = counter

    async def ainvoke(self, tool_call):
        self._counter["n"] += 1
        return await self._inner.ainvoke(tool_call)


def test_retry_whitelist_excludes_create_ticket():
    """create_ticket 是写操作,重试会建出两张工单 —— 必须在白名单之外。"""
    assert "create_ticket" not in RETRYABLE_TOOLS
    assert RETRYABLE_TOOLS == {
        "query_order",
        "query_product",
        "query_logistics",
        "query_faq",
    }


@pytest.mark.anyio
async def test_unknown_tool_is_recoverable_not_fatal():
    outcome = await execute_tool(
        tool_call=_tc("nope", {}), registry={}, settings=_settings()
    )
    assert outcome.ok is False
    assert "nope" in outcome.content
    assert "不存在" in outcome.content


@pytest.mark.anyio
async def test_tool_not_found_is_recoverable():
    @tool
    async def query_order(order_id: str) -> str:
        """替身。"""
        raise ToolNotFound("未找到订单 9999")

    outcome = await execute_tool(
        tool_call=_tc("query_order", {"order_id": "9999"}),
        registry={"query_order": query_order},
        settings=_settings(),
    )
    assert outcome.ok is False
    assert "未找到订单" in outcome.content


@pytest.mark.anyio
async def test_validation_error_is_recoverable():
    @tool
    async def query_order(order_id: str) -> str:
        """替身。"""
        return "ok"

    outcome = await execute_tool(
        tool_call=_tc("query_order", {}),   # 缺必填参数
        registry={"query_order": query_order},
        settings=_settings(),
    )
    assert outcome.ok is False
    assert "ValidationError" in outcome.content or "参数" in outcome.content


@pytest.mark.anyio
async def test_timeout_is_recoverable_and_retries_whitelisted_tool():
    calls = {"n": 0}

    @tool
    async def query_order(order_id: str) -> str:
        """替身。"""
        calls["n"] += 1
        await asyncio.sleep(5)
        return "never"

    outcome = await execute_tool(
        tool_call=_tc("query_order", {"order_id": "1001"}),
        registry={"query_order": query_order},
        settings=_settings(tool_timeout_seconds=0.05, tool_retry_attempts=1,
                           tool_retry_delay_seconds=0.01),
    )
    assert outcome.ok is False
    assert "超时" in outcome.content
    assert calls["n"] == 2      # 首次 + 1 次重试


@pytest.mark.anyio
async def test_non_whitelisted_tool_is_never_retried():
    """create_ticket 超时后必须**恰好调用 1 次**。"""
    calls = {"n": 0}

    @tool
    async def create_ticket(description: str, ticket_type: str) -> str:
        """替身。"""
        calls["n"] += 1
        await asyncio.sleep(5)
        return "never"

    outcome = await execute_tool(
        tool_call=_tc("create_ticket", {"description": "换货", "ticket_type": "换货"}),
        registry={"create_ticket": create_ticket},
        settings=_settings(tool_timeout_seconds=0.05, tool_retry_attempts=1),
    )
    assert outcome.ok is False
    assert calls["n"] == 1


@pytest.mark.anyio
async def test_validation_error_does_not_retry():
    """参数错误重试无意义 —— 单轮下模型也没有第二次改参数的机会。"""
    calls = {"n": 0}

    @tool
    async def query_order(order_id: str) -> str:
        """替身。"""
        return "ok"

    await execute_tool(
        tool_call=_tc("query_order", {}),
        registry={"query_order": _CountingTool(query_order, calls)},
        settings=_settings(tool_retry_attempts=3),
    )
    assert calls["n"] == 1      # 首次即校验失败,不再重放


@pytest.mark.anyio
async def test_tool_not_found_does_not_retry():
    """业务性未找到是决定性结果 —— 重放同样的参数只会同样落空。

    FAQ 查不到是本章最常见的落空路径,重试白搭一次 DB 往返加
    tool_retry_delay_seconds 的等待,可恢复路径本该是最便宜的那条。
    """
    calls = {"n": 0}

    @tool
    async def query_faq(keyword: str) -> str:
        """替身。"""
        raise ToolNotFound("没有匹配的条目")

    outcome = await execute_tool(
        tool_call=_tc("query_faq", {"keyword": "邮费"}),
        registry={"query_faq": _CountingTool(query_faq, calls)},
        settings=_settings(tool_retry_attempts=3, tool_retry_delay_seconds=0.01),
    )
    assert outcome.ok is False
    assert "没有匹配的条目" in outcome.content
    assert calls["n"] == 1      # 确定性落空,不重放


@pytest.mark.anyio
async def test_database_error_is_fatal():
    """DB 故障不能被伪装成"你的订单号查不到"。"""
    @tool
    async def query_order(order_id: str) -> str:
        """替身。"""
        raise OperationalError("SELECT 1", {}, Exception("连接断开"))

    with pytest.raises(ToolInfrastructureError):
        await execute_tool(
            tool_call=_tc("query_order", {"order_id": "1001"}),
            registry={"query_order": query_order},
            settings=_settings(),
        )


@pytest.mark.anyio
async def test_unexpected_exception_is_fatal():
    """未预期异常按 spec §6.7 判不可恢复。"""
    @tool
    async def query_order(order_id: str) -> str:
        """替身。"""
        raise RuntimeError("没预料到的坏事")

    with pytest.raises(ToolInfrastructureError):
        await execute_tool(
            tool_call=_tc("query_order", {"order_id": "1001"}),
            registry={"query_order": query_order},
            settings=_settings(),
        )


@pytest.mark.anyio
async def test_summary_is_truncated_to_200_chars():
    @tool
    async def query_order(order_id: str) -> str:
        """替身。"""
        return "中" * 500

    outcome = await execute_tool(
        tool_call=_tc("query_order", {"order_id": "1001"}),
        registry={"query_order": query_order},
        settings=_settings(),
    )
    assert outcome.ok is True
    assert len(outcome.summary) == 201      # 200 字符 + 省略号
    assert len(outcome.content) == 500      # 回灌给模型的仍是完整内容
```

- [ ] **Step 2: 运行测试,确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_executor.py`
Expected: FAIL —— `ModuleNotFoundError: No module named 'app.tools.executor'`

- [ ] **Step 3: 实现**

创建 `app/tools/executor.py`:

```python
"""工具执行:超时、重试、错误分类。"""

import asyncio
import logging
from dataclasses import dataclass

from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from app.tools.errors import ToolInfrastructureError, ToolNotFound

logger = logging.getLogger(__name__)

# 重试用**白名单**:只有幂等的查询类工具可重试。
# create_ticket 是写操作,超时后重试会建出两张工单 —— 而"超时"恰恰意味着
# 我们不知道第一次到底成没成。默认不重试、显式声明可重试,比反过来安全。
RETRYABLE_TOOLS = frozenset(
    {"query_order", "query_product", "query_logistics", "query_faq"}
)

# tool_result 事件里给前端展示的摘要上限(spec §5.2)。
SUMMARY_MAX_CHARS = 200


@dataclass(frozen=True)
class ToolOutcome:
    tool_call_id: str
    name: str
    ok: bool
    content: str    # 完整内容,回灌给模型
    summary: str    # 截断后的展示用摘要


def _summarize(text: str) -> str:
    text = text.strip()
    if len(text) <= SUMMARY_MAX_CHARS:
        return text
    return text[:SUMMARY_MAX_CHARS] + "…"


async def execute_tool(*, tool_call: dict, registry: dict, settings) -> ToolOutcome:
    """执行一次工具调用。

    可恢复的失败返回 ok=False 的 ToolOutcome(调用方回灌给模型);
    基础设施故障抛 ToolInfrastructureError(调用方推 error 帧终止流)。
    """
    name = tool_call.get("name", "")
    tool_call_id = tool_call.get("id", "")

    tool = registry.get(name)
    if tool is None:
        message = (
            f"工具 {name} 不存在。可用工具:{', '.join(sorted(registry))}"
        )
        return ToolOutcome(tool_call_id, name, False, message, _summarize(message))

    attempts = 1 + (
        settings.tool_retry_attempts if name in RETRYABLE_TOOLS else 0
    )
    last_message = ""

    for attempt in range(attempts):
        try:
            message = await asyncio.wait_for(
                tool.ainvoke(tool_call), timeout=settings.tool_timeout_seconds
            )
            return ToolOutcome(
                tool_call_id, name, True, message.content, _summarize(message.content)
            )
        except TimeoutError:
            last_message = (
                f"工具 {name} 执行超时(超过 {settings.tool_timeout_seconds} 秒)"
            )
            logger.warning("工具 %s 超时,第 %d/%d 次尝试", name, attempt + 1, attempts)
            if attempt + 1 < attempts:
                await asyncio.sleep(settings.tool_retry_delay_seconds)
        except ValidationError as exc:
            # 参数不合 schema。重试无意义 —— 同一个工具调用重放一次还是同样的参数。
            last_message = f"工具 {name} 的参数不合法:{exc}"
            logger.warning("工具 %s 参数校验失败:%s", name, exc)
            break
        except ToolNotFound as exc:
            # 业务性未找到是决定性结果:重放同一个 tool_call 送的是同样的参数,
            # 只会同样落空。而且这是本章最常见的落空路径(FAQ 查不到),重试
            # 白搭一次 DB 往返加 tool_retry_delay_seconds 的等待 —— 可恢复路径
            # 本该是最便宜的那条。
            last_message = str(exc)
            break
        except SQLAlchemyError as exc:
            logger.exception("工具 %s 命中数据库故障", name)
            raise ToolInfrastructureError("数据服务暂时不可用") from exc
        except Exception as exc:
            logger.exception("工具 %s 抛出未预期异常", name)
            raise ToolInfrastructureError("工具执行失败") from exc

    return ToolOutcome(tool_call_id, name, False, last_message, _summarize(last_message))
```

- [ ] **Step 4: 运行测试,确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_executor.py`
Expected: PASS(10 passed)

- [ ] **Step 5: 验证三条关键断言真的能区分错误实现**

1. 把 `RETRYABLE_TOOLS` 改成包含 `"create_ticket"`,重跑
   `test_non_whitelisted_tool_is_never_retried` → 必须 **FAIL**。
2. 改回后,把 `except ValidationError` 分支里的 `break` 去掉,重跑
   `test_validation_error_does_not_retry` → 必须 **FAIL**。
3. 改回后,把 `except ToolNotFound` 分支里的 `break` 去掉,重跑
   `test_tool_not_found_does_not_retry` → 必须 **FAIL**
   (计次壳在 `ainvoke` 边界,`tool_retry_attempts=3` 故应报 `assert 4 == 1`)。

三次都确认后改回正确实现,把输出记进任务报告。

**为什么计次不能写在工具体里**:`@tool` 把函数体包在 pydantic `validate_arguments`
之下(`StructuredTool.from_function` → `create_schema_from_function`),参数不合法时
**工具体根本不进入** —— 按工具体计数在正确实现与错误实现下**都是 0**,
`assert calls["n"] == 1` 对任何实现都是红的。计次必须落在 `ainvoke` 边界。
(这条是本章实现过程中实测得出的,见 `dev-notes/ch02.md`。)

- [ ] **Step 6: 提交**

```bash
git add app/tools/executor.py tests/test_executor.py
git commit -m "feat: 工具执行器,超时重试白名单与错误分类"
```

---

## Task 7: `Message` 扩展与裁剪分轮规则改为 user 边界

**Files:**
- Modify: `app/schemas.py`
- Modify: `app/memory/trim.py`
- Modify: `app/prompts.py`(`_to_lc_message` → `to_lc_messages`,支持 tool 角色)
- Test: `tests/test_trim.py`(追加), `tests/test_prompts.py`(追加)

**Interfaces:**
- Consumes: 无
- Produces:
  - `app.schemas.Message(role: Literal["user","assistant","tool"], content: str = "", tool_calls: list[dict] | None = None, tool_call_id: str | None = None)`
  - `app.prompts.to_lc_messages(history: Sequence[Message]) -> list`
  - `app.memory.trim.select_history(history, available_tokens)` 行为变更(分轮规则)

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_trim.py`:

```python
def _pairing_intact(messages) -> bool:
    """每条 tool 消息前面都必须有带对应 tool_call_id 的 assistant 消息。"""
    pending: set[str] = set()
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            pending |= {tc["id"] for tc in m.tool_calls}
        elif m.role == "tool":
            if m.tool_call_id not in pending:
                return False
    return True


def test_select_history_never_separates_tool_from_its_assistant():
    """裁剪不能把 tool 消息与它的 assistant 父亲切开。

    OpenAI 兼容 API 要求 tool 消息前面必须紧跟着带对应 tool_call_id 的
    assistant 消息,切开就会 400 —— 而且只在历史长到触发裁剪时偶发。

    **注意这条断言区分不出旧规则**(实测已推翻本计划原先「必然失败」的说法):
    旧规则把一轮工具往返切成 [user, assistant(tool_calls)] 与 [tool, assistant]
    两半,而本用例的预算恰好**同时够得着这两半** —— 丢弃线落在两者的共同边界上,
    于是新旧两条规则输出逐字节相同,切开确实发生了、但这条测试看不见。
    真正钉住旧规则的是紧接着的那条
    test_select_history_drops_a_whole_tool_round_instead_of_its_tail。
    """
    old_call = {"id": "call_old", "name": "query_order", "args": {"order_id": "9001"}}
    new_call = {"id": "call_new", "name": "query_logistics", "args": {"order_id": "1001"}}
    history = [
        Message(role="user", content="很早的问题" * 60),
        Message(role="assistant", content="", tool_calls=[old_call]),
        Message(role="tool", content="很老的工具结果" * 60, tool_call_id="call_old"),
        Message(role="assistant", content="很早的回答" * 60),
        Message(role="user", content="订单 1001 的物流到哪了"),
        Message(role="assistant", content="", tool_calls=[new_call]),
        Message(role="tool", content="已揽件", tool_call_id="call_new"),
        Message(role="assistant", content="您的包裹已揽件。"),
    ]
    kept = select_history(history, available_tokens=40)

    assert _pairing_intact(kept)
    assert [m.role for m in kept] == ["user", "assistant", "tool", "assistant"]


def test_select_history_drops_a_whole_tool_round_instead_of_its_tail():
    """裁剪线落在 tool 与它的 assistant 父亲之间时,必须整轮丢掉。

    这是上一条测试没能覆盖到的情形:只有当预算"够得着后半段、够不着前半段"
    时,旧规则(遇 assistant 收轮)才会真的留下以 tool 打头的残缺序列。

    触发条件是"用户消息贵、工具往返便宜" —— 用户提问越大段,越容易命中;
    这也正是它只在长对话里偶发的原因。
    """
    call = {"id": "call_1", "name": "query_logistics", "args": {"order_id": "1001"}}
    history = [
        Message(role="user", content="很早的问题" * 60),
        Message(role="assistant", content="很早的回答" * 60),
        # 这一轮的用户消息很贵,工具往返很便宜
        Message(role="user", content="订单 1001 的物流到哪了" * 30),
        Message(role="assistant", content="", tool_calls=[call]),
        Message(role="tool", content="已揽件", tool_call_id="call_1"),
        Message(role="assistant", content="您的包裹已揽件。"),
        # 最新一轮很便宜,裁剪后应当只剩它
        Message(role="user", content="那什么时候到"),
        Message(role="assistant", content="预计明天送达。"),
    ]

    kept = select_history(history, available_tokens=40)

    assert _pairing_intact(kept)
    assert [m.role for m in kept] == ["user", "assistant"]


def test_round_definition_is_user_delimited():
    """一轮 = 从一条 user 开始,到(不含)下一条 user 为止。"""
    history = [
        Message(role="user", content="q1"),
        Message(role="assistant", content="", tool_calls=[{"id": "c1", "name": "t", "args": {}}]),
        Message(role="tool", content="r1", tool_call_id="c1"),
        Message(role="assistant", content="a1"),
        Message(role="user", content="q2"),
        Message(role="assistant", content="a2"),
    ]
    rounds = _to_rounds(history)
    assert len(rounds) == 2
    assert [m.role for m in rounds[0]] == ["user", "assistant", "tool", "assistant"]
    assert [m.role for m in rounds[1]] == ["user", "assistant"]
```

追加到 `tests/test_prompts.py`:

```python
def test_to_lc_messages_handles_tool_role():
    from langchain.messages import AIMessage, HumanMessage, ToolMessage

    history = [
        Message(role="user", content="订单 1001 到哪了"),
        Message(role="assistant", content="", tool_calls=[{"id": "c1", "name": "query_logistics", "args": {"order_id": "1001"}}]),
        Message(role="tool", content="已揽件", tool_call_id="c1"),
        Message(role="assistant", content="已揽件。"),
    ]
    converted = to_lc_messages(history)
    assert isinstance(converted[0], HumanMessage)
    assert isinstance(converted[1], AIMessage)
    assert converted[1].tool_calls[0]["id"] == "c1"
    assert isinstance(converted[2], ToolMessage)
    assert converted[2].tool_call_id == "c1"
```

- [ ] **Step 2: 运行测试,确认失败**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_trim.py tests/test_prompts.py
```
Expected: FAIL —— `Message` 不接受 `role="tool"` / `to_lc_messages` 不存在 / `_to_rounds` 行为不符

- [ ] **Step 3: 改 `app/schemas.py`**

```python
class Message(BaseModel):
    """会话历史中的一条消息。纯数据,不依赖 LangChain。

    role 含 "tool" 是因为模型可能要求调用工具 —— 那一轮的历史由
    assistant(带 tool_calls)+ tool(带 tool_call_id)两条构成。
    """

    role: Literal["user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
```

- [ ] **Step 4: 改 `app/memory/trim.py` 的 `_to_rounds`**

```python
def _to_rounds(history: Sequence[Message]) -> list[list[Message]]:
    """把消息序列切成整轮。

    一轮 = **从一条 user 消息开始,到(不含)下一条 user 消息为止**。

    为什么不用"遇到 assistant 就收一轮"(ch01 的旧规则):引入 tool 角色后,
    后者会把 tool 消息与它的 assistant 父亲切到不同轮里。OpenAI 兼容 API
    要求 tool 消息前面必须紧跟着带对应 tool_call_id 的 assistant 消息,
    切开就会 400,且只在历史长到触发裁剪时复现,极难定位。
    """
    rounds: list[list[Message]] = []
    for msg in history:
        if msg.role == "user" or not rounds:
            rounds.append([msg])
        else:
            rounds[-1].append(msg)
    return rounds
```

`select_history` 本体不变(仍是整轮累加 + `break`)。

- [ ] **Step 5: 改 `app/prompts.py`**

把 `_to_lc_message` 换为公开的 `to_lc_messages`,并让 `build_messages` 使用它:

```python
from collections.abc import Sequence

from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.schemas import Message


def to_lc_messages(history: Sequence[Message]) -> list:
    """把纯数据 Message 转成 LangChain 消息。本模块是唯一的转换点。

    role="assistant" 且带 tool_calls 时,content 通常是空串 —— 这在
    上游是合法的(模型只申请调用、还没产出文字)。
    """
    converted = []
    for message in history:
        if message.role == "user":
            converted.append(HumanMessage(message.content))
        elif message.role == "tool":
            converted.append(
                ToolMessage(
                    content=message.content,
                    tool_call_id=message.tool_call_id or "",
                )
            )
        else:
            converted.append(
                AIMessage(
                    content=message.content,
                    tool_calls=message.tool_calls or [],
                )
            )
    return converted
```

`build_messages` 内的 `[_to_lc_message(m) for m in history]` 改为 `to_lc_messages(history)`。
`render_system_prompt` 与 `build_extract_messages` 不变。

- [ ] **Step 6: 运行测试,确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_trim.py tests/test_prompts.py`
Expected: PASS

- [ ] **Step 7: 验证「tool 配对」那条断言真能区分错误实现**

把 `_to_rounds` 临时改回 ch01 的旧规则(`pending.append(msg)`,遇 assistant 收轮,末尾 `if pending` 补轮),重跑:

Run: `.venv/Scripts/python.exe -m pytest tests/test_trim.py -m ""`
Expected: **`test_select_history_drops_a_whole_tool_round_instead_of_its_tail` 与 `test_round_definition_is_user_delimited` FAIL**

**注意验证的是这两条,不是 `test_select_history_never_separates_tool_from_its_assistant`。**
后者在本计划最初被写成「旧规则下必然失败」,实现时实测**推翻**了:它的预算
(`available_tokens=40`)恰好同时够得着被切开的那对的两半,丢弃线落在共同边界上,
新旧规则输出逐字节相同 —— 切开确实发生,只是那条测试看不见。它的 docstring 已
如实订正。**这条经验对 Tasks 9/10 同样适用**:一条「看起来在守护某行为」的断言,
必须实测它在错误实现下会红,否则它只是在执行代码路径。

- [ ] **Step 8: 提交**

```bash
git add app/schemas.py app/memory/trim.py app/prompts.py tests/test_trim.py tests/test_prompts.py
git commit -m "feat: Message 支持 tool 角色,裁剪分轮改为 user 边界"
```

---

## Task 8: `SessionStore` 瘦身为锁注册表

**Files:**
- Modify: `app/memory/store.py`
- Test: `tests/test_store.py`(**删除**已退役行为的测试,改写锁相关测试)

**Interfaces:**
- Consumes: 无
- Produces: `SessionStore(*, ttl_seconds: float, max_sessions: int)`,公开方法 `lock_for(session_id) -> asyncio.Lock` 与 `active_lock_count() -> int`(替代 ch01 的 `history` / `append` / `active_session_count`)

**背景**:历史迁到 MySQL 后,`_sessions`、`history()`、`append()`、`_enforce_capacity` 对历史的作用全部退役。`MAX_SESSIONS` 改用于**限制锁表**(用户裁决,见 spec §6.4 订正),ch01 spec §9 的「锁表无硬数量上限」风险由此关闭。

- [ ] **Step 1: 先删除已退役行为的测试**

打开 `tests/test_store.py`,删除针对以下**已不存在的行为**的测试:

- 会话历史的读写(`append` / `history` 相关)
- TTL 过期**淘汰历史**
- LRU 容量上限作用于**历史**(`max_sessions` 淘汰 `_sessions`)
- `active_session_count`

**不要保留空壳**。ch01 复盘明确反对「为了绿而绿」的测试 —— 被测对象没了,测试就该删。

**保留并保留原样**:锁的串行化、同会话并发不丢消息、`lock_for` 幂等、被持锁的会话不被 TTL 清扫。

- [ ] **Step 2: 写新测试**

追加到 `tests/test_store.py`:

```python
def test_lock_for_refreshes_recency():
    """lock_for 必须把条目移到 LRU 末尾。

    否则淘汰的是插入序而非最近使用序,刚建的锁会被优先选中,
    破坏「同一 session 两次 lock_for 返回同一把锁」的幂等性
    —— ch01 Task 4 已经在这上面栽过一次。

    断言用**持有的引用**比对而非 `is not None`:后者恒真,区分不了
    正确与错误实现(去掉 _touch 里的 move_to_end,它照样通过)。
    """
    store = SessionStore(ttl_seconds=600, max_sessions=3)
    lock_s1 = store.lock_for("s1")
    lock_s2 = store.lock_for("s2")
    store.lock_for("s3")
    store.lock_for("s1")            # s1 变成最近使用
    store.lock_for("s4")            # 触发淘汰:应淘汰 s2,而不是 s1

    assert store.lock_for("s1") is lock_s1      # s1 存活
    assert store.lock_for("s2") is not lock_s2  # s2 已被淘汰(插入序下会淘汰 s1)


def test_capacity_bounds_the_lock_table():
    """MAX_SESSIONS 现在约束的是锁表 —— ch01 那条遗留风险在本章关闭。"""
    store = SessionStore(ttl_seconds=600, max_sessions=3)
    for i in range(20):
        store.lock_for(f"s{i}")
    assert store.active_lock_count() <= 3


def test_lock_for_is_idempotent_under_capacity_pressure():
    store = SessionStore(ttl_seconds=600, max_sessions=1)
    first = store.lock_for("s1")
    second = store.lock_for("s1")
    assert first is second


@pytest.mark.anyio
async def test_capacity_never_evicts_a_held_lock():
    """淘汰遇到被持锁的必须**整个停下**,不跳过。

    跳过会去删比它更新的条目,把 LRU 语义弄反 —— ch01 自审抓到过这个 bug。
    """
    store = SessionStore(ttl_seconds=600, max_sessions=2)
    held = store.lock_for("oldest")
    await held.acquire()
    for i in range(5):
        store.lock_for(f"new{i}")

    # 顺序**不能反**。`lock_for("oldest")` 自身有副作用:它把 oldest 移到 LRU 末尾,
    # 于是它不再是队首、不再挡住淘汰,后续淘汰会一路删到容量上限,count 掉回 2。
    # 先断言身份再断言数量,在正确实现下必然失败 —— 而插入序(无 move_to_end)的
    # 错误实现反而通过。实测两个方向都验过。
    assert store.active_lock_count() > 2             # 宁可短暂超容量
    assert store.lock_for("oldest") is held          # 持锁条目仍在
```

- [ ] **Step 3: 运行测试,确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_store.py`
Expected: FAIL —— `active_lock_count` 不存在;容量测试不通过(旧实现不限制锁表)

- [ ] **Step 4: 重写 `app/memory/store.py`**

```python
import asyncio
import time
from collections import OrderedDict


class SessionStore:
    """每会话互斥锁的注册表。

    ch01 里它还管进程内会话历史(TTL + LRU 淘汰);ch02 历史迁到 MySQL 后
    那部分退役,只剩锁与清扫。

    `max_sessions` 因此**改为限制锁表大小** —— 它当初唯一的作用是限制
    历史条数,历史一走就成了死配置(读 .env.example 的人会以为它在管事)。
    这也顺带关闭了 ch01 spec §9 记录的「_locks/_touched 无硬数量上限」风险。

    锁仍然必要:同会话的并发请求会各自读到同一份历史、各自追加,
    不串行化就会后写覆盖先写。锁在进程内,历史在 MySQL —— 两者正交。

    多进程 / 多 worker 部署下本锁失效,本章不在范围内。
    """

    def __init__(self, *, ttl_seconds: float, max_sessions: int) -> None:
        self._ttl = ttl_seconds
        self._max = max_sessions
        # OrderedDict 而非普通 dict:淘汰要按 LRU 序,不能按插入序。
        self._touched: OrderedDict[str, float] = OrderedDict()
        self._locks: dict[str, asyncio.Lock] = {}

    # ---- 公开 ----

    def lock_for(self, session_id: str) -> asyncio.Lock:
        """取该会话的锁,按需创建。

        同时刷新使用时间并移到 LRU 末尾 —— 刚建的锁因此不会被紧接着的
        容量淘汰选中,幂等性得以保持。
        """
        self._purge()
        lock = self._locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session_id] = lock
        self._touch(session_id)
        self._enforce_capacity()
        return lock

    def active_lock_count(self) -> int:
        self._purge()
        return len(self._locks)

    # ---- 内部 ----

    def _touch(self, session_id: str) -> None:
        self._touched[session_id] = time.monotonic()
        self._touched.move_to_end(session_id)

    def _is_locked(self, session_id: str) -> bool:
        lock = self._locks.get(session_id)
        return lock is not None and lock.locked()

    def _drop(self, session_id: str) -> None:
        self._touched.pop(session_id, None)
        self._locks.pop(session_id, None)

    def _purge(self) -> None:
        """清扫超过 TTL 且未被持有的条目。持锁的不动 —— 它正在流式。"""
        now = time.monotonic()
        for session_id in list(self._touched.keys()):
            if now - self._touched[session_id] <= self._ttl:
                continue
            if self._is_locked(session_id):
                continue
            self._drop(session_id)

    def _enforce_capacity(self) -> None:
        """超容量时淘汰最久未使用的条目。

        遇到被持锁的**整个停下**,而不是跳过它去淘汰更新的 ——
        跳过会删掉比它更新的条目,把 LRU 语义彻底弄反。
        代价:极端情况下会短暂超出 max_sessions,上界仍由"同时进行中的
        流数量"兜住。

        淘汰一个**未被持有**的锁是安全的:没有持锁者就不存在被破坏的
        互斥,后续请求会拿到一把全新的、未锁定的锁。
        """
        while len(self._touched) > self._max:
            oldest = next(iter(self._touched))
            if self._is_locked(oldest):
                return
            self._drop(oldest)
```

- [ ] **Step 5: 运行测试,确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_store.py`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add app/memory/store.py tests/test_store.py
git commit -m "refactor: SessionStore 瘦身为锁注册表,MAX_SESSIONS 改限锁表"
```

---

## Task 9: 历史服务(读写 MySQL)

**Files:**
- Create: `app/services/history.py`
- Test: `tests/test_history.py`

**Interfaces:**
- Consumes: `app.db.models`(Task 2)、`app.schemas.Message`(Task 7)
- Produces:
  - `app.services.history.ensure_conversation(*, session, session_id, user_id) -> Conversation`
  - `app.services.history.load_history(*, session, conversation_id) -> list[Message]`
  - `app.services.history.append_turn(*, session, conversation_id, messages) -> None`

- [ ] **Step 1: 写失败测试**
创建 `tests/test_history.py`:


```python
"""历史服务测试。需要 MySQL。"""

import asyncio

import pytest
from sqlalchemy import select, text

from app.db.base import get_sessionmaker
from app.db.models import Conversation
from app.schemas import Message
from app.services.history import append_turn, ensure_conversation, load_history

pytestmark = pytest.mark.db

SCRATCH = "histtest000000000000000000000000"


def asyncio_run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _cleanup():
    async def _drop():
        async with get_sessionmaker()() as session:
            await session.execute(
                text("DELETE FROM messages WHERE conversation_id = :c"), {"c": SCRATCH}
            )
            await session.execute(
                text("DELETE FROM conversations WHERE id = :c"), {"c": SCRATCH}
            )
            await session.commit()

    asyncio_run(_drop())
    yield
    asyncio_run(_drop())


def test_ensure_conversation_creates_then_reuses():
    async def run():
        async with get_sessionmaker()() as session:
            first = await ensure_conversation(
                session=session, session_id=SCRATCH, user_id="alice"
            )
            assert first.user == "alice"
            assert first.status == "active"
            # 已存在时忽略新的 user_id,以创建时为准
            again = await ensure_conversation(
                session=session, session_id=SCRATCH, user_id="mallory"
            )
            assert again.user == "alice"

        # 换一个**全新 session** 重查,断的是行里存的值。
        # 同一个 session 里读会被身份映射兜住(expire_on_commit=False,二次
        # select 不会用行覆盖已加载的属性),那样只证明"内存对象没被改过",
        # 证明不了"库里还是 alice" —— 回写若用 UPDATE 实现,照样全绿。
        # Ruling 1 是安全裁决,必须钉在行上。
        async with get_sessionmaker()() as session:
            stored = (
                await session.execute(
                    select(Conversation).where(Conversation.id == SCRATCH)
                )
            ).scalars().one()
            return stored.user

    assert asyncio_run(run()) == "alice"


def test_append_turn_then_load_history_roundtrips_tool_messages():
    tool_calls = [{"id": "c1", "name": "query_logistics", "args": {"order_id": "1001"}}]

    async def run():
        async with get_sessionmaker()() as session:
            await ensure_conversation(session=session, session_id=SCRATCH, user_id="u")
            await append_turn(
                session=session,
                conversation_id=SCRATCH,
                messages=[
                    Message(role="user", content="订单 1001 到哪了"),
                    Message(role="assistant", content="", tool_calls=tool_calls),
                    Message(role="tool", content="已揽件", tool_call_id="c1"),
                    Message(role="assistant", content="您的包裹已揽件。"),
                ],
            )
        # 另开 session 读回。同一 session 里读到的值**未必**来自 MySQL:身份映射
        # 持**弱**引用,只要还有东西引用那几个 ORM 实例,select 就命中缓存、直接把
        # 原来那个 Python list 交回来 —— 实测把行改成坏值时断言照样通过;而这个
        # 引用是"恰好"消失的(原实现没引用返回对象),等于一条往返靠 refcount 走运。
        # 这条往返是 Ruling 6 明令要钉的,不能建立在"恰好没人引用"上。
        async with get_sessionmaker()() as session:
            return await load_history(session=session, conversation_id=SCRATCH)

    history = asyncio_run(run())
    assert [m.role for m in history] == ["user", "assistant", "tool", "assistant"]
    assert history[1].tool_calls[0]["id"] == "c1"
    # 深层结构整体比对:只断言某字段的话,"args 内层字典被拍平成字符串"
    # 这类有损往返照样全绿。
    assert history[1].tool_calls == tool_calls
    assert history[2].tool_call_id == "c1"
    assert history[2].content == "已揽件"              # 中文往返


def test_load_history_returns_chronological_order():
    async def run():
        async with get_sessionmaker()() as session:
            await ensure_conversation(session=session, session_id=SCRATCH, user_id="u")
            await append_turn(
                session=session, conversation_id=SCRATCH,
                messages=[Message(role="user", content="第一句")],
            )
            await append_turn(
                session=session, conversation_id=SCRATCH,
                messages=[Message(role="user", content="第二句")],
            )
            return await load_history(session=session, conversation_id=SCRATCH)

    contents = [m.content for m in asyncio_run(run())]
    assert contents == ["第一句", "第二句"]


def test_load_history_orders_by_id_when_created_at_disagrees():
    """id 序 ≠ created_at 序时,必须按 id(插入序)返回。

    上一条顺序用例**区分不出两种实现**:两条消息在同一秒内写入,created_at
    打平,MySQL 恰好按插入序返回 —— 把 order_by(MessageRecord.id) 换成
    created_at 它依然全绿(已实测,7 个变异里它是唯一活下来的那个)。
    这里把 created_at 人为倒挂,让「按 id」与「按时间戳」给出相反结果,
    那条不变式才真正被钉住。
    """

    async def run():
        async with get_sessionmaker()() as session:
            await ensure_conversation(session=session, session_id=SCRATCH, user_id="u")
            # 同一批写入,但先写的那条 created_at 更晚:id 序与时间戳序相反。
            await session.execute(
                text(
                    "INSERT INTO messages (conversation_id, role, content, created_at) "
                    "VALUES (:c, 'user', '先写的', '2999-01-01 00:00:00'),"
                    "       (:c, 'user', '后写的', '2000-01-01 00:00:00')"
                ),
                {"c": SCRATCH},
            )
            await session.commit()
            return await load_history(session=session, conversation_id=SCRATCH)

    assert [m.content for m in asyncio_run(run())] == ["先写的", "后写的"]
```

**注**:`SCRATCH` 必须恰好 **32 字符** —— `conversations.id` 是 `varchar(32)`,
在 MySQL 严格模式下超长会抛 `DataError 1406`,断言根本到不了。
(本章此前的计划文本在这里写成 33 字符,是错的。)

- [ ] **Step 2: 运行测试,确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_history.py`
Expected: FAIL —— `ModuleNotFoundError: No module named 'app.services.history'`

- [ ] **Step 3: 实现**

创建 `app/services/history.py`:

```python
"""会话历史的读写。只做 DB I/O —— Message -> BaseMessage 的转换在 prompts.py。"""

from collections.abc import Sequence

from sqlalchemy import select

from app.db.models import Conversation, MessageRecord
from app.schemas import Message


async def ensure_conversation(*, session, session_id: str, user_id: str) -> Conversation:
    """取会话,不存在则新建。

    已存在时**忽略传入的 user_id**,以创建时记录的为准 —— 否则任何客户端
    都能改掉一条会话的归属(本章端点没有认证)。
    """
    conversation = (
        await session.execute(
            select(Conversation).where(Conversation.id == session_id)
        )
    ).scalars().one_or_none()

    if conversation is None:
        conversation = Conversation(id=session_id, user=user_id, status="active")
        session.add(conversation)
        await session.commit()
    return conversation


async def load_history(*, session, conversation_id: str) -> list[Message]:
    """按时间正序读出该会话的全部消息。"""
    rows = (
        await session.execute(
            select(MessageRecord)
            .where(MessageRecord.conversation_id == conversation_id)
            .order_by(MessageRecord.id)
        )
    ).scalars().all()

    return [
        Message(
            role=row.role,
            content=row.content,
            tool_calls=row.tool_calls,
            tool_call_id=row.tool_call_id,
        )
        for row in rows
    ]


async def append_turn(
    *, session, conversation_id: str, messages: Sequence[Message]
) -> None:
    """把一轮的消息一次性写入。

    调用方只在**流完整走完后**才调它 —— 与 ch01「半截回复不污染历史」
    的语义一致,也避免留下"有问无答"的孤儿行。
    """
    for message in messages:
        session.add(
            MessageRecord(
                conversation_id=conversation_id,
                role=message.role,
                content=message.content,
                tool_calls=message.tool_calls,
                tool_call_id=message.tool_call_id,
            )
        )
    await session.commit()
```

- [ ] **Step 4: 运行测试,确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_history.py`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add app/services/history.py tests/test_history.py
git commit -m "feat: 历史服务,读写 messages 表并保全工具调用信息"
```

---

## Task 10: 单轮工具编排

**Files:**
- Modify: `app/services/chat.py`
- Test: `tests/test_chat_service.py`(改写)

**Interfaces:**
- Consumes: `prompts.to_lc_messages` / `build_messages` / `render_system_prompt`(Task 7)、`trim`(Task 7)、`executor.execute_tool`(Task 6)、`history.append_turn`(Task 9)
- Produces:
  - `app.services.chat.prepare_turn(*, settings, history, user_input) -> list`(**签名变更**:不再收 store,改为收 history)
  - `app.services.chat.stream_turn(*, settings, model, session, conversation_id, user_input, messages, tools, registry) -> AsyncIterator[tuple[str, dict]]`

- [ ] **Step 1: 写失败测试**

改写 `tests/test_chat_service.py`,保留原有的预算/溢出测试并适配新签名,追加编排测试:

```python
"""对话编排测试。全部用替身,不联网、不碰 DB。"""

import asyncio

import pytest
from langchain.tools import tool

from app.config import Settings
from app.schemas import Message
from app.services.chat import prepare_turn, stream_turn
from app.tools.errors import ToolInfrastructureError

REQUIRED = {
    "openai_base_url": "https://example.invalid/v1",
    "openai_api_key": "sk-test",
    "openai_model": "test-model",
    "database_url": "mysql+asyncmy://u:p@h:3306/db",
}


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **REQUIRED, **overrides)


class FakeChunk:
    """模拟 AIMessageChunk:支持 + 累加,累加后携带 tool_calls。"""

    def __init__(self, text="", tool_calls=None, usage=None):
        self.text = text
        self.tool_calls = list(tool_calls or [])
        self.usage_metadata = usage

    def __add__(self, other):
        return FakeChunk(
            text=self.text + other.text,
            tool_calls=self.tool_calls + other.tool_calls,
            usage=other.usage_metadata or self.usage_metadata,
        )


class _BoundModel:
    def __init__(self, inner):
        self._inner = inner

    async def astream(self, messages):
        self._inner.calls.append(("bound", list(messages)))
        for chunk in self._inner.batches.pop(0):
            yield chunk


class ScriptedModel:
    """按顺序回放预置 chunk 批次的替身。

    记录每次 astream 走的是**绑了工具**还是**未绑工具**的入口 ——
    这正是"只做单轮"的结构保证所在,必须有断言钉住。
    """

    def __init__(self, batches):
        self.batches = list(batches)
        self.calls = []
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return _BoundModel(self)

    async def astream(self, messages):
        self.calls.append(("unbound", list(messages)))
        for chunk in self.batches.pop(0):
            yield chunk


@tool
async def query_logistics(order_id: str) -> str:
    """替身:查物流。"""
    return '{"status": "已揽件"}'


@tool
async def create_ticket(description: str, ticket_type: str) -> str:
    """替身:建工单。"""
    return '{"ticket_no": "T-1"}'


class RecordingSession:
    """记录落库内容的替身 DB 会话。"""

    def __init__(self):
        self.appended = []

    def add(self, obj):        # 供 MessageRecord 构造期调用,此处不关心
        pass

    async def commit(self):
        pass


def _collect(model, session, registry, tools=None):
    async def run():
        return [
            event
            async for event in stream_turn(
                settings=_settings(),
                model=model,
                session=session,
                conversation_id="s1",
                user_input="订单 1001 的物流到哪了",
                messages=[Message(role="user", content="订单 1001 的物流到哪了")],
                tools=tools if tools is not None else list(registry.values()),
                registry=registry,
            )
        ]

    return asyncio.run(run())


# ---------- 单轮结构保证 ----------

def test_second_round_uses_unbound_model():
    """**本章最关键的一条结构断言。**

    「只做单轮」不能靠提示词求模型自觉,必须靠第二轮不绑 tools。
    谁把第二轮改成 model_with_tools,这条就挂。
    """
    model = ScriptedModel(
        [
            [FakeChunk(tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}])],
            [FakeChunk("已揽件。")],
        ]
    )
    _collect(model, RecordingSession(), {"query_logistics": query_logistics})

    assert [kind for kind, _ in model.calls] == ["bound", "unbound"]


def test_no_tool_call_means_single_api_call():
    """不调工具时只发一次请求,且文本已经流式推出。"""
    model = ScriptedModel([[FakeChunk("您"), FakeChunk("好")]])
    events = _collect(model, RecordingSession(), {})

    assert len(model.calls) == 1
    assert events[0] == ("token", {"text": "您"})
    assert events[-1][0] == "done"


def test_tool_event_order_is_call_then_result_then_answer():
    model = ScriptedModel(
        [
            [FakeChunk(tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}])],
            [FakeChunk("包裹"), FakeChunk("已揽件。")],
        ]
    )
    events = _collect(model, RecordingSession(), {"query_logistics": query_logistics})
    kinds = [kind for kind, _ in events]

    assert kinds == ["tool_call", "tool_result", "token", "token", "done"]
    assert events[0][1]["name"] == "query_logistics"
    assert events[0][1]["tool_call_id"] == "c1"
    assert events[1][1]["ok"] is True


def test_tool_result_is_fed_back_as_tool_message():
    model = ScriptedModel(
        [
            [FakeChunk(tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}])],
            [FakeChunk("已揽件。")],
        ]
    )
    _collect(model, RecordingSession(), {"query_logistics": query_logistics})

    second_round_messages = model.calls[1][1]
    from langchain.messages import ToolMessage

    tool_messages = [m for m in second_round_messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "c1"
    assert "已揽件" in tool_messages[0].content


# ---------- 错误分类 ----------

def test_recoverable_tool_failure_still_converges():
    """可恢复失败(查无此单)要回灌给模型,流正常 done,不是 error 帧。"""

    @tool
    async def query_order(order_id: str) -> str:
        """替身:总是找不到订单。"""
        from app.tools.errors import ToolNotFound
        raise ToolNotFound("未找到订单 9999")

    model = ScriptedModel(
        [
            [FakeChunk(tool_calls=[{"name": "query_order", "args": {"order_id": "9999"}, "id": "c1"}])],
            [FakeChunk("没查到该订单,请核对单号。")],
        ]
    )
    events = _collect(model, RecordingSession(), {"query_order": query_order})
    kinds = [kind for kind, _ in events]

    assert "tool_result" in kinds
    assert kinds[-1] == "done"
    assert "error" not in kinds
    result_payload = next(payload for kind, payload in events if kind == "tool_result")
    assert result_payload["ok"] is False


def test_infrastructure_failure_propagates():
    """基础设施故障必须向上抛,由 API 层推 error 帧 —— 不能伪装成"查不到"。"""

    @tool
    async def query_order(order_id: str) -> str:
        """替身:DB 挂了。"""
        from sqlalchemy.exc import OperationalError
        raise OperationalError("SELECT 1", {}, Exception("连接断开"))

    model = ScriptedModel(
        [[FakeChunk(tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}])]]
    )
    with pytest.raises(ToolInfrastructureError):
        _collect(model, RecordingSession(), {"query_order": query_order})


def test_history_is_not_written_when_second_round_breaks():
    """第二轮炸了 → 整轮不落库,不留孤儿行(沿用 ch01 语义)。"""
    recorded = []

    class ExplodingSession:
        def add(self, obj):
            recorded.append(obj)

        async def commit(self):
            raise AssertionError("不应提交")

    class ExplodingModel(ScriptedModel):
        async def astream(self, messages):
            self.calls.append(("unbound", list(messages)))
            yield FakeChunk("前半句")
            raise RuntimeError("上游炸了")

    model = ExplodingModel(
        [[FakeChunk(tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}])]]
    )
    with pytest.raises(RuntimeError):
        _collect(model, ExplodingSession(), {"query_logistics": query_logistics})

    assert recorded == []


def test_successful_turn_appends_user_tool_and_answer():
    """成功一轮落库四条:user / assistant(带 tool_calls) / tool / assistant。"""
    captured = []

    class CapturingSession:
        def add(self, obj):
            captured.append(obj)

        async def commit(self):
            pass

    model = ScriptedModel(
        [
            [FakeChunk(tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}])],
            [FakeChunk("已揽件。")],
        ]
    )
    _collect(model, CapturingSession(), {"query_logistics": query_logistics})

    roles = [obj.role for obj in captured]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert captured[1].tool_calls[0]["id"] == "c1"
    assert captured[2].tool_call_id == "c1"
    assert captured[3].content == "已揽件。"
```

- [ ] **Step 2: 运行测试,确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chat_service.py`
Expected: FAIL —— `prepare_turn` 签名不符、`stream_turn` 不存在

- [ ] **Step 3: 改写 `app/services/chat.py`**

```python
from collections.abc import AsyncIterator, Sequence

from langchain.messages import AIMessage, ToolMessage

from app.config import Settings
from app.memory import trim
from app.prompts import build_messages, render_system_prompt
from app.schemas import Message
from app.services.history import append_turn
from app.tools.executor import execute_tool


def prepare_turn(
    *,
    settings: Settings,
    history: Sequence[Message],
    user_input: str,
) -> list:
    """组装本轮要发给模型的消息。

    历史由调用方从 MySQL 读出后传入 —— 本函数不做 IO,便于单测。

    预算不足时抛 ContextOverflowError,调用方在响应开始前处理,
    因此能返回 400 而不是一个已经开始的 SSE 流。
    """
    system_prompt = render_system_prompt(settings.brand_name)
    available = trim.compute_available_tokens(
        system_prompt=system_prompt,
        user_input=user_input,
        context_budget_tokens=settings.context_budget_tokens,
        reserved_output_tokens=settings.reserved_output_tokens,
        safety_margin_tokens=settings.safety_margin_tokens,
    )
    if available < 0:
        budget = (
            settings.context_budget_tokens
            - settings.reserved_output_tokens
            - settings.safety_margin_tokens
        )
        raise trim.ContextOverflowError(
            used=trim.count_tokens(system_prompt) + trim.count_tokens(user_input),
            budget=budget,
        )

    kept = trim.select_history(history, available)
    return build_messages(
        brand_name=settings.brand_name,
        history=kept,
        user_input=user_input,
    )


async def stream_turn(
    *,
    settings: Settings,
    model,
    session,
    conversation_id: str,
    user_input: str,
    messages: Sequence,
    tools: Sequence,
    registry: dict,
) -> AsyncIterator[tuple[str, dict]]:
    """单轮工具调用编排。

    产出 (event_name, payload),event_name 取值:
    token / tool_call / tool_result / done。

    本函数不负责加锁解锁 —— 锁由 API 层持有。基础设施故障
    (ToolInfrastructureError)向上抛,由 API 层转成 error 帧并终止流。

    **「只做单轮」是结构保证,不是提示词约定**:第二轮用未绑定工具的
    model,模型在结构上无法再调工具。实测确认过"第二轮仍绑 tools 时
    模型这次没再调",但那是模型行为、不是保证,故不采用。
    """
    model_with_tools = model.bind_tools(list(tools))

    # ---- 第一轮:边流边分拣 ----
    # 实测:调工具的提问产出 0 个文本 chunk + 若干 tool_call chunk;
    # 不调工具的提问产出 0 个 tool_call chunk。两者零重叠,
    # 故不需要"先缓冲再判断"的试探逻辑。
    accumulated = None
    first_usage = None
    async for chunk in model_with_tools.astream(messages):
        accumulated = chunk if accumulated is None else accumulated + chunk
        first_usage = getattr(chunk, "usage_metadata", None) or first_usage
        if chunk.text:
            yield ("token", {"text": chunk.text})

    first_text = (getattr(accumulated, "text", "") or "") if accumulated is not None else ""
    tool_calls = list(getattr(accumulated, "tool_calls", None) or [])

    if not tool_calls:
        # 没调工具:第一轮的文本已经流式推完,单次 API 调用即完成。
        await append_turn(
            session=session,
            conversation_id=conversation_id,
            messages=[
                Message(role="user", content=user_input),
                Message(role="assistant", content=first_text),
            ],
        )
        yield ("done", {"finish_reason": "stop", "usage": first_usage})
        return

    # ---- 执行工具 ----
    round_two = list(messages) + [AIMessage(content=first_text, tool_calls=tool_calls)]
    tool_messages: list[Message] = []

    for tool_call in tool_calls:
        yield (
            "tool_call",
            {
                "name": tool_call["name"],
                "args": tool_call["args"],
                "tool_call_id": tool_call["id"],
            },
        )
        outcome = await execute_tool(
            tool_call=tool_call, registry=registry, settings=settings
        )
        yield (
            "tool_result",
            {
                "tool_call_id": outcome.tool_call_id,
                "ok": outcome.ok,
                "summary": outcome.summary,
            },
        )
        round_two.append(
            ToolMessage(content=outcome.content, tool_call_id=tool_call["id"])
        )
        tool_messages.append(
            Message(role="tool", content=outcome.content, tool_call_id=tool_call["id"])
        )

    # ---- 第二轮:不绑 tools,强制收敛为文本 ----
    parts: list[str] = []
    usage = None
    async for chunk in model.astream(round_two):
        usage = getattr(chunk, "usage_metadata", None) or usage
        if chunk.text:
            parts.append(chunk.text)
            yield ("token", {"text": chunk.text})

    reply = "".join(parts)

    # 只有流完整走完才会执行到这里。中途抛异常时下面的写入不会发生,
    # 半截回复不会污染历史。
    await append_turn(
        session=session,
        conversation_id=conversation_id,
        messages=[
            Message(role="user", content=user_input),
            Message(role="assistant", content=first_text, tool_calls=tool_calls),
            *tool_messages,
            Message(role="assistant", content=reply),
        ],
    )
    yield ("done", {"finish_reason": "stop", "usage": usage})
```

- [ ] **Step 4: 运行测试,确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chat_service.py`
Expected: PASS

- [ ] **Step 5: 验证结构断言真能区分错误实现**

把第二轮那行 `async for chunk in model.astream(round_two)` 临时改成
`async for chunk in model_with_tools.astream(round_two)`,重跑:

Run: `.venv/Scripts/python.exe -m pytest tests/test_chat_service.py::test_second_round_uses_unbound_model`
Expected: **FAIL**

确认后改回。把两次输出记进任务报告。

- [ ] **Step 6: 提交**

```bash
git add app/services/chat.py tests/test_chat_service.py
git commit -m "feat: 单轮工具编排,第二轮不绑 tools 作结构保证"
```

---

## Task 11: API 层与 SSE 事件扩展

**Files:**
- Modify: `app/api/chat.py`
- Test: `tests/test_api_chat.py`(改写)

**Interfaces:**
- Consumes: `services.history`(Task 9)、`services.chat`(Task 10)、`tools.registry`(Task 5)、`db.session.get_session`(Task 2)
- Produces: `POST /api/chat/stream` 新增 `tool_call` / `tool_result` 两个 SSE 事件;请求体新增可选 `user_id`

- [ ] **Step 1: 写失败测试**

改写 `tests/test_api_chat.py`,保留原有用例并适配,追加:

```python
def test_chat_stream_emits_tool_call_event(client_factory):
    """验收 1 的后端一半:必须推出 tool_call 帧,且 name 正确。"""
    client, model = client_factory(
        batches=[
            [FakeChunk(tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}])],
            [FakeChunk("已揽件。")],
        ]
    )
    with client as c:
        resp = c.post("/api/chat/stream", json={"message": "订单 1001 的物流到哪了"})
        events = _parse_sse(resp.text)

    kinds = [name for name, _ in events]
    assert "tool_call" in kinds
    assert "tool_result" in kinds
    payload = next(p for name, p in events if name == "tool_call")
    assert payload["name"] == "query_logistics"
    assert payload["args"] == {"order_id": "1001"}
    assert kinds[-1] == "done"


def test_chat_stream_accepts_optional_user_id(client_factory):
    client, _ = client_factory(batches=[[FakeChunk("您好")]])
    with client as c:
        resp = c.post(
            "/api/chat/stream",
            json={"message": "你好", "user_id": "alice"},
        )
    assert resp.status_code == 200


def test_infrastructure_failure_emits_error_frame(client_factory):
    """DB 故障 → error 帧终止流,不是 done;且错误文本不回显密钥。"""
    from sqlalchemy.exc import OperationalError

    @tool
    async def query_order(order_id: str) -> str:
        """替身:DB 挂了。"""
        raise OperationalError("SELECT 1", {}, Exception("连接断开"))

    client, _ = client_factory(
        batches=[
            [FakeChunk(tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}])]
        ],
        registry={"query_order": query_order},
    )
    with client as c:
        resp = c.post("/api/chat/stream", json={"message": "订单 1001 的状态"})
        events = _parse_sse(resp.text)

    kinds = [name for name, _ in events]
    assert "error" in kinds
    assert "done" not in kinds

    payload = next(p for name, p in events if name == "error")
    assert "sk-test" not in payload["message"]   # 脱敏生效
```

`client_factory` 夹具需替换 `get_session` 与模型依赖。因 `get_store` / `get_chat_model` / `get_session` 都走 `Depends`,用 `app.dependency_overrides` 覆盖:

```python
@pytest.fixture
def client_factory():
    created = []

    def make(batches, session=None):
        model = ScriptedModel(batches)
        created.append(model)

        class SessionStub:
            async def execute(self, *a, **kw):
                class R:
                    def scalars(self):
                        return self
                    def one_or_none(self):
                        return None
                    def all(self):
                        return []
                return R()

            def add(self, obj):
                pass

            async def commit(self):
                pass

        app.dependency_overrides[get_chat_model] = lambda: model
        app.dependency_overrides[get_session] = lambda: session or SessionStub()
        store = SessionStore(ttl_seconds=60, max_sessions=10)
        app.dependency_overrides[get_store] = lambda: store
        return TestClient(app), model

    yield make
    app.dependency_overrides.clear()
```

**注意**:`get_session` 是 async generator 依赖,`dependency_overrides` 里需要用 async generator 形式覆盖。若直接返回对象报错,改用:

```python
async def _session_override():
    yield SessionStub()

app.dependency_overrides[get_session] = _session_override
```

- [ ] **Step 2: 运行测试,确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api_chat.py`
Expected: FAIL —— 没有 `tool_call` 事件

- [ ] **Step 3: 改写 `app/api/chat.py`**

关键改动(其余沿用 ch01:锁、预算校验在流开始前、`finally` 释放锁、`redact_api_key` 脱敏):

```python
@router.post("/api/chat/stream")
async def chat_stream(
    request: ChatRequest,
    settings: Settings = Depends(get_settings),
    store: SessionStore = Depends(get_store),
    model=Depends(get_chat_model),
    session: AsyncSession = Depends(get_session),
) -> EventSourceResponse:
    session_id = request.session_id or uuid.uuid4().hex
    user_id = request.user_id or "demo-user"

    lock = store.lock_for(session_id)
    try:
        await asyncio.wait_for(
            lock.acquire(), timeout=settings.session_lock_timeout_seconds
        )
    except TimeoutError as exc:
        raise HTTPException(
            status_code=409, detail="该会话正在处理另一条消息,请稍后重试"
        ) from exc

    # 预算校验必须在响应开始前完成 —— 一旦开始流式就改不了状态码。
    # 会话读取与历史加载也在锁内,避免同会话并发时读到半轮历史。
    try:
        await ensure_conversation(
            session=session, session_id=session_id, user_id=user_id
        )
        history = await load_history(session=session, conversation_id=session_id)
        messages = prepare_turn(
            settings=settings, history=history, user_input=request.message
        )
    except ContextOverflowError as exc:
        lock.release()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except BaseException:
        # 拿到锁之后,凡是不返回 EventSourceResponse 的退出都必须释放锁,
        # 否则该会话会永久 409。用 BaseException 而非 Exception 兜住一切 ——
        # 客户端在响应开始前断开会让 Starlette 取消端点任务,抛出的
        # asyncio.CancelledError 是 BaseException 的子类,不是 Exception。
        lock.release()
        raise

    tools = build_tools(session=session, conversation_id=session_id)
    registry = registry_for(tools)

    async def generate():
        try:
            yield _frame(
                "meta",
                {"session_id": session_id, "model": settings.openai_model},
            )
            async for event, payload in stream_turn(
                settings=settings,
                model=model,
                session=session,
                conversation_id=session_id,
                user_input=request.message,
                messages=messages,
                tools=tools,
                registry=registry,
            ):
                yield _frame(event, payload)
        except Exception as exc:
            # 上游异常文本可能带着密钥(见 app/sanitize.py),出站前抹掉。
            yield _frame(
                "error",
                {"message": redact_api_key(str(exc), settings.openai_api_key)},
            )
        finally:
            lock.release()

    return EventSourceResponse(
        generate(),
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

`app/schemas.py` 的 `ChatRequest` 追加:

```python
class ChatRequest(BaseModel):
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    message: str = Field(min_length=1)
    user_id: str | None = Field(default=None, min_length=1, max_length=128)
```

- [ ] **Step 4: 运行测试,确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api_chat.py`
Expected: PASS

- [ ] **Step 5: 全量回归**

Run: `.venv/Scripts/python.exe -m pytest`
Expected: PASS(含 db 标记的测试,MySQL 在跑)

- [ ] **Step 6: 提交**

```bash
git add app/api/chat.py app/schemas.py tests/test_api_chat.py
git commit -m "feat: SSE 新增工具事件,聊天接口接入单轮编排"
```

---

## Task 12: 工具选择评估集

**Files:**
- Create: `evals/tool_selection_cases.jsonl`, `evals/run_tool_selection_eval.py`

**Interfaces:**
- Consumes: `tools.registry.build_tools`(Task 5)、`db.base.get_sessionmaker`(Task 2)、`llm.create_chat_model`
- Produces: 可执行的评估脚本,输出工具选择准确率

**说明**:这是「非可单测产出用评估集验证」那一步。评分口径是**闭式精确匹配**(工具名是枚举),不会重蹈 ch01 `expected_solution` 关键词口径被样本拟合的覆辙。

- [ ] **Step 1: 写用例文件**

创建 `evals/tool_selection_cases.jsonl`(**每行一个 JSON 对象**,`expected` 为 `null` 表示**不该调工具**):

```jsonl
{"text": "订单 1001 的物流到哪了", "expected": "query_logistics", "note": "诱饵:句里有『订单』二字,容易误选 query_order,但问的是物流"}
{"text": "帮我查下物流,单号 1002", "expected": "query_logistics", "note": ""}
{"text": "订单 1002 为什么还没发货", "expected": "query_logistics", "note": ""}
{"text": "订单 1001 现在是什么状态", "expected": "query_order", "note": ""}
{"text": "查一下订单 1003 的金额", "expected": "query_order", "note": ""}
{"text": "无线耳机多少钱", "expected": "query_product", "note": ""}
{"text": "运动鞋还有货吗", "expected": "query_product", "note": ""}
{"text": "保温杯有什么规格", "expected": "query_product", "note": ""}
{"text": "退货政策是什么", "expected": "query_faq", "note": "验收标准 2 的用例"}
{"text": "发票怎么开", "expected": "query_faq", "note": ""}
{"text": "退款多久到账", "expected": "query_faq", "note": ""}
{"text": "我要投诉,给我转人工", "expected": "create_ticket", "note": ""}
{"text": "你们客服态度太差了,我要投诉", "expected": "create_ticket", "note": ""}
{"text": "你好", "expected": null, "note": "诱饵:闲聊不该调工具。防的是『模型见谁都调工具』"}
{"text": "你们几点下班", "expected": null, "note": "诱饵:没有工具能回答,不该硬调"}
```

- [ ] **Step 2: 实现评估脚本**

创建 `evals/run_tool_selection_eval.py`:

```python
"""工具选择评估集。

用法:.venv/Scripts/python.exe evals/run_tool_selection_eval.py(需真实 key 与 MySQL)

评分口径:闭式精确匹配 —— 工具名是枚举,不存在"措辞不同"的模糊地带。
与 ch01 的 expected_solution(自由文本、关键词口径最终被证明是样本拟合)
形成对比:本口径的分数可以直接引用。

只跑「工具选择」这一段,不执行工具 —— 故不产生任何 DB 写入。
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.db.base import get_sessionmaker
from app.llm import create_chat_model
from app.tools.registry import build_tools

CASES = Path(__file__).with_name("tool_selection_cases.jsonl")


def load_cases() -> list[dict]:
    lines = CASES.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


async def select_tool(model, tools, text: str) -> str | None:
    """跑一轮工具选择,返回模型选中的工具名;没调工具则返回 None。"""
    bound = model.bind_tools(tools)
    accumulated = None
    async for chunk in bound.astream([{"role": "user", "content": text}]):
        accumulated = chunk if accumulated is None else accumulated + chunk
    tool_calls = list(getattr(accumulated, "tool_calls", None) or [])
    return tool_calls[0]["name"] if tool_calls else None


async def main() -> int:
    settings = get_settings()
    model = create_chat_model(settings)
    cases = load_cases()

    async with get_sessionmaker()() as session:
        tools = build_tools(session=session, conversation_id="_eval")

        print(f"模型:{settings.openai_model}  用例数:{len(cases)}\n")
        hits = 0
        misses: list[tuple[dict, str | None]] = []

        for case in cases:
            actual = await select_tool(model, tools, case["text"])
            ok = actual == case["expected"]
            hits += ok
            print(
                f"  [{'✓' if ok else '✗'}] {case['text'][:30]:<32}"
                f" 期望={case['expected']}  实际={actual}"
            )
            if not ok:
                misses.append((case, actual))

    total = len(cases)
    print(f"\n工具选择准确率:{hits}/{total} = {hits / total:.1%}")

    if misses:
        print(f"\n未命中 {len(misses)} 条(附备注,便于判断是标注问题还是模型问题):")
        for case, actual in misses:
            print(f"  - {case['text']}")
            print(f"      期望={case['expected']}  实际={actual}")
            if case.get("note"):
                print(f"      备注:{case['note']}")

    return 1 if misses else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
```

- [ ] **Step 3: 运行**

Run: `.venv/Scripts/python.exe evals/run_tool_selection_eval.py`
Expected: 打印逐条结果与准确率。

**把结果如实记入 dev-notes**,包括未命中的条目。若准确率不理想,**不要调评分口径去凑** —— ch01 的教训是那样做出来的数字不可引用。

- [ ] **Step 4: 提交**

```bash
git add evals/tool_selection_cases.jsonl evals/run_tool_selection_eval.py
git commit -m "test: 工具选择评估集,闭式精确匹配口径"
```

---

## Task 13: 端到端验收脚本扩展

**Files:**
- Modify: `scripts/acceptance.sh`

**Interfaces:**
- Consumes: 运行中的服务(需真实 key 与 MySQL)
- Produces: 三条验收的自动化断言

**前置**:该文件头部已记录两个平台陷阱的规避方式(**含中文的请求体走 stdin**;**断言比对拼回后的文本而非原始 SSE 流**),新代码必须继续遵守。改之前先读那段注释。

- [ ] **Step 1: 追加两个 SSE 解析助手**

在 `scripts/acceptance.sh` 的 `has_cjk()` 之后追加:

```bash
# 取出本轮实际调用了哪些工具(按出现顺序,逗号分隔)。
# 直接对原始 SSE 流 grep 工具名是不行的 —— 事件名与 data 分处两行,
# 且工具名只出现在 JSON 里。按 SSE 块解析才可靠。
called_tools() {
  "$PYTHON" -c '
import json, sys

names = []
for block in sys.stdin.buffer.read().decode("utf-8", "replace").split("\n\n"):
    event = data = None
    for line in block.splitlines():
        if line.startswith("event: "):
            event = line[len("event: "):]
        elif line.startswith("data: "):
            data = line[len("data: "):]
    if event == "tool_call" and data:
        try:
            names.append(json.loads(data)["name"])
        except (json.JSONDecodeError, KeyError):
            pass
print(",".join(names))
'
}

# 取出每个 tool_result 的成败(ok/fail,逗号分隔)。
tool_result_states() {
  "$PYTHON" -c '
import json, sys

states = []
for block in sys.stdin.buffer.read().decode("utf-8", "replace").split("\n\n"):
    event = data = None
    for line in block.splitlines():
        if line.startswith("event: "):
            event = line[len("event: "):]
        elif line.startswith("data: "):
            data = line[len("data: "):]
    if event == "tool_result" and data:
        try:
            states.append("ok" if json.loads(data).get("ok") else "fail")
        except json.JSONDecodeError:
            pass
print(",".join(states))
'
}

# 从工具实现里取订单 1001 的确定性物流状态。
# 动态取值而非写死 —— 工具改了种子函数也不必改脚本;
# 而"工具到底返回什么"由 Tier 1 的跨进程确定性测试守护。
expected_logistics_status() {
  "$PYTHON" -c '
import asyncio, json, sys

from app.tools.business import query_logistics

tool_call = {
    "name": "query_logistics",
    "args": {"order_id": "1001"},
    "id": "probe",
    "type": "tool_call",
}
payload = json.loads(asyncio.run(query_logistics.ainvoke(tool_call)).content)
sys.stdout.buffer.write(payload["status"].encode("utf-8"))
'
}
```

- [ ] **Step 2: 追加三条验收**

在文件末尾统计行之前插入:

```bash
echo
echo "=== 验收 4:工具调用链路(需求 4) ==="
EXPECTED_STATUS=$(expected_logistics_status)
echo "  订单 1001 的确定性物流状态:$EXPECTED_STATUS"

SID_TOOL="acceptance-tool-$$"
OUT4=$(curl -sN -X POST "$BASE/api/chat/stream" \
  -H 'Content-Type: application/json' \
  --data-binary @- <<JSON
{"session_id":"$SID_TOOL","message":"订单 1001 的物流到哪了"}
JSON
)

TOOLS4=$(echo "$OUT4" | called_tools)
STATES4=$(echo "$OUT4" | tool_result_states)
REPLY4=$(echo "$OUT4" | join_tokens)
echo "  调用的工具:[$TOOLS4]  结果:[$STATES4]"
echo "  回复:$REPLY4"

if [ "$TOOLS4" = "query_logistics" ]; then
  pass "模型选中 query_logistics(验收标准 1 的后端一半)"
else
  fail "期望选中 query_logistics,实际调用了 [$TOOLS4]"
fi

if [ "$STATES4" = "ok" ]; then
  pass "工具执行成功"
else
  fail "工具执行未成功,结果状态为 [$STATES4]"
fi

# 断言模型真的**读懂了工具结果**而不是自己编。
# 比的是工具返回的确定性状态词 —— 模型没拿到结果就不可能说对。
if echo "$REPLY4" | grep -q "$EXPECTED_STATUS"; then
  pass "回复中复述了工具返回的状态「$EXPECTED_STATUS」(工具结果真的被用上了)"
else
  fail "回复中没有「$EXPECTED_STATUS」—— 模型没有用上工具结果"
fi

echo
echo "=== 验收 5:FAQ 查表(验收标准 2) ==="
OUT5=$(curl -sN -X POST "$BASE/api/chat/stream" \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{"message":"退货政策是什么"}
JSON
)

TOOLS5=$(echo "$OUT5" | called_tools)
STATES5=$(echo "$OUT5" | tool_result_states)
REPLY5=$(echo "$OUT5" | join_tokens)
echo "  调用的工具:[$TOOLS5]  结果:[$STATES5]"
echo "  回复:$REPLY5"

if [ "$TOOLS5" = "query_faq" ] && [ "$STATES5" = "ok" ]; then
  pass "query_faq 查到了退货政策"
else
  fail "期望 query_faq 命中,实际工具=[$TOOLS5] 结果=[$STATES5]"
fi

if REASON=$(echo "$REPLY5" | has_cjk); then
  pass "回复非空且含 $REASON"
else
  fail "回复$REASON"
fi

echo
echo "=== 验收 6:邮费漏召回(验收标准 3,预期失败) ==="
OUT6=$(curl -sN -X POST "$BASE/api/chat/stream" \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{"message":"邮费是多少"}
JSON
)

TOOLS6=$(echo "$OUT6" | called_tools)
STATES6=$(echo "$OUT6" | tool_result_states)
REPLY6=$(echo "$OUT6" | join_tokens)
echo "  调用的工具:[$TOOLS6]  结果:[$STATES6]"
echo "  回复:$REPLY6"

# 反向断言:这一条**期望查不到**。若它竟然查到了,说明 faq 种子里混进了
# 「邮费」条目,验收标准 3 的前提被破坏 —— 那才是失败。
if [ "$TOOLS6" = "query_faq" ] && [ "$STATES6" = "ok" ]; then
  fail "「邮费」竟然从 faq 查到了 —— 种子数据污染了,验收标准 3 失去意义"
else
  pass "「邮费」未能查到(预期漏召回),工具=[$TOOLS6] 结果=[$STATES6]"
fi
```

- [ ] **Step 3: 端到端跑一遍**

先起服务(另一个终端):

```bash
.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000
```

再跑:

```bash
bash scripts/acceptance.sh
```

Expected: 全部通过,失败 0 项。**原样跑一遍再看结果** —— ch01 的教训是验收脚本自己最容易写错(当时原样跑是 0/7)。

- [ ] **Step 4: 提交**

```bash
git add scripts/acceptance.sh
git commit -m "test: 端到端验收补工具链路、FAQ 查表与邮费漏召回"
```

---

## Task 14: 聊天页(前端 · Vibe Coding)

**本任务按用户规则 1 的例外条款执行**:用 Vibe Coding 方式直接做,**不套 TDD、不写单测、不进 code review**。验收方式是浏览器里肉眼看 —— 这与前 13 个任务的流程不同,是有意为之。

**Files:**
- Create: `app/static/index.html`
- Modify: `app/main.py`(挂载静态目录)

**Interfaces:**
- Consumes: `POST /api/chat/stream` 的六个 SSE 事件(见 spec §5.2)
- Produces: 可在浏览器打开的聊天页

### 设计依据

参照 `asserts/img.png` 原型,**配色按要求改为浅蓝系**:

| 元素 | 原型 | 改成 |
|---|---|---|
| 页面底色 | 米黄网格纸 | **浅蓝**(如 `#e8f2fb`),保留细网格纹理 |
| 顶栏 | 橙色 | 深蓝(如 `#2f6fb0`),白字 |
| 用户气泡 | 橙色右对齐 | 同顶栏的蓝,白字 |
| 助手气泡 | 白底左对齐 | 保持白底,描边改深蓝 |
| 发送按钮 | 橙色方块 | 同顶栏的蓝 |
| 工具徽章 | 虚线描边胶囊 | 保持虚线胶囊,描边与文字用深蓝 |

原型里的其它元素都要保留:头像(🐱)、「小喵 · 智能客服」/「ONLINE · 喵喵优选」、「＋ 新对话」按钮、输入框提示语「输入消息,和小喵聊聊吧~(回车发送 / Shift+回车换行)」、页脚「小喵是 AI 助手,涉及具体订单会为你转接人工核实」。

### 关键技术约束

**不能用 `EventSource`。** 它只支持 GET,而我们的接口是 POST。必须用 `fetch` + `ReadableStream` 手工解析 SSE:

```js
const resp = await fetch("/api/chat/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ session_id, message }),
});
const reader = resp.body.getReader();
const decoder = new TextDecoder("utf-8");
let buffer = "";
for (;;) {
  const { value, done } = await reader.read();
  if (done) break;
  buffer += decoder.decode(value, { stream: true });
  let idx;
  while ((idx = buffer.indexOf("\n\n")) !== -1) {
    const block = buffer.slice(0, idx);
    buffer = buffer.slice(idx + 2);
    handleBlock(block);   // 解析 event: / data: 两行
  }
}
```

注意 `decoder.decode(value, { stream: true })` —— 不加 `stream: true` 会让**多字节中文被切断在 chunk 边界上**变成乱码(SSE 分片与 UTF-8 字符边界无关)。这是本任务最容易翻车的一处。

### 事件到 UI 的映射

| 事件 | UI 行为 |
|---|---|
| `meta` | 记下 `session_id`(后续轮次复用) |
| `tool_call` | 在助手气泡**顶部**插入虚线徽章:`🔧 调用了 {name}`,状态为「调用中」 |
| `tool_result` | 把徽章状态改为「已返回」;`ok=false` 时徽章转为警示色 |
| `token` | 追加到当前助手气泡正文 |
| `done` | 结束本轮,恢复输入框可用 |
| `error` | 在当前气泡显示错误文案(红色),结束本轮 |

### 步骤

- [ ] **Step 1: 写 `app/static/index.html`**

按上面的设计依据与约束实现单页。无需构建工具链,原生 HTML + CSS + JS。

- [ ] **Step 2: `app/main.py` 挂载静态目录**

```python
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.chat import router as chat_router
from app.api.extract import router as extract_router

app = FastAPI(title="电商智能客服 ch02")
app.include_router(chat_router)
app.include_router(extract_router)

_static = Path(__file__).parent / "static"
if _static.is_dir():
    app.mount("/", StaticFiles(directory=_static, html=True), name="static")
```

**注意挂载顺序**:`app.mount("/", ...)` 会接管根路径,必须在 `include_router` **之后**,否则 `/api/*` 会被静态目录抢走。

- [ ] **Step 3: 起服务,浏览器验证**

```bash
.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000 --reload
```

打开 `http://localhost:8000`,依次确认:

1. 底色浅蓝,顶栏与气泡配色协调(**对照原型图**)
2. 问「订单 1001 的物流到哪了」→ 气泡顶部出现「🔧 调用了 query_logistics」徽章,随后逐字吐出回复
3. 问「退货政策是什么」→ 徽章显示 `query_faq`,回复正确
4. 问「邮费是多少」→ 徽章显示调用失败或未调用,回复**没有**编造价目表
5. 追问第二轮能接住上下文(历史来自 MySQL)
6. 中文全程不乱码

- [ ] **Step 4: 提交**

```bash
git add app/static/index.html app/main.py
git commit -m "feat: 浅蓝配色聊天页,气泡内显示工具调用徽章"
```

---

## 完成后的收尾(不属于任何单个任务)

- [ ] **全量测试**

```bash
.venv/Scripts/python.exe -m pytest
```
全部通过(含 `@pytest.mark.db`)。

- [ ] **端到端验收**

```bash
bash scripts/acceptance.sh
```
失败 0 项。

- [ ] **评估集**

```bash
.venv/Scripts/python.exe evals/run_tool_selection_eval.py
```
如实记录准确率与未命中条目。

- [ ] **dev-notes 收尾**

在 `dev-notes/ch02.md` 追加实施阶段的留痕(每个任务一段),以及**最终整支审查**的结论。不得事后一次性补记 —— 每个任务完成时就该写。

- [ ] **交付物**

向用户给出:功能演示命令、测试结果、dev-notes 路径。
