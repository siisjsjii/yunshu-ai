# ch06 分流器正式版 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 ch05 里占位的分流器做成正式版 —— LLM prompt 意图识别(八类含「其他」)、指代消解与 Query 改写、退款退货/售后的确定性子流程(含 interrupt/resume 的订单卡片回填)、Query 扩写多路检索。

**Architecture:** 沿用 ch05 的 LangGraph 确定性骨架。主图新增一条 REFUND 出口指向**子流程**;子流程每一步写死在图上,只在「这一单能不能退」处调一次主力 Agent。槽位缺失时用 LangGraph `interrupt()` 真暂停,前端渲染订单卡片,点选后经 `Command(resume=...)` 同 thread 续跑。

**Tech Stack:** Python 3.13 / FastAPI / LangGraph 1.2.11 / LangChain 1.x / SQLAlchemy(async)/ MySQL / Milvus / 原生 HTML+CSS+JS(无构建工具链)

**Spec:** `docs/superpowers/specs/2026-09-20-ecommerce-cs-ch06-intent-router-design.md`
目标流程图:`asserts/ch06workflow.png`

## Global Constraints

以下每条都是**硬约束**,每个任务的要求都隐含包含它:

- **依赖方向单向**:`api → services → {tools, db, memory, prompts, llm}`;本章新增 `agent → refund`(单向)。`refund_nodes` 必须放 `app/agent/` 内,否则会出现 `refund → agent` 反边成环。
- **`app/prompts.py` 是消息组装的唯一出口**。任何新的模型消息组装都写在这里,不在节点里拼。
- **`memory/` 与 `services/history.py` 不依赖 LangChain**;`Message → BaseMessage` 只经 `prompts.py:to_lc_messages`。
- **`services/` 的函数接收 llm 实例作参数**;测试靠 `dependency_overrides` 替换。
- **错误语义边界**:`422` 只表示「请求/模型输出不合约定」;上游与基础设施故障一律 `502` + 固定文案。**所有出站错误文本**(SSE `error` 帧、`tool_result` 失败 `summary`、422/502 detail)必须过 `app/sanitize.py:redact_api_key`。
- **`ToolInfrastructureError` 必须一路上抛**,绝不回灌给模型 —— 数据库故障不能伪装成「你的订单号查不到」。
- **单测全程不联网**。`Settings(...)` 构造必须传 `_env_file=None`。db 测试读真实 `.env` 并打 `@pytest.mark.db`。
- **不要再往命令行加 `-q`**:`pytest.ini` 的 `addopts` 已有一个,叠加成 `-qq` 会**整行不打印 `N passed`**。
- **含中文的请求体不走 `curl` 的 argv**(MSYS2 按 CP936 重编码):一律 stdin heredoc 或 httpx。
- **本仓头号风险是「假绿测试」**:写断言前先问「实现改错了,这条的输出会不会不同?」。
- **F1/F3 是实测事实,不是设计偏好**:`astream(stream_mode="custom")` **会吞掉 interrupt**;resume 时**节点从头重跑**。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `app/refund/__init__.py` | 包标记 |
| `app/refund/categories.py` | 退款原因固定类目(单一来源,前后端共用) |
| `app/refund/orders.py` | 候选订单(会话历史扫描 + 会话固定演示集) |
| `db/ch06.sql` | `refund_requests` 建表 |
| `app/db/models.py` | 增加 `RefundRequest` ORM |
| `app/api/refund.py` | `POST /api/refund` |
| `app/agent/state.py` | 新通道 + `IntentResult.confidence` |
| `app/agent/routing.py` | `REFUND` 出口 + 八类路由表 |
| `app/agent/nodes.py` | `resolve_references` 改消解+改写;`classify_intent` 出 confidence |
| `app/agent/refund_nodes.py` | 子流程五个节点 |
| `app/agent/graph.py` | 接子流程 |
| `app/retrieval/expand.py` | 扩写 → 多路检索 → 去重合并 |
| `app/api/chat.py` | resume 分支 + `["custom","updates"]` + interrupt 转帧 |
| `app/schemas.py` | `ChatRequest.resume` |
| `app/prompts.py` | 三个新 prompt |
| `app/static/index.html` | 订单卡片 + 退款表单(Vibe Coding) |
| `scripts/acceptance_ch06.sh` | 章级端到端验收 |

**任务依赖**:T1 → T3;T2 → T3;T4/T5/T6 独立;T4+T5+T6+T7 → T8;T8 → T9/T10。

---

### Task 1: 退款原因类目 + 候选订单(纯函数)

**Files:**
- Create: `app/refund/__init__.py`, `app/refund/categories.py`, `app/refund/orders.py`
- Test: `tests/test_refund_categories.py`, `tests/test_refund_orders.py`

**Interfaces:**
- Produces:
  - `app.refund.categories.REFUND_REASON_CATEGORIES: tuple[str, ...]`
  - `app.refund.categories.is_valid_category(value: str) -> bool`
  - `app.refund.orders.candidate_orders(history: list[Message], conversation_id: str) -> list[str]`
  - `app.refund.orders.DEMO_ORDERS: tuple[str, ...]`

- [ ] **Step 1: 写失败测试**

`tests/test_refund_categories.py`:

```python
import pytest

from app.refund.categories import REFUND_REASON_CATEGORIES, is_valid_category


def test_categories_are_a_closed_nonempty_set():
    assert len(REFUND_REASON_CATEGORIES) >= 3
    assert len(set(REFUND_REASON_CATEGORIES)) == len(REFUND_REASON_CATEGORIES)


def test_valid_category_accepts_only_exact_members():
    first = REFUND_REASON_CATEGORIES[0]
    assert is_valid_category(first) is True


@pytest.mark.parametrize("bad", ["", "不存在的类目", " 商品质量问题 "])
def test_valid_category_rejects_non_members_without_trimming(bad):
    """不做 strip 归一:带空格的输入更可能是一次真实的传参错误,不是同一个类目。"""
    assert is_valid_category(bad) is False
```

`tests/test_refund_orders.py`:

```python
from app.refund.orders import DEMO_ORDERS, candidate_orders
from app.schemas import Message


def _msg(role, content):
    return Message(role=role, content=content)


def test_history_order_numbers_come_first():
    history = [_msg("user", "我刚才问的是订单 20240915 的物流")]
    got = candidate_orders(history, "s1")
    assert got[0] == "20240915"


def test_history_scan_finds_numbers_in_assistant_messages_too():
    history = [_msg("assistant", "您的订单 12345678 已发货")]
    assert "12345678" in candidate_orders(history, "s1")


def test_no_history_order_numbers_falls_back_to_demo_set():
    got = candidate_orders([_msg("user", "这个能退吗")], "s1")
    assert got == list(DEMO_ORDERS)


def test_candidates_are_deduped_and_capped():
    history = [_msg("user", "订单 111111 和 111111 还有 222222")]
    got = candidate_orders(history, "s1")
    assert len(got) == len(set(got))
    assert len(got) <= 4


def test_demo_set_is_session_stable():
    """同一会话两次调用给出同一批 —— 否则卡片上的号码会跳。"""
    a = candidate_orders([], "conv-A")
    b = candidate_orders([], "conv-A")
    assert a == b


def test_demo_numbers_are_usable_order_numbers():
    """演示集里的号码必须能真的被 query_order 查出来(4-32 位数字)。"""
    for no in DEMO_ORDERS:
        assert no.isdigit() and 4 <= len(no) <= 32
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_refund_categories.py tests/test_refund_orders.py`
Expected: FAIL —— `ModuleNotFoundError: No module named 'app.refund'`

- [ ] **Step 3: 实现**

`app/refund/__init__.py`:

```python
"""退款子流程的纯数据与纯函数。

**不依赖 LangChain、不依赖 app.agent** —— 这样 `app/agent/refund_nodes.py`
与 `app/api/refund.py` 都能引用它而不产生反向依赖边。
"""
```

`app/refund/categories.py`:

```python
"""退款原因固定类目 —— **单一来源**。

前后端共用:`POST /api/refund` 用它做校验,`refund_offer` 帧用它下发选项。
**前端不得硬编码这份列表**(选项从帧里来),否则两处会漂移。
"""

#: 用户提交退款单时从这几项里自选。**不追问原因**,这是产品的明确选择。
REFUND_REASON_CATEGORIES: tuple[str, ...] = (
    "商品质量问题",
    "不想要了",
    "发错货",
    "少发/漏发",
    "与描述不符",
)


def is_valid_category(value: str) -> bool:
    """**精确匹配,不做 strip/大小写归一**。

    与 `app/agent/routing.py` 的 `route_by_intent` 同一理由:清洗会让
    「商品质量问题 」这种近乎正确的输入静默通过,而它更可能是一次真实的
    传参错误。宁可 422。
    """
    return value in REFUND_REASON_CATEGORIES
```

`app/refund/orders.py`:

```python
"""候选订单 —— 补图里没有的一环。

系统里**没有 orders 表**:订单是 `app/tools/business.py` 的 `_order_record()`
用 hashlib 按订单号**现算**的,不存在「某用户的订单」这个概念。
所以订单选择器的候选要另找来源,顺序如下:

1. **会话历史里出现过的订单号**(用户之前提过的);
2. 一个都没有时,给**会话固定的一组演示订单**。

⚠️ 第 2 条是**演示数据,不是真实用户订单** —— 本仓没有用户-订单关系表,
这里只是让「不带订单号问退款」这条验收路径能走完。号码本身是真能查出单的
(订单由哈希派生,任何 4-32 位数字都有确定数据)。
"""

import hashlib
import re

from app.schemas import Message

#: 演示订单号池。取 8 位数字,与既有演示号(如 20240915)同量级。
_DEMO_POOL: tuple[str, ...] = (
    "20240915",
    "20240901",
    "20240818",
    "20240808",
    "20240726",
    "20240712",
)

_ORDER_RE = re.compile(r"\b\d{4,32}\b")

#: 卡片最多给几个候选。
MAX_CANDIDATES = 4

#: 兼容测试与外部引用的名字。
DEMO_ORDERS = _DEMO_POOL[:MAX_CANDIDATES]


def _scan(history: list[Message], limit: int) -> list[str]:
    """从**由近及远**的历史里取订单号,去重保序。"""
    found: list[str] = []
    for msg in reversed(history):          # 越近的越可能是用户当下关心的
        for no in _ORDER_RE.findall(msg.content or ""):
            if no not in found:
                found.append(no)
            if len(found) >= limit:
                return found
    return found


def _demo_for(conversation_id: str) -> list[str]:
    """按会话 id 稳定取一组演示订单。

    固定顺序 + 会话哈希偏移,保证**同一会话每次拿到同一批**(否则卡片上的
    号码每次刷新都变,用户点不到同一个)。
    """
    if not _DEMO_POOL:
        return []
    start = int(hashlib.sha256(conversation_id.encode("utf-8")).hexdigest(), 16) % len(_DEMO_POOL)
    return [_DEMO_POOL[(start + i) % len(_DEMO_POOL)] for i in range(MAX_CANDIDATES)]


def candidate_orders(history: list[Message], conversation_id: str) -> list[str]:
    """给订单卡片用的候选订单号(已去重、已限长)。"""
    from_history = _scan(history, MAX_CANDIDATES)
    if from_history:
        return from_history
    return _demo_for(conversation_id)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_refund_categories.py tests/test_refund_orders.py`
Expected: PASS(全部)

> ⚠️ **注意 `_ORDER_RE` 的假绿风险**:`\b\d{4,32}\b` 会匹配到金额、日期等数字。
> 若测试只断「历史里有个号码」而不构造干扰数字,这条断言在错误实现下照样绿。
> 上面的 `test_candidates_are_deduped_and_capped` 用纯数字串构造,属于**必要但不充分**;
> 实现者须自行补一条**含金额干扰**的用例(如「订单 111111 花了 99 元」),
> 并断言 `candidate_orders` 不把 `99` 当候选。

- [ ] **Step 5: 提交**

```bash
git add app/refund tests/test_refund_categories.py tests/test_refund_orders.py
git commit -m "feat(ch06): 退款原因固定类目 + 候选订单(会话历史扫描/演示集)"
```

---

### Task 2: `refund_requests` 表 + ORM

**Files:**
- Create: `db/ch06.sql`
- Modify: `app/db/models.py`(文件末尾追加)
- Test: `tests/test_db_models.py`(追加一条)

**Interfaces:**
- Produces: `app.db.models.RefundRequest`(表 `refund_requests`)

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_db_models.py`(模块已 import `pytest` / `text` / `get_sessionmaker`,
需在顶部 import 里加 `RefundRequest`):

```python
@pytest.mark.anyio
async def test_refund_request_roundtrips_chinese_and_defaults_status():
    """新表建得出来、中文往返不炸、status 有默认值。"""
    async with get_sessionmaker()() as session:
        session.add(
            RefundRequest(
                conversation_id=SCRATCH_CONVERSATION,
                order_no="20240915",
                reason_category="商品质量问题",
            )
        )
        await session.commit()
        await session.execute(
            text("DELETE FROM refund_requests WHERE conversation_id = :c"),
            {"c": SCRATCH_CONVERSATION},
        )
```

> 注意:断言**必须**在新 session 里读回(身份映射持弱引用,同 session 重读可能
> 走缓存而不是打库 —— 本仓已记过这条)。

```python
@pytest.mark.anyio
async def test_refund_request_persists_expected_columns():
    async with get_sessionmaker()() as session:
        session.add(
            RefundRequest(
                conversation_id=SCRATCH_CONVERSATION,
                order_no="20240915",
                reason_category="商品质量问题",
            )
        )
        await session.commit()
    async with get_sessionmaker()() as session:      # 新 session 读回
        row = (
            await session.execute(
                select(RefundRequest).where(
                    RefundRequest.conversation_id == SCRATCH_CONVERSATION
                )
            )
        ).scalars().one()
        assert row.order_no == "20240915"
        assert row.reason_category == "商品质量问题"
        assert row.status == "pending"               # 服务端默认值
        await session.execute(
            text("DELETE FROM refund_requests WHERE conversation_id = :c"),
            {"c": SCRATCH_CONVERSATION},
        )
        await session.commit()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_db_models.py -k refund`
Expected: FAIL —— `Table 'mewhelp.refund_requests' doesn't exist` 或 ImportError

- [ ] **Step 3: 建表 + ORM**

`db/ch06.sql`(**由人手工执行一次**,与 `db/ch04.sql` 同规矩):

```sql
-- ch06:退款单。用户在前端退款表单里确认后才写入。
CREATE TABLE IF NOT EXISTS refund_requests (
  id              BIGINT      NOT NULL AUTO_INCREMENT PRIMARY KEY,
  conversation_id VARCHAR(32) NOT NULL,
  order_no        VARCHAR(32) NOT NULL,
  reason_category VARCHAR(64) NOT NULL,
  status          VARCHAR(32) NOT NULL DEFAULT 'pending',
  created_at      DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_refund_conv (conversation_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='退款单(ch06)';
```

`app/db/models.py` 追加:

```python
class RefundRequest(Base):
    """退款单(ch06)。用户在前端表单确认后由 /api/refund 写入。

    `reason_category` 必须是 `app/refund/categories.REFUND_REASON_CATEGORIES`
    里的一个 —— 校验在端点层做(DB 不建 CHECK,与既有表的做法一致)。
    """

    __tablename__ = "refund_requests"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    order_no: Mapped[str] = mapped_column(String(32), nullable=False)
    reason_category: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
```

执行建表(把 `.env` 的连接串喂给 mysql 客户端,或用 docker):

```bash
docker exec -i mysql mysql -uroot -p"$MYSQL_ROOT_PASSWORD" mewhelp < db/ch06.sql
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_db_models.py -k refund`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add db/ch06.sql app/db/models.py tests/test_db_models.py
git commit -m "feat(ch06): refund_requests 表与 ORM 模型"
```

---

### Task 3: `POST /api/refund`

**Files:**
- Create: `app/api/refund.py`
- Modify: `app/main.py`(include_router)
- Modify: `app/schemas.py`(追加 `RefundRequestIn`;注意与 ORM 的 `RefundRequest` **同名不同物**,别混)
- Test: `tests/test_api_refund.py`

**Interfaces:**
- Consumes: `app.refund.categories.is_valid_category`、`app.db.models.RefundRequest`
- Produces: `POST /api/refund`;`app.schemas.RefundRequestIn`

- [ ] **Step 1: 写失败测试**

`tests/test_api_refund.py`:

```python
"""POST /api/refund 的端点测试。需要 MySQL。"""

import pytest
from sqlalchemy import select, text

from app.db.base import get_sessionmaker
from app.db.models import RefundRequest
from app.refund.categories import REFUND_REASON_CATEGORIES

pytestmark = pytest.mark.db

SCRATCH = "refundtest0000000000000000000000"


@pytest.fixture(autouse=True)
async def _cleanup():
    yield
    async with get_sessionmaker()() as s:
        await s.execute(
            text("DELETE FROM refund_requests WHERE conversation_id = :c"), {"c": SCRATCH}
        )
        await s.commit()


@pytest.mark.anyio
async def test_creates_row_with_valid_category(client_factory):
    client = client_factory()
    r = await client.post(
        "/api/refund",
        json={
            "session_id": SCRATCH,
            "order_no": "20240915",
            "reason_category": REFUND_REASON_CATEGORIES[0],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["order_no"] == "20240915"
    assert body["status"] == "pending"

    async with get_sessionmaker()() as s:
        row = (
            await s.execute(
                select(RefundRequest).where(RefundRequest.conversation_id == SCRATCH)
            )
        ).scalars().one()
        assert row.reason_category == REFUND_REASON_CATEGORIES[0]


@pytest.mark.anyio
async def test_rejects_category_outside_the_closed_set(client_factory):
    """不在固定类目里 → 422。且**不得落库**。"""
    client = client_factory()
    r = await client.post(
        "/api/refund",
        json={"session_id": SCRATCH, "order_no": "20240915", "reason_category": "随便写的原因"},
    )
    assert r.status_code == 422
    async with get_sessionmaker()() as s:
        n = (
            await s.execute(
                select(RefundRequest).where(RefundRequest.conversation_id == SCRATCH)
            )
        ).scalars().all()
        assert n == []


@pytest.mark.anyio
async def test_infra_failure_is_502_not_500(client_factory, monkeypatch):
    """基础设施故障一律 502 + 固定文案,不是 FastAPI 默认的 500。"""
    import app.api.refund as mod

    async def boom(*a, **k):
        from app.tools.errors import ToolInfrastructureError

        raise ToolInfrastructureError("数据库暂时不可用")

    monkeypatch.setattr(mod, "_persist", boom)
    client = client_factory()
    r = await client.post(
        "/api/refund",
        json={
            "session_id": SCRATCH,
            "order_no": "20240915",
            "reason_category": REFUND_REASON_CATEGORIES[0],
        },
    )
    assert r.status_code == 502
```

> `client_factory` 是本仓既有 fixture(`tests/conftest.py`),若签名不同**先读再用**。

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api_refund.py`
Expected: FAIL —— 404(端点不存在)

- [ ] **Step 3: 实现**

`app/schemas.py` 追加:

```python
class RefundRequestIn(BaseModel):
    """退款单提交入参。字段名与前端表单一一对应。"""

    session_id: str = Field(min_length=1, max_length=32)
    order_no: str = Field(min_length=4, max_length=32)
    reason_category: str = Field(min_length=1, max_length=64)
```

`app/api/refund.py`:

```python
"""退款单提交端点。

为什么是**独立端点**而不是模型工具:按钮点击是 HTTP 请求,够不到模型工具 ——
与 ch05 的 `/api/ticket` 同一个理由。

守护栏与对话端点一致:同一把会话锁串行化;写库**不重试**(重试会建出两张单)。
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.chat import get_store          # 复用同一把会话锁的单例依赖
from app.config import Settings, get_settings
from app.db.models import RefundRequest
from app.db.session import get_session
from app.memory.store import SessionStore
from app.refund.categories import is_valid_category
from app.sanitize import redact_api_key
from app.schemas import RefundRequestIn
from app.services.history import ensure_conversation
from app.tools.errors import ToolInfrastructureError

router = APIRouter()


async def _persist(session, request: RefundRequestIn) -> RefundRequest:
    """写一行退款单。单独成函数,便于测试注入故障。"""
    row = RefundRequest(
        conversation_id=request.session_id,
        order_no=request.order_no,
        reason_category=request.reason_category,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row


@router.post("/api/refund")
async def create_refund(
    request: RefundRequestIn,
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
    store: SessionStore = Depends(get_store),
) -> dict:
    # 类目校验**先于**任何 IO:不在固定集里就是请求语义错,422。
    if not is_valid_category(request.reason_category):
        raise HTTPException(status_code=422, detail="退款原因不在可选范围内")

    lock = store.lock_for(request.session_id)
    try:
        await asyncio.wait_for(lock.acquire(), timeout=settings.session_lock_timeout_seconds)
    except TimeoutError as exc:
        raise HTTPException(status_code=409, detail="该会话正在处理另一条消息,请稍后重试") from exc

    try:
        await ensure_conversation(
            session=session, session_id=request.session_id, user_id="demo-user"
        )
        row = await _persist(session, request)
        return {
            "id": row.id,
            "conversation_id": row.conversation_id,
            "order_no": row.order_no,
            "reason_category": row.reason_category,
            "status": row.status,
            "created_at": row.created_at.isoformat(),
        }
    except ToolInfrastructureError as exc:
        raise HTTPException(
            status_code=502, detail=redact_api_key(str(exc), settings.openai_api_key)
        ) from exc
    finally:
        lock.release()
```

> `get_store` 的取法照实现时 `app/api/chat.py` 的既有写法来(用 `Depends(get_store)`
> 直接 import 即可 —— 上面那行 `__import__` 只是示意,实现者按仓库风格写干净的导入)。

`app/main.py` 加一行 `app.include_router(refund_api.router)`(**必须在 `mount("/")` 之前** —— 否则静态目录会抢走 `/api/*`)。

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api_refund.py`
Expected: PASS(3 条)

- [ ] **Step 5: 提交**

```bash
git add app/api/refund.py app/schemas.py app/main.py tests/test_api_refund.py
git commit -m "feat(ch06): POST /api/refund —— 固定类目校验 + 502 语义"
```

---

### Task 4: 意图识别改八类 + `confidence`(评估集验证)

> **本任务是非可单测产出(Prompt)**,按项目规矩把 TDD 换成**标注样例评估**;
> 但 `IntentResult` 的形状与 `classify_intent` 的降级分支**仍走单测**。

**Files:**
- Modify: `app/agent/state.py`(`IntentResult` 加 `confidence`)
- Modify: `app/prompts.py`(`INTENT_SYSTEM_PROMPT` 改四件套)
- Modify: `app/agent/nodes.py`(`classify_intent` 带出 confidence)
- Modify: `app/agent/routing.py`(`"其他": FALLBACK` 进表)
- Modify: `evals/intent_cases.jsonl`、`scripts/run_intent_eval.py`
- Test: `tests/test_agent_intent.py`

**Interfaces:**
- Produces:`IntentResult(intent: str, confidence: float)`;`classify_intent` 返回 `{"intent", "confidence", "trace"}`

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_agent_intent.py`:

```python
@pytest.mark.anyio
async def test_confidence_is_written_to_state():
    """confidence 必须进 state(本章只用于日志/后续降级路,不改变路由)。"""
    node = make_classify_intent_node(model=_FakeIntentModel("物流", 0.87))
    out = await node({"user_input": "包裹到哪了"})
    assert out["intent"] == "物流"
    assert out["confidence"] == pytest.approx(0.87)


@pytest.mark.anyio
async def test_confidence_defaults_when_model_omits_it():
    """模型没给 confidence 时不得炸 —— 落一个保守值。"""
    node = make_classify_intent_node(model=_FakeIntentModel("物流", None))
    out = await node({"user_input": "包裹到哪了"})
    assert out["intent"] == "物流"
    assert out["confidence"] == 0.0
```

`_FakeIntentModel` 由实现者按本文件既有替身风格补(必须让**缺 confidence** 这一路真的走到)。

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_intent.py`
Expected: FAIL —— `TypeError: IntentResult.__init__() got an unexpected keyword argument 'confidence'`

- [ ] **Step 3: 改形状与节点**

`app/agent/state.py`:

```python
class IntentResult(BaseModel):
    """意图识别的结构化出参。取值越界由 routing 兜底,这里只描述形状。"""

    intent: str = Field(
        description="物流 / 订单 / 商品咨询 / 退款退货 / 售后 / 投诉 / 闲聊 之一;"
        "无法归入任何一类时为「其他」。"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="对本次判断的把握程度。**本章只用于日志**,不改变路由结果。",
    )
```

> **预检订正(PF-1)**:`confidence` 不能只停在 state —— spec §4.2 要求它**进 done 帧**。
> 本任务负责把 `log_turn` 的 `trace` 帧载荷加上它(T8 再透到 done 帧),
> 否则这条链路只做了一半。

`app/agent/nodes.py` 的 `log_turn` 里那一帧加 `confidence`(现为 `nodes.py:360-362`,
只带 `trace` / `intent` / `gate_passed` / `agent_steps`):

```python
        emit({"frame": "trace", "trace": full_trace,
              "intent": state.get("intent"), "gate_passed": state.get("gate_passed"),
              "agent_steps": state.get("agent_steps") or 0,
              "confidence": state.get("confidence")})
```

`app/agent/nodes.py` 的 `classify_intent` 返回值改为:

```python
        return {
            "intent": intent,
            "confidence": float(result.confidence or 0.0),
            "trace": [f"classify_intent:{intent}"],
        }
```

并在解析失败的 `except` 分支里同样带上 `"confidence": 0.0`。

`app/agent/routing.py` 的表加一行(**行为不变**:`其他` 本来就落在 `.get()` 的默认值上,
显式写进去是为了让 `INTENT_LABELS` 自动带上它 —— 提示词与路由表同源):

```python
    "闲聊": CHITCHAT,
    OTHER: FALLBACK,          # 显式:兜底标签进表,INTENT_LABELS 才带得上它
```

- [ ] **Step 4: 改 Prompt(四件套)**

`app/prompts.py` 的 `INTENT_SYSTEM_PROMPT` 换成:

```python
INTENT_SYSTEM_PROMPT = """你是电商客服的意图识别助手。
判断用户这一句话属于下面八类中的哪一类,并以 JSON 对象输出。

八类(逐条读完再选,不要看到「退」字就选退款退货):
- 物流:查询包裹位置、发货进度、配送时效
- 订单:查询订单状态、金额、下单时间
- 商品咨询:咨询商品价格、库存、规格、功能
- 退款退货:申请退款、退货、换货,或询问退款退货相关政策
- 售后:商品质量问题、维修、补发、安装
- 投诉:表达不满、要求赔偿、要求人工处理
- 闲聊:问候、感谢等日常寒暄(如「你好」「谢谢」)
- 其他:无法归入以上任何一类

**输出一个 JSON 对象,只有 intent 和 confidence 两个字段。**
- intent:上述八类之一的**原文**;
- confidence:0 到 1 之间的小数,表示你对本次判断的把握。

边界样例(照着判):
- 「你们的客服太差了,我要投诉」→ 投诉(不是售后)
- 「这个能退吗」→ 退款退货(问的是退货规则)
- 「订单 1001 买的是什么」→ 订单(问的是订单内容,不是商品本身)
- 「这手机壳好看吗」→ 其他(本店不卖手机壳,且这问题客服答不了)
- 「帮我写一首诗」→ 其他(不是寒暄,是与购物无关的具体请求)
- 「你好呀」→ 闲聊

**拿不准就归「其他」,不要硬塞进业务意图。** 不要输出 JSON 以外的任何内容。"""
```

- [ ] **Step 5: 扩充标注样例并跑评估**

`evals/intent_cases.jsonl` **追加**(保留既有 24 条;这些是**刻意构造的边界与怪问题**):

```jsonl
{"text": "这个能退吗", "expected": "退款退货"}
{"text": "你们客服态度太差了", "expected": "投诉"}
{"text": "订单 1001 买的是什么", "expected": "订单"}
{"text": "帮我写一首诗", "expected": "其他"}
{"text": "今天股市怎么样", "expected": "其他"}
{"text": "asdfghjkl", "expected": "其他"}
{"text": "机械键盘和猫砂盆哪个好养猫", "expected": "其他"}
{"text": "我要退货但是我不知道订单号", "expected": "退款退货"}
```

`scripts/run_intent_eval.py` 增加**置信度**输出与**「其他」专项**统计:

```python
        got = (await node({"user_input": row["text"]}))
        ok = got["intent"] == row["expected"]
        hit += ok
        emit(f"{'OK ' if ok else 'MISS'} 期望={row['expected']:<6} "
             f"实际={got['intent']:<6} conf={got.get('confidence', 0):.2f}  {row['text']}")
```

跑:

```bash
.venv/Scripts/python.exe scripts/run_intent_eval.py
```

Expected:整体准确率打印出来;**「其他」那几条必须判对**(验收标准 2)。
准确率与置信度分布**写进 `dev-notes/ch06.md`**,并由它定
`intent_confidence_threshold`(spec §9 的待实测项)。

- [ ] **Step 6: 全量单测 + 提交**

Run: `.venv/Scripts/python.exe -m pytest -m "not db"`
Expected: PASS

```bash
git add app/agent/state.py app/agent/nodes.py app/agent/routing.py app/prompts.py \
        evals/intent_cases.jsonl scripts/run_intent_eval.py tests/test_agent_intent.py
git commit -m "feat(ch06): 意图识别八类 + confidence,四件套 prompt + 边界标注样例"
```

---

### Task 5: 指代消解 + Query 改写(评估集验证)

**Files:**
- Modify: `app/prompts.py`(新 `RESOLVE_SYSTEM_PROMPT` + `build_resolve_messages`)
- Modify: `app/agent/nodes.py`(`make_resolve_references_node` 收 model 参数)
- Modify: `app/agent/graph.py`(传 model)
- Create: `evals/resolve_cases.jsonl`、`scripts/run_resolve_eval.py`
- Test: `tests/test_agent_resolve.py`

**Interfaces:**
- Consumes:`IntentResult` 无关;`to_lc_messages`
- Produces:`build_resolve_messages(*, history, user_input) -> list`;
  `make_resolve_references_node(*, model) -> Callable`(签名变化!)

> ⚠️ **本节点还兼任「每轮重置」**(见其 docstring):checkpointer 是进程级单例、
> thread_id=session_id,不清零的话上一轮的 `gate_passed` / `agent_steps` /
> `evidence` / `order_no` 会**被当成本轮的**。改实现时**必须保留重置职责**。

- [ ] **Step 1: 写失败测试**

`tests/test_agent_resolve.py`:

```python
def test_reset_still_clears_per_turn_channels():
    """加了消解之后,每轮重置**不能丢** —— 丢了会静默串轮。"""
    node = make_resolve_references_node(model=_EchoModel())
    out = asyncio.run(node({"user_input": "这个能退吗", "gate_passed": True,
                            "agent_steps": 3, "order_no": "1001"}))
    assert out["gate_passed"] is None
    assert out["agent_steps"] == 0
    assert out["order_no"] == ""


def test_resolution_failure_passes_through_original():
    """消解失败**原样透传** —— 绝不因为改写失败就答不出话。"""
    node = make_resolve_references_node(model=_BoomModel())
    out = asyncio.run(node({"user_input": "这个能退吗"}))
    assert out["resolved_input"] == "这个能退吗"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_resolve.py`
Expected: FAIL —— 签名不接受 `model=`

- [ ] **Step 3: 实现**

`app/prompts.py` 加:

```python
RESOLVE_SYSTEM_PROMPT = """你在做电商客服对话的**指代消解与问题改写**。

给你对话历史和用户这一轮的原话,输出**一句不依赖上下文也能看懂的完整问题**。

规则:
1. 「它能退吗」「这个多少钱」这类带**指代**的话,从历史里找出「它/这个」指的是什么,
   补全成完整问题;
2. 口语、模糊的问法**归一成标准问法**(如「多久能到」→「发货后多久能送达」);
3. **问题本身已经完整、指代已经明确的,原样输出,不要强行改写**;
4. 只输出改写后的那句话,**不要解释、不要加引号、不要输出 JSON**。
"""
```

`app/agent/nodes.py` 的 `make_resolve_references_node` 改为收 `model`,
try 里调 `build_resolve_messages`,失败则透传原话;重置逻辑**原样保留**。

> **预检订正(PF-2)**:本任务**只清既有通道**(`intent` / `evidence` / `gate_passed` /
> `agent_steps` / `tool_calls_made` / `reply` / `citations` / `choices`)。
> **不要**在这里清 `order_no` / `order_data` / `refund_decision` ——
> 那三个通道是 **T7** 才加进 `ChatState` 的,现在写它们会被 LangGraph 忽略或报错,
> 而那只在**跑图时**才暴露(典型「报错指向别处」)。
> **通道与它的清零同处一地**:那三个的清零并入 T7。

`app/agent/graph.py` 相应传 `model=model`。

- [ ] **Step 4: 标注样例 + 评估脚本 + 跑**

`evals/resolve_cases.jsonl`(每条含 `history` / `input` / `must_contain`):

```jsonl
{"history": [{"role": "user", "content": "我买的猫砂盆不想要了"}, {"role": "assistant", "content": "好的,请问是哪一张订单?"}], "input": "它能退吗", "must_contain": "猫砂盆"}
{"history": [{"role": "user", "content": "订单 20240915 什么时候发货"}], "input": "那它现在到哪了", "must_contain": "20240915"}
{"history": [], "input": "退货政策是怎么规定的", "must_contain": "退货政策"}
```

> 口径:`must_contain` 是**闭式**的 —— 只要求改写结果里出现该关键词,
> 不做字符串全等(模型在 temperature=0 下依然非确定,本仓已记账)。

`scripts/run_resolve_eval.py` 照 `run_intent_eval.py` 的结构写(字节输出、只出数字)。

```bash
.venv/Scripts/python.exe scripts/run_resolve_eval.py
```

Expected:逐条 OK/MISS + 准确率。**第 3 条(本来就完整的)必须原样透传**。

- [ ] **Step 5: 提交**

```bash
git add app/prompts.py app/agent/nodes.py app/agent/graph.py \
        evals/resolve_cases.jsonl scripts/run_resolve_eval.py tests/test_agent_resolve.py
git commit -m "feat(ch06): 指代消解 + Query 改写(失败透传,保留每轮重置)"
```

---

### Task 6: Query 扩写 + 多路检索去重合并

**Files:**
- Create: `app/retrieval/expand.py`
- Modify: `app/prompts.py`(`EXPAND_SYSTEM_PROMPT` + `build_expand_messages`)
- Create: `evals/expand_cases.jsonl`、`scripts/run_expand_eval.py`
- Test: `tests/test_retrieval_expand.py`

**Interfaces:**
- Produces:`expand_queries(model, *, text, max_queries) -> list[str]`;
  `multi_search(retriever, queries) -> list[RetrievedChunk]`(按 `chunk_id` 去重,
  保留首次出现顺序,**并按 score 降序**)

- [ ] **Step 1: 写失败测试**

`tests/test_retrieval_expand.py`:

```python
@pytest.mark.anyio
async def test_multi_search_merges_and_dedupes_by_chunk_id():
    """同一个块被两条查询召回 → 只留一份。"""
    r = _FakeRetriever({
        "q1": [_chunk(1, 0.9), _chunk(2, 0.5)],
        "q2": [_chunk(2, 0.7), _chunk(3, 0.8)],
    })
    got = await multi_search(r, ["q1", "q2"])
    assert [c.chunk_id for c in got] == [1, 3, 2]      # 按 score 降序
    assert len([c for c in got if c.chunk_id == 2]) == 1


@pytest.mark.anyio
async def test_multi_search_tolerates_one_query_failing():
    """一条查询挂掉不该让整次检索失败 —— 用剩下的。"""
    r = _FlakyRetriever(fail_on="q2")
    got = await multi_search(r, ["q1", "q2"])
    assert [c.chunk_id for c in got] == [1]


def test_expand_failure_falls_back_to_original_question():
    """扩写失败 → 退回单路原问题,绝不让检索空转。"""
    got = asyncio.run(expand_queries(_BoomModel(), text="这个能退吗", max_queries=3))
    assert got == ["这个能退吗"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_retrieval_expand.py`
Expected: FAIL —— ModuleNotFoundError

- [ ] **Step 3: 实现**

`app/prompts.py`:

```python
EXPAND_SYSTEM_PROMPT = """你要把用户的一个问题**泛化成多条侧重不同的检索查询**。

输出一个 JSON 对象,**只有 queries 一个字段**,值是字符串数组。

要求:
- 每条查询从**不同角度**切入同一件事(如:政策依据 / 时效 / 费用 / 例外情况);
- 不要复述原问题多遍,不要输出近义改写;
- 条数不超过 {max_queries} 条;
- 不要输出 JSON 以外的任何内容。
"""
```

`app/retrieval/expand.py`:

```python
"""Query 扩写与多路检索合并。

**扩写只发生在检索侧、现查现用** —— 库里的知识只留一份,不在入库侧拆存多份
(用户明确要求)。本模块因此不碰 chunker / writer。

错误语义:`multi_search` 里**单条查询失败不升级为整体失败**(用剩下的),
但 `ToolInfrastructureError`(Milvus/嵌入挂了)必须照原样上抛 ——
那是另一类故障,不能降级成「没搜到」。
"""

from app.tools.errors import ToolInfrastructureError


async def multi_search(retriever, queries: list[str]) -> list:
    """多路检索 → 按 chunk_id 去重 → 按 score 降序。"""
    merged: dict[int, object] = {}
    for q in queries:
        try:
            chunks = await retriever.search(q)
        except ToolInfrastructureError:
            raise                       # 基础设施故障照抛,不降级
        except Exception:
            continue                    # 单条查询的其他故障:跳过,用剩下的
        for c in chunks:
            old = merged.get(c.chunk_id)
            if old is None or c.score > old.score:
                merged[c.chunk_id] = c
    return sorted(merged.values(), key=lambda c: -c.score)


async def expand_queries(model, *, text: str, max_queries: int) -> list[str]:
    """扩写;任何失败都退回 [原问题] —— 绝不返回空列表(那会让检索空转)。"""
    ...
```

- [ ] **Step 4: 标注样例 + 跑评估**

`evals/expand_cases.jsonl`:

```jsonl
{"text": "这个能退吗", "min_queries": 2, "must_cover_any": ["退货", "退款", "无理由"]}
{"text": "我要退货,运费谁出", "min_queries": 2, "must_cover_any": ["运费", "谁承担"]}
{"text": "保修多久", "min_queries": 2, "must_cover_any": ["保修", "质保"]}
```

```bash
.venv/Scripts/python.exe scripts/run_expand_eval.py
```

Expected:每条输出生成条数与是否覆盖关键词;并**打印 JSON 可解析率**
(验收标准 2 的同类要求)。

- [ ] **Step 5: 提交**

```bash
git add app/retrieval/expand.py app/prompts.py evals/expand_cases.jsonl \
        scripts/run_expand_eval.py tests/test_retrieval_expand.py
git commit -m "feat(ch06): Query 扩写 + 多路检索去重合并(检索侧现查现用)"
```

---

### Task 7: 退款子流程节点 + 接线(含 interrupt/resume)

**Files:**
- Create: `app/agent/refund_nodes.py`
- Modify: `app/agent/state.py`(新通道)
- Modify: `app/agent/routing.py`(`REFUND` 出口;退款退货/售后 改指它)
- Modify: `app/agent/graph.py`(接子流程 + 路由表)
- Test: `tests/test_agent_refund.py`

**Interfaces:**
- Consumes:Task 1 的 `candidate_orders` / `REFUND_REASON_CATEGORIES`;Task 6 的 `multi_search` / `expand_queries`
- Produces:`make_refund_pick_order_node` / `make_refund_fetch_order_node` /
  `make_refund_expand_retrieve_node` / `make_refund_judge_node` /
  `make_refund_offer_node` / `make_refund_explain_node`;`REFUND` 常量

- [ ] **Step 1: 写失败测试**

`tests/test_agent_refund.py`(用**真 checkpointer + 替身模型**,不联网):

```python
@pytest.mark.anyio
async def test_missing_order_no_interrupts_with_cards():
    """缺订单号 → interrupt,载荷是 order_choice 帧的形状。"""
    graph = _build_refund_graph()          # 见下
    cfg = {"configurable": {"thread_id": "t-refund-1"}}
    async for mode, chunk in graph.astream(
        {"conversation_id": "t-refund-1", "user_input": "这个能退吗",
         "history": [], "trace": []},
        config=cfg, stream_mode=["custom", "updates"],
    ):
        if mode == "updates" and "__interrupt__" in chunk:
            payload = chunk["__interrupt__"][0].value
            assert payload["frame"] == "order_choice"
            assert payload["options"]                # 有候选
            return
    pytest.fail("没有出现 interrupt —— 订单卡片永远不会显示")


@pytest.mark.anyio
async def test_resume_with_order_no_drives_flow_to_offer():
    """resume 回填订单号 → 子流程走完并给出可提交退款的信号。"""
    graph = _build_refund_graph()
    cfg = {"configurable": {"thread_id": "t-refund-2"}}
    async for _ in graph.astream({..., "user_input": "这个能退吗"},
                                 config=cfg, stream_mode=["custom", "updates"]):
        pass
    frames = []
    async for mode, chunk in graph.astream(
        Command(resume="20240915"), config=cfg, stream_mode=["custom", "updates"]
    ):
        if mode == "custom":
            frames.append(chunk)
    assert any(f.get("frame") == "refund_offer" for f in frames)


@pytest.mark.anyio
async def test_context_already_has_order_no_does_not_interrupt():
    """问题里已经带了订单号 → 不弹卡片,直接走。"""
```

**必须再补一条**(F3 的回归防线):

```python
@pytest.mark.anyio
async def test_fetch_runs_exactly_once_across_resume():
    """resume 会把节点从头重跑 —— 取订单**不能**因此执行两次。

    做法:让替身 query_order 计数,断言 resume 之后计数只 +1。
    """
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_refund.py`
Expected: FAIL —— 节点不存在

- [ ] **Step 3: 实现(关键节点)**

`app/agent/state.py` 加通道:

```python
    # ---- 退款子流程(ch06)----
    order_no: str                 # 槽位;空 = 待回填
    order_data: dict              # query_order 的返回
    refund_decision: bool | None  # None = 还没判
```

> **预检订正(PF-2,承 T5)**:这三个通道的**清零也在这里做** ——
> 在 `make_resolve_references_node` 既有那组重置里加上 `order_no=""`、
> `order_data={}`、`refund_decision=None`。
> **通道与它的清零必须同处一地**:T5 已经明确不清它们(那时它们还不存在)。
> 漏了清零的后果是**静默串轮** —— 上一轮填过的订单号会被这一轮当成本轮槽位,
> 用户明明在问别的却直接跳进退款子流程。T7 的测试须覆盖这一点。

`app/agent/refund_nodes.py` 的**第一个节点必须极薄**:

```python
def make_refund_pick_order_node(*, emit):
    """订单号槽位闸。**interrupt 之外不干任何事。**

    ⚠️ 实测(F3):resume 时**节点从头重跑**。把这个节点里放进「取订单数据」,
    那笔查询就会执行两遍。所以这里只做一件事:决定要不要打断。
    """

    async def refund_pick_order(state) -> dict:
        if state.get("order_no"):
            return {"trace": [f"refund:已有订单号 {state['order_no']}"]}

        options = _candidate_options(state)      # 来自 app/refund/orders.py
        picked = interrupt({"frame": "order_choice", "options": options})
        return {
            "order_no": str(picked or "").strip(),
            "trace": ["refund:resume 回填订单号"],
        }

    return refund_pick_order
```

`refund_fetch_order` 调 `query_order`(工具),把 `order_data` 写进 state;
`order_not_found` 时置 `refund_decision=False` 并让路由去 `refund_explain`。

`refund_judge` **不新建 Agent**:

```python
def make_refund_judge_node(*, model, emit):
    """「这一单能不能退」—— 同一个主力模型,**一次** ainvoke,不绑工具、不进 ReAct。"""

    async def refund_judge(state) -> dict:
        msgs = build_refund_judge_messages(
            order=state.get("order_data") or {},
            evidence=state.get("evidence") or [],
            user_input=state["resolved_input"],
        )
        reply = await _stream_once(model, msgs, emit)     # 复用 agent 的流式取文本
        can_refund = "**不能退**" not in reply and "不能退" not in reply
        return {"refund_decision": can_refund, "reply": reply,
                "trace": [f"refund:judge={'能退' if can_refund else '不能退'}"]}
    return refund_judge
```

> 判据的**具体形态**(用不用结构化输出、怎么判能退)由实现者定,
> 但**必须可测**:测试里替身模型给「不能退」的文本时,路由必须走 `refund_explain`。

`app/agent/routing.py`:

```python
REFUND = "refund"
...
    "退款退货": REFUND,
    "售后": REFUND,
```

`app/agent/graph.py` 把 REFUND 接进条件边映射,并在 `_OUTLETS` 里加上子流程的
两个终节点(`refund_offer` 走 `log_turn`;**`refund_pick_order` 不再连回 log_turn**,
它可能停在 interrupt)。

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_refund.py`
Expected: PASS(含 `test_fetch_runs_exactly_once_across_resume`)

- [ ] **Step 5: 提交**

```bash
git add app/agent/state.py app/agent/routing.py app/agent/refund_nodes.py \
        app/agent/graph.py tests/test_agent_refund.py
git commit -m "feat(ch06): 退款子流程(interrupt 回填订单号)+ REFUND 路由出口"
```

---

### Task 8: 端点 —— resume 分支 + `["custom","updates"]` + interrupt 转帧

**Files:**
- Modify: `app/api/chat.py`
- Modify: `app/schemas.py`(`ChatRequest` 加 `resume`)
- Test: `tests/test_api_chat.py`

**Interfaces:**
- Consumes:Task 7 的子流程
- Produces:`POST /api/chat/stream` 支持 `{"resume": {"order_no": "..."}}`

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_api_chat.py`:

```python
@pytest.mark.anyio
async def test_interrupt_surfaces_as_order_choice_frame(client_factory):
    """F1 的回归防线:只给 custom 会把 interrupt 吞掉,前端永远看不到卡片。"""
    client = client_factory()
    r = await client.post("/api/chat/stream", json={"message": "这个能退吗"})
    assert "event: order_choice" in r.text


@pytest.mark.anyio
async def test_resume_request_continues_the_flow(client_factory):
    """带 resume 的请求走 Command(resume=...),并把流程推完。"""
    client = client_factory()
    first = await client.post("/api/chat/stream", json={"session_id": "s-resume",
                                                        "message": "这个能退吗"})
    assert "event: order_choice" in first.text
    second = await client.post("/api/chat/stream",
                               json={"session_id": "s-resume",
                                     "resume": {"order_no": "20240915"}})
    assert "event: order_choice" not in second.text     # 不该再问一次
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api_chat.py -k "order_choice or resume"`
Expected: FAIL

- [ ] **Step 3: 实现**

`app/schemas.py`:

```python
class ChatRequest(BaseModel):
    session_id: str | None = None
    user_id: str | None = None
    message: str | None = None
    #: 从挂起点续跑。给订单号即 resume;不给则开新一轮(见 spec F4)。
    resume: dict | None = None
```

`app/api/chat.py` 的 `generate()` 里:

```python
            stream_input = (
                Command(resume=request.resume)
                if request.resume
                else {
                    "conversation_id": session_id,
                    "user_input": request.message,
                    "history": history,
                    "trace": [],
                }
            )
            async for mode, chunk in graph.astream(
                stream_input,
                config={"configurable": {"thread_id": session_id}},
                stream_mode=["custom", "updates"],       # ← 必须带上 updates(F1)
            ):
                if mode == "updates":
                    if "__interrupt__" in chunk:
                        payload = chunk["__interrupt__"][0].value
                        yield _frame(payload.get("frame", "interrupt"),
                                     {k: v for k, v in payload.items() if k != "frame"})
                    continue                                  # 其余 updates 一律不外推
                ...                                           # 原有的 custom 分支不动
```

**三处必须一并处理**:

1. `request.message` 为 None 且无 `resume` → `422`(请求语义错),别让 `prepare_turn` 拿到 None;
2. **挂起时 `log_turn` 没跑**,所以这一轮的锁与历史都要照常释放/不落库 —— 现有
   `finally: lock.release()` 已经覆盖,但**不要**在 interrupt 分支里额外写库。
3. **PF-1 的后半截**:done 帧加上 confidence(承 T4 —— 它已把 confidence 放进
   `trace` 帧的载荷里):

   ```python
           yield _frame("done", {
               "finish_reason": "stop",
               "usage": None,
               "trace": final.get("trace") or [],
               "intent": final.get("intent"),
               "confidence": final.get("confidence"),      # ← 新增
               "gate_passed": final.get("gate_passed"),
               "agent_steps": final.get("agent_steps") or 0,
           })
   ```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api_chat.py`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add app/api/chat.py app/schemas.py tests/test_api_chat.py
git commit -m "feat(ch06): 端点支持 interrupt 转帧与 resume 续跑(stream_mode 加 updates)"
```

---

### Task 9: 前端 —— 订单卡片 + 退款表单(**Vibe Coding,不派实现者、不审查**)

> 按用户明确要求:前端**用 Vibe Coding 直接做**,我描述效果、实现随对话迭代,
> **不套 brainstorm / TDD / code review**。本任务在计划里只留坐标。

**Files:**
- Modify: `app/static/index.html`

**要做出来的效果**:
- 收到 `order_choice` 帧 → 在聊天气泡里渲染**可点的订单卡片**(号码 + 状态 + 商品 + 金额);
- 点一张卡 → `POST /api/chat/stream` 带 `{"session_id": …, "resume": {"order_no": …}}`,
  回复**接着流进同一个气泡**;
- 收到 `refund_offer` 帧 → 渲染退款表单(原因**下拉**,选项来自帧的 `categories`)+ 提交按钮;
- 提交 → `POST /api/refund` → 回执。

- [ ] **Step 1: 实现**(直接改,随时找我调效果)
- [ ] **Step 2: 我人工验**(浏览器,验收标准 4)
- [ ] **Step 3: 提交**

```bash
git add app/static/index.html
git commit -m "feat(ch06): 聊天页订单卡片 + 退款表单(Vibe Coding)"
```

---

### Task 10: ch06 端到端验收脚本

**Files:**
- Create: `scripts/acceptance_ch06.sh`

**Interfaces:** 无(脚本)

- [ ] **Step 1: 写脚本**

照 `scripts/acceptance_ch05.sh` 的既有骨架写(必读它再动手),要点:

- `set -uo pipefail`(**不加 `-e`**);
- `ask()` 走 **stdin heredoc**(含中文的请求体不能走 curl argv);
- 取帧的 helper **按字节读写**(`sys.stdin.buffer`);
- **预检**:`POST /api/ticket` 必须回 **422**,否则判定 8000 上不是当前代码;
- 服务没起时 curl 给 `000`,**先分这一支**(别让人去 netstat 一个不存在的 PID)。

覆盖四条验收:

```
验收 1:多轮意图(物流 → 退款 → 物流)每轮 intent 判对
验收 2:意图 JSON 可解析(拿 done 帧的 intent 非空)+ 怪问题落「其他」
验收 3:「这个能退吗」→ trace 里有 resolve_references 且走了 refund 子流程,
        且拿到订单与政策(evidence 非空)
验收 4:不带订单号问退款 → 收到 order_choice 帧 → 带 resume 再发一次 → 走完,
        且 refund_requests 里有对应行(直接查库)
```

- [ ] **Step 2: 跑**

```bash
bash -n scripts/acceptance_ch06.sh
bash scripts/acceptance_ch06.sh
```

Expected:全部通过;**红的时候如实贴原始输出,不要重试到绿**。

- [ ] **Step 3: 提交**

```bash
git add scripts/acceptance_ch06.sh
git commit -m "test(ch06): 端到端验收脚本(四条验收标准)"
```

---

### Task 11: 章级收尾

- [ ] **Step 1: 全量测试(含 db)**

Run: `.venv/Scripts/python.exe -m pytest`
Expected:全绿。若有红,**先查是不是环境问题**(`faq` 表已废弃;Windows 上
`test_tools_random.py` 那条 2000 次 `asyncio.run` 的用例会因套接字耗尽偶发红,
与本章无关 —— ch05 已记账)。

- [ ] **Step 2: 老验收回归网**

Run: `bash scripts/acceptance.sh`
Expected:**27 通过 / 0 失败**(ch05 收尾时的基线)。ch06 改了意图 Prompt 与路由,
**这条网会真的可能红** —— 红了先读 `intent` 与 `trace`(验收 1/2/5/6 都依赖路由,
脚本里已有标注说明红该往哪看)。

- [ ] **Step 3: 文档同步**

- `CLAUDE.md`:架构图加 REFUND 子流程与八类路由;「若干不读多文件就会踩的硬约束」补
  **F1(astream custom 吞 interrupt)** 与 **F3(resume 重跑节点)** 两条 —— 这两条
  都是「报错指向别处、极难定位」的类型。
- spec §12 记齐实现订正。

- [ ] **Step 4: 提交 + 交付**

```bash
git add CLAUDE.md docs/superpowers/specs/2026-09-20-ecommerce-cs-ch06-intent-router-design.md
git commit -m "docs(ch06): 章级文档同步(REFUND 子流程 + 两条实测硬约束)"
```

交付三样:**功能演示命令、测试结果、`dev-notes/ch06.md` 路径**。

---

## 自检记录

**1. Spec 覆盖**:spec §3 图结构 → T7/T8;§4 三个 prompt → T4/T5/T6;§5 接口 → T3/T8;
§6 数据 → T1/T2;§7 模块布局 → 各任务 Files 块;§8 错误语义 → T3 的 502 用例与
Global Constraints;§9 配置 → T4(intent 模型/阈值);§10 测试口径 → 各任务;
§11 风险 → Global Constraints 与 T1 的假绿提示。**无遗漏**。

**2. 占位符扫描**:无 TBD/TODO;每个代码步骤都有可执行内容。两处"由实现者定"
(`refund_judge` 的判据形态、`_FakeIntentModel` 的写法)都**给了必须满足的可测约束**,
不是「自行发挥」。

**3. 类型一致性**:`REFUND_REASON_CATEGORIES`(T1)→ `is_valid_category`(T3);
`candidate_orders`(T1)→ `refund_pick_order`(T7);
`multi_search` / `expand_queries`(T6)→ `refund_expand_retrieve`(T7);
`IntentResult.confidence`(T4)→ `classify_intent` 返回键(同名同型);
`Command` / `interrupt`(T7)→ 端点 resume(T8)。**一致**。
