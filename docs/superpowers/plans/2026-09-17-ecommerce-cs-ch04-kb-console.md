# 知识库管理台(ch04)实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 ch03 知识库加在线管理台:查看/上传源 Markdown 文档、后台触发向量化与从会话挖知识,独立管理页。

**Architecture:** 新增 `app/kb/jobs.py`(内存任务注册表)+ `app/kb/orchestrate.py`(后台任务:专用线程 + 自建独立 engine)+ `app/api/kb.py`(7 个端点)。把 `scripts/mine_qa.py` 的编排抽成可复用的 `mining.mine_knowledge`。前端 `app/static/admin.html`。

**Tech Stack:** 复用 ch03 全栈(FastAPI / SQLAlchemy asyncmy / BGE-M3 / Milvus / vanilla HTML+JS,无构建)。

**Spec:** `docs/superpowers/specs/2026-09-17-ecommerce-cs-ch04-kb-console-design.md`

## Global Constraints

- 解释器一律 `.venv/Scripts/python.exe`(Windows + Git Bash,locale cp936)。
- 单测不联网、不连 MySQL/Milvus、不加载 BGE-M3;`Settings(...)` 测试构造一律 `_env_file=None`;db 测试 `@pytest.mark.db` 读真实 `.env`;异步测试 `@pytest.mark.anyio`。
- **跑测试不加 CLI `-q`**(addopts 已有)。脚本/接口打印非 ASCII 用 `sys.stdout.buffer.write(...encode("utf-8"))` 或 `emit()`;含中文 HTTP 请求体走 stdin heredoc。
- 出站错误文本必须过 `app/sanitize.py:redact_api_key(text, api_key)`。
- **不改任何表结构**(ch03 DDL 既定)。文档↔块映射靠文件系统,不靠 DB 列。
- 依赖方向:新增 `api/kb.py → app/kb/` 是授权边;`app/kb/orchestrate.py` 不得被 `api/chat.py` 或 `services/` 引用。
- 提交信息中文 `type: 描述`;每任务一提交,先红后绿。
- 后台任务不阻塞事件循环:向量化/挖知识跑在专用线程,自建 engine,不用 `get_engine()` 单例。

## File Structure

- **Create** `app/kb/jobs.py` — `Job` dataclass + `JobStore`(threading.Lock,串行)。
- **Create** `app/kb/orchestrate.py` — `start_job` / 内部 `_run_vectorize` / `_run_mine`(线程 + 独立 engine)。
- **Create** `app/api/kb.py` — 7 个端点 + 纯函数 helper。
- **Modify** `app/kb/mining.py` — 抽 `mine_knowledge`。
- **Modify** `scripts/mine_qa.py` — 薄壳,调 `mine_knowledge`。
- **Modify** `app/kb/writer.py` — `vectorize_pending` 加可选 `progress` 回调。
- **Modify** `app/retrieval/embedder.py` — 加 `_encode_lock` 串行化 encode。
- **Modify** `app/schemas.py` — `UploadDocumentRequest`。
- **Modify** `app/main.py` — 挂 kb 路由。
- **Modify** `app/static/index.html` — 顶栏加「知识库」链接。
- **Create** `app/static/admin.html` — 管理页。
- **Modify** `scripts/acceptance.sh` — 增补验收。
- **Test** `tests/test_kb_jobs.py`、`tests/test_kb_orchestrate.py`、`tests/test_api_kb.py`;扩 `tests/test_kb_mining.py`、`tests/test_retrieval_embedder.py`。

---

### Task 1: JobStore(内存任务注册表)

**Files:**
- Create: `app/kb/jobs.py`
- Test: `tests/test_kb_jobs.py`

**Interfaces:**
- Produces: `Job`(dataclass:id, type, status, message, result, created_at, finished_at)、`JobStore`(start/update/get/list/is_busy)、`get_job_store()`(lru_cache 单例)。后续 orchestrate.py 与 api/kb.py 依赖这些。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_kb_jobs.py
import time
from app.kb.jobs import JobStore, get_job_store


def test_start_returns_job_with_running_status():
    store = JobStore()
    job = store.start("mine")
    assert job is not None
    assert job.type == "mine"
    assert job.status == "running"
    assert job.message == ""


def test_busy_store_rejects_second_job():
    store = JobStore()
    assert store.start("mine") is not None
    assert store.start("vectorize") is None      # 忙 → 拒绝
    assert store.is_busy() is True


def test_update_changes_status_message_result():
    store = JobStore()
    job = store.start("vectorize")
    store.update(job.id, message="处理中")
    store.update(job.id, status="done", message="完成", result={"processed": 3})
    got = store.get(job.id)
    assert got.status == "done"
    assert got.message == "完成"
    assert got.result == {"processed": 3}
    assert got.finished_at is not None


def test_get_unknown_returns_none():
    assert JobStore().get("nope") is None


def test_list_newest_first_and_empty_after_done_allows_new_job():
    store = JobStore()
    a = store.start("mine"); store.update(a.id, status="done")
    b = store.start("vectorize")
    assert [j.id for j in store.list()] == [b.id, a.id]
    assert store.is_busy() is True  # b 仍在跑
    store.update(b.id, status="failed")
    assert store.is_busy() is False


def test_get_job_store_is_singleton():
    assert get_job_store() is get_job_store()
    get_job_store.cache_clear()
```

- [ ] **Step 2: 跑测试确认红**

Run: `.venv/Scripts/python.exe -m pytest tests/test_kb_jobs.py -v`
Expected: FAIL / collection error(`app.kb.jobs` 不存在)

- [ ] **Step 3: 实现**

```python
# app/kb/jobs.py
import threading
import time
import uuid
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass
class Job:
    id: str
    type: str                    # "vectorize" | "mine"
    status: str                  # "running" | "done" | "failed"
    message: str = ""
    result: dict | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None


class JobStore:
    """内存任务注册表。threading.Lock 保护;同时只允许一个 running 任务。

    进程重启即失(与 SessionStore 同模式,管理/演示工具,不持久化)。
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._running_id: str | None = None

    def start(self, job_type: str) -> Job | None:
        with self._lock:
            if self._running_id is not None:
                return None
            job = Job(id=uuid.uuid4().hex[:12], type=job_type, status="running")
            self._jobs[job.id] = job
            self._running_id = job.id
            return job

    def update(self, job_id: str, *, status=None, message=None, result=None) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if status is not None:
                job.status = status
            if message is not None:
                job.message = message
            if result is not None:
                job.result = result
            if job.status in ("done", "failed"):
                job.finished_at = time.time()
                if self._running_id == job_id:
                    self._running_id = None

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def is_busy(self) -> bool:
        with self._lock:
            return self._running_id is not None


@lru_cache(maxsize=1)
def get_job_store() -> JobStore:
    return JobStore()
```

- [ ] **Step 4: 跑测试确认绿**

Run: `.venv/Scripts/python.exe -m pytest tests/test_kb_jobs.py -v`
Expected: PASS(6 passed)

- [ ] **Step 5: 提交**

```bash
git add app/kb/jobs.py tests/test_kb_jobs.py
git commit -m "feat: ch04 T1 JobStore(内存任务注册表,threading.Lock 串行,6 测试先行)"
```

---

### Task 2: embedder encode 锁(跨线程串行化)

**Files:**
- Modify: `app/retrieval/embedder.py:20-56`
- Test: `tests/test_retrieval_embedder.py`

**Interfaces:**
- Consumes: 现有 `BgeM3Embedder.encode(texts) -> list[list[float]]`(签名不变)。
- Produces: encode 加 `_encode_lock` 串行;外部接口不变(仅内部并发安全)。ch04 后台任务与聊天检索共享单例 embedder,靠此锁避免 torch 模型跨线程并发。

- [ ] **Step 1: 写失败测试(可证伪)**

在 `tests/test_retrieval_embedder.py` 追加:

```python
def test_encode_is_serialized_across_threads(monkeypatch):
    """并发 encode 必须串行 —— 后台任务与聊天共用同一 torch 模型实例。

    去掉锁时,4 个线程会同时进入 encode(计数 4);加锁后串行(峰值并发 1)。
    """
    import threading
    import time

    class _SerialProbe:
        active = 0
        peak = 0
        lock = threading.Lock()

        def __init__(self, path, **kw):
            pass

        def encode(self, texts, **kw):
            with _SerialProbe.lock:
                _SerialProbe.active += 1
                _SerialProbe.peak = max(_SerialProbe.peak, _SerialProbe.active)
            time.sleep(0.02)          # 撑开窗口,让并发若存在则重叠
            with _SerialProbe.lock:
                _SerialProbe.active -= 1
            return {"dense_vecs": [[0.0] for _ in texts]}

    import sys, types
    mod = types.ModuleType("FlagEmbedding")
    mod.BGEM3FlagModel = _SerialProbe
    monkeypatch.setitem(sys.modules, "FlagEmbedding", mod)

    from app.retrieval.embedder import BgeM3Embedder
    e = BgeM3Embedder("p")
    barrier = threading.Barrier(4)
    def worker():
        barrier.wait()
        e.encode(["x"])
    ts = [threading.Thread(target=worker) for _ in range(4)]
    for t in ts: t.start()
    for t in ts: t.join()
    assert _SerialProbe.peak == 1, f"并发 encode 峰值应=1,实际 {_SerialProbe.peak}"
```

- [ ] **Step 2: 跑测试确认红**

Run: `.venv/Scripts/python.exe -m pytest tests/test_retrieval_embedder.py::test_encode_is_serialized_across_threads -v`
Expected: FAIL(`peak == 4`,因为还没锁)

- [ ] **Step 3: 实现**

`app/retrieval/embedder.py`:`__init__` 里加 `self._encode_lock = threading.Lock()`;`encode` 主体包进 `with self._encode_lock:`。

```python
    def __init__(self, model_path, max_length=1024, batch_size=16):
        ...
        self._load_lock = threading.Lock()
        # encode 也要串行:后台任务与聊天检索共享同一个 torch 模型实例,
        # 并发调用同一模型的前向(跨线程)不保证安全。锁的代价是任务批量
        # encode 时聊天查询可能等当前一批(约数秒,spec §9)。
        self._encode_lock = threading.Lock()

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with self._encode_lock:
            return [list(v) for v in self._ensure_model().encode(
                texts, batch_size=self._batch_size, max_length=self._max_length,
                return_dense=True, return_sparse=False, return_colbert_vecs=False,
            )["dense_vecs"]]
```

- [ ] **Step 4: 跑测试确认绿 + 全套 embedder 回归**

Run: `.venv/Scripts/python.exe -m pytest tests/test_retrieval_embedder.py -v`
Expected: PASS(8 passed,含新增 1)

- [ ] **Step 5: 提交**

```bash
git add app/retrieval/embedder.py tests/test_retrieval_embedder.py
git commit -m "feat: ch04 T2 embedder encode 串行化(后台任务与聊天共享模型实例,防跨线程并发)"
```

---

### Task 3: mine_knowledge 抽取(编排从脚本搬进可调用函数)

**Files:**
- Modify: `app/kb/mining.py`(追加 `mine_knowledge`)
- Modify: `scripts/mine_qa.py`(改成薄壳)
- Test: `tests/test_kb_mining.py`(追加编排级用例)

**Interfaces:**
- Consumes: 现有 `mining` 内部函数(`load_turns`/`mine_batch`/`dedupe_pairs`/`find_near_duplicate`/`keep_pairs`/`finalize_staging`/`write_staging`/`existing_fingerprints`/`staging_fingerprints`/`batch_conversations`/`render_conversations`)。
- Produces:
  ```python
  async def mine_knowledge(*, session_factory, batch_size, dedupe_threshold, model,
                           store, embedder, batch_no=None, dry_run=False,
                           progress=None) -> dict
  # 返回 {"extracted","kept","discarded","near_dup_dropped","inserted",
  #        "kept_qa":[{"question","answer","category"}]}
  ```
  `session_factory` 是 `async_sessionmaker`(脚本传 `get_sessionmaker()`,web 传自建 engine 的 maker)。`progress` 是可选 `async callable(message: str)`。

- [ ] **Step 1: 写失败测试(db,真 MySQL + fake 三件套)**

在 `tests/test_kb_mining.py` 追加(复用本文件已有的 `_FakeModel`/`_FakeChain`/`_FakeStore`/`_FakeEmbedder` 或就地定义):

```python
@pytest.mark.db
@pytest.mark.anyio
async def test_mine_knowledge_end_to_end_result_shape():
    """编排级:抽取→去重→入库→结果字典,全程 fake 模型/向量库。

    只造 1 个会话、模型固定吐 2 条问答对(1 条与已入库重复)。断言:
    - 结果字典 key 齐全且计数自洽;
    - kept 的那条真进了 knowledge_chunks(pending);
    - staging 状态被 finalize 成 kept/discarded。
    """
    from app.kb.mining import mine_knowledge

    await _cleanup()  # 本文件已有的清理
    saved = await _saved_foreign_pending()  # 若本文件有;没有则跳过
    try:
        async with get_sessionmaker()() as session:
            session.add(Conversation(id=SCRATCH_CONVERSATION, user="t", status="active"))
            session.add_all([
                MessageRecord(conversation_id=SCRATCH_CONVERSATION, role="user", content="怎么退货"),
                MessageRecord(conversation_id=SCRATCH_CONVERSATION, role="assistant", content="七天无理由。"),
            ])
            await session.commit()

        # 预置一条已入库知识,让第二条问答对与它重复
        async with get_sessionmaker()() as session:
            session.add(KnowledgeChunk(category=SCRATCH_CATEGORY, questions="怎么换货", answer="能换。"))
            await session.commit()

        class _M:
            def with_structured_output(self, schema, **kw):
                return _FakeChain(MinedQaBatch(items=[
                    MinedQaItem(question="怎么退货", answer="七天无理由。", category="退换货"),
                    MinedQaItem(question="怎么换货", answer="能换。", category="退换货"),
                ]))

        progress_messages = []
        async def progress(msg): progress_messages.append(msg)

        result = await mine_knowledge(
            session_factory=get_sessionmaker(),
            batch_size=5, dedupe_threshold=0.95,
            model=_M(), store=_FakeStore(), embedder=_FakeEmbedder(),
            batch_no=SCRATCH_BATCH, dry_run=False, progress=progress,
        )
        assert set(result) == {"extracted","kept","discarded","near_dup_dropped","inserted","kept_qa"}
        assert result["extracted"] == 2
        assert result["inserted"] == 1            # 与已入库重复的那条被去重
        assert result["kept_qa"] == [{"question":"怎么退货","answer":"七天无理由。","category":"退换货"}]
        assert progress_messages                 # 分批时至少报过一次进度

        async with get_sessionmaker()() as session:
            rows = (await session.execute(
                select(KnowledgeChunk).where(KnowledgeChunk.category == "退换货",
                    KnowledgeChunk.questions == "怎么退货"))).scalars().all()
            assert len(rows) == 1 and rows[0].vectorize_status == "pending"
    finally:
        await _cleanup()
        await get_engine().dispose()
```

(注:测试里 `_FakeModel`/`_FakeChain`/`_FakeStore`/`_FakeEmbedder` 若本文件已有则复用,否则按 `test_kb_mining.py` 顶部既有定义照抄一份。)

- [ ] **Step 2: 跑测试确认红**

Run: `.venv/Scripts/python.exe -m pytest tests/test_kb_mining.py::test_mine_knowledge_end_to_end_result_shape -v`
Expected: FAIL(`mine_knowledge` 未定义)

- [ ] **Step 3: 实现 mine_knowledge**

追加到 `app/kb/mining.py`(核心逻辑 = 现在 `scripts/mine_qa.py:_run` 的搬运,**去重快照必须在抽取循环之前取**):

```python
async def mine_knowledge(*, session_factory, batch_size, dedupe_threshold, model,
                         store, embedder, batch_no=None, dry_run=False,
                         progress=None) -> dict:
    """从会话挖知识的完整编排。脚本与 web 任务共用这一份。

    结果计数含义与 ch03 一致:kept = 最终保留并入库的问答对数;
    discarded = 字面去重 + 向量近重复总共丢掉的。去重快照必须在抽取**之前**
    取 —— 否则本轮刚写进 staging 的产物会把自己全判重(ch03 命门)。
    """
    import secrets
    import time as _time

    batch_no = batch_no or f"mine-{_time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"

    async def _report(msg: str) -> None:
        if progress is not None:
            await progress(msg)

    async with session_factory() as session:
        turns = await load_turns(session)

    # 去重基准快照(抽取之前!)
    async with session_factory() as session:
        seen = await existing_fingerprints(session)
        seen |= await staging_fingerprints(session)

    all_pairs: list[QaPair] = []
    failed_batches = 0
    batch_index = 0
    for group in batch_conversations(turns, batch_size):
        batch_index += 1
        text = render_conversations(group)
        try:
            pairs = await mine_batch(model=model, conversation_text=text)
        except MineParseError:
            failed_batches += 1
            await _report(f"第 {batch_index} 批解析失败,跳过")
            continue
        all_pairs.extend(pairs)
        if not dry_run:
            async with session_factory() as session:
                await write_staging(session, batch_no=batch_no,
                                    source_ref=group[0].conversation_id if group else None,
                                    pairs=pairs)
        await _report(f"第 {batch_index} 批:{len(group)} 轮 → {len(pairs)} 条")

    kept, dropped = dedupe_pairs(all_pairs, seen)
    near_duplicates = []
    for pair in kept:
        if await find_near_duplicate(store=store, embedder=embedder,
                                     question=pair.question, threshold=dedupe_threshold):
            near_duplicates.append(pair)
    if near_duplicates:
        near_prints = {question_fingerprint(p.question) for p in near_duplicates}
        kept = [p for p in kept if question_fingerprint(p.question) not in near_prints]
        dropped = dropped + near_duplicates

    inserted = 0
    if not dry_run:
        async with session_factory() as session:
            inserted = await keep_pairs(session, kept)
        async with session_factory() as session:
            await finalize_staging(session, batch_no=batch_no, kept=kept)
        if inserted:
            await _report(f"入库 {inserted} 条,待向量化")

    return {
        "extracted": len(all_pairs),
        "kept": len(kept),
        "discarded": len(dropped),
        "near_dup_dropped": len(near_duplicates),
        "inserted": inserted,
        "kept_qa": [{"question": p.question, "answer": p.answer, "category": p.category}
                    for p in kept],
    }
```

- [ ] **Step 4: 改 `scripts/mine_qa.py` 为薄壳**

`_run` 只做:读 settings/args → 造 model/store/embedder + 一个把消息写 stdout 的 `progress` → 调 `mine_knowledge` → 打印 result。删除脚本内已搬走的分批/去重/入库逻辑,`mine_qa.py` 不再定义 `_render_batch`/`_batched`(改 import `render_conversations`/`batch_conversations`/`mine_knowledge`)。

- [ ] **Step 5: 跑测试确认绿 + 既有 mining 回归**

Run: `.venv/Scripts/python.exe -m pytest tests/test_kb_mining.py -v`
Expected: PASS(23 passed:22 既有 + 1 新增)

- [ ] **Step 6: 手动冒烟脚本仍可用(dry-run)**

Run: `.venv/Scripts/python.exe scripts/mine_qa.py --dry-run`(需真实 key)
Expected: 输出「抽取完成…」「最终保留 K 条…」,不写库。

- [ ] **Step 7: 提交**

```bash
git add app/kb/mining.py scripts/mine_qa.py tests/test_kb_mining.py
git commit -m "feat: ch04 T3 mine_knowledge 编排抽取(脚本与 web 共用,去重快照时序不变)"
```

---

### Task 4: 后台编排(专用线程 + 独立 engine)

**Files:**
- Create: `app/kb/orchestrate.py`
- Modify: `app/kb/writer.py:108`(`vectorize_pending` 加 `progress`)
- Test: `tests/test_kb_orchestrate.py`

**Interfaces:**
- Consumes: `JobStore`(T1)、`mine_knowledge`(T3)、`vectorize_pending`、`get_embedder`、`get_vector_store`、`create_extract_model`、`redact_api_key`。
- Produces:
  ```python
  def start_job(job_store, job_type: str, settings, *, store=None, embedder=None, model=None) -> Job | None
  # job_type ∈ {"vectorize","mine"};返回 None 表示忙;store/embedder/model 缺省时用真实单例(测试注入 fake)
  ```

- [ ] **Step 1: 写失败测试(db:真 MySQL + fake 三件套,验证线程+engine 胶水)**

```python
# tests/test_kb_orchestrate.py
import asyncio
import time

import pytest
from sqlalchemy import select

from app.db.base import get_engine, get_sessionmaker
from app.db.models import KnowledgeChunk
from app.kb.jobs import JobStore
from app.kb.orchestrate import start_job
from app.kb.chunker import Chunk
from app.kb.writer import write_chunks

pytestmark = pytest.mark.db

SCRATCH_CATEGORY = "ch04-probe-orch"


class _FakeEmbedder:
    def encode(self, texts):
        return [[float(len(t))] * 3 for t in texts]

class _FakeStore:
    def __init__(self): self.calls = []
    def ensure_collection(self): pass
    def upsert(self, ids, vectors): self.calls.append(tuple(ids))
    def flush(self): pass
    def search(self, vector, top_k): return []
    def count(self): return len(self.calls)
    def query(self, **kw): return [{"count(*)": len(self.calls)}]


async def _cleanup():
    from sqlalchemy import delete
    async with get_sessionmaker()() as s:
        await s.execute(delete(KnowledgeChunk).where(KnowledgeChunk.category == SCRATCH_CATEGORY))
        await s.commit()


@pytest.mark.anyio
async def test_vectorize_job_runs_in_background_and_completes():
    await _cleanup()
    try:
        async with get_sessionmaker()() as s:
            await write_chunks(s, [Chunk(SCRATCH_CATEGORY, "问", "答", "路径", "policy")])
        store = _FakeStore()
        js = JobStore()
        job = start_job(js, "vectorize", _settings(), store=store, embedder=_FakeEmbedder())
        assert job is not None
        # 轮询直到完成(后台线程)
        for _ in range(200):
            j = js.get(job.id)
            if j.status != "running":
                break
            await asyncio.sleep(0.05)
        j = js.get(job.id)
        assert j.status == "done", j.message
        assert j.result["processed"] == 1
        async with get_sessionmaker()() as s:
            row = (await s.execute(select(KnowledgeChunk).where(KnowledgeChunk.category == SCRATCH_CATEGORY))).scalars().one()
            assert row.vectorize_status == "done"
            assert row.vector_id == str(row.id)
        assert js.is_busy() is False
    finally:
        await _cleanup()
        await get_engine().dispose()
```

`_settings()` = `Settings(_env_file=None, **REQUIRED)`(按 `tests/test_config.py` 里的 `REQUIRED` 口径);`database_url` 用真实 `.env` —— 故本文件标 `pytest.mark.db` 并**读真实 .env**,`_settings()` 里 `database_url` 从 `get_settings()` 取。

- [ ] **Step 2: 跑测试确认红**

Run: `.venv/Scripts/python.exe -m pytest tests/test_kb_orchestrate.py -v`
Expected: FAIL(`app.kb.orchestrate` 不存在)

- [ ] **Step 3: writer.vectorize_pending 加 progress**

`app/kb/writer.py:108` 签名改 `async def vectorize_pending(session, store, embedder, *, batch_size: int = 16, progress=None) -> int`;循环内每批后:

```python
        await vectorize_rows(session, store, embedder, rows)
        total += len(rows)
        if progress is not None:
            await progress(f"已向量化 {total} 行")
```

(默认 None,`build_kb.py` 与既有测试不受影响。)

- [ ] **Step 4: 实现 orchestrate.py**

```python
# app/kb/orchestrate.py
import asyncio
import threading

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.models import KnowledgeChunk
from app.kb.jobs import Job, JobStore
from app.kb.mining import mine_knowledge
from app.kb.writer import vectorize_pending
from app.llm import create_extract_model
from app.retrieval.embedder import get_embedder
from app.retrieval.milvus import get_vector_store
from app.sanitize import redact_api_key


def _spawn(job_store: JobStore, job: Job, coro_fn) -> None:
    def target() -> None:
        try:
            asyncio.run(coro_fn())
        except Exception as exc:
            # 正常路径里 coro 自己会置 done/failed;这里兜住线程级意外,
            # 避免 job 永远停在 running。
            job_store.update(job.id, status="failed",
                             message=redact_api_key(str(exc), _settings().openai_api_key))
    threading.Thread(target=target, name=f"kb-job-{job.id}", daemon=True).start()


def _fresh_factory(settings):
    # 独立 engine:不用 get_engine() 的 lru_cache 单例(绑主事件循环)。
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def _count(session, status: str) -> int:
    stmt = select(func.count(KnowledgeChunk.id))
    if status == "pending":
        stmt = stmt.where(KnowledgeChunk.vectorize_status == "pending")
    elif status == "done":
        stmt = stmt.where(KnowledgeChunk.vectorize_status == "done")
    return (await session.execute(stmt)).scalar_one()


async def _run_vectorize(job_store, job_id, settings, store, embedder) -> None:
    engine, factory = _fresh_factory(settings)
    try:
        async def progress(msg): job_store.update(job_id, message=msg)
        async with factory() as session:
            pending = await _count(session, "pending")
        store.ensure_collection()
        if pending:
            await progress(f"待向量化 {pending} 行")
            async with factory() as session:
                processed = await vectorize_pending(session, store, embedder,
                                                    batch_size=settings.embedding_batch_size,
                                                    progress=progress)
        else:
            processed = 0
        async with factory() as session:
            done = await _count(session, "done")
        job_store.update(job_id, status="done", message=f"完成,处理 {processed} 行",
                         result={"processed": processed, "total_pending_before": pending,
                                 "milvus_count": store.count(), "done_rows": done})
    finally:
        await engine.dispose()


async def _run_mine(job_store, job_id, settings, store, embedder, model) -> None:
    engine, factory = _fresh_factory(settings)
    try:
        async def progress(msg): job_store.update(job_id, message=msg)
        result = await mine_knowledge(
            session_factory=factory, batch_size=settings.mine_batch_conversations,
            dedupe_threshold=settings.dedupe_threshold, model=model,
            store=store, embedder=embedder, dry_run=False, progress=progress)
        job_store.update(job_id, status="done",
                         message=f"抽出 {result['extracted']} 条,保留 {result['kept']} 条",
                         result=result)
    finally:
        await engine.dispose()


def start_job(job_store: JobStore, job_type: str, settings, *,
              store=None, embedder=None, model=None) -> Job | None:
    job = job_store.start(job_type)
    if job is None:
        return None
    store = store or get_vector_store(settings.milvus_uri, settings.milvus_collection)
    embedder = embedder or get_embedder(settings.embedding_model_path,
                                        settings.embedding_max_length,
                                        settings.embedding_batch_size)
    if job_type == "vectorize":
        _spawn(job_store, job, lambda: _run_vectorize(job_store, job.id, settings, store, embedder))
    elif job_type == "mine":
        model = model or create_extract_model(settings)
        _spawn(job_store, job, lambda: _run_mine(job_store, job.id, settings, store, embedder, model))
    else:
        job_store.update(job.id, status="failed", message=f"未知任务类型 {job_type}")
    return job
```

注:`_settings()` 是模块内 `from app.config import get_settings; _settings = get_settings`(兜底报错时取 api_key 用)。

- [ ] **Step 5: 跑测试确认绿**

Run: `.venv/Scripts/python.exe -m pytest tests/test_kb_orchestrate.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add app/kb/orchestrate.py app/kb/writer.py tests/test_kb_orchestrate.py
git commit -m "feat: ch04 T4 后台任务编排(专用线程 + 独立 engine;vectorize_pending 加 progress)"
```

---

### Task 5: api/kb.py 端点 + schemas

**Files:**
- Create: `app/api/kb.py`
- Modify: `app/schemas.py`(加 `UploadDocumentRequest`)
- Modify: `app/main.py:58-64`(挂路由)
- Test: `tests/test_api_kb.py`

**Interfaces:**
- Consumes: `JobStore`/`get_job_store`(T1)、`start_job`(T4)、`ingest.parse_corpus_file`、`writer.write_chunks`、`MilvusVectorStore.count`、`get_session`、`get_settings`。
- Produces(端点,JSON):
  - `GET /api/kb/stats` → `{total,done,pending,milvus_count}`
  - `GET /api/kb/documents` → `[{name,type,title,chunk_count}]`
  - `GET /api/kb/documents/{name}` → `{name,type,content}`
  - `POST /api/kb/documents` → 201 `{name,type,title,chunks_added,chunks_skipped}`
  - `POST /api/kb/jobs/vectorize` / `POST /api/kb/jobs/mine` → 201 `{job_id}`
  - `GET /api/kb/jobs` → `[{id,type,status,message,created_at,finished_at}]`
  - `GET /api/kb/jobs/{id}` → job 详情

- [ ] **Step 1: schemas 加请求模型**

`app/schemas.py` 追加:

```python
from typing import Literal

class UploadDocumentRequest(BaseModel):
    """上传知识文档请求。文件名只收安全 Markdown;type 限 ch03 三类。"""

    filename: str
    type: Literal["policy", "faq", "manual"]
    content: str

    @field_validator("filename")
    @classmethod
    def _safe_md_name(cls, v: str) -> str:
        if not re.fullmatch(r"[\w\-]+\.md", v):
            raise ValueError("文件名须为 [字母数字_-\] .md,不含路径分隔符")
        return v
```

(需在文件顶部 `import re`。)

- [ ] **Step 2: 写失败测试(纯函数 helper 直测 + TestClient 端点)**

```python
# tests/test_api_kb.py
import json
import pytest
from pathlib import Path

from app.kb.jobs import get_job_store


def _settings():
    from app.config import Settings
    return Settings(_env_file=None, openai_base_url="http://x", openai_api_key="k",
                    openai_model="m", database_url="mysql+asyncmy://u:p@localhost/db")


# ── 文档列表/读取 helper(纯函数,不连 DB)──
from app.api.kb import list_documents, read_document, KNOWN_TYPES


def test_list_documents_parses_type_title_and_chunk_count(tmp_path):
    (tmp_path / "a.md").write_text("<!--type: policy-->\n\n# 退货政策\n\n满 99 包邮。\n", encoding="utf-8")
    docs = list_documents(tmp_path, max_chars=100, overlap_chars=10)
    assert len(docs) == 1
    d = docs[0]
    assert d["name"] == "a.md"
    assert d["type"] == "policy"
    assert d["title"] == "退货政策"
    assert d["chunk_count"] >= 1


def test_list_documents_skips_non_markdown(tmp_path):
    (tmp_path / "x.txt").write_text("hi", encoding="utf-8")
    assert list_documents(tmp_path, 100, 10) == []


def test_upload_filename_validation():
    from app.schemas import UploadDocumentRequest
    with pytest.raises(Exception):
        UploadDocumentRequest(filename="../../etc/passwd", type="policy", content="x")
    with pytest.raises(Exception):
        UploadDocumentRequest(filename="a.txt", type="policy", content="x")
    ok = UploadDocumentRequest(filename="退货.md", type="faq", content="正文")
    assert ok.filename == "退货.md"
```

端点级(用 `fastapi.testclient.TestClient` + `dependency_overrides` 覆盖 `get_settings`/`get_session`/`get_job_store`;沿用 `test_api_chat.py` 的 override 手法)再补 2 条:上传重名 409、任务忙时 409。端点测试若过重,退化为直接调 helper + `start_job` 的单测(在 Task 4 已覆盖),本文件聚焦纯函数与校验即可 —— 但**至少一条**走 TestClient 验路由挂载正确。

- [ ] **Step 3: 实现 api/kb.py**

结构:纯函数 helper(`list_documents(dirpath, max_chars, overlap_chars)`、`read_document(dirpath, name)`) + 薄端点。上传落盘到 `KNOWLEDGE_DIR`(与 `scripts/build_kb.py` 的 `KNOWLEDGE_DIR` 同值 = 项目根 `knowledge/`),注入 `<!--type: X -->` 首行,再 `parse_corpus_file` → `write_chunks`。文档名用 `Path` 校验,读取用 `Path(...).resolve()` 防越出 `KNOWLEDGE_DIR`。job 端点调 `start_job(get_job_store(), "vectorize"|"mine", settings)`,None → 409。

- [ ] **Step 4: main.py 挂路由**

```python
from app.api.kb import router as kb_router
...
app.include_router(chat_router)
app.include_router(extract_router)
app.include_router(kb_router)   # 加在 mount("/") 之前
```

- [ ] **Step 5: 跑测试确认绿**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api_kb.py tests/test_kb_jobs.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add app/api/kb.py app/schemas.py app/main.py tests/test_api_kb.py
git commit -m "feat: ch04 T5 api/kb 端点(stats/文档列表读/上传/jobs)+ UploadDocumentRequest"
```

---

### Task 6: 前端 admin.html + 聊天页入口

**Files:**
- Create: `app/static/admin.html`
- Modify: `app/static/index.html:215-217`(header 加链接)

**Interfaces:**
- Consumes: T5 的 7 个端点。
- Produces: 独立管理页,无构建。`index.html` 顶栏「知识库」链接到 `/admin.html`。

- [ ] **Step 1: index.html 加链接**

`header` 里 `＋ 新对话` 按钮旁加:

```html
<a href="/admin.html" style="color:#fff;font-size:13px;padding:7px 12px;text-decoration:none;
   border:2px solid #fff;border-radius:3px;">知识库</a>
```

- [ ] **Step 2: 写 admin.html**

复用 `index.html` 的 `:root` 配色与网格纸 body。三块布局:

1. 顶部:标题「知识库管理」+ 回聊天链接 + `GET /api/kb/stats` 的四个数。
2. 文档区:列表(名/类型徽章/标题/块数),点开 `GET /api/kb/documents/{name}` 显示原文;上传表单(文件名 input + 类型 select + textarea,提交 `POST /api/kb/documents`)。
3. 任务区:两个按钮 → `POST /api/kb/jobs/vectorize` / `mine`,拿到 `job_id` 后 `setInterval` 轮询 `GET /api/kb/jobs/{id}`(1s),running 显示 message,done 渲染 result(mine 的 `kept_qa` 列表 + 计数),failed 显示 message。忙时接口 409 → 提示「已有任务在跑」。

页面上传成功/向量化/挖知识后都刷新 stats。纯 vanilla JS + `fetch`,无依赖。

- [ ] **Step 3: 起服务手验**

Run: `.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000`,浏览器开 `http://localhost:8000/admin.html`
Expected: 文档列表显示 3 份现有文档;上传一份测试 .md 后列表 +1,点开可见原文;点向量化/挖知识,轮询显示状态。

- [ ] **Step 4: 提交**

```bash
git add app/static/admin.html app/static/index.html
git commit -m "feat: ch04 T6 管理页 admin.html(文档查看/上传/任务轮询)+ 聊天页入口"
```

---

### Task 7: 端到端验收增补 + 冒烟

**Files:**
- Modify: `scripts/acceptance.sh`

**Interfaces:**
- Consumes: T5 端点。
- Produces: 验收 8(上传→向量化→改说法召回)、验收 9(挖知识幂等)。

- [ ] **Step 1: acceptance.sh 增补**

新增两段(放在验收 7 之后):

- **验收 8(上传 → 向量化 → 改说法召回)**:`POST /api/kb/documents`(文件名 `运费补充.md`、type `policy`、content 含「顺丰到付运费由买家承担」)→ 拿返回 → `POST /api/kb/jobs/vectorize` → 轮询 `GET /api/kb/jobs/{id}` 至 done → 再走一次聊天「到付运费谁出」断言回复含「买家承担」。
- **验收 9(挖知识幂等)**:`POST /api/kb/jobs/mine` → 轮询 done → 记录 result.kept;再 `POST /api/kb/jobs/mine` 一次 → 断言第二次 `inserted` == 0(已入库,幂等)。

脚本内新增轮询 helper:

```bash
poll_job() {
  JOBID="$1" "$PYTHON" -c '
import asyncio, os, sys, time, urllib.request
job = os.environ["JOBID"]
for _ in range(600):
    data = urllib.request.urlopen(f"http://localhost:8000/api/kb/jobs/{job}").read()
    p = __import__("json").loads(data)
    if p["status"] != "running":
        sys.stdout.buffer.write(data)
        sys.exit(0)
    time.sleep(1)
sys.stdout.buffer.write(b"{\"status\":\"running\"}")
sys.exit(1)
'
}
```

(HTTP 请求体含中文 → 一律 stdin heredoc,沿用脚本既有头注。)

- [ ] **Step 2: 起服务跑全套验收**

Run: `bash scripts/acceptance.sh`(前置:服务已启动、MySQL + Milvus 在跑)
Expected: 验收 1–9 全绿(通过数 > 此前 19)。

- [ ] **Step 3: 提交**

```bash
git add scripts/acceptance.sh
git commit -m "feat: ch04 T7 验收增补(上传→向量化→召回;挖知识幂等)"
```

---

### Task 8: 收尾

**Files:**
- Modify: `dev-notes/ch04.md`(新建)、`docs/superpowers/specs/...ch04...design.md`(§12 订正)、`AGENTS.md`、`CLAUDE.md`

- [ ] **Step 1: dev-notes/ch04.md 分阶段补记**(每任务一段,四样:原话/产出/纠偏/翻车)
- [ ] **Step 2: spec §12 实现订正回填**(如有偏离:encode 锁代价、后台线程独立 engine、任务串行 409 等已实现的点)
- [ ] **Step 3: AGENTS.md / CLAUDE.md 状态行更新**(ch04 已交付;架构边界加 `api/kb → kb` 与后台任务模型;高频命令加 ch04 端点/验收)
- [ ] **Step 4: 全量回归**

Run: `.venv/Scripts/python.exe -m pytest`
Expected: 全绿(此前 283 + 本章新增)

- [ ] **Step 5: 提交**

```bash
git add -A
git commit -m "docs: ch04 T8 收尾(dev-notes/spec §12/AGENTS/CLAUDE 回填)"
```

---

## 依赖与顺序

T1 → T2(独立)→ T3 → T4 → T5 → T6 → T7 → T8。T2 与 T1 无依赖可并行;T3 依赖 mining 既有代码、独立于 T1/T2;T4 依赖 T1+T3;T5 依赖 T1+T4;T7 依赖 T5+T6。

## 风险与回退

- 后台线程独立 engine 跨线程问题 → 已用「每任务自建 engine + dispose」规避;若实测仍炸,回退方案是把任务改成 subprocess 调 `scripts/*.py`(但 cp936 子进程编码是本仓库复发型坑,不作为首选)。
- encode 锁使任务批量编码时聊天查询等待 → 实测可接受(§9);不可接受再考虑给后台任务单独一个 embedder 实例(代价 2×2.2GB,不优先)。
- 上传文件名/类型校验 → 已在 `UploadDocumentRequest` 用 Pydantic validator 收口,422 语义诚实。
