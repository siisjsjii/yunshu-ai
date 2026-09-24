"""T15 / H5 实测:**一次 approve 在请求路径上会冻住事件循环多久**。

为什么必须量:审核端点的向量化是**同步**做的(ch09 spec §9.2 要求立刻入库),
而 `vectorize_rows` 走 `embedder.encode` —— 那是 torch 前向,同步 CPU 代码。
它跑在**事件循环里**(FastAPI 的 `async def` 端点里直接调同步函数),
所以这期间**整个服务**(包括聊天那条 SSE)都停摆 —— 这正是 ch04 把向量化丢进
专用线程的理由(`app/kb/orchestrate.py` 的模块 docstring 第一段)。

三件事各量一次,分开报:
  ① BGE-M3 冷加载(只在冷进程第一次,不等于稳态);
  ② **单条文本 encode**(稳态,这才是 approve 的主要代价);
  ③ encode 期间事件循环的**最大冻结时长**(一个 10ms 心跳的最大间隔 ——
     直接就是「聊天接口被冻住多久」);
  ④ 端点整链路的延迟(真嵌入 + **替身 Milvus**)。

⚠️ **Milvus 那一段没量**:本机 19530 没起(实测 connect 直接超时),
所以 `store.ensure_collection()`(一次 `has_collection` RPC)与
`store.upsert`(一次写 + 一次 flush)的真实耗时**未知** —— 报告里按「未量」写,
别拿本文件的数去推它。

用法:`.venv/Scripts/python.exe .superpowers/probe_t15_blocking.py`
（**不是 pytest 用例**:单测不加载 2.2GB 权重是硬规矩。）
"""

import asyncio
import io
import sys
import time

import httpx

from app.api import review as review_api
from app.config import get_settings
from app.db.models import KnowledgeChunk, ReviewQueue
from app.db.session import get_session
from app.main import app
from app.retrieval.embedder import get_embedder


def out(msg: str) -> None:
    sys.stdout.buffer.write((msg + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()


class FakeStore:
    def __init__(self):
        self.ensure = 0
        self.upserts = []

    def ensure_collection(self):
        self.ensure += 1

    def upsert(self, ids, texts, categories, vectors):
        self.upserts.append(list(ids))


class FakeEmbedder:
    """只为端点整链路那一步留个空位(端点那一步注入的是**真**嵌入器)。"""


class FakeResult:
    def __init__(self, rows):
        self._rows = list(rows)

    def scalars(self):
        return self

    def all(self):
        return list(self._rows)


class FakeSession:
    """端点要的那一小块接口面(本探针**不碰真库**,从而也不动那张 30 行的池子)。"""

    def __init__(self, queue_row, kb_rows):
        self.queue_row = queue_row
        self.kb_rows = kb_rows
        self.commits = 0

    async def get(self, model, pk):
        return self.queue_row

    async def execute(self, stmt):
        if ReviewQueue.__tablename__ in str(stmt):
            return FakeResult([self.queue_row])
        if KnowledgeChunk.__tablename__ in str(stmt):
            return FakeResult(self.kb_rows)
        raise AssertionError(str(stmt))

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        pass


QUESTION = "t15 探针:换货到底要不要自己出运费"
ANSWER = "换货由我们承担运费,拒收产生的费用另计。"


async def main() -> None:
    settings = get_settings()
    embedder = get_embedder(
        settings.embedding_model_path,
        settings.embedding_max_length,
        settings.embedding_batch_size,
    )

    t0 = time.perf_counter()
    embedder.warmup()
    out(f"① BGE-M3 冷加载(冷进程第一次): {(time.perf_counter() - t0) * 1000:.0f} ms")

    text = f"faq\n{QUESTION}\n{ANSWER}"
    t0 = time.perf_counter()
    embedder.encode([text])
    first = (time.perf_counter() - t0) * 1000

    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        embedder.encode([text])
        times.append((time.perf_counter() - t0) * 1000)
    out(f"② 单条 encode:第一次 {first:.0f} ms;稳态 5 次 {[round(t) for t in times]} ms")

    # ③ 事件循环在 encode 期间被冻多久:一个 10ms 心跳的最大间隔
    ticks: list[float] = []

    async def heartbeat():
        while True:
            ticks.append(time.perf_counter())
            await asyncio.sleep(0.01)

    task = asyncio.create_task(heartbeat())
    await asyncio.sleep(0.05)
    t0 = time.perf_counter()
    embedder.encode([text])          # ← 与端点在事件循环里做的**同一件事**
    blocking = (time.perf_counter() - t0) * 1000
    # ⚠️ **必须先让心跳再跑一次**再取消:取消只在下一个 await 点生效,而心跳在
    # 冻结期间**一个点都打不出来** —— 直接 `task.cancel()` 的话,「冻结」这一段
    # 在 `gaps` 里根本不会出现(那正是这次测量第一版的错:量出来 16ms,
    # 而调用本身 90ms —— 两个数自相矛盾,矛盾本身就是线索)。
    await asyncio.sleep(0.05)
    task.cancel()
    gaps = [b - a for a, b in zip(ticks, ticks[1:])]
    out(f"③ encode 期间事件循环冻结:调用本身 {blocking:.0f} ms,"
        f"心跳最大间隔 {max(gaps) * 1000:.0f} ms(基线 {0.01 * 1000:.0f} ms)")

    # ④ 端点整链路:真嵌入 + 替身 Milvus + 替身 session(**不碰真库**)
    store = FakeStore()
    review_api.get_vector_store = lambda uri, coll: store       # 探针专用猴补
    review_api.get_embedder = lambda *a, **k: embedder
    session = FakeSession(
        ReviewQueue(id=1, standard_question=QUESTION, example_answer=ANSWER,
                    occurrences=1, status="pending", first_raw_question=QUESTION,
                    source_conversation_id="t15probe-blocking"),
        [],
    )

    async def fake_write_chunks(sess, chunks):
        row = KnowledgeChunk(id=90210, category="faq", questions=QUESTION,
                             answer=ANSWER, section_path=None, content_type="faq",
                             is_key_clause=False, vectorize_status="pending")
        sess.kb_rows.append(row)
        return 1

    review_api.write_chunks = fake_write_chunks
    app.dependency_overrides[get_session] = lambda: session

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        t0 = time.perf_counter()
        r = await client.post("/api/review/1/approve", json={})
        endpoint_ms = (time.perf_counter() - t0) * 1000
    out(f"④ approve 整链路(真嵌入 + 替身 Milvus):{endpoint_ms:.0f} ms,"
        f"响应 {r.status_code} {r.json()},upserts={store.upserts}")
    out("   ⚠️ Milvus 那两段(ensure_collection / upsert)本机没起,**未量**。")


if __name__ == "__main__":
    asyncio.run(main())
