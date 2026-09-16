"""writer 的纯逻辑单测:向量化文本拼接、整批一次 encode、标记与提交的**先后**。

不碰 MySQL(行、store、embedder、session 全是替身);SQL 往返与中断重跑
由 `test_kb_writer_db.py` 用真库覆盖。
"""

import pytest

from app.kb.writer import vector_text, vectorize_rows


class _FakeRow:
    def __init__(self, row_id, category="分类", questions="问法", answer="正文"):
        self.id = row_id
        self.category = category
        self.questions = questions
        self.answer = answer
        self.vector_id = None
        self.vectorize_status = "pending"


class _FakeEmbedder:
    def __init__(self):
        self.calls: list = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return [[float(len(t))] for t in texts]


class _FakeStore:
    """计数器放在 upsert 边界(而不是某个函数体里)—— ch02 教训。"""

    def __init__(self, fail_at: int | None = None):
        self.calls: list = []
        self.fail_at = fail_at

    def ensure_collection(self):
        self.calls.append(("ensure_collection",))

    def upsert(self, ids, vectors):
        if self.fail_at is not None and len(self.calls) + 1 >= self.fail_at:
            # 抛在记录**之前**:模拟「这次写根本没落地」(进程被打断),
            # 重跑必须原样重写这一批。
            raise RuntimeError("模拟中断")
        self.calls.append(("upsert", list(ids), list(vectors)))


class _FakeSession:
    def __init__(self):
        self.commits = 0

    async def commit(self):
        self.commits += 1


def test_vector_text_keeps_field_order_and_joins_with_newline():
    assert vector_text("分类", "问法", "正文") == "分类\n问法\n正文"


def test_vector_text_includes_category():
    """证伪「只用 questions + answer 拼」的实现:category 变了文本必须变。"""
    assert vector_text("A", "问", "答") != vector_text("B", "问", "答")


def test_vector_text_skips_empty_parts():
    assert vector_text("", "问", "答") == "问\n答"


@pytest.mark.anyio
async def test_vectorize_rows_marks_done_and_backfills_vector_id():
    rows = [_FakeRow(11), _FakeRow(12)]
    session, store, embedder = _FakeSession(), _FakeStore(), _FakeEmbedder()
    await vectorize_rows(session, store, embedder, rows)
    assert [(r.vector_id, r.vectorize_status) for r in rows] == [
        ("11", "done"),
        ("12", "done"),
    ]
    assert store.calls[0][0] == "upsert"
    assert store.calls[0][1] == ["11", "12"]        # pk = str(MySQL id)
    assert len(store.calls[0][2]) == 2              # 一行一个向量,顺序对应
    assert session.commits == 1


@pytest.mark.anyio
async def test_vectorize_rows_encodes_the_whole_batch_in_one_call():
    """整批一次 encode —— 逐行 encode 会把 BGE-M3 的批处理白白丢掉。"""
    rows = [_FakeRow(1, questions="aaa"), _FakeRow(2, questions="bb")]
    embedder = _FakeEmbedder()
    await vectorize_rows(_FakeSession(), _FakeStore(), embedder, rows)
    assert len(embedder.calls) == 1
    assert embedder.calls[0] == [vector_text("分类", "aaa", "正文"),
                                 vector_text("分类", "bb", "正文")]


@pytest.mark.anyio
async def test_vectorize_rows_empty_is_noop():
    embedder, store, session = _FakeEmbedder(), _FakeStore(), _FakeSession()
    await vectorize_rows(session, store, embedder, [])
    assert embedder.calls == [] and store.calls == [] and session.commits == 0


@pytest.mark.anyio
async def test_vectorize_rows_marks_nothing_when_upsert_fails():
    """写 Milvus 失败时,行**不能**被标记 done、**不能**提交。

    这是中断窗口 (a) 的安全前提:先 upsert 成功、再改状态提交。顺序反了的话,
    Milvus 里没有向量的行会被记成 done,重跑再也不会补 —— 静默永久缺失。
    """
    rows = [_FakeRow(11), _FakeRow(12)]
    session, store = _FakeSession(), _FakeStore(fail_at=1)
    with pytest.raises(RuntimeError):
        await vectorize_rows(session, store, _FakeEmbedder(), rows)
    assert [(r.vector_id, r.vectorize_status) for r in rows] == [
        (None, "pending"),
        (None, "pending"),
    ]
    assert session.commits == 0
