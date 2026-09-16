"""Milvus 封装单元测试:懒连接、建集合幂等、upsert 形状与 flush、search 扁平化、count 口径。

全部注入假 client —— 单测不连 Milvus(spec §8.1)。真实链路的连通性由 T7/T8
的 db/脚本任务与 acceptance.sh 验收 6 覆盖,单元测试只管参数与返回形状。
"""

import sys

from app.retrieval.milvus import MilvusVectorStore, get_vector_store


class _FakeClient:
    """记录每次调用的形状。参数名与 pymilvus MilvusClient 一致(实现用关键字传参)。"""

    def __init__(self, *, exists=False, count=0, hits=(), raw=None):
        self.calls: list = []
        self._exists = exists
        self._count = count
        self._hits = list(hits)
        self._raw = raw

    def has_collection(self, collection_name, **kw):
        self.calls.append(("has_collection", collection_name))
        return self._exists

    def create_collection(self, collection_name, **kw):
        self.calls.append(("create_collection", collection_name, kw))
        self._exists = True

    def upsert(self, collection_name, data, **kw):
        self.calls.append(("upsert", collection_name, data))
        return {"upsert_count": len(data)}

    def flush(self, collection_name, **kw):
        self.calls.append(("flush", collection_name))

    def search(self, collection_name, data, limit, output_fields, **kw):
        self.calls.append(("search", collection_name, data, limit, output_fields))
        if self._raw is not None:
            return self._raw
        return [self._hits]

    def query(self, collection_name, filter="", output_fields=None, **kw):
        self.calls.append(("query", collection_name, filter, output_fields))
        return [{"count(*)": self._count}]

    def get_collection_stats(self, collection_name, **kw):
        raise AssertionError(
            "行数核对必须走 query(count(*));stats 的 row_count 未扣 delete(spec §12)"
        )


def _store(fake, **kw):
    return MilvusVectorStore(
        "http://127.0.0.1:19530", "knowledge", dim=1024, client=fake, **kw
    )


def test_construction_does_not_import_pymilvus(monkeypatch):
    """懒连接第一环:构造不 import pymilvus。

    把 pymilvus 在 sys.modules 里置 None 会让任何 `import pymilvus` 抛
    ImportError —— 顶部 import 即加载的实现会在这里直接炸,故这条断言可证伪。
    """
    monkeypatch.setitem(sys.modules, "pymilvus", None)
    MilvusVectorStore("http://127.0.0.1:19530", "knowledge")


def test_ensure_collection_creates_with_ip_metric_and_string_pk():
    fake = _FakeClient(exists=False)
    _store(fake).ensure_collection()
    creates = [c for c in fake.calls if c[0] == "create_collection"]
    assert len(creates) == 1, f"应恰好建一次集合,实际调用序列:{fake.calls}"
    _, name, kw = creates[0]
    assert name == "knowledge"
    assert kw["dimension"] == 1024
    assert kw["metric_type"] == "IP"  # dense 已归一化,IP 等价余弦
    assert kw["id_type"] == "string"  # pk = str(MySQL id),不是自增整数
    # VARCHAR 主键必须自带 max_length,否则 Milvus 直接 1101 拒建(真机实测);
    # 下界 20 = BIGINT UNSIGNED 的十进制最大位数。
    assert kw["max_length"] >= 20
    assert kw["primary_field_name"] == "id"
    assert kw["vector_field_name"] == "vector"
    assert kw["auto_id"] is False


def test_ensure_collection_is_idempotent():
    """已存在则一次都不 create;连调两次也只 create 一次。"""
    fake = _FakeClient(exists=True)
    store = _store(fake)
    store.ensure_collection()
    store.ensure_collection()
    assert [c for c in fake.calls if c[0] == "create_collection"] == []

    fake2 = _FakeClient(exists=False)
    store2 = _store(fake2)
    store2.ensure_collection()
    store2.ensure_collection()
    kinds = [c[0] for c in fake2.calls]
    assert kinds == ["has_collection", "create_collection", "has_collection"]


def test_upsert_sends_id_vector_rows_then_flushes():
    """行形状 = [{"id": str(id), "vector": [...]}];写完必须 flush(T0 实测)。"""
    fake = _FakeClient()
    _store(fake).upsert(["7", "8"], [[0.1, 0.2], [0.3, 0.4]])
    assert fake.calls == [
        ("upsert", "knowledge", [
            {"id": "7", "vector": [0.1, 0.2]},
            {"id": "8", "vector": [0.3, 0.4]},
        ]),
        ("flush", "knowledge"),
    ]


def test_upsert_empty_is_noop():
    """空批不打扰 Milvus:既不 upsert 也不白 flush 一次。"""
    fake = _FakeClient()
    _store(fake).upsert([], [])
    assert fake.calls == []


def test_search_returns_id_score_pairs():
    fake = _FakeClient(hits=[{"id": "7", "distance": 0.91}, {"id": "3", "distance": 0.42}])
    result = _store(fake).search([0.5, 0.5], top_k=2)
    assert result == [("7", 0.91), ("3", 0.42)]
    assert fake.calls == [
        ("search", "knowledge", [[0.5, 0.5]], 2, ["id"]),
    ]


def test_search_empty_hit_list_returns_empty():
    assert _store(_FakeClient(hits=[])).search([0.5, 0.5], top_k=3) == []


def test_search_no_result_lists_returns_empty():
    """pymilvus 在无命中/无 query 向量时可能给空列表 —— 不能 IndexError。"""
    assert _store(_FakeClient(raw=[])).search([0.5, 0.5], top_k=3) == []


def test_count_uses_query_count_star():
    """§12 订正:行数一律 count(*),绝不读 get_collection_stats。"""
    fake = _FakeClient(count=34)
    assert _store(fake).count() == 34
    assert fake.calls == [("query", "knowledge", "", ["count(*)"])]


def test_get_vector_store_is_singleton():
    a = get_vector_store("http://127.0.0.1:19530", "knowledge")
    b = get_vector_store("http://127.0.0.1:19530", "knowledge")
    assert a is b
    get_vector_store.cache_clear()
