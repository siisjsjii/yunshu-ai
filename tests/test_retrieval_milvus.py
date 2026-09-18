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

    def create_index(self, collection_name, index_params, **kw):
        self.calls.append(("create_index", collection_name, index_params))

    def prepare_index_params(self, field_name, index_type, metric_type, **kw):
        return {"field_name": field_name, "index_type": index_type, "metric_type": metric_type}

    def load_collection(self, collection_name, **kw):
        self.calls.append(("load_collection", collection_name))

    def upsert(self, collection_name, data, **kw):
        self.calls.append(("upsert", collection_name, data))
        return {"upsert_count": len(data)}

    def flush(self, collection_name, **kw):
        self.calls.append(("flush", collection_name))

    def drop_collection(self, collection_name, **kw):
        self.calls.append(("drop_collection", collection_name))
        self._exists = False

    def search(self, collection_name, data, filter="", limit=10, output_fields=None,
               search_params=None, anns_field=None, **kw):
        self.calls.append(("search", collection_name, data, filter, limit, output_fields,
                           anns_field, search_params))
        if self._raw is not None:
            return self._raw
        return [self._hits]

    def hybrid_search(self, collection_name, reqs, ranker, limit, output_fields, **kw):
        self.calls.append(("hybrid_search", collection_name, reqs, ranker, limit, output_fields))
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


def test_ensure_collection_creates_schema_with_text_category_and_bm25():
    fake = _FakeClient(exists=False)
    _store(fake).ensure_collection()
    creates = [c for c in fake.calls if c[0] == "create_collection"]
    assert len(creates) == 1, f"应恰好建一次集合,实际调用序列:{fake.calls}"
    _, name, kw = creates[0]
    assert name == "knowledge"
    schema = kw["schema"]
    assert [f.name for f in schema.fields] == ["id", "text", "category", "vector", "text_bm25"]
    # text 字段带 jieba analyzer(ch04 混合检索,T0 实测)
    text_field = next(f for f in schema.fields if f.name == "text")
    assert "jieba" in str(text_field.params.get("analyzer_params", ""))
    # BM25 函数 text → text_bm25
    fn = schema.functions[0]
    assert fn.input_field_names == ["text"]
    assert fn.output_field_names == ["text_bm25"]
    # 建 dense + 稀疏两个索引,再 load
    assert len([c for c in fake.calls if c[0] == "create_index"]) == 2
    assert [c for c in fake.calls if c[0] == "load_collection"]


def test_ensure_collection_is_idempotent():
    """已存在则一次都不 create;连调两次也只建一次(含索引/load)。"""
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
    assert kinds == ["has_collection", "create_collection", "create_index", "create_index",
                     "load_collection", "has_collection"]


def test_drop_collection_is_noop_when_absent():
    fake = _FakeClient(exists=False)
    _store(fake).drop_collection()
    assert fake.calls == [("has_collection", "knowledge")]


def test_drop_collection_drops_when_present():
    fake = _FakeClient(exists=True)
    _store(fake).drop_collection()
    assert fake.calls == [("has_collection", "knowledge"), ("drop_collection", "knowledge")]
    # drop 之后 ensure 会重新建集合 —— 这正是「重建索引」的入口
    _store(fake).ensure_collection()
    assert "create_collection" in [c[0] for c in fake.calls]


def test_upsert_sends_text_category_vector_rows_then_flushes():
    """行形状含 text/category/vector;写完必须 flush(T0 实测)。"""
    fake = _FakeClient()
    _store(fake).upsert(
        ["7", "8"],
        ["文本一", "文本二"],
        ["分类A", "分类B"],
        [[0.1, 0.2], [0.3, 0.4]],
    )
    assert fake.calls == [
        ("upsert", "knowledge", [
            {"id": "7", "text": "文本一", "category": "分类A", "vector": [0.1, 0.2]},
            {"id": "8", "text": "文本二", "category": "分类B", "vector": [0.3, 0.4]},
        ]),
        ("flush", "knowledge"),
    ]


def test_upsert_empty_is_noop():
    """空批不打扰 Milvus:既不 upsert 也不白 flush 一次。"""
    fake = _FakeClient()
    _store(fake).upsert([], [], [], [])
    assert fake.calls == []


def test_search_returns_id_score_pairs():
    fake = _FakeClient(hits=[{"id": "7", "distance": 0.91}, {"id": "3", "distance": 0.42}])
    result = _store(fake).search([0.5, 0.5], top_k=2)
    assert result == [("7", 0.91), ("3", 0.42)]
    _, name, data, flt, limit, of, anns, sp = fake.calls[-1]
    assert name == "knowledge" and data == [[0.5, 0.5]] and limit == 2
    assert anns == "vector" and sp == {"metric_type": "IP"} and flt == ""


def test_search_passes_category_filter():
    fake = _FakeClient(hits=[])
    _store(fake).search([0.5, 0.5], top_k=2, category="商品规格手册")
    _, _, _, flt, _, _, _, _ = fake.calls[-1]
    assert flt == 'category == "商品规格手册"'


def test_hybrid_search_sends_dense_and_bm25_legs_with_rrf():
    fake = _FakeClient(hits=[{"id": "7", "distance": 0.9}])
    result = _store(fake).hybrid_search([0.5, 0.5], "猫砂盆", top_k=50)
    assert result == [("7", 0.9)]
    call = [c for c in fake.calls if c[0] == "hybrid_search"][0]
    _, name, reqs, ranker, limit, output_fields = call
    assert name == "knowledge" and limit == 50
    assert len(reqs) == 2
    assert [r.anns_field for r in reqs] == ["vector", "text_bm25"]
    assert reqs[1].data == ["猫砂盆"]  # BM25 腿 data = 原始文本(T0 实测)
    assert ranker.dict()["strategy"] == "rrf"


def test_bm25_search_sends_single_bm25_leg():
    fake = _FakeClient(hits=[{"id": "1", "distance": 0.016}])
    result = _store(fake).bm25_search("MH-LP100", top_k=10)
    assert result == [("1", 0.016)]
    call = [c for c in fake.calls if c[0] == "hybrid_search"][0]
    reqs = call[2]
    assert len(reqs) == 1
    assert reqs[0].anns_field == "text_bm25"
    assert reqs[0].data == ["MH-LP100"]


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
