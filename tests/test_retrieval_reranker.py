"""bge-reranker-v2-m3 重排封装:懒加载、compute_score 形状、单例。假 FlagEmbedding 模块。"""

import sys
import types

from app.retrieval.reranker import Reranker, get_reranker


class _FakeFlagReranker:
    calls: list = []

    def __init__(self, path, **kw):
        _FakeFlagReranker.calls.append(("init", path, kw))

    def compute_score(self, pairs, **kw):
        _FakeFlagReranker.calls.append(("compute", pairs, kw))
        return [float(i) for i in range(len(pairs))]


def _install(monkeypatch):
    _FakeFlagReranker.calls = []
    mod = types.ModuleType("FlagEmbedding")
    mod.FlagReranker = _FakeFlagReranker
    monkeypatch.setitem(sys.modules, "FlagEmbedding", mod)


def test_construction_does_not_import_flagembedding():
    Reranker("fake-path")
    assert "FlagEmbedding" not in sys.modules


def test_rerank_loads_model_once_and_returns_scores(monkeypatch):
    _install(monkeypatch)
    r = Reranker("fake-path")
    scores = r.rerank("query", [("1", "文本一"), ("2", "文本二")])
    assert scores == [0.0, 1.0]  # fake 按 index 返回
    assert len([c for c in _FakeFlagReranker.calls if c[0] == "init"]) == 1
    compute = [c for c in _FakeFlagReranker.calls if c[0] == "compute"][0]
    assert compute[1] == [["query", "文本一"], ["query", "文本二"]]
    assert compute[2]["normalize"] is True   # 0-1 分数,用于置信度/阈值


def test_rerank_empty_is_noop(monkeypatch):
    _install(monkeypatch)
    r = Reranker("p")
    assert r.rerank("q", []) == []
    assert _FakeFlagReranker.calls == []


def test_get_reranker_is_singleton():
    assert get_reranker("p") is get_reranker("p")
    get_reranker.cache_clear()
