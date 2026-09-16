"""embedder 单元测试:懒加载契约、单次初始化、空输入、单例工厂。

全部用假 FlagEmbedding 模块替身 —— 真模型 2.2GB,单测绝不加载。
"""

import sys
import types

from app.retrieval.embedder import BgeM3Embedder, get_embedder


class _FakeModel:
    calls: list = []

    def __init__(self, path, **kw):
        _FakeModel.calls.append(("init", path))

    def encode(self, texts, **kw):
        _FakeModel.calls.append(("encode", tuple(texts), tuple(sorted(kw.items()))))
        # 只出 dense —— 出 sparse/colbert 会把键检查打红
        assert kw["return_dense"] is True
        assert kw["return_sparse"] is False
        assert kw["return_colbert_vecs"] is False
        return {"dense_vecs": [[0.25, 0.5] for _ in texts]}


def _install_fake(monkeypatch):
    _FakeModel.calls = []
    mod = types.ModuleType("FlagEmbedding")
    mod.BGEM3FlagModel = _FakeModel
    monkeypatch.setitem(sys.modules, "FlagEmbedding", mod)


def test_construction_does_not_import_flagembedding():
    """懒加载第一环:构造不 import FlagEmbedding(torch 在里面,重得很)。"""
    BgeM3Embedder("fake-path")
    assert "FlagEmbedding" not in sys.modules


def test_encode_loads_model_exactly_once(monkeypatch):
    """懒加载第二环:首次 encode 才初始化,再 encode 复用同一实例。"""
    _install_fake(monkeypatch)
    e = BgeM3Embedder("fake-path", max_length=512, batch_size=4)
    assert e.encode(["a", "b"]) == [[0.25, 0.5], [0.25, 0.5]]
    e.encode(["c"])
    inits = [c for c in _FakeModel.calls if c[0] == "init"]
    assert len(inits) == 1 and inits[0][1] == "fake-path"


def test_encode_passes_length_and_batch(monkeypatch):
    _install_fake(monkeypatch)
    e = BgeM3Embedder("p", max_length=77, batch_size=3)
    e.encode(["x"])
    kw = dict(_FakeModel.calls[-1][2])
    assert kw["max_length"] == 77 and kw["batch_size"] == 3


def test_encode_empty_list_skips_model(monkeypatch):
    """空列表不打扰模型:既不加载也不调用。"""
    _install_fake(monkeypatch)
    e = BgeM3Embedder("p")
    assert e.encode([]) == []
    assert _FakeModel.calls == []


def test_get_embedder_is_singleton():
    a = get_embedder("p", 8, 2)
    b = get_embedder("p", 8, 2)
    assert a is b
    get_embedder.cache_clear()
