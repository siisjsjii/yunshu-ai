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


def test_warmup_loads_model_without_encoding(monkeypatch):
    """预热:把权重读进来但不跑推理 —— 启动时就是这么用的。"""
    _install_fake(monkeypatch)
    e = BgeM3Embedder("fake-path")
    e.warmup()
    assert [c[0] for c in _FakeModel.calls] == ["init"]
    e.encode(["x"])
    assert [c[0] for c in _FakeModel.calls] == ["init", "encode"]


def test_concurrent_first_use_loads_model_exactly_once(monkeypatch):
    """并发首用只能加载一次。

    预热线程与首请求(或两个并发首请求)会同时进 `_ensure_model`;没有锁
    就是**两份 2.2GB 权重**常驻,而且两边都"成功",不会有任何报错。
    这个窗口靠 __init__ 里的 sleep 撑开,再用 Barrier 让线程同时出发 ——
    去掉锁时四个线程都会读到 `self._model is None`,计数变 4。
    """
    import threading
    import time

    class _SlowFakeModel:
        inits = 0

        def __init__(self, path, **kw):
            time.sleep(0.05)  # 撑开竞态窗口
            _SlowFakeModel.inits += 1

        def encode(self, texts, **kw):
            return {"dense_vecs": [[0.0] for _ in texts]}

    mod = types.ModuleType("FlagEmbedding")
    mod.BGEM3FlagModel = _SlowFakeModel
    monkeypatch.setitem(sys.modules, "FlagEmbedding", mod)

    embedder = BgeM3Embedder("fake-path")
    barrier = threading.Barrier(4)

    def worker():
        barrier.wait()
        embedder.encode(["x"])

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert _SlowFakeModel.inits == 1


def test_get_embedder_is_singleton():
    a = get_embedder("p", 8, 2)
    b = get_embedder("p", 8, 2)
    assert a is b
    get_embedder.cache_clear()
