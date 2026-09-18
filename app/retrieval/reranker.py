"""bge-reranker-v2-m3 重排封装(懒加载单例)。

与 embedder 同一套懒加载 + 锁:构造不 import FlagEmbedding、首次 rerank 才加载
权重。`compute_score(pairs, normalize=True)` 输出 0-1 分数(sigmoid),用于置信度
与阈值判定。单测用假 FlagReranker 模块,不加载真模型。
"""

import os
import threading
from functools import lru_cache


class Reranker:
    def __init__(self, model_path: str, use_fp16: bool = False):
        self._model_path = model_path
        self._use_fp16 = use_fp16
        self._model = None
        self._load_lock = threading.Lock()

    def rerank(self, query: str, chunks: list[tuple[str, str]]) -> list[float]:
        """query 与每个 chunk 文本成对打分,返回同序 0-1 分数。

        chunks 是 [(chunk_id, text)];空列表直接返回空(不加载模型)。
        """
        if not chunks:
            return []
        pairs = [[query, text] for _, text in chunks]
        return list(self._ensure_model().compute_score(pairs, normalize=True))

    def _ensure_model(self):
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    # 权重本地提供(同 bge-m3),钉死离线模式,防止去连 huggingface。
                    os.environ.setdefault("HF_HUB_OFFLINE", "1")
                    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                    from FlagEmbedding import FlagReranker

                    self._model = FlagReranker(self._model_path, use_fp16=self._use_fp16)
        return self._model


@lru_cache(maxsize=1)
def get_reranker(model_path: str, use_fp16: bool = False) -> Reranker:
    """进程内单例(同 get_embedder)。"""
    return Reranker(model_path, use_fp16)
