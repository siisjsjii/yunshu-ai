"""BGE-M3 嵌入封装(离线在线共用)。

懒加载两段式:**构造不 import FlagEmbedding、首次 encode 才加载权重** ——
单测与不碰检索的服务启动都不能背上 torch 的导入开销。离线在线必须走同一
配置(max_length/batch_size),保证两头的向量在同一空间。
"""

import os
import threading
from functools import lru_cache


class BgeM3Embedder:
    """BGEM3FlagModel 的薄封装,只出 dense。"""

    def __init__(self, model_path: str, max_length: int = 1024, batch_size: int = 16):
        self._model_path = model_path
        self._max_length = max_length
        self._batch_size = batch_size
        self._model = None
        # 加载要十几秒,期间必须挡住第二个加载者。预热线程与首请求、
        # 或两个并发首请求,都会同时进 _ensure_model —— 没有锁就是**两份**
        # 2.2GB 权重(4.4GB 常驻),而且两边都"成功",不会有任何报错。
        self._load_lock = threading.Lock()
        # encode 也要串行:ch04 的后台任务与聊天检索共享同一个 torch 模型
        # 实例,并发调同一模型的前向(跨线程)不保证安全。锁的代价是任务
        # 批量 encode 时聊天查询可能等当前一批(约数秒,spec §9)。
        self._encode_lock = threading.Lock()

    def encode(self, texts: list[str]) -> list[list[float]]:
        """文本列表 → 1024 维已归一化稠密向量(顺序对应)。空列表直接返回。"""
        if not texts:
            return []
        with self._encode_lock:
            return [list(v) for v in self._ensure_model().encode(
                texts,
                batch_size=self._batch_size,
                max_length=self._max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )["dense_vecs"]]

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        with self._load_lock:
            # 双重检查:等锁期间可能已经被别人加载好了。
            if self._model is None:
                # 权重由 models/ 目录本地提供(T0 裁决),钉死离线模式,
                # 防止加载时去连被墙的 huggingface.co。
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                from FlagEmbedding import BGEM3FlagModel

                self._model = BGEM3FlagModel(self._model_path, use_fp16=False)
        return self._model

    def warmup(self) -> None:
        """把权重读进内存,供启动时预热调用(阻塞,十几秒)。

        存在的理由:首次加载远超工具超时(`tool_timeout_seconds` 默认 10 秒),
        冷进程的第一个检索请求必然超时。预热让它不在请求路径上发生。
        见 app/main.py 的 lifespan。
        """
        self._ensure_model()


@lru_cache(maxsize=1)
def get_embedder(model_path: str, max_length: int, batch_size: int) -> BgeM3Embedder:
    """进程内单例:2.2GB 权重只加载一次(同 get_settings 模式)。"""
    return BgeM3Embedder(model_path, max_length, batch_size)
