import asyncio
import time
from collections import OrderedDict


class SessionStore:
    """每会话互斥锁的注册表。

    ch01 里它还管进程内会话历史(TTL + LRU 淘汰);ch02 历史迁到 MySQL 后
    那部分退役,只剩锁与清扫。

    `max_sessions` 因此**改为限制锁表大小** —— 它当初唯一的作用是限制
    历史条数,历史一走就成了死配置(读 .env.example 的人会以为它在管事)。
    这也顺带关闭了 ch01 spec §9 记录的「_locks/_touched 无硬数量上限」风险。

    锁仍然必要:同会话的并发请求会各自读到同一份历史、各自追加,
    不串行化就会后写覆盖先写。锁在进程内,历史在 MySQL —— 两者正交。

    多进程 / 多 worker 部署下本锁失效,本章不在范围内。
    """

    def __init__(self, *, ttl_seconds: float, max_sessions: int) -> None:
        self._ttl = ttl_seconds
        self._max = max_sessions
        # OrderedDict 而非普通 dict:淘汰要按 LRU 序,不能按插入序。
        self._touched: OrderedDict[str, float] = OrderedDict()
        self._locks: dict[str, asyncio.Lock] = {}

    # ---- 公开 ----

    def lock_for(self, session_id: str) -> asyncio.Lock:
        """取该会话的锁,按需创建。

        同时刷新使用时间并移到 LRU 末尾 —— 刚建的锁因此不会被紧接着的
        容量淘汰选中,幂等性得以保持。
        """
        self._purge()
        lock = self._locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session_id] = lock
        self._touch(session_id)
        self._enforce_capacity()
        return lock

    def active_lock_count(self) -> int:
        self._purge()
        return len(self._locks)

    # ---- 内部 ----

    def _touch(self, session_id: str) -> None:
        self._touched[session_id] = time.monotonic()
        self._touched.move_to_end(session_id)

    def _is_locked(self, session_id: str) -> bool:
        lock = self._locks.get(session_id)
        return lock is not None and lock.locked()

    def _drop(self, session_id: str) -> None:
        self._touched.pop(session_id, None)
        self._locks.pop(session_id, None)

    def _purge(self) -> None:
        """清扫超过 TTL 且未被持有的条目。持锁的不动 —— 它正在流式。"""
        now = time.monotonic()
        for session_id in list(self._touched.keys()):
            if now - self._touched[session_id] <= self._ttl:
                continue
            if self._is_locked(session_id):
                continue
            self._drop(session_id)

    def _enforce_capacity(self) -> None:
        """超容量时淘汰最久未使用的条目。

        遇到被持锁的**整个停下**,而不是跳过它去淘汰更新的 ——
        跳过会删掉比它更新的条目,把 LRU 语义彻底弄反。
        代价:极端情况下会短暂超出 max_sessions,上界仍由"同时进行中的
        流数量"兜住。

        淘汰一个**未被持有**的锁是安全的:没有持锁者就不存在被破坏的
        互斥,后续请求会拿到一把全新的、未锁定的锁。
        """
        while len(self._touched) > self._max:
            oldest = next(iter(self._touched))
            if self._is_locked(oldest):
                return
            self._drop(oldest)
