import asyncio
import time
from collections import OrderedDict
from collections.abc import Sequence

from app.schemas import Message


class SessionStore:
    """进程内会话存储:惰性 TTL + LRU 上限 + 每会话一把互斥锁。

    不做后台清理任务 —— LRU 上限已给出内存硬上界,后台任务只改变
    "何时释放",不改变"是否释放"。

    被持锁的会话不会被淘汰(无论 TTL 还是 LRU)。否则正在流式的会话
    被淘汰后,新请求会拿到另一把锁,两个流并发写同一会话。
    """

    def __init__(self, *, ttl_seconds: float, max_sessions: int) -> None:
        self._ttl = ttl_seconds
        self._max = max_sessions
        self._sessions: OrderedDict[str, list[Message]] = OrderedDict()
        self._touched: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    # ---- 查询 ----

    def history(self, session_id: str) -> list[Message]:
        self._purge()
        if session_id in self._sessions:
            self._sessions.move_to_end(session_id)
            self._touched[session_id] = time.monotonic()
        return list(self._sessions.get(session_id, []))

    def active_session_count(self) -> int:
        self._purge()
        return len(self._sessions)

    # ---- 写入 ----

    def append(self, session_id: str, messages: Sequence[Message]) -> None:
        self._purge()
        history = list(self._sessions.get(session_id, []))
        history.extend(messages)
        self._sessions[session_id] = history
        self._sessions.move_to_end(session_id)
        self._touched[session_id] = time.monotonic()
        self._enforce_capacity()

    # ---- 锁 ----

    def lock_for(self, session_id: str) -> asyncio.Lock:
        """取该会话的锁,按需创建。

        同时刷新活跃时间 —— 这是"持锁会话不被 TTL 淘汰"的实现方式:
        请求一开始就会调 lock_for,等它被淘汰时锁已被持有。
        """
        self._purge()
        lock = self._locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session_id] = lock
        self._touched[session_id] = time.monotonic()
        return lock

    # ---- 内部 ----

    def _is_locked(self, session_id: str) -> bool:
        lock = self._locks.get(session_id)
        return lock is not None and lock.locked()

    def _drop(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._touched.pop(session_id, None)
        self._locks.pop(session_id, None)

    def _purge(self) -> None:
        """淘汰超时会话。被持锁的不动 —— 它正在流式。"""
        now = time.monotonic()
        for session_id in list(self._sessions.keys()):
            if now - self._touched.get(session_id, now) > self._ttl:
                if not self._is_locked(session_id):
                    self._drop(session_id)
        # 清理孤儿条目:lock_for 建过锁/时间戳、但从未 append 的会话。
        # 否则 _touched/_locks 不受 max_sessions 约束,长期运行会无界增长。
        # 同样按 TTL 判惰性 —— 立刻扫掉刚建的锁会破坏"同一 session 两次
        # lock_for 返回同一把锁"的幂等性;持锁的不动,否则请求刚拿锁就 400
        # 时,并发请求会拿到第二把锁,破坏互斥。
        for session_id in list(self._touched.keys()):
            if (
                session_id not in self._sessions
                and not self._is_locked(session_id)
                and now - self._touched[session_id] > self._ttl
            ):
                self._touched.pop(session_id, None)
                self._locks.pop(session_id, None)

    def _enforce_capacity(self) -> None:
        """超容量时从最老的开始淘汰。

        遇到被持锁的就**整个停下**,而不是跳过它去淘汰更新的 ——
        跳过会删掉比它更新的会话,把 LRU 语义彻底弄反。
        代价:极端情况下会短暂超出 max_sessions,这是有意的取舍,
        上界仍由"同时进行中的流数量"兜住。
        """
        while len(self._sessions) > self._max:
            oldest = next(iter(self._sessions))
            if self._is_locked(oldest):
                return
            self._drop(oldest)
