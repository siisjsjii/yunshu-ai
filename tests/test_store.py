"""SessionStore 的测试 —— 它现在只是一个「每会话互斥锁注册表」。

ch02 把会话历史迁到 MySQL 后,ch01 里针对进程内历史的测试(读写、TTL 淘汰、
LRU 上限、`active_session_count`)全部删除 —— 被测对象不存在了,留着就是空壳。
历史自身的读写测试由 `tests/test_history.py`(Task 9)覆盖。

以下四类保证不因历史迁移而消失,必须继续成立:
  - 锁的串行化;
  - `lock_for` 幂等(同一 session 两次调用返回同一把锁);
  - 被持锁的条目不被 TTL 清扫、也不被容量淘汰;
  - 容量淘汰遇到被持锁的条目时整个停下(而不是跳过它去删更新的)。
"""

import asyncio

import pytest

from app.memory.store import SessionStore


@pytest.fixture
def clock(monkeypatch):
    """可控时钟,避免测试里真的 sleep。"""
    state = {"now": 1000.0}
    monkeypatch.setattr(
        "app.memory.store.time.monotonic", lambda: state["now"], raising=True
    )
    return state


# ---- 基本语义 ----


def test_lock_for_returns_same_lock_for_same_session():
    store = SessionStore(ttl_seconds=60, max_sessions=10)
    assert store.lock_for("s1") is store.lock_for("s1")


def test_lock_for_returns_different_locks_for_different_sessions():
    store = SessionStore(ttl_seconds=60, max_sessions=10)
    assert store.lock_for("s1") is not store.lock_for("s2")


@pytest.mark.anyio
async def test_same_session_lock_serialises_access():
    store = SessionStore(ttl_seconds=60, max_sessions=10)
    lock = store.lock_for("s1")
    order: list[str] = []

    async def worker(tag: str):
        async with lock:
            order.append(f"{tag}-in")
            await asyncio.sleep(0.01)
            order.append(f"{tag}-out")

    await asyncio.gather(worker("a"), worker("b"))

    assert order in (
        ["a-in", "a-out", "b-in", "b-out"],
        ["b-in", "b-out", "a-in", "a-out"],
    )


@pytest.mark.anyio
async def test_different_sessions_do_not_block_each_other():
    store = SessionStore(ttl_seconds=60, max_sessions=10)
    done: list[str] = []

    async def worker(sid: str):
        async with store.lock_for(sid):
            await asyncio.sleep(0.01)
            done.append(sid)

    await asyncio.wait_for(
        asyncio.gather(worker("s1"), worker("s2")), timeout=0.5
    )
    assert sorted(done) == ["s1", "s2"]


# ---- TTL 清扫 ----


def test_lock_for_orphan_is_swept_after_ttl(clock):
    """lock_for 建过锁/时间戳的会话,TTL 后应被清扫。

    否则 _touched/_locks 长期运行会无界增长。
    """
    store = SessionStore(ttl_seconds=60, max_sessions=10)
    store.lock_for("ghost")

    assert "ghost" in store._locks
    assert "ghost" in store._touched

    clock["now"] += 61  # 超过 TTL
    store.active_lock_count()  # 触发 _purge 清扫

    assert "ghost" not in store._locks
    assert "ghost" not in store._touched


def test_held_lock_orphan_is_not_swept(clock):
    """持锁的条目不能被清扫 —— 否则请求刚拿锁就 400 时,
    并发请求会拿到第二把锁,破坏互斥。"""
    store = SessionStore(ttl_seconds=60, max_sessions=10)
    lock = store.lock_for("ghost")

    async def hold():
        async with lock:
            clock["now"] += 61  # 超过 TTL,但锁被持有
            store.active_lock_count()
            return store._locks.get("ghost")

    assert asyncio.run(hold()) is lock


def test_locked_session_is_not_evicted_by_ttl(clock):
    """契约层:正在流式的会话不能被 TTL 清扫。

    上面那条查的是内部状态,这条查的是对外契约 —— 清扫之后 `lock_for`
    仍必须返回**同一把**锁。返回新锁就意味着一份历史被两个流并发写。
    """
    store = SessionStore(ttl_seconds=60, max_sessions=10)
    lock = store.lock_for("s1")

    async def hold():
        async with lock:
            clock["now"] += 3600  # 持锁期间时间大幅前进
            return store.lock_for("s1")  # 内部触发 _purge

    assert asyncio.run(hold()) is lock


# ---- 容量上限(MAX_SESSIONS 现在约束锁表) ----


def test_capacity_bounds_the_lock_table():
    """MAX_SESSIONS 现在约束的是锁表 —— ch01 那条遗留风险在本章关闭。"""
    store = SessionStore(ttl_seconds=600, max_sessions=3)
    for i in range(20):
        store.lock_for(f"s{i}")
    assert store.active_lock_count() <= 3


def test_lock_for_refreshes_recency():
    """lock_for 必须把条目移到 LRU 末尾。

    否则淘汰的是插入序而非最近使用序,刚建的锁会被优先选中,
    破坏「同一 session 两次 lock_for 返回同一把锁」的幂等性
    —— ch01 Task 4 已经在这上面栽过一次。

    断言用**持有的引用**比对而非 `is not None`:后者恒真,区分不了
    正确与错误实现(去掉 _touch 里的 move_to_end,它照样通过)。
    """
    store = SessionStore(ttl_seconds=600, max_sessions=3)
    lock_s1 = store.lock_for("s1")
    lock_s2 = store.lock_for("s2")
    store.lock_for("s3")
    store.lock_for("s1")  # s1 变成最近使用
    store.lock_for("s4")  # 触发淘汰:应淘汰 s2,而不是 s1

    assert store.lock_for("s1") is lock_s1  # s1 存活
    assert store.lock_for("s2") is not lock_s2  # s2 已被淘汰(插入序下会淘汰 s1)


def test_lock_for_is_idempotent_under_capacity_pressure():
    store = SessionStore(ttl_seconds=600, max_sessions=1)
    first = store.lock_for("s1")
    second = store.lock_for("s1")
    assert first is second


@pytest.mark.anyio
async def test_capacity_never_evicts_a_held_lock():
    """淘汰遇到被持锁的必须**整个停下**,不跳过。

    跳过会去删比它更新的条目,把 LRU 语义弄反 —— ch01 自审抓到过这个 bug。
    """
    store = SessionStore(ttl_seconds=600, max_sessions=2)
    held = store.lock_for("oldest")
    await held.acquire()
    for i in range(5):
        store.lock_for(f"new{i}")

    # 两行的顺序不能颠倒,也不能改回 brief 里的原样:
    # `lock_for("oldest")` 自己会触发一次容量淘汰,把 5 个未被持有的 new*
    # 一路删到 max=2。所以"超容量"这个状态只存在于这次调用**之前**,必须
    # 先把数量断言掉。反过来写,第二行只会看到 2,而正确实现必然挂。
    assert store.active_lock_count() > 2  # 宁可短暂超容量
    assert store.lock_for("oldest") is held  # 持锁条目仍在


@pytest.mark.anyio
async def test_locked_session_is_not_evicted_by_lru():
    """「整个停下」的另一面:更新的条目不能被当成代价删掉。

    跳过式实现会让 s2 的条目消失(它比被持锁的 s1 新),于是第二次
    `lock_for("s2")` 建出一把新锁 —— 同一会话的并发请求失去互斥。
    """
    store = SessionStore(ttl_seconds=10_000, max_sessions=1)
    held = store.lock_for("s1")
    await held.acquire()

    lock_s2 = store.lock_for("s2")  # 触发容量淘汰:s1 被持锁 → 整个停下

    # 同样不能颠倒:先问 s1 会给它做一次 touch,反过来把 s2 顶成最老的
    # 未持锁条目、被合法淘汰 —— 那是 LRU 在正常工作,不是要测的行为。
    assert store.lock_for("s2") is lock_s2
    assert store.lock_for("s1") is held
