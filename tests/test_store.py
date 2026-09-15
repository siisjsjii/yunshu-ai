import asyncio

import pytest

from app.memory.store import SessionStore
from app.schemas import Message


def _msg(text: str) -> Message:
    return Message(role="user", content=text)


@pytest.fixture
def clock(monkeypatch):
    """可控时钟,避免测试里真的 sleep。"""
    state = {"now": 1000.0}
    monkeypatch.setattr(
        "app.memory.store.time.monotonic", lambda: state["now"], raising=True
    )
    return state


def test_history_is_empty_for_unknown_session():
    store = SessionStore(ttl_seconds=60, max_sessions=10)
    assert store.history("nope") == []


def test_append_then_history_roundtrips():
    store = SessionStore(ttl_seconds=60, max_sessions=10)
    store.append("s1", [_msg("你好")])
    assert store.history("s1") == [_msg("你好")]


def test_append_accumulates_in_order():
    store = SessionStore(ttl_seconds=60, max_sessions=10)
    store.append("s1", [_msg("一")])
    store.append("s1", [_msg("二")])
    assert [m.content for m in store.history("s1")] == ["一", "二"]


def test_history_returns_a_copy():
    """调用方改动返回值不应污染存储。"""
    store = SessionStore(ttl_seconds=60, max_sessions=10)
    store.append("s1", [_msg("你好")])
    store.history("s1").append(_msg("注入"))
    assert len(store.history("s1")) == 1


def test_sessions_are_isolated():
    store = SessionStore(ttl_seconds=60, max_sessions=10)
    store.append("s1", [_msg("甲")])
    store.append("s2", [_msg("乙")])
    assert [m.content for m in store.history("s1")] == ["甲"]
    assert [m.content for m in store.history("s2")] == ["乙"]


def test_expired_session_is_dropped(clock):
    store = SessionStore(ttl_seconds=60, max_sessions=10)
    store.append("s1", [_msg("你好")])

    clock["now"] += 61

    assert store.history("s1") == []
    assert store.active_session_count() == 0


def test_session_survives_within_ttl(clock):
    store = SessionStore(ttl_seconds=60, max_sessions=10)
    store.append("s1", [_msg("你好")])

    clock["now"] += 59

    assert store.history("s1") == [_msg("你好")]


def test_lru_evicts_oldest_when_over_capacity():
    store = SessionStore(ttl_seconds=10_000, max_sessions=2)
    store.append("s1", [_msg("一")])
    store.append("s2", [_msg("二")])
    store.append("s3", [_msg("三")])

    assert store.active_session_count() == 2
    assert store.history("s1") == []
    assert store.history("s3") == [_msg("三")]


def test_reading_a_session_refreshes_its_lru_position():
    store = SessionStore(ttl_seconds=10_000, max_sessions=2)
    store.append("s1", [_msg("一")])
    store.append("s2", [_msg("二")])

    store.history("s1")  # s1 变成最近使用
    store.append("s3", [_msg("三")])

    assert store.history("s1") == [_msg("一")]
    assert store.history("s2") == []
    assert store.history("s3") == [_msg("三")]


def test_lock_for_returns_same_lock_for_same_session():
    store = SessionStore(ttl_seconds=60, max_sessions=10)
    assert store.lock_for("s1") is store.lock_for("s1")


def test_lock_for_returns_different_locks_for_different_sessions():
    store = SessionStore(ttl_seconds=60, max_sessions=10)
    assert store.lock_for("s1") is not store.lock_for("s2")


def test_locked_session_is_not_evicted_by_ttl(clock):
    """正在流式的会话不能被 TTL 淘汰,否则并发保护会失效。"""
    store = SessionStore(ttl_seconds=60, max_sessions=10)
    store.append("s1", [_msg("你好")])

    lock = store.lock_for("s1")
    assert not lock.locked()

    async def hold():
        async with lock:
            clock["now"] += 3600  # 持锁期间时间大幅前进
            return store.history("s1")  # history() 内部会触发淘汰

    # 用 asyncio.run 直接驱动,绕开 pytest 的异步夹具
    assert asyncio.run(hold()) == [_msg("你好")]


def test_locked_session_is_not_evicted_by_lru():
    store = SessionStore(ttl_seconds=10_000, max_sessions=1)
    store.append("s1", [_msg("一")])

    lock = store.lock_for("s1")

    async def hold():
        async with lock:
            store.append("s2", [_msg("二")])
            return store.history("s1")

    assert asyncio.run(hold()) == [_msg("一")]
    # 关键:不能为了守容量就越过被锁的 s1 去删更新的 s2 ——
    # 那会把 LRU 语义弄反。宁可短暂超容量。
    assert store.history("s2") == [_msg("二")]


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
