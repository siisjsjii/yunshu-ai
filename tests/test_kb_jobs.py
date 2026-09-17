"""JobStore 单元测试:串行、状态机、单例。不碰 DB / 线程外的副作用。"""

from app.kb.jobs import JobStore, get_job_store


def test_start_returns_job_with_running_status():
    store = JobStore()
    job = store.start("mine")
    assert job is not None
    assert job.type == "mine"
    assert job.status == "running"
    assert job.message == ""
    assert job.result is None


def test_busy_store_rejects_second_job():
    store = JobStore()
    assert store.start("mine") is not None
    assert store.start("vectorize") is None      # 忙 → 拒绝
    assert store.is_busy() is True


def test_update_changes_status_message_result():
    store = JobStore()
    job = store.start("vectorize")
    store.update(job.id, message="处理中")
    store.update(job.id, status="done", message="完成", result={"processed": 3})
    got = store.get(job.id)
    assert got.status == "done"
    assert got.message == "完成"
    assert got.result == {"processed": 3}
    assert got.finished_at is not None


def test_get_unknown_returns_none():
    assert JobStore().get("nope") is None


def test_list_newest_first_and_busy_released_on_terminal_status():
    store = JobStore()
    a = store.start("mine")
    store.update(a.id, status="done")
    b = store.start("vectorize")
    assert [j.id for j in store.list()] == [b.id, a.id]   # 新在前
    assert store.is_busy() is True                          # b 仍在跑
    store.update(b.id, status="failed")
    assert store.is_busy() is False
    # 释放后能再起
    c = store.start("mine")
    assert c is not None


def test_get_job_store_is_singleton():
    assert get_job_store() is get_job_store()
    get_job_store.cache_clear()
