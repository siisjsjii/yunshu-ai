"""ORM 元数据的形状检查。**纯单测:不联网、不碰数据库。**

为什么不放进 `tests/test_db_models.py`:那个文件整体带 `pytestmark = pytest.mark.db`,
断言的是**库里的**形状;这里读的是 `Base.metadata`,问的是**另一条建库路径**
(`scripts/init_db.py` 的 `create_all`)会建出什么。
两者缺一不可 —— 库对不代表两条路径一致,而「谁建的库」差异只在全新环境上才现形。
"""


def test_orm_declares_the_unique_key_that_the_ddl_also_creates():
    """两条建库路径必须建出**同样的约束**。

    只断「撞唯一键会 IntegrityError」是不够的:那条跑在由 db/ch07.sql 建好的
    库上,而 `create_all` 那条路径建出来的表**可能根本没有这个键** ——
    于是全新环境按 `init_db.py` → `db/ch07.sql` 走一遍,
    唯一键永远不存在,而没有任何东西报错。

    (2026-09-22 实测过这个洞是真的:`UniqueConstraint` / `__table_args__` 在本文件
    修好之前于 `app/db/models.py` 里**零命中**,而 `db/ch07.sql` 当时写的是
    `CREATE TABLE IF NOT EXISTS` —— create_all 抢建之后那一句静默跳过。)
    """
    from app.db.models import ConversationSummary

    names = {c.name for c in ConversationSummary.__table__.constraints}
    assert "uk_conv_seq" in names
    cols = {
        tuple(sorted(c.name for c in ct.columns))
        for ct in ConversationSummary.__table__.constraints
        if ct.name == "uk_conv_seq"
    }
    assert cols == {("conversation_id", "seq")}
