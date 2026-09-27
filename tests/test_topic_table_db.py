"""`topic_classifications` 的**表形状**走真库(`db` 标记,需要 MySQL 在跑)。

**为什么必须走真库**:这个文件要验的三件事**都只存在于 SQL 层**,替身一个都测不出 ——
① 唯一键存不存在(它是「重跑 = 覆盖」的全部保证);② `labels` / `scores` 到底是不是
`json` 列;③ 真正建出来的表有哪些列、可空性如何。单测里 `Base.metadata` 是我们自己写的,
拿它去核对它自己等于同义反复。

⚠️ **没有 `session` fixture** —— 本仓 `tests/conftest.py` 不提供它。照
`tests/test_api_conversations_db.py` 的既有写法:自建 `get_sessionmaker()` 会话,
**探针行用完即删、删除必须 `commit`**(`async with session` 退出是 rollback)。

⚠️ **探针 id 恒为 `PROBE_ID`,插入前先删一遍**(这张表是只追加的结果表,上一轮若有
残留 —— 超时、Ctrl-C、断言中途红 —— 会撞唯一键让下一轮**红在别处**)。所有查询都按
`low_confidence_question_id` 过滤,不写「查最近这几条」那种会被历史数据污染的形式。
"""

import pytest
from sqlalchemy import delete, select, text
from sqlalchemy.exc import IntegrityError

from app.db.base import get_engine, get_sessionmaker
from app.db.models import TopicClassification

pytestmark = pytest.mark.db

#: 探针 id 用高位段,不会撞真实数据(池子今天是**很小**的自增值,远不到这里)。
#: ⚠️ 关联列**故意不挂外键**(见 ORM docstring),所以这个 id 不必真的在池子里存在。
PROBE_ID = 999001


async def _purge(session) -> None:
    """删掉本文件的探针行并**提交**(不提交的话下一轮会看到残留)。"""
    await session.execute(
        delete(TopicClassification).where(
            TopicClassification.low_confidence_question_id == PROBE_ID
        )
    )
    await session.commit()


async def _rows() -> list[TopicClassification]:
    """**新 session** 读回(同 session 重读是否打到库要靠身份映射的 refcount 走运)。"""
    async with get_sessionmaker()() as session:
        found = await session.execute(
            select(TopicClassification).where(
                TopicClassification.low_confidence_question_id == PROBE_ID
            )
        )
        return list(found.scalars().all())


@pytest.mark.anyio
async def test_unique_key_rejects_a_second_row_for_the_same_question():
    """唯一键 `uk_pool_question` 是「重跑 = 覆盖」的保证:同一条池子行**只能有一行**。

    ⚠️ 这条断言**不是**「插入两次会报错」那么弱 —— 三个断言缺一不可:
    ① **第一行真的落库了**(否则「第二次插入失败」可能来自别的约束,比如
       `model_version` 的 NOT NULL,那这条用例就变成了在验一件别的事);
    ② 第二次写**抛 `IntegrityError`**(窄类型,不是 `Exception`)—— 若唯一键被删掉,
       这里**不抛**,`pytest.raises` 当场红;
    ③ 被拒之后库里的**那一行仍是先写的那条**(`model_version == "probe-a"`)——
       「静默追加成两行」与「覆盖成新值」都在这条上红,而它们正是幂等语义的两种错法。

    与 ch09 的 `review_queue` **刻意不加唯一键**规矩相反,而这是对的:
    那边判的是**语义**(两句话是不是一个意思),字面唯一键会在一次合理的归并上
    响亮地 1062;这边判的是**确定性重算** —— 同一输入就该覆盖旧值,唯一键正是
    「覆盖」这种写法的前提。
    """
    engine = get_engine()
    try:
        async with get_sessionmaker()() as session:
            await _purge(session)
            session.add(
                TopicClassification(
                    low_confidence_question_id=PROBE_ID,
                    labels=["尺码", "退换货"],
                    scores={"尺码": 0.9},
                    model_version="probe-a",
                )
            )
            await session.commit()

        first = await _rows()
        assert len(first) == 1, f"前提:第一行应当落库(唯一键下恰 1 行),实际 {len(first)}"
        assert first[0].model_version == "probe-a"

        async with get_sessionmaker()() as session:
            session.add(
                TopicClassification(
                    low_confidence_question_id=PROBE_ID,
                    labels=["运费"],
                    scores={},
                    model_version="probe-b",
                )
            )
            with pytest.raises(IntegrityError):
                await session.commit()
            await session.rollback()

        after = await _rows()
        assert len(after) == 1, (
            f"同一池子行只许有一行,实际 {len(after)} 行 —— "
            "唯一键没生效(静默追加了一条)"
        )
        assert after[0].model_version == "probe-a", (
            f"被拒的第二次写不该改到已有一行;实际 model_version={after[0].model_version!r}"
        )
    finally:
        async with get_sessionmaker()() as session:
            await _purge(session)
        await engine.dispose()


@pytest.mark.anyio
async def test_labels_and_scores_are_json_columns_reading_with_json_type():
    """`labels` / `scores` 是 **JSON 列**,读法一律 `JSON_TYPE()`。

    ⚠️ 判别力全靠第一条断言:`JSON_TYPE('[]')` 对一个 **TEXT 列**同样答 `ARRAY`
    (MySQL 会把合法 JSON 文本当文档解析)—— 只断 `JSON_TYPE` 的话,把这两列写成
    `Text` 的实现**照样绿**。真正分开两者的是 `information_schema` 的 `DATA_TYPE`
    (json vs text)。后面两条断的是**取值往返**(落进去的 Python 列表/字典读回来
    还是它),既钉住序列化,也顺带排除「列是 json 但塞的不是 JSON 文档」。

    ⚠️ **本表两列都是 NOT NULL,所以「JSON `null` 不是 SQL NULL」那个坑在这里够不到**
    (读法照旧钉成 `JSON_TYPE()` —— 它是 `app/kb/assess.py` 记过血的那一条,
    ch09 的 T19 拿 `IS NOT NULL` 去数「有快照的行」,把 JSON `null` 数成了非空)。
    """
    engine = get_engine()
    try:
        async with get_sessionmaker()() as session:
            await _purge(session)
            session.add(
                TopicClassification(
                    low_confidence_question_id=PROBE_ID,
                    labels=["尺码", "退换货"],
                    scores={"尺码": 0.9, "退换货": 0.4},
                    model_version="probe-json",
                )
            )
            await session.commit()

            col_types = dict(
                (
                    await session.execute(
                        text(
                            "SELECT COLUMN_NAME, DATA_TYPE FROM information_schema.COLUMNS "
                            "WHERE TABLE_SCHEMA = DATABASE() "
                            "AND TABLE_NAME = 'topic_classifications' "
                            "AND COLUMN_NAME IN ('labels', 'scores')"
                        )
                    )
                ).all()
            )
            assert col_types == {"labels": "json", "scores": "json"}, (
                f"两列必须真的是 json 列(Text 列也能骗过 JSON_TYPE),实际 {col_types}"
            )

            json_types = (
                await session.execute(
                    text(
                        "SELECT JSON_TYPE(labels), JSON_TYPE(scores) "
                        "FROM topic_classifications "
                        "WHERE low_confidence_question_id = :qid"
                    ),
                    {"qid": PROBE_ID},
                )
            ).one()
            assert tuple(json_types) == ("ARRAY", "OBJECT"), f"实际 {json_types}"

        # 取值往返:Python 的 list / dict 落库再读回来还是它
        # (**新 session**,不吃身份映射里的缓存)。
        rows = await _rows()
        assert len(rows) == 1, f"前提:探针行应当落库,实际 {len(rows)}"
        assert rows[0].labels == ["尺码", "退换货"], f"实际 {rows[0].labels!r}"
        assert rows[0].scores == {"尺码": 0.9, "退换货": 0.4}, f"实际 {rows[0].scores!r}"
    finally:
        async with get_sessionmaker()() as session:
            await _purge(session)
        await engine.dispose()


@pytest.mark.anyio
async def test_shape_matches_the_model():
    """**真表**的逐列形状:列名 / 序 / 类型 / 可空性 + 唯一键的名字。

    这条钉的是「两条建库路径形状一致」里**能自动化的那一半**(`scripts/init_db.py`
    的 create_all 与 `db/ch10.sql` 手工执行)。⚠️ 它只能验**当前库里这一份** ——
    所以 `db/ch10.sql` 那条路径必须另外**手工跑一遍**再 `SHOW CREATE TABLE` 核对
    (读数记在 ORM docstring 的差异清单与 T12 报告里)。

    判别力来源:列名少一个/多一个、类型从 `bigint` 变 `int`、`NOT NULL` 变可空、
    唯一键被删掉 —— 四种改动各自让下面某一条红。**列序也断**:两条路径本表都没有
    ALTER,列序应当相同,断了才能保证「有人往 DDL 里插一列到中间」被看见。
    """
    engine = get_engine()
    try:
        async with get_sessionmaker()() as session:
            cols = (
                await session.execute(
                    text(
                        "SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE "
                        "FROM information_schema.COLUMNS "
                        "WHERE TABLE_SCHEMA = DATABASE() "
                        "AND TABLE_NAME = 'topic_classifications' "
                        "ORDER BY ORDINAL_POSITION"
                    )
                )
            ).all()
            assert [c.COLUMN_NAME for c in cols] == [
                "id",
                "low_confidence_question_id",
                "labels",
                "scores",
                "model_version",
                "classified_at",
            ], f"列名/列序不符:{[(c.COLUMN_NAME, c.DATA_TYPE) for c in cols]}"
            assert [c.DATA_TYPE for c in cols] == [
                "bigint",
                "bigint",
                "json",
                "json",
                "varchar",
                "datetime",
            ], f"列类型不符:{[(c.COLUMN_NAME, c.DATA_TYPE) for c in cols]}"
            # 六列**全部** NOT NULL:labels/scores 也是(空标签该让脚本响亮失败,
            # 而不是落一行「没有主题」的结果 —— spec §9.2)。
            assert [c.IS_NULLABLE for c in cols] == ["NO"] * 6, (
                f"可空性不符:{[(c.COLUMN_NAME, c.IS_NULLABLE) for c in cols]}"
            )

            uniques = (
                await session.execute(
                    text(
                        "SELECT INDEX_NAME, NON_UNIQUE, COLUMN_NAME "
                        "FROM information_schema.STATISTICS "
                        "WHERE TABLE_SCHEMA = DATABASE() "
                        "AND TABLE_NAME = 'topic_classifications' "
                        "AND NON_UNIQUE = 0 AND COLUMN_NAME = 'low_confidence_question_id'"
                    )
                )
            ).all()
            # 唯一键**两侧同名**(DDL 的 `UNIQUE KEY uk_pool_question` / ORM 的
            # `UniqueConstraint(..., name="uk_pool_question")`)—— 照 ch07
            # `uk_conv_seq` 的做法,把 ch08/ch09 那种「索引名不同」的差异**消掉**。
            assert [u.INDEX_NAME for u in uniques] == ["uk_pool_question"], (
                f"`low_confidence_question_id` 上应恰有一个名为 uk_pool_question 的"
                f"唯一索引,实际 {[u.INDEX_NAME for u in uniques]}"
            )
    finally:
        await engine.dispose()
