"""ch09 两张新表 + 两列 —— 真库往返。

db 标记:读**真实** .env(不加 `_env_file=None`),需要 MySQL 已起 + db/ch09.sql 已应用。
三条用例各自**自清**(跑完删掉自己那一行)—— 池子那张表被验收脚本读,不许留垃圾。

这里不测模型逻辑(没有逻辑),测的是「ORM 与库里的形状对得上」:
列名写错、少一列、JSON 列类型不对,都会在这三条里红。
"""

import pytest
from sqlalchemy import text

from app.db.base import get_sessionmaker
from app.db.models import EvalRun, LowConfidenceQuestion, ReviewQueue

pytestmark = pytest.mark.db


@pytest.mark.anyio
async def test_new_columns_round_trip():
    maker = get_sessionmaker()
    async with maker() as s:
        row = LowConfidenceQuestion(
            question="T9 往返探针",
            entry_point="生成自评",
            reject_reason="r",
            evidence_snapshot={"chunks": [{"id": 1, "score": 0.5}]},
        )
        s.add(row)
        await s.commit()
        rid = row.id
        await s.refresh(row)
        assert row.evidence_snapshot == {"chunks": [{"id": 1, "score": 0.5}]}
        # NULL = 尚未进流水线 —— 流水线的幂等标记,所以未归并时必须是 None
        assert row.matched_review_id is None
        await s.execute(
            text("DELETE FROM low_confidence_questions WHERE id = :i"), {"i": rid}
        )
        await s.commit()


@pytest.mark.anyio
async def test_review_queue_round_trip():
    maker = get_sessionmaker()
    async with maker() as s:
        rq = ReviewQueue(
            standard_question="T9 标准问题",
            example_answer="示例",
            occurrences=1,
            first_raw_question="原话",
        )
        s.add(rq)
        await s.commit()
        rid = rq.id
        assert rq.status == "pending" and rq.occurrences == 1
        await s.execute(text("DELETE FROM review_queue WHERE id = :i"), {"i": rid})
        await s.commit()


@pytest.mark.anyio
async def test_eval_run_round_trip():
    maker = get_sessionmaker()
    async with maker() as s:
        r = EvalRun(trigger_by="manual", case_count=3, metrics={"top_k": 10})
        s.add(r)
        await s.commit()
        rid = r.id
        await s.refresh(r)
        assert r.metrics == {"top_k": 10}
        await s.execute(text("DELETE FROM eval_runs WHERE id = :i"), {"i": rid})
        await s.commit()
