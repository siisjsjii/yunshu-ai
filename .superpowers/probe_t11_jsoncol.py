"""探针:evidence_snapshot 在库里到底是什么 —— 为 T11 的读回断言定口径。

问三件事:
1. `text()` 裸 SQL 读出来是 str 还是 list?(JSON 结果处理器只作用于**带类型的**列)
2. 不传快照(= 传 None)落的是 **SQL NULL** 还是 **JSON null**?
3. 走 ORM 列读,两种形状各自读回什么?
"""

import asyncio
import json

from sqlalchemy import select, text

from app.db.base import get_engine, get_sessionmaker
from app.db.models import LowConfidenceQuestion
from app.kb.assess import record_low_confidence

Q = "ch09-T11-probe-jsoncol"
SNAP = [{"chunk_id": 7, "score": 0.31, "section_path": "s", "answer": "a"}]


async def main():
    engine = get_engine()
    ids = []
    async with get_sessionmaker()() as s:
        ids.append(await record_low_confidence(
            s, question=Q, source_conversation_id=None, entry_point="探针",
            reject_reason="带快照", evidence_snapshot=SNAP))
    async with get_sessionmaker()() as s:
        ids.append(await record_low_confidence(
            s, question=Q, source_conversation_id=None, entry_point="探针",
            reject_reason="不带快照"))

    async with get_sessionmaker()() as s:
        for i in ids:
            raw = (await s.execute(
                text("SELECT evidence_snapshot, "
                     "evidence_snapshot IS NULL, JSON_TYPE(evidence_snapshot) "
                     "FROM low_confidence_questions WHERE id=:i"), {"i": i})).one()
            orm = (await s.execute(
                select(LowConfidenceQuestion.evidence_snapshot)
                .where(LowConfidenceQuestion.id == i))).scalar_one()
            print(f"id={i} 裸SQL={raw[0]!r} ({type(raw[0]).__name__}) "
                  f"IS_NULL={raw[1]} JSON_TYPE={raw[2]} ORM读回={orm!r}")
            if isinstance(raw[0], str):
                print(f"        json.loads(裸SQL) == SNAP ? {json.loads(raw[0]) == SNAP}")

        await s.execute(text("DELETE FROM low_confidence_questions WHERE question=:q"), {"q": Q})
        await s.commit()
        left = (await s.execute(
            text("SELECT COUNT(*) FROM low_confidence_questions WHERE question=:q"),
            {"q": Q})).scalar_one()
        print("清理后剩余:", left)
    await engine.dispose()


asyncio.run(main())
