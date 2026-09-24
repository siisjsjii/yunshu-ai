"""复核 F1 那条算式(生产默认旋钮,真 `evidence_detail`):单块 top1 的合成分与过闸临界。

顺带量一下共享表在一天里长了多少(F2/CLAUDE.md 要引用的读数 —— 引用前先自己读一遍)。
"""

import asyncio
import sys

from sqlalchemy import text

from app.config import Settings
from app.db.base import get_engine
from app.kb.evidence import evidence_detail
from app.retrieval.search import RetrievedChunk


def chunk(score: float) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=1, question="q", answer="a", category="c",
                          section_path="s", score=score)


def main() -> None:
    from app.config import get_settings
    s = get_settings()   # 生产默认:只读旋钮,不打印任何凭据
    print("旋钮: w_top1=%s w_count=%s w_gap=%s max_count=%s gate_threshold=%s "
          "min_score=%s retrieval_score_threshold=%s"
          % (s.w_evidence_top1, s.w_evidence_count, s.w_evidence_gap,
             s.evidence_max_count, s.evidence_confidence_threshold,
             s.evidence_min_score, s.retrieval_score_threshold))
    for score in (0.15, 0.16, 0.1666, 0.1667, 0.19, 0.25):
        d = evidence_detail([chunk(score)], settings=s)
        verdict = "过闸" if d["confidence"] >= s.evidence_confidence_threshold else "被拦"
        print("  单块 top1=%.4f -> confidence=%.4f  %s" % (score, d["confidence"], verdict))
    # 解出来的临界值
    c = s.evidence_confidence_threshold
    w = s.w_evidence_top1 + s.w_evidence_gap
    fixed = s.w_evidence_count * (1.0 / s.evidence_max_count)
    print("  解: 0.8*top1 + %.4f >= %.4f  ->  临界 top1 >= %.4f  (几何: %.4f)"
          % (fixed, c, (c - fixed) / w, (c - fixed) / w))


async def tables() -> None:
    engine = get_engine()
    try:
        async with engine.connect() as conn:
            for label, sql in (
                ("tool_audit_logs 总数", "SELECT COUNT(*) FROM tool_audit_logs"),
                ("tool_audit_logs 今天(2026-09-24)",
                 "SELECT COUNT(*) FROM tool_audit_logs WHERE DATE(created_at)='2026-09-24'"),
                ("conversations", "SELECT COUNT(*) FROM conversations"),
                ("messages", "SELECT COUNT(*) FROM messages"),
            ):
                r = await conn.execute(text(sql))
                print("%-28s %s" % (label, r.scalar()))
    finally:
        await engine.dispose()


main()
asyncio.run(tables())
