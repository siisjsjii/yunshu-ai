import asyncio, sys, os
sys.path.insert(0, os.getcwd())
from sqlalchemy import text
from app.db.base import get_engine

Q = {
 "low_confidence_questions 总行数": "SELECT COUNT(*) FROM low_confidence_questions",
 "  ├ 按 entry_point": "SELECT entry_point, COUNT(*) c FROM low_confidence_questions GROUP BY entry_point ORDER BY c DESC",
 "  ├ 去重后问题数": "SELECT COUNT(DISTINCT question) FROM low_confidence_questions",
 "  └ question 长度分布": "SELECT MIN(CHAR_LENGTH(question)) mn, ROUND(AVG(CHAR_LENGTH(question))) avg, MAX(CHAR_LENGTH(question)) mx FROM low_confidence_questions",
 "review_queue 总行数": "SELECT COUNT(*) FROM review_queue",
 "  ├ 按 status": "SELECT status, COUNT(*) c FROM review_queue GROUP BY status ORDER BY c DESC",
 "messages 里 user 行数": "SELECT COUNT(*) FROM messages WHERE role='user'",
 "  └ 去重 user 行数": "SELECT COUNT(DISTINCT content) FROM messages WHERE role='user'",
 "conversations": "SELECT COUNT(*) FROM conversations",
}

async def main():
    eng = get_engine()
    async with eng.connect() as conn:
        for label, sql in Q.items():
            try:
                rows = (await conn.execute(text(sql))).all()
                print(f"{label}: {[tuple(r) for r in rows]}")
            except Exception as e:
                print(f"{label}: ERR {type(e).__name__} {e}")
    await eng.dispose()

asyncio.run(main())
