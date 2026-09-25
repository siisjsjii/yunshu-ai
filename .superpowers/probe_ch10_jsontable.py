import asyncio, sys, os
sys.path.insert(0, os.getcwd())
from sqlalchemy import text
from app.db.base import get_engine

SQL = """
SELECT jt.label AS label, COUNT(*) AS n
FROM (SELECT CAST('["尺码","退换货"]' AS JSON) AS labels) t,
     JSON_TABLE(t.labels, '$[*]' COLUMNS (label VARCHAR(64) PATH '$')) jt
GROUP BY jt.label ORDER BY n DESC
"""
async def main():
    eng = get_engine()
    async with eng.connect() as conn:
        ver = (await conn.execute(text("SELECT VERSION()"))).scalar()
        print("MySQL:", ver)
        try:
            rows = (await conn.execute(text(SQL))).all()
            print("JSON_TABLE 可用 ->", [tuple(r) for r in rows])
        except Exception as e:
            print("JSON_TABLE 不可用 ->", type(e).__name__, str(e)[:300])
    await eng.dispose()
asyncio.run(main())
