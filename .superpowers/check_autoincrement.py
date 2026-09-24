"""**可复跑**的证据:`tests/test_flywheel_pipeline.py` 占用 id 1–9 会不会动 AUTO_INCREMENT。

为什么要留这个文件:报告里那句「不会推进 AUTO_INCREMENT」原先只有**当天读数**
(579),而它**依赖被删的行、每天都在变** —— 那样的一句「实测」别人没法复核
(评审判为 Minor:「要么给产物,要么把『实测』改成『自述 + 现场旁证』」)。

用法(不需要网络,只需要 MySQL):
    .venv/Scripts/python.exe .superpowers/check_autoincrement.py

它做四件事并**逐步打印**:
1. 读插入前的 AUTO_INCREMENT;
2. **显式**插一行 id=1(照测试的做法),再读 —— 看有没有被推进;
3. **不指定 id** 插一行(模拟生产/验收的真实插入),看它拿到的 id 是不是
   「与 1–9 无关的、正常递增值」;
4. 删掉两行探针并复查残留为 0,再读一次(看删除会不会把计数器拉回去)。

⚠️ **不要把输出里的具体数字当成恒量** —— 它是**当天**的读数。要引用的是
**关系**(前后相等、真插入拿到的 id 不在 1–9、删除不回退),不是 579/590 这种值。
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from app.db.base import get_engine, get_sessionmaker

PROBE_CONV = "t13ai"          # 探针会话 id(≤32 字符,结构上撞不到真实数据)

_AUTO_INCREMENT = text(
    "SELECT AUTO_INCREMENT FROM information_schema.TABLES "
    "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'low_confidence_questions'"
)
_COUNT = text(
    "SELECT COUNT(*) FROM low_confidence_questions WHERE source_conversation_id = :c"
)


def emit(line: str = "") -> None:
    """**必须钉输出编码**:重定向到文件时 Python 按 locale(cp936)编码 stdout,
    中文直接变成乱码(本仓平台陷阱;第一次跑出来的证据文件就是一坨乱码)。
    """
    stream = getattr(sys.stdout, "buffer", None)
    if stream is None:
        print(line)                 # 没有 buffer(某些被嵌进别的宿主时)时的退路
        return
    stream.write(line.encode("utf-8") + b"\n")
    stream.flush()


async def main() -> int:
    async with get_sessionmaker()() as session:
        before = (await session.execute(_AUTO_INCREMENT)).scalar_one()
        emit(f"1. AUTO_INCREMENT 插入前            = {before}")

        await session.execute(
            text(
                "INSERT INTO low_confidence_questions "
                "(id, question, source_conversation_id, entry_point, reject_reason) "
                "VALUES (1, 't13 显式主键探针', :c, '置信度闸', 'probe')"
            ),
            {"c": PROBE_CONV},
        )
        await session.commit()
        after_explicit = (await session.execute(_AUTO_INCREMENT)).scalar_one()
        emit(
            f"2. **显式**插 id=1 之后             = {after_explicit}"
            f"  → 被推进了吗:{after_explicit != before}"
        )

        await session.execute(
            text(
                "INSERT INTO low_confidence_questions "
                "(question, source_conversation_id, entry_point, reject_reason) "
                "VALUES ('t13 自增探针', :c, '置信度闸', 'probe')"
            ),
            {"c": PROBE_CONV},
        )
        await session.commit()
        auto_id = (
            await session.execute(
                text(
                    "SELECT MAX(id) FROM low_confidence_questions "
                    "WHERE source_conversation_id = :c"
                ),
                {"c": PROBE_CONV},
            )
        ).scalar_one()
        emit(
            f"3. **不指定 id** 的插入拿到的 id     = {auto_id}"
            f"  → 与探针占用的 1–9 撞了吗:{1 <= auto_id <= 9}"
        )

        await session.execute(
            text("DELETE FROM low_confidence_questions WHERE source_conversation_id = :c"),
            {"c": PROBE_CONV},
        )
        await session.commit()
        after_delete = (await session.execute(_AUTO_INCREMENT)).scalar_one()
        leftover = (await session.execute(_COUNT, {"c": PROBE_CONV})).scalar_one()
        emit(f"4. 删掉探针之后                     = {after_delete}  (残留 {leftover} 行)")

        verdict = (
            after_explicit == before          # 显式小 id 不推进计数器
            and not 1 <= auto_id <= 9         # 真插入拿到的 id 不在探针那一段
            and leftover == 0                 # 探针清干净
        )
        emit(f"\n结论:占 id 1–9 对后续真实插入{'无影响' if verdict else '**有影响**'}")
    await get_engine().dispose()
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
