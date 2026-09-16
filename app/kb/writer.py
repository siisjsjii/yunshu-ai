"""双写落库:MySQL 原文权威源 + Milvus 向量索引(spec §6.2)。

时序:`INSERT (pending)` → 嵌入 → `Milvus upsert(同 pk 覆盖)` →
`UPDATE vector_id + status=done`。两个中断窗口都安全:

(a) Milvus 写完、MySQL 回填前挂 → 行仍 pending,重跑重写**同 pk**(upsert
    幂等)再回填;
(b) MySQL 已 done → 重跑跳过。

**重跑 = 重扫 pending 行**,这就是验收 2 的实现。
"""

from sqlalchemy import select

from app.db.models import KnowledgeChunk
from app.kb.chunker import Chunk


def vector_text(category: str, questions: str, answer: str) -> str:
    """category + questions + answer 拼成向量化文本(spec §6.3)。

    拼接符与顺序固定,且只此一处 —— 离线建库与在线检索必须走同一条路径。
    两边各拼一次的话,同一条知识入库时的文本与查询时的文本会悄悄漂移,
    而这种漂移不报错,只让召回变差。
    """
    return "\n".join(part for part in (category, questions, answer) if part)


async def write_chunks(session, chunks: list[Chunk]) -> int:
    """Chunk 列表 → knowledge_chunks 行(pending)。返回**新增**行数。

    查重口径 = `(category, questions, answer)` 三元组全等(spec §6.2):
    knowledge_chunks 没有 source 列(DDL 已定,不改),而语料量级小,
    应用层查重足够。已存在的行**原样保留**(含它的向量化状态与前后指针),
    所以「重复导入 = 无操作」,不会把已 done 的行打回 pending。
    """
    existing = {
        (row[0], row[1], row[2])
        for row in (
            await session.execute(
                select(
                    KnowledgeChunk.category,
                    KnowledgeChunk.questions,
                    KnowledgeChunk.answer,
                )
            )
        ).all()
    }

    fresh: list[KnowledgeChunk] = []
    for c in chunks:
        key = (c.category, c.questions, c.answer)
        if key in existing:
            continue
        existing.add(key)  # 批内重复同样挡掉,不靠数据库唯一约束(表上没有)
        row = KnowledgeChunk(
            category=c.category,
            questions=c.questions,
            answer=c.answer,
            section_path=c.section_path,
            content_type=c.content_type,
            is_key_clause=c.is_key_clause,
        )
        session.add(row)
        fresh.append(row)

    if not fresh:
        return 0
    await session.flush()  # 取回自增主键,才能串前后指针
    _link_neighbours(fresh)
    await session.commit()
    return len(fresh)


def _link_neighbours(rows: list[KnowledgeChunk]) -> None:
    """同章节的相邻新块双向串联(元数据,不进向量)。

    只串**本批新插入**的相邻块 —— 重跑时已存在的行保留原链接不动,所以
    往已建好的章节中间增量补录不会自动接到老邻居上。前后指针是溯源用的
    元数据、不参与检索,这个缺口可接受(spec §12 已记)。
    """
    for prev, cur in zip(rows, rows[1:]):
        if prev.section_path is None or prev.section_path != cur.section_path:
            continue
        prev.next_chunk_id = cur.id
        cur.prev_chunk_id = prev.id


async def vectorize_rows(session, store, embedder, rows: list[KnowledgeChunk]) -> None:
    """把一批行向量化并回填。

    顺序是**先写 Milvus、成功了再改状态并提交**。反过来的话,写 Milvus
    失败的那批行已经被记成 done,重跑再也不会补 —— 静默的永久缺失,
    而且它不会报错,只会让某些知识永远检索不到。
    """
    if not rows:
        return
    vectors = embedder.encode(
        [vector_text(r.category, r.questions, r.answer) for r in rows]
    )
    store.upsert([str(r.id) for r in rows], vectors)
    for r in rows:
        r.vector_id = str(r.id)  # 与 Milvus 主键对齐(spec §6.2)
        r.vectorize_status = "done"
    await session.commit()


async def vectorize_pending(session, store, embedder, *, batch_size: int = 16) -> int:
    """把所有 pending 行补齐,返回处理行数。**重跑 = 再跑一次这个函数。**

    每批一次 encode + 一次批量 upsert + 一次提交。批内中断的话这批仍是
    pending,重跑按同 pk 重写(Milvus upsert 幂等)后照常回填。

    batch_size 只管「一次处理多少行」,与 embedder 内部的 batch_size 无关。
    """
    store.ensure_collection()
    total = 0
    while True:
        rows = list(
            (
                await session.execute(
                    select(KnowledgeChunk)
                    .where(KnowledgeChunk.vectorize_status == "pending")
                    .order_by(KnowledgeChunk.id)
                    .limit(batch_size)
                )
            )
            .scalars()
            .all()
        )
        if not rows:
            return total
        await vectorize_rows(session, store, embedder, rows)
        total += len(rows)
