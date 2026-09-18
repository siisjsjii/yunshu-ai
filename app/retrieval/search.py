"""在线检索:问题向量化 → 混合检索(dense+BM25,RRF)→ 回查 → 重排 → 阈值过滤。

ch04 起检索换成混合 + 重排:

1. 嵌入 query → dense 向量;
2. `hybrid_search`(dense 腿 + BM25 腿,RRF 融合)→ 候选 Top-N;
3. 回 MySQL 取原文(拿 text 供重排);
4. bge-reranker-v2-m3 精排 → Top-K(0-1 sigmoid 分数);
5. 阈值过滤 → 返回带 chunk_id / section_path / score 的块。

本模块是错误语义的**翻译边界**:Milvus 连不上、嵌入失败、重排失败都是基础设施
故障,一律翻成 `ToolInfrastructureError`(502),绝不降级成「没搜到」。
"""

from dataclasses import dataclass

from sqlalchemy import select

from app.db.models import KnowledgeChunk
from app.tools.errors import ToolInfrastructureError

#: 翻译后的对外文案。**不含底层异常文本** —— openai/pymilvus 的 str(exc)
#: 可能带上端点信息,统一由 api 层的 redact_api_key 再兜一道。
_INFRA_MESSAGE = "知识检索服务暂时不可用"


@dataclass(frozen=True)
class RetrievedChunk:
    """一条召回结果。chunk_id / section_path 供引用定位回原文(ch04)。

    question/answer/category 在前保持旧位置构造兼容;chunk_id/section_path/score
    是 ch04 新增、带默认值(检索器用关键字构造,不受字段序影响)。
    """

    question: str
    answer: str
    category: str
    chunk_id: int = 0
    section_path: str | None = None
    score: float = 0.0


class KnowledgeRetriever:
    def __init__(self, session, store, embedder, reranker, *, top_k: int,
                 score_threshold: float, hybrid_top_k: int = 50):
        self._session = session
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._top_k = top_k
        self._score_threshold = score_threshold
        self._hybrid_top_k = hybrid_top_k

    async def search(self, query: str) -> list[RetrievedChunk]:
        try:
            return await self._search(query)
        except BaseException:
            # 工具超时靠 `asyncio.wait_for` 取消协程,取消点随机。取消后会话停在
            # 未收尾事务上,而 query_faq 会重试复用同一 session → PendingRollbackError
            # 把「超时」升级成 502。这里回滚再抛。BaseException 覆盖 CancelledError。
            if self._session is not None:
                await self._session.rollback()
            raise

    async def _search(self, query: str) -> list[RetrievedChunk]:
        hits = self._hybrid_hits(query)  # [(chunk_id, rrf_score)]
        if not hits:
            return []

        rows = await self._load_rows([chunk_id for chunk_id, _ in hits])
        ranked = self._rerank(query, hits, rows)  # [(chunk_id, sigmoid_score)]
        kept = [(cid, score) for cid, score in ranked if score >= self._score_threshold]
        if not kept:
            return []

        out: list[RetrievedChunk] = []
        for chunk_id, score in kept:
            row = rows.get(chunk_id)
            if row is None:
                # Milvus 里残留的陈旧向量(对应行已从 MySQL 删掉)。跳过。
                continue
            out.append(
                RetrievedChunk(
                    chunk_id=int(chunk_id),
                    question=row.questions,  # 全文,多个问法含换行,不截断
                    answer=row.answer,
                    category=row.category,
                    section_path=row.section_path,
                    score=score,
                )
            )
        return out

    def _hybrid_hits(self, query: str) -> list[tuple[str, float]]:
        """嵌入 + 混合检索。两处失败都是基础设施故障,统一翻译。"""
        try:
            vector = self._embedder.encode([query])[0]
        except ToolInfrastructureError:
            raise
        except Exception as exc:
            raise ToolInfrastructureError(_INFRA_MESSAGE) from exc

        try:
            return self._store.hybrid_search(vector, query, self._hybrid_top_k)
        except ToolInfrastructureError:
            raise
        except Exception as exc:
            raise ToolInfrastructureError(_INFRA_MESSAGE) from exc

    def _rerank(self, query: str, hits: list[tuple[str, float]],
                rows: dict[str, KnowledgeChunk]) -> list[tuple[str, float]]:
        """对候选块用 bge-reranker-v2-m3 精排,返回 Top-K 的 [(chunk_id, score)]。

        排序依据是重排分数(0-1 sigmoid);分数同时当置信度供阈值过滤。
        """
        candidates = [(cid, rows[cid].answer) for cid, _ in hits if cid in rows]
        if not candidates:
            return []
        try:
            scores = self._reranker.rerank(query, candidates)
        except ToolInfrastructureError:
            raise
        except Exception as exc:
            raise ToolInfrastructureError(_INFRA_MESSAGE) from exc
        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
        return [(cid, score) for (cid, _), score in ranked[: self._top_k]]

    async def _load_rows(self, chunk_ids: list[str]) -> dict[str, KnowledgeChunk]:
        """按 id 回查原文,返回 {str(id): row}。非数字 id 直接丢掉。"""
        numeric_ids = []
        for chunk_id in chunk_ids:
            try:
                numeric_ids.append(int(chunk_id))
            except (TypeError, ValueError):
                continue
        if not numeric_ids:
            return {}
        rows = (
            (
                await self._session.execute(
                    select(KnowledgeChunk).where(KnowledgeChunk.id.in_(numeric_ids))
                )
            )
            .scalars()
            .all()
        )
        return {str(row.id): row for row in rows}
