"""在线检索:问题向量化 → Milvus Top-K → 阈值过滤 → MySQL 回查原文。

Milvus 只当索引(spec §6.1),命中的 id 一律回 MySQL 取原文,所以在返回
之前有两件事必须做对:

- **顺序跟 Milvus 的相似度**,不跟 MySQL 的返回顺序 —— 综合结果按分数
  排序才对用户/模型有意义,而 SQL 的 `IN` 不保证任何顺序。
- **阈值以下的命中直接丢**,全丢光就返回空 —— dense 单路没有重排兜底,
  阈值是「不相关也硬凑答案」的唯一闸门(spec §6.6)。返回空由调用方
  (query_faq)翻成 ToolNotFound。

本模块是错误语义的**翻译边界**:Milvus 连不上、嵌入失败都是基础设施故障,
一律翻成 `ToolInfrastructureError`(502),绝不降级成「没搜到」——
那会把「向量库挂了」伪装成「这条知识没收录」,模型会如实转告用户「暂未
收录」,而我们永远不知道检索其实一直在失败。
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
    """一条召回结果 —— 字段与 query_faq 出参的 items 一一对应(spec §5.1)。"""

    question: str
    answer: str
    category: str


class KnowledgeRetriever:
    def __init__(self, session, store, embedder, *, top_k: int, score_threshold: float):
        self._session = session
        self._store = store
        self._embedder = embedder
        self._top_k = top_k
        self._score_threshold = score_threshold

    async def search(self, query: str) -> list[RetrievedChunk]:
        try:
            return await self._search(query)
        except BaseException:
            # 工具超时是靠 `asyncio.wait_for` **取消协程**实现的,取消点是随机的
            # (冷启动时第一次 encode 要加载 2.2GB 权重,十几秒都回不来,极容易
            # 落在那里)。被取消之后这条会话会停在一个没有收尾的事务上,而
            # executor 对 query_faq 是**会重试**的 —— 重试复用同一个 session,
            # 于是撞上 `PendingRollbackError`,把一次"超时重试"升级成用户可见的
            # 502(实测:验收 1 冷启动那一次就是这个形态)。
            #
            # 这里把事务收干净再往上抛。用 BaseException 而不是 Exception:
            # CancelledError 是 BaseException,只抓 Exception 正好漏掉这个场景。
            #
            # session 判空:单测里"全被阈值滤掉"这类用例根本不碰库,传的就是
            # None。不判空的话,真正的故障会被 `None.rollback()` 的
            # AttributeError 顶掉 —— 报错指向本行,而根因是上游的 Milvus 故障。
            if self._session is not None:
                await self._session.rollback()
            raise

    async def _search(self, query: str) -> list[RetrievedChunk]:
        kept = [
            (chunk_id, score)
            for chunk_id, score in self._top_hits(query)
            if score >= self._score_threshold
        ]
        if not kept:
            return []

        rows = await self._load_rows([chunk_id for chunk_id, _ in kept])
        out: list[RetrievedChunk] = []
        for chunk_id, _score in kept:  # 保持 Milvus 给的降序
            row = rows.get(chunk_id)
            if row is None:
                # Milvus 里残留的陈旧向量(对应行已从 MySQL 删掉)。跳过即可
                # —— 这正是「Milvus 只是索引」的代价,两边可能短暂不一致。
                continue
            out.append(
                RetrievedChunk(
                    question=row.questions,  # 全文,多个问法含换行,不截断
                    answer=row.answer,
                    category=row.category,
                )
            )
        return out

    def _top_hits(self, query: str) -> list[tuple[str, float]]:
        """嵌入 + 向量检索。两处失败都是基础设施故障,统一翻译。"""
        try:
            # 查询侧**不拼** category/questions/answer:那是入库时对 chunk 做的
            # 事(见 writer.vector_text)。查询就是问题原文。
            vector = self._embedder.encode([query])[0]
        except ToolInfrastructureError:
            raise
        except Exception as exc:
            raise ToolInfrastructureError(_INFRA_MESSAGE) from exc

        try:
            return self._store.search(vector, self._top_k)
        except ToolInfrastructureError:
            raise
        except Exception as exc:
            raise ToolInfrastructureError(_INFRA_MESSAGE) from exc

    async def _load_rows(self, chunk_ids: list[str]) -> dict[str, KnowledgeChunk]:
        """按 id 回查原文。

        返回 {str(id): row} —— 键是**字符串**,与 Milvus 的主键类型一致,
        免得调用方在 int/str 之间来回转。非数字 id 直接丢掉(理论上不会出现,
        真出现也不能变成 500)。
        """
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
