"""Query 扩写与多路检索合并。

**扩写只发生在检索侧、现查现用** —— 库里的知识只留一份,不在入库侧拆存多份
(用户明确要求)。本模块因此不碰 chunker / writer。

错误语义:`multi_search` 里**单条查询失败不升级为整体失败**(用剩下的),
但 `ToolInfrastructureError`(Milvus/嵌入挂了)必须照原样上抛 ——
那是另一类故障,不能降级成「没搜到」。
"""

import logging

from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.exc import SQLAlchemyError

from app.prompts import build_expand_messages
from app.retrieval.search import _INFRA_MESSAGE, RetrievedChunk
from app.tools.errors import ToolInfrastructureError

logger = logging.getLogger(__name__)


class ExpandQueries(BaseModel):
    """扩写结果。**只有 queries 一个字段**(spec §4.3)。

    走 `with_structured_output(method="json_mode")`:本项目端点上
    `function_calling` / `json_schema` 均返回 400,与抽取/意图同源。
    """

    queries: list[str] = Field(default_factory=list)


async def multi_search(retriever, queries: list[str]) -> list[RetrievedChunk]:
    """多路检索 → 按 `chunk_id` 去重 → 按 `score` 降序。

    **同一 chunk 被多路召回、两次分数不同时:留分数更高的那一份**
    (`c.score > old.score`),不是首次出现的那一份。理由是**顺序无关性**:
    扩写的几条查询来自模型输出的 JSON 数组,谁排第一是**任意的**;若改成
    「首次出现者胜」,同一个块最终报出的分数就取决于这个任意顺序。而检索分数在
    本项目里当**置信度**用(`confidence_gate` 取 max 与
    `settings.retrieval_score_threshold` 比),于是「多扩写一路」反而可能把某块
    的分数压低到阈值以下、挤掉单路本来能命中的结果。取 max 保证扩写**只补召回、
    不削弱**已有证据。`tests/test_retrieval_expand.py` 里有一条**能区分两种取法**
    的用例钉着它。

    分数**并列**时(`>` 是严格比较)名次回落到**首次出现顺序** —— `merged` 是
    dict(插入序即首次出现序),`sorted` 又是稳定排序,两者叠加即得。

    单路失败的处理见模块 docstring:非基础设施故障**跳过**(留日志),基础设施
    故障**照抛**。三个 `except` 的**顺序不能换** —— 换过来就是「静默吞掉 502」。
    """
    merged: dict[int, RetrievedChunk] = {}
    for query in queries:
        try:
            chunks = await retriever.search(query)
        except ToolInfrastructureError:
            raise  # 基础设施故障照抛,不降级(见模块 docstring)
        except SQLAlchemyError as exc:
            # **纵深防御**:未翻译的数据库故障同样不许降级成「这一路没搜到」。
            # 根修在 `KnowledgeRetriever._load_rows`(它自己的 docstring 早就承诺了
            # 这个翻译边界),但检索器是**注入**的 —— 别的实现、或将来又漏翻译一处,
            # 这里得认得出裸 SQLAlchemy 错误。
            #
            # **必须翻成 `ToolInfrastructureError`,不能原样重抛**:
            # `app/api/chat.py` 只按这个类型给 502,原样抛出去是 **500**(而 500 把
            # 「服务端出问题」说成「你的请求有问题」)。文案借用 search.py 的同一条
            # —— 出站文本必须一致,不在这里复制一遍字面量(会静默漂移)。
            raise ToolInfrastructureError(_INFRA_MESSAGE) from exc
        except Exception as exc:
            # 单条查询的其他故障:跳过,用剩下的。**必须留痕** —— 否则这一路的
            # 召回悄悄消失,日志里什么都没有,形态与「库里本来就没有」一致。
            #
            # 这里只记**异常类型名**,不记 str(exc):本子句是宽口子,可能看到
            # 上游响应体原文(openai SDK 的 str(exc) 就是响应体),而它可能带着
            # 密钥 —— 出站文本要过 redact_api_key,日志没这道闸,索性不给它原文。
            logger.warning(
                "多路检索:查询 %r 失败,跳过该路(异常类型 %s)",
                query, type(exc).__name__,
            )
            continue
        for chunk in chunks:
            old = merged.get(chunk.chunk_id)
            if old is None or chunk.score > old.score:
                merged[chunk.chunk_id] = chunk
    return sorted(merged.values(), key=lambda c: -c.score)


async def expand_queries(model, *, text: str, max_queries: int) -> list[str]:
    """把一个问题泛化成多条侧重不同的检索查询。

    契约(逐条都有用例钉着):

    - **`max_queries >= 1` 时绝不返回空列表** —— 扩写失败退回 `[text]`,让检索
      至少有一条路走。空列表 = 检索空转,而它看起来与「库里没有这条知识」
      一模一样。`max_queries < 1` 时这个不变式**结构上无法成立**(见下面那道护栏),
      所以它不在承诺内 —— 那是一个会让空列表变成合法返回值的入参;
    - **条数上限是结构保证,不是提示词约定**:`max_queries` 既进 prompt 也在此处
      截断 —— 模型不照做时,截断发生在代码里;
    - **先去重(保序)再截断**,顺带清空白条目。顺序反了会白丢角度:
      `["退货", "退货", "运费"]` 配 `max_queries=2`,先截断就只剩一个角度。
      重复查询多跑一路 embed+search 也只是白花钱(合并结果一致)。

    **只接「模型出参不可用」这一族**(`OutputParserException` / `ValidationError`),
    与同一章的 `classify_intent` / `make_resolve_references_node` 一致。上游故障
    (超时/401/限流)与我自己代码的实现缺陷**都不接** —— 前者按 spec §8 该一路抛到
    端点的 error 帧,后者更不能被伪装成「这轮没扩写」。宽 `except Exception` 会把
    这两类都变成静默降级,而本仓栽在这上面太多次。
    """
    if max_queries < 1:
        # 「绝不返回空列表」在 `max_queries < 1` 时**结构上不可能成立**,而两种
        # 难看形态都实测过:`0` → `[]`(正是本模块通篇在防的「检索空转」);
        # `-1` → **静默少一条**(Python 负切片是「去掉最后 N 条」,不是空也不是截断)
        # —— 后者看起来完全正常,是最坏的一种。所以直接抛,把配置错误变成响声。
        # 生产不可达:spec §9 的配置项按 `Field(ge=1)` 在**启动时**拒(接线任务落地),
        # 这里是那道校验之外的最后一道。
        raise ValueError(f"max_queries 必须 >= 1,收到 {max_queries}")

    # 装配在 try 之外:它出错是我自己的实现缺陷(签名/模板),必须响。
    messages = build_expand_messages(text=text, max_queries=max_queries)
    chain = model.with_structured_output(ExpandQueries, method="json_mode")
    try:
        result = await chain.ainvoke(messages)
    except (OutputParserException, ValidationError) as exc:
        # 这里可以打 exc 原文:本子句只覆盖解析/校验失败,其 str 是**模型输出**,
        # 不含上游响应体(与 multi_search 那个宽子句不同)。
        logger.warning("Query 扩写失败,退回单路原问题:%s", exc)
        return [text]

    # 以下在 try 之外:上面接的是「模型出参不可用」,这里的 AttributeError 之类
    # 是我自己的实现缺陷,必须响(别用 getattr 兜底把它抹平)。
    queries = [q.strip() for q in result.queries if q.strip()]
    if not queries:
        logger.warning("Query 扩写返回空,退回单路原问题:%r", text)
        return [text]
    return list(dict.fromkeys(queries))[:max_queries]
