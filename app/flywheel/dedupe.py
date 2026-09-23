"""语义查重:候选问题是不是与待审队列里已有的某一条**是同一个意思**。

**纯 Prompt ⇒ 按项目规矩不套 TDD**,与 `normalize.py` 同款(解析退化那一支有单测,
因为它是个可单测的**分支**,不是提示词质量)。

⚠️ 提示词里必须出现字面 `JSON` 字样(json_mode 硬约束,spec §2.3),
且**不得使用裸花括号**(ChatPromptTemplate 按 f-string 解析)。

**为什么必须是语义判断而不是字面比对**:同一条知识缺口会以多种说法落进池子 ——
同一个问题被 `生成自评` 收一次、又被 👎 收一次,就是**两行**(`entry_point` 不是池子
的身份的一部分,spec §7.1);再加上口语与书面语的差别,字面全等在真实数据上几乎
从不命中。查重命中时累加的 `occurrences` 因此是**这条缺口有多真**的读数。
"""

import logging
from collections.abc import Sequence

from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.db.models import ReviewQueue

logger = logging.getLogger(__name__)


class DedupeResult(BaseModel):
    match_index: int = Field(
        description="与候选问题同义的那条在列表里的编号(从 1 开始);都不匹配填 -1"
    )


DEDUPE_SYSTEM_PROMPT = """你是电商客服知识库的去重助手。
给定一个候选问题和一份已有问题列表,判断候选问题与列表里的**哪一条问的是同一件事**。

判定标准:两句话说的是**同一件事**才算同义 —— 说法、语序、口语与书面语的差别
都不影响;只是话题相近(同属退换货、同属物流)但问的不是同一件事,**不算**同义。

输出一个 JSON 对象,只含一个键:
- match_index:整数。同义时填**列表里那一条的编号(从 1 开始)**;
  没有任何一条同义就填 -1。最多只算一条同义,拿不准时一律按 -1 处理。

不要输出 JSON 以外的任何内容。"""

_DEDUPE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", DEDUPE_SYSTEM_PROMPT),
        # 编号从 1 开始(给人看的列表就该这么编)。**返回的是编号不是下标**,
        # 所以下面那句 `pending[idx - 1]` 的减一是承重的;写错的话查重会把
        # 「命中第 2 条」记成第 1 条 —— 归并到了**另一条**不相关的问题上,
        # 而两边都不报错(`occurrences` 照样加一,匹配也照样发生了)。
        ("human", "候选问题:{candidate}\n\n已有问题列表:\n{listed}"),
    ]
)


async def find_duplicate(
    candidate: str, pending: Sequence[ReviewQueue], *, model
) -> ReviewQueue | None:
    """在 `pending` 里找**与 candidate 同义**的那一条;没有就返回 `None`。

    ⚠️ **解析失败 / 编号越界 / 列表为空,一律按「不匹配」处理**:宁可多建一行,
    也不要把两个不同的问题并成一个 —— 并错的代价是**永久**的(那两条缺口此后
    共用一行 `occurrences`,审核人删掉其中一条会把另一条一起删掉),而多建一行的
    代价只是审核人多看一眼。收益不对称时,退路要选**便宜且可逆**的那一边
    (与 ch07「摘要失败等于什么都没发生」同款取向)。

    上游故障(401/超时)**不在这里兜**,原样抛给流水线记成该行失败 —— 与
    `normalize_question` 同款:故障不许伪装成「不重复」这个业务判断。
    """
    # 没有可比对象 ⇒ 不花这次往返。**不只是省一次调用**:空列表交给模型判,
    # 它会从「什么都没有」里被要求给出一个编号,退化成一次没有依据的猜测。
    if not pending:
        return None

    listed = "\n".join(
        f"[{i}] {rq.standard_question}" for i, rq in enumerate(pending, start=1)
    )
    chain = model.with_structured_output(DedupeResult, method="json_mode")
    try:
        result = await chain.ainvoke(
            _DEDUPE_PROMPT.format_messages(candidate=candidate, listed=listed)
        )
    except OutputParserException:
        logger.warning("查重解析失败,按不匹配处理:%r", candidate, exc_info=True)
        return None

    index = result.match_index
    if index == -1:
        return None
    # 越界**响亮地**按不匹配处理,而不是 `pending[index]` 硬取:模型给出越界编号
    # 说明那一次判断本来就不可信(`IndexError` 会被流水线记成该行失败,越界之外
    # 的编号则会静默并到**无辜的**那一条上)。这里留痕但**不抛** —— 抛出去会让
    # 一个「模型数错了行」的问题升级成「这条缺口处理不了」。
    if not 1 <= index <= len(pending):
        logger.warning(
            "查重返回了越界编号 %s(列表只有 %s 条),按不匹配处理:%r",
            index, len(pending), candidate,
        )
        return None
    return pending[index - 1]
