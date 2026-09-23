"""生成质量控制(ch04 spec §5.5):自评召回够不够 + 拒答落池。

自评是生成阶段的结构化一步(json_mode):给用户原话 + 召回的知识块,让模型
判断「够不够答」;不够则拒答,并把问题落 `low_confidence_questions` 池。
"""

from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.db.models import LowConfidenceQuestion


class AssessResult(BaseModel):
    sufficient: bool = Field(description="召回的知识是否足以回答用户问题")
    reason: str = Field(description="不足时的一句话原因")


ASSESS_SYSTEM_PROMPT = """你是电商客服的知识充分性判断助手。
给定用户问题和召回的知识块,判断这些知识是否足以直接回答用户的问题。

输出一个 JSON 对象,含两个字段:
1. sufficient:布尔值。true 表示足以回答;false 表示不足以回答,应当拒答。
2. reason:字符串。不足时用一句话说明原因。

判断标准:知识块里有没有覆盖用户问题的关键信息。没有相关内容、只有部分信息、
或信息与问题无关时,sufficient 为 false。不要输出 JSON 以外的任何内容。"""

_ASSESS_PROMPT = ChatPromptTemplate.from_messages(
    [("system", ASSESS_SYSTEM_PROMPT),
     ("human", "用户问题:{question}\n\n召回的知识块:\n{chunks}")]
)


class AssessError(Exception):
    """自评解析失败(与上游故障区分;退化为「够」,不因自评失败而误拒)。"""


async def assess_sufficiency(question: str, chunks: list, model) -> dict:
    """自评召回是否足够。返回 {"sufficient": bool, "reason": str}。

    解析失败退化为 sufficient=True(宁可放行由生成阶段兜底,也不因自评故障
    误拒一个能答的问题)。上游故障(401/超时)原样向上抛。

    **当前不在请求路径上**(ch05 spec §50):ch05 起「召回够不够」改由
    `app/agent/nodes.py` 的**置信度闸在事前**判定,ch04 这套「生成后再自评」
    被整段替换。T8 删掉 `services/chat.py:stream_turn` 后,本函数的生产调用方
    为零,只剩 `tests/test_kb_assess.py` 在跑它 —— 那个文件全绿**不代表**线上
    有这条链路。函数本身没坏,也不删(删它等于在 ch05 里改掉 ch04 已交付的
    接口面);留着是为了将来需要「生成后二次自评」时有现成的、有测试的实现。
    """
    chunk_text = "\n\n".join(f"[{i + 1}] {c.answer}" for i, c in enumerate(chunks))
    chain = model.with_structured_output(AssessResult, method="json_mode")
    try:
        result = await chain.ainvoke(
            _ASSESS_PROMPT.format_messages(question=question, chunks=chunk_text)
        )
    except OutputParserException:
        return {"sufficient": True, "reason": ""}
    return {"sufficient": result.sufficient, "reason": result.reason}


async def record_low_confidence(session, *, question: str, source_conversation_id: str | None,
                                entry_point: str, reject_reason: str,
                                evidence_snapshot: list | None = None) -> None:
    """问题落低置信度池。

    `evidence_snapshot` 是 ch09 加的:落池当轮的召回片段(Top-N 的
    id / 得分 / 章节 / 原文)。审核人靠它判「知识库真缺这块,还是有、但没检到」——
    没有它,池子里只有一句问题,那两件事看起来一模一样(ch09 spec §7.1)。

    ⚠️ **本参数由 T10 先落**(2026-09-23)。计划把「加这个参数**并返回新行 id**」
    整条记在 T11 名下,而 T10 的契约里已经写着「Consumes T5 的
    `record_low_confidence(evidence_snapshot=…)`」—— T5 只落了 ORM 那一列,
    函数签名没动,而 T10 排在 T11 前面。⇒ T10 只补**它当下就需要的那一半**
    (收下快照);**返回新行 id 仍是 T11 的活**。详见 task-10-report.md 的 concerns。
    """
    session.add(
        LowConfidenceQuestion(
            question=question,
            source_conversation_id=source_conversation_id,
            entry_point=entry_point,
            reject_reason=reject_reason,
            evidence_snapshot=evidence_snapshot,
        )
    )
    await session.commit()
