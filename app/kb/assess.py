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
                                entry_point: str, reject_reason: str) -> None:
    """问题落低置信度池。"""
    session.add(
        LowConfidenceQuestion(
            question=question,
            source_conversation_id=source_conversation_id,
            entry_point=entry_point,
            reject_reason=reject_reason,
        )
    )
    await session.commit()
