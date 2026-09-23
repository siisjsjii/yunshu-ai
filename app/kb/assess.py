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
                                evidence_snapshot: list | None = None) -> int:
    """问题落低置信度池。**返回新行的 id。**

    ⚠️ **边界:真 session 上返回新行 id,替身 session 上返回 `None`。**
    主键是 `commit()` 那一刻分配的,而本仓一大批单测替身 session(闸 / ch09 闸 /
    协议 / 节点 / 写流程,九处以上)的 `commit()` 只把对象收进 `added` 列表,
    **不 flush、也不给 id** ⇒ **照本仓惯例写的单测会从一个正确的实现上拿到
    `None`**。T12/T13 若在替身上断言返回值,先看这一条,别去查那个不存在的
    缺陷:要么把用例换成真 session(`@pytest.mark.db`,本文件那两条就是这么做的),
    要么就在替身上认下 `None`(那些用例该断的是「落池的对象长什么样」,
    不是 id —— 那条由真库用例守)。

    返回 id 的理由:ch09 起这张表不再是「只落不读」的池子,而是**数据飞轮的
    入口**(spec §6)。流水线要沿着「刚归并的是哪一行」往下走(`review_queue`
    的 `matched_review_id` 得指回它),审核页要按 id 定位;测试也用它**就地
    清理**自己写的那一行,不必再按问题文本反查(文本是自由文本,反查容易
    误伤别人写的行)。

    `evidence_snapshot` 是 ch09 加的:落池当轮的召回片段(Top-N 的
    id / 得分 / 章节 / 原文)。审核人靠它判「知识库真缺这块,还是有、但没检到」——
    没有它,池子里只有一句问题,那两件事看起来一模一样(ch09 spec §7.1)。
    不传 ⇒ 落 **JSON null**(不是空列表):「当轮零召回」与「没人记这件事」
    在审核页上是两件事。

    ⚠️ **它落的是 JSON 的 `null`,不是 SQL 的 NULL**(实测 2026-09-23:
    `evidence_snapshot IS NULL` = **0**、`JSON_TYPE(evidence_snapshot)` = `'NULL'`)。
    这是 SQLAlchemy `JSON` 列的默认行为(`none_as_null=False`:Python 的 `None`
    被写成 JSON `null`)。后果:**别拿 `WHERE evidence_snapshot IS NULL` 筛
    「这条没记快照」** —— 一行都筛不出来。经 ORM 读回是 Python `None`,
    所以审核页那侧看不出差别。

    ⚠️ **这两半是分两次落地的**(2026-09-23)。计划把「加 `evidence_snapshot`
    参数**并返回新行 id**」整条记在 T11 名下,而 T10 的契约里已经写着
    「Consumes T5 的 `record_low_confidence(evidence_snapshot=…)`」——
    T5 只落了 ORM 那一列,函数签名没动,而 T10 排在 T11 前面。
    ⇒ **T10 补了「收下快照」那半,T11(本次)补「返回新行 id」那半**。
    连带的覆盖度也分两次补:T10 那次只在**替身 session** 上验了「形参进了
    ORM 对象」,**真列上一条用例都没有**(T11 实测:把形参写死成 `None`,
    既有那条 db 用例照样绿),T11 补了真库往返那条。
    """
    row = LowConfidenceQuestion(
        question=question,
        source_conversation_id=source_conversation_id,
        entry_point=entry_point,
        reject_reason=reject_reason,
        evidence_snapshot=evidence_snapshot,
    )
    session.add(row)
    await session.commit()
    # 提交后取 id:本仓的 session 工厂是 `expire_on_commit=False`
    #(`app/db/base.py` 的注释写明了理由),属性在提交后仍可读,不会再发一次
    # SELECT。**不调 `session.flush()`**:那会加宽本函数对 session 的接口要求,
    # 而**九条**既有用例(闸 3 + ch09 闸 4 + 协议 2)用的是只实现 `add` + `commit`
    # 的替身 session —— 多要一个方法就把那些用例全打红,而它们与「返回 id」
    # 这件事毫无关系(踩过:见 task-11-report.md 的返工记录)。
    #
    # ⚠️ 这个数**重跑数过**(`pytest` 全量 → 9 failed, 873 passed;明细在报告 §3)。
    # 它一度被写成「七条」;在把「引用没数过的数」列为复发型缺陷的仓库里,
    # 留一个错的数比不留数更坏。
    return row.id
