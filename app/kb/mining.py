"""对话挖知识(spec §6.5):历史会话 → LLM 抽问答对 → 暂存表 → 整体去重 → 入库。

离线管线,不在请求路径上,分两段:

1. **抽取按批**(每 `mine_batch_conversations` 个会话喂一次 LLM)—— 分批防串味,
   按 `batch_no` 可追溯,一批解析失败不影响其它批。
2. **去重放在全部批次抽完之后整体做** —— 批内去重挡不住跨批重复,而
   「同一句话被两个会话各问一遍」恰恰是最常见的重复来源。

两级去重:① 归一化后 sha256 精确去重(对 staging 内 + 已入库
`knowledge_chunks`);② 向量近重复(候选问法对库检索,相似度 ≥
`dedupe_threshold` 判重)。dense 单路下 ② 是 ① 的必要补充 —— 换个说法
问同一件事,字面指纹完全不同。
"""

import hashlib
import re
from dataclasses import dataclass

from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy import select

from app.db.models import KnowledgeChunk, MessageRecord, QaExtractionStaging
from app.kb.chunker import Chunk
from app.kb.writer import write_chunks
from app.schemas import MinedQaBatch

MINE_SYSTEM_PROMPT = """你是电商客服知识库的整理助手。
下面是若干段真实客服对话。请从中提取可以沉淀进知识库的问答对,并以 JSON 输出。

输出一个 JSON 对象,对象里只有一个字段 items,它的值是一个数组;
数组中每个元素都是一个对象,含 question、answer、category 三个字段:
1. question:字符串。用户在对话中提出的问法,尽量保留用户的原话措辞。
2. answer:字符串。客服在对话中给出的答案,**必须来自对话原文**,
   不允许你自己补充、推测或改写事实。对话里没有明确答案的,不要提取这一条。
3. category:字符串。这条知识所属的分类,如 退换货、物流异常、发票问题、商品咨询。

知识库收的是**对所有用户都成立的通用规则**,不是这一次对话的记录。以下一律不要提取:
- 客服没有正面回答的:表示查不到、无法确认、需要用户补充信息、或者只是让用户自己去操作的;
- 只对某个具体订单或商品才成立的**个案信息**:订单号、金额、库存数量、
  快递单号、物流轨迹、工单号、下单时间等具体数值;
- 寒暄、自我介绍、感谢、道歉这类没有信息量的客套;
- 与本次对话状态有关的回应(例如"你刚才说的是…""这是我们对话的开始")。

只提取对话中真实出现的信息,没有可提取的内容时 items 给空数组。
同一段对话里重复的问法只保留一条。
不要输出 JSON 以外的任何内容。"""

MINE_PROMPT = ChatPromptTemplate.from_messages(
    [("system", MINE_SYSTEM_PROMPT), ("human", "{conversations}")]
)


class MineParseError(Exception):
    """本批模型输出无法解析为问答对。

    与上游故障区分开:编排层对它是**记数后继续**(一批没抽好不该拖垮整跑),
    而上游故障(401/超时)会原样抛出、把整跑崩掉 —— 那才是该人工介入的。
    """


@dataclass(frozen=True)
class QaPair:
    question: str
    answer: str
    category: str


@dataclass(frozen=True)
class Turn:
    """一问一答。conversation_id 供 source_ref 溯源。"""

    conversation_id: str
    question: str
    answer: str


def build_mine_messages(conversation_text: str):
    return MINE_PROMPT.format_messages(conversations=conversation_text)


async def mine_batch(*, model, conversation_text: str) -> list[QaPair]:
    """一批对话文本 → 问答对。走 json_mode(该端点上另两种方法直接 400)。"""
    chain = model.with_structured_output(MinedQaBatch, method="json_mode")
    try:
        result = await chain.ainvoke(build_mine_messages(conversation_text))
    except OutputParserException as exc:
        raise MineParseError(f"模型输出无法解析为问答对:{exc}") from exc

    pairs: list[QaPair] = []
    for item in result.items or []:
        question = item.question.strip()
        answer = item.answer.strip()
        if not question or not answer:
            continue  # 模型偶尔回空壳,不能让它带着空问题进库
        pairs.append(QaPair(question, answer, item.category.strip()))
    return pairs


# ---- 归一化与去重 ----


#: `\W` 在 str 模式是 Unicode 感知的:中文算 word 字符、标点与空白不算,
#: 正好是「去标点去空白」想要的语义。
_NON_WORD_RE = re.compile(r"[\W_]+")


def normalize_question(text: str) -> str:
    """去空白、去标点、统一小写。

    目的是让「怎么退货?」与「 怎么退货 」判为同一条 —— 归一化**只**用于
    去重比对,入库保留的仍是模型给的原文。
    """
    return _NON_WORD_RE.sub("", text).lower()


def question_fingerprint(text: str) -> str:
    return hashlib.sha256(normalize_question(text).encode("utf-8")).hexdigest()


def dedupe_pairs(
    pairs: list[QaPair], seen: set[str]
) -> tuple[list[QaPair], list[QaPair]]:
    """按归一化指纹精确去重。`seen` 会被**就地更新**,返回 (保留, 丢弃)。

    就地更新是有意的:同一批里后面再出现同样的问法也要挡掉,而不是只跟
    传进来的历史比。
    """
    kept: list[QaPair] = []
    dropped: list[QaPair] = []
    for pair in pairs:
        fingerprint = question_fingerprint(pair.question)
        if fingerprint in seen:
            dropped.append(pair)
            continue
        seen.add(fingerprint)
        kept.append(pair)
    return kept, dropped


async def find_near_duplicate(*, store, embedder, question: str, threshold: float) -> bool:
    """向量近重复:候选问法与库里最相近的一条比相似度(≥ 阈值即判重)。"""
    vector = embedder.encode([question])[0]
    hits = store.search(vector, 1)
    return bool(hits) and hits[0][1] >= threshold


# ---- 分批与渲染 ----


def batch_conversations(turns: list[Turn], size: int) -> list[list[Turn]]:
    """按**会话**分批,同一会话的轮次绝不跨批。

    按"每 size 轮"切会在会话中间断开,于是一批里读到半段对话:模型看到
    问题没有对应答案,要么抽不出、要么把隔壁的问题配上去 —— 抽出的是
    **不存在**的知识,而且它看起来很合理。
    """
    ids: list[str] = []
    for turn in turns:
        if not ids or ids[-1] != turn.conversation_id:
            ids.append(turn.conversation_id)
    return [
        [t for t in turns if t.conversation_id in set(ids[start : start + size])]
        for start in range(0, len(ids), size)
    ]


def render_conversations(turns: list[Turn]) -> str:
    """一批对话 → 一段文本。带会话标题与角色前缀,会话之间用分隔线隔开。

    分隔线不是装饰:没有它,模型会把相邻两段对话连起来读成一段,把 A 会话
    的问题配 B 会话的答案 —— 与上个函数是同一类错误的两个入口。
    """
    blocks: list[str] = []
    current: str | None = None
    lines: list[str] = []
    for turn in turns:
        if turn.conversation_id != current:
            if lines:
                blocks.append("\n".join(lines))
            current = turn.conversation_id
            lines = [f"【会话 {turn.conversation_id}】"]
        lines.append(f"用户:{turn.question}")
        lines.append(f"客服:{turn.answer}")
    if lines:
        blocks.append("\n".join(lines))
    return "\n\n---\n\n".join(blocks)


# ---- 数据访问 ----


async def load_turns(session) -> list[Turn]:
    """全部会话 → 一问一答列表。不足一轮的会话自然被跳过。

    assistant 的空应答**不**清空待配对的 user 提问:走工具调用的那一轮,
    assistant 先推一条 content 为空的 tool_call 消息,真正的答复在后面
    那条 —— 在这里把它当"回答完了"会让所有查过数据的对话都配不上。
    """
    rows = (
        (
            await session.execute(
                select(MessageRecord).order_by(
                    MessageRecord.conversation_id, MessageRecord.id
                )
            )
        )
        .scalars()
        .all()
    )
    turns: list[Turn] = []
    current: str | None = None
    pending: str | None = None
    for row in rows:
        if row.conversation_id != current:
            current, pending = row.conversation_id, None
        if row.role == "user":
            pending = row.content
        elif row.role == "assistant":
            if not row.content.strip():
                continue
            if pending:
                turns.append(Turn(current, pending, row.content))
            pending = None
    return turns


async def existing_fingerprints(session) -> set[str]:
    """已入库知识的问法指纹。

    `questions` 可能是多行(块有多个问法),**每一行**都要参与去重 ——
    只取整串的话,"库里第二条问法"与挖出来的问法重复就检不出来。
    """
    rows = (await session.execute(select(KnowledgeChunk.questions))).scalars().all()
    prints: set[str] = set()
    for value in rows:
        for line in str(value).splitlines():
            if line.strip():
                prints.add(question_fingerprint(line))
    return prints


async def staging_fingerprints(session) -> set[str]:
    """暂存表里出现过的问法指纹。

    **三态都算**,不只是 kept:否则重跑会把上一轮已经判为 discarded 的
    再抽一遍、再判一遍,白烧 LLM 而且输出里全是重复的丢弃记录。
    """
    rows = (await session.execute(select(QaExtractionStaging.question))).scalars().all()
    return {question_fingerprint(q) for q in rows if q and q.strip()}


async def write_staging(
    session, *, batch_no: str, source_ref: str | None, pairs: list[QaPair]
) -> list[QaExtractionStaging]:
    """问答对写入暂存表,状态 extracted。返回 ORM 行(供后续改状态)。"""
    rows = [
        QaExtractionStaging(
            batch_no=batch_no,
            source_ref=source_ref,
            question=pair.question,
            answer=pair.answer,
        )
        for pair in pairs
    ]
    session.add_all(rows)
    await session.commit()
    return rows


async def mark_staging(session, rows, status: str) -> None:
    for row in rows:
        row.status = status
    await session.commit()


async def finalize_staging(
    session, *, batch_no: str, kept: list[QaPair]
) -> tuple[int, int]:
    """把某批次的 staging 行按最终结论标成 kept / discarded,返回 (kept, discarded)。

    按**问法指纹**匹配而不是行对象:写入 staging 用的是另一个已关闭的 session,
    拿着那些 ORM 对象到新 session 上改属性是不生效的(detached),重新按
    batch_no 查回来再匹配才可靠。

    同一个问法在多批里各抽到一次时**只有一条标 kept**,其余标 discarded ——
    `kept` 的含义是「这一行进了 knowledge_chunks」,而不是「这个问法活过了
    去重」。否则同一句话被 5 个会话问过就会有 5 行 kept,而库里其实只多了
    1 条,统计口径对不上(实测踩过:入 19 条而 staging 报 kept 37)。

    仍然停在 extracted 的行一律判 discarded:本轮的抽取结果都已经有了结论。
    """
    rows = (
        (
            await session.execute(
                select(QaExtractionStaging)
                .where(QaExtractionStaging.batch_no == batch_no)
                .order_by(QaExtractionStaging.id)
            )
        )
        .scalars()
        .all()
    )
    kept_prints = {question_fingerprint(pair.question) for pair in kept}
    claimed: set[str] = set()
    kept_count = discarded_count = 0
    for row in rows:
        fingerprint = question_fingerprint(row.question)
        if fingerprint in kept_prints and fingerprint not in claimed:
            claimed.add(fingerprint)
            row.status, kept_count = "kept", kept_count + 1
        else:
            row.status, discarded_count = "discarded", discarded_count + 1
    await session.commit()
    return kept_count, discarded_count


async def keep_pairs(session, pairs: list[QaPair]) -> int:
    """保留的问答对 → `knowledge_chunks`。

    **复用 writer.write_chunks**:因此自动获得三元组查重幂等,并且落成
    `pending` 行、由 build_kb 的向量化步骤统一补齐(spec §6.5)。

    注意 staging 表**没有** category 列(DDL 既定,不改),所以分类只活在
    本次运行的 QaPair 里 —— 跨进程续跑不支持,重跑会重抽(幂等,不会重复入库)。
    """
    chunks = [
        Chunk(
            category=pair.category,
            questions=pair.question,
            answer=pair.answer,
            section_path=None,  # 挖出来的知识没有章节结构
            content_type="faq",
            is_key_clause=False,
        )
        for pair in pairs
    ]
    return await write_chunks(session, chunks)
