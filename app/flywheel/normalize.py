"""口语原话 → 标准 FAQ 式问题 + 示例答案。

**纯 Prompt ⇒ 按项目规矩不套 TDD**,用 `evals/flywheel_cases.jsonl` 的标注样例验证
(见 `scripts/run_flywheel_eval.py`,与 `run_resolve_eval.py` 同款)。

⚠️ 提示词里必须出现字面 `JSON` 字样(json_mode 硬约束,spec §2.3),
且**不得使用裸花括号**(ChatPromptTemplate 按 f-string 解析)。

**为什么这一层值得存在**:池子里躺的是用户原话(「我买的那个猫砂盆寄到新疆要不要
另外加钱啊」),而审核队列是给人看的知识缺口清单 —— 一行一句口语原话,审核人既
判不出该不该补、也没法拿去知识库里查重。标准化把三件事一次做掉:去口语、补指代、
纠正错别字;顺带留一句示例答案,让审核那一侧有个起点(审核人改比从零写快)。

**它刻意不做的事**:不许把**业务性问题**(查订单、催发货)改写成知识问法 ——
「这一单到哪了」不是知识缺口,把它标准化成「物流时效政策是什么」会让审核队列里
混进一堆本来就有答案的伪缺口。这类问题照原意保留(含单号),示例答案留空。
"""

import logging

from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class NormalizeResult(BaseModel):
    """标准化的产物(两段都是**自由文本**,闭式判据在评估集里,不在这里)。"""

    standard_question: str = Field(description="整理后的标准 FAQ 式问题,一句话")
    example_answer: str = Field(
        description="示例答案;业务性提问(查单/催发货)返回空字符串"
    )


NORMALIZE_SYSTEM_PROMPT = """你是电商客服知识库的整理助手。
用户会把口语化的提问丢给你,你要把它整理成**一句标准 FAQ 式的问题**,
再补一句示例答案,供人工审核时判断「知识库该不该补这一条」。

规则:
1. standard_question:一句话,不超过 30 字。去掉寒暄、语气词与个人化的生活细节;
   补全指代(把「它」「那个」换成具体的商品名);纠正错别字;保留商品名、型号、
   订单号、地区这些业务实体。**不要复述用户原话的句式**。
2. 如果用户问的是**自己某一单的具体事务**(查订单到哪了、催发货、问这一单能不能退),
   那**不是**一个知识问题:standard_question 保留它的业务原意与单号原样,
   **不要**把它改写成「某某政策是什么」这种知识问法。
3. example_answer:不超过 40 字的示例答案。**只有第 2 条不适用时**才写;
   适用第 2 条时返回空字符串 —— 那类问题没有知识可补,写出来的都是编的。

输出一个 JSON 对象,只含两个键:
- standard_question:字符串
- example_answer:字符串

不要输出 JSON 以外的任何内容。"""

_NORMALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", NORMALIZE_SYSTEM_PROMPT),
        ("human", "用户原话:{question}"),
    ]
)


async def normalize_question(question: str, *, model) -> dict:
    """口语原话 → `{"standard_question": str, "example_answer": str}`。

    与 `app/kb/assess.py:assess_sufficiency` 同款(structured output + json_mode)。

    ⚠️ **解析失败退化为「原话入库」**,而不是往上抛:那一行是池子里的真问题,
    丢掉它就等于**静默地少了一个知识缺口**;原话虽然没被整理,至少信息没丢,
    审核人照样看得懂。往上抛的代价落在另一处 —— 流水线会把这一行记成失败,
    于是「模型偶尔吐了一句废话」变成「这条问题要等下一轮」,没有任何好处。

    上游故障(401/超时/连接失败)**不在这里兜**:那是 `except OutputParserException`
    只认解析失败的另一个分支,会原样抛到流水线,由那边记成该行失败、留在池子里
    等下一轮 —— 「基础设施故障绝不伪装成一条正常结果」,与 ch03 的检索边界同款。
    """
    chain = model.with_structured_output(NormalizeResult, method="json_mode")
    try:
        result = await chain.ainvoke(_NORMALIZE_PROMPT.format_messages(question=question))
    except OutputParserException:
        logger.warning("标准化解析失败,退化为原话入库:%r", question, exc_info=True)
        return {"standard_question": question, "example_answer": ""}

    standard = (result.standard_question or "").strip()
    # 空 / 纯空白 = 与解析失败**同一类**(模型没给出可用结果)。放行的话,
    # `review_queue.standard_question` 会落一条空串 —— 那一行在审核页上是**空白行**,
    # 而审核人既不知道原话是什么、也无从判断(原话在 `first_raw_question` 里,
    # 但那是次要字段)。**宁可退回原话**,与 ch07「空梗概不许落库」同一条道理:
    # 「有个占位的坏值」比「缺值」更坏,因为它看起来是正常的一行。
    if not standard:
        logger.warning("标准化返回了空问题,退化为原话入库:%r", question)
        return {"standard_question": question, "example_answer": ""}

    return {
        "standard_question": standard,
        "example_answer": (result.example_answer or "").strip(),
    }
