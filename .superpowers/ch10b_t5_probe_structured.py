"""T5 探针:`with_structured_output(method="json_mode")` 在本网关可用吗?打网络。

拿两条真实问句,把「裸 json_mode 路径」与「with_structured_output 路径」各跑一遍,
打印原始文本 / 解析结果 / 异常。**不是单测。**
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel

from app.config import get_settings
from app.llm import create_extract_model

ROOT = Path(__file__).resolve().parents[1]
PROMPT = """你在为一个电商客服系统做**多标签主题标注**。

权威的类目表(每类的边界说明 + 正例 + 反例):
- 退换货:要退货、换货、退款,或询问退换货的规则与流程
    · 例:「七天无理由怎么算」
    · 「保修期内坏了能修吗」归 保修维修,不归 退换货
- 尺码:尺寸、大小、肥瘦、身高体重对应的码数
    · 例:「175 穿什么码」
- 退货运费谁承担 -> 退换货 + 运费

请判断下面这句话**字面提到了几个诉求**,每个诉求打一个标签。

输出一个 JSON 对象,两个字段:
- labels:标签数组,标签名逐字来自上表
- evidence:对象,键是标签名,值是**这句话里支持该标签的连续片段**(照抄)

用户这句话是:
{question}

只输出 JSON,不要别的内容。"""


class _Out(BaseModel):
    labels: list[str]
    evidence: dict[str, str]


async def main() -> None:
    settings = get_settings()
    model = create_extract_model(settings)
    questions = ["买大了想退", "退货运费谁承担", "我要转人工"]
    for q in questions:
        prompt = PROMPT.format(question=q)
        from langchain_core.messages import HumanMessage

        resp = await model.ainvoke([HumanMessage(content=prompt)])
        text = (resp.text or "").strip()
        plain = "OK" if _safe(text) else "DECODE_FAIL"
        print(f"\n=== {q} ===")
        print(f"[plain]  {plain}  raw={text[:120]!r}")
        try:
            chain = model.with_structured_output(_Out, method="json_mode")
            out = await chain.ainvoke([HumanMessage(content=prompt)])
            print(f"[struct] OK  labels={out.labels} evidence={out.evidence}")
        except Exception as exc:  # noqa: BLE001
            print(f"[struct] {type(exc).__name__}: {str(exc)[:200]}")


def _safe(text: str) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except json.JSONDecodeError:
        return False


asyncio.run(main())
