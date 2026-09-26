"""大模型预标:照术语表打标签,**每个标签必须附一句原文证据**。

**打网络** —— 不是单测。

断点续跑:`evals/topic/prelabeled.jsonl` 是 append-only,
重跑时跳过已经有结果的 id(照 `build_kb.py`「重跑 = 幂等补齐」的先例)。
⚠️ 正因为是 append-only,**改了 prompt 就必须先删产物**再重跑 ——
已写出的行不会被覆盖。

用法:
    .venv/Scripts/python.exe scripts/prelabel_topics.py --limit 5
    .venv/Scripts/python.exe scripts/prelabel_topics.py
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.llm import create_extract_model
from app.topic.labeling import validate_evidence
from app.topic.taxonomy import render_taxonomy_for_prompt

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "evals" / "topic" / "corpus.jsonl"
SYNTH = ROOT / "evals" / "topic" / "synthetic.jsonl"
OUT = ROOT / "evals" / "topic" / "prelabeled.jsonl"

#: ⚠️ **正例自(订正 D,2026-09-26)起也在渲染块里**(`render_taxonomy_for_prompt`),
#: 所以这里写的是「边界说明、正例与反例」而不是旧版的「边界说明与反例」——
#: 这句是**描述渲染块含什么**的陈述,不跟着改就是第三类「让别处的陈述变假」。
PROMPT = """你在为一款电商客服系统做**多标签主题标注**。

权威类目表(含每类的边界说明、正例与容易混的反例):

{taxonomy}

请判断下面这句话**字面提到了几个诉求**,每个诉求打一个标签。

判据是「**字面提到**」,不是「用户真正想要什么」:
- 「买大了想退」→ 尺码、退换货(字面两个诉求)
- 「退货运费谁承担」→ 退换货、运费
- 「保修期内坏了能退吗」→ 保修维修、退换货
- 「满多少钱包邮」→ 运费(**不是**优惠活动)
- 「我要转人工」→ 其他(这不是主题)

**一个不多,一个不少。**

输出一个 JSON 对象,两个字段:
- labels:标签数组,每个标签名**逐字来自上面的类目表**
- evidence:对象,键是标签名,值是**这句话里支持该标签的那几个字**(必须是原句的**连续片段**,照抄,不要改写)

用户这句话是:
{question}

只输出 JSON,不要别的内容。"""


async def run(limit: int | None) -> None:
    done = _already_done()
    rows = []
    for path in (CORPUS, SYNTH):
        if path.exists():
            rows += [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    todo = [r for r in rows if r["id"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"总 {len(rows)} 条,已完成 {len(done)} 条,本次处理 {len(todo)} 条")

    settings = get_settings()
    model = _bind_json(create_extract_model(settings))
    flagged = 0        # 有标签被证据校验拒掉 —— **质量读数**
    parse_failed = 0   # JSON 根本没解出来(或形状不是约定那样)—— **故障读数**
    zero_label = 0     # 解出来了,但一个标签都没落下(含上面那种,也含模型真判了零诉求)
    with OUT.open("a", encoding="utf-8") as f:
        for row in todo:
            labels, evidence, bad = await _label(model, row["question"])
            if bad:
                parse_failed += 1
            accepted, rejected = validate_evidence(row["question"], labels, evidence)
            if rejected:
                flagged += 1
            if not accepted:
                zero_label += 1
            f.write(json.dumps(
                {**row, "labels": accepted, "rejected_labels": rejected,
                 # ⚠️ 「分得开」的**唯一**保证就是这一列(见 `_label` 的 docstring)
                 "parse_failed": bad,
                 "evidence": {k: v for k, v in evidence.items() if k in accepted}},
                ensure_ascii=False) + "\n")
            # 每条都 flush:断点续跑靠它 —— 中途崩了,已处理的行必须已经在盘上。
            f.flush()
    # ⚠️ **三个读数都要打,别只打一个**:
    #    `flagged`      = 有标签被证据校验拒掉 —— **质量读数**,直接进 spec §6.3
    #    `parse_failed` = JSON 没解出来 —— **故障读数,它是 0 才正常**
    #    `zero_label`   = 一个标签都没落下(含上面那种,也含「模型真的判了零诉求」)
    print(f"完成。被证据校验拒掉标签的 {flagged} 条({flagged / max(1, len(todo)):.1%});"
          f"**JSON 解析失败 {parse_failed} 条**;空标签 {zero_label} 条")
    if parse_failed:
        print("⚠️ 解析失败不为 0 ⇒ **先停下看产物,别直接进 Task 6** ——"
              "这些行的标签是空的,而不是「判定了没有主题」。")


def _bind_json(model):
    """把模型绑成「输出必是 JSON 对象」。

    ⚠️ **为什么是 `bind(response_format=…)` 而不是 `with_structured_output(method="json_mode")`**
    (2026-09-26 实测 + 读锁定版本的轮子源码,不是照文档猜的):

    - 实测:`with_structured_output(_Out, method="json_mode")` 在本网关**可用**
      (3/3 条问句正常返回),所以那句「不可用」的顾虑不成立;
    - 但读 `langchain_openai/chat_models/base.py` 的 `with_structured_output`:json_mode 那一支
      在 `schema` **不是 pydantic 类**时,输出解析器是**裸的 `JsonOutputParser()`** ——
      **零 schema 校验**(它给的是 `response_format={"type":"json_object"}` 而已);
      传 pydantic 类时换 `PydanticOutputParser`,而它**把两种失败并成一个**
      —— 源码逐字:「Raises `OutputParserException`: If the result is not valid JSON
      **or does not conform** to the Pydantic model」,且**原始文本被丢掉**。
      两个后果都不可接受:① `parse_failed` 是**故障读数**,并进来之后它就成了混合量;
      ② 「模型吐了散文」与「吐了 JSON 但字段形状不对」变得**分不开** ——
      而 `parse_failed` 这一列存在的全部理由就是分得开(订正 C)。
    - ⇒ 只取 json_mode 真正加的那一样东西(`response_format`,网关侧保证是个 JSON 对象),
      解析自己做。这也是 `app/services/extract.py` 的 docstring 描述的那条真实链路。
    """
    return model.bind(response_format={"type": "json_object"})


async def _label(model, question: str) -> tuple[list[str], dict[str, str], bool]:
    """返回 `(labels, evidence, parse_failed)`。

    ⚠️ **`parse_failed` 必须一路传出去**(计划订正 C,controller 2026-09-26)。
    本节原稿在 `JSONDecodeError` 时 `return [], {}`,调用方写出的行是
    `{"labels": [], "rejected_labels": [], "evidence": {}}` —— 与「模型**真的**返回零标签」
    (合法,例如闲聊)写出来的行**逐字节相同**。而原稿那行注释写的是
    「它与「有标签但被拒」是两回事,**分开记**才能在人审时看出是哪一种」—— **那句是假的**。

    后果不是「少个字段」:① 空标签行进 Task 6 人审、进 Task 7 分层抽样时**看起来是正常行**,
    而它可能是「模型吐了散文、JSON 没解出来」;② `flagged` 那个「质量读数」
    (要进 spec §6.3 的训练标签错误率)会**系统性偏小**。

    ⚠️ **形状也要挡**(brief 的原稿只挡了 JSON 语法):模型把 `evidence` 吐成字符串、
    或把 `labels` 吐成一个字符串,原稿会 ① 把 `"尺码,退货"` 按**字符**迭代、
    每个字符都判「不是原句子串」⇒ 全部被拒(读数看起来像模型的错);
    ② 在 `validate_evidence` 里对非 str 调 `.strip()` ⇒ **AttributeError 打断整跑**。
    这两种都不是「JSON 没解出来」,但都不是正常结果 ⇒ 归 `parse_failed`。
    单个 evidence 的值不是 str 时**只丢那一个键**(它那一条会被证据校验拒掉,
    进 `flagged` —— 质量读数,正是该看见的地方),不整行作废。
    """
    from langchain_core.messages import HumanMessage

    prompt = PROMPT.format(taxonomy=render_taxonomy_for_prompt(), question=question)
    resp = await model.ainvoke([HumanMessage(content=prompt)])
    text = (resp.text or "").strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text[4:] if text.lower().startswith("json") else text
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [], {}, True
    if not isinstance(data, dict):
        return [], {}, True
    raw_labels = data.get("labels")
    raw_ev = data.get("evidence")
    if not isinstance(raw_labels, list) or not isinstance(raw_ev, dict):
        return [], {}, True
    return (
        [x for x in raw_labels if isinstance(x, str)],
        {k: v for k, v in raw_ev.items() if isinstance(v, str)},
        False,
    )


def _already_done() -> set[str]:
    if not OUT.exists():
        return set()
    return {
        json.loads(l)["id"]
        for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()
    }


def _pin_stdout_encoding() -> None:
    """把 stdout 钉成 UTF-8。**这是本机的硬约束,不是美化。**

    本机 locale 是 cp936,而 `⚠️`(U+26A0)与 `⇒`(U+21D2)**编不进 GBK** ——
    它们只出现在「解析失败不为 0」那条 print 里 ⇒ 不钉的话,**恰恰在最需要它输出的
    那条路径上**抛 `UnicodeEncodeError`。实测(2026-09-26):

    ```
    $ python -c "print('⚠️ 解析失败不为 0 ⇒ 停下')" > out.txt
    UnicodeEncodeError: 'gbk' codec can't encode character '\\u26a0' in position 0
    ```

    产物是逐行 flush 的,所以**不会丢数据**;但脚本以退出码 1 结束、那行警告消失 ——
    而「产物是好的」与「日志里没有警告」叠在一起是最难分辨的一种假绿。
    (`tests/test_topic_labeling.py::test_main_pins_stdout_encoding` 用子进程钉住它。)
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    _pin_stdout_encoding()
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    asyncio.run(run(args.limit or None))


if __name__ == "__main__":
    main()
