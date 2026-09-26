"""合成语料:配额 + 形态强制 + 禁词自检。

**打网络** —— 不是单测,不进 `pytest` 默认集。

用法:
    .venv/Scripts/python.exe scripts/gen_topic_data.py --dry-run   # 只打印 prompt
    .venv/Scripts/python.exe scripts/gen_topic_data.py

⚠️ **输出是 append**(`"a"`),为的是断点续跑;所以**重跑之前先删掉
`evals/topic/synthetic.jsonl`**,否则同一批问题会进两遍。脚本不替调用方做这件事 ——
删文件是不可逆的,交给人来决定。
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.llm import create_extract_model
from app.topic.synth import FORMS, QUOTA, shape_ok, violates_forbidden
from app.topic.taxonomy import OTHER, render_taxonomy_for_prompt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evals" / "topic" / "synthetic.jsonl"

#: 一次要几条。**不要一次要 200 条** —— 长输出里模型会开始重复自己,
#: 而重复的样本对训练没有增量,却会占掉配额。
#:
#: ⚠️ **今天它没有被接线**:`run` 里每次要的条数是 `batch_size(label, form)`
#: (即 `max(2, round(QUOTA[label] * FORMS[form]))`),头四类的 single 那一批
#: 因此要 **50 条**,而不是 12 条。留在这里是把它要表达的那条经验留下 ——
#: 但**别读成「已经按 12 条一批发了」**(本仓「引用了教训 ≠ 免疫于它」那一类)。
#: 实测(2026-09-26,962/962 全收、零重复问句)显示 50 条一批**这批数据上没出问题**,
#: 所以没有为了对齐这行注释去改行为 —— 那会让已量到的读数失效。
BATCH = 12


def build_prompt(label: str, form: str, n: int) -> str:
    """组装一批合成用的提示词。

    ⚠️ 三条硬约束写进 prompt,它们对应 `test_gen_topic_data.py` 的三条形态测试:
    ① **不许出现类目名与边界说明的原词**;
    ② `multi` 形态必须**真的带 2–3 个诉求**,且每个诉求在句子里**字面可指**;
    ③ 说人话 —— 像真实买家在客服窗口里打出来的,带口语与省略。

    ⚠️ 本仓硬约束:提示词里必须出现字面 `JSON` 字样,且**不得有裸花括号**。
    """
    form_desc = {
        "single": f"只涉及「{label}」**一个**诉求",
        "multi": f"涉及「{label}」**以及另外 1–2 个**不同的诉求(共 2–3 个)",
        "boundary": f"涉及「{label}」,而且**与它最容易混的那个类目**同时出现,"
                    f"让人必须读完才分得清",
    }[form]
    return f"""你在为一个电商客服系统造**训练语料**。

下面是本系统权威的类目表(含每类的边界说明与容易混的反例):

{render_taxonomy_for_prompt()}

请造 {n} 条**真实的买家问句**,要求:
1. 这些问句的主诉求是「{label}」,且{form_desc}。
2. **绝对不许出现任何类目名**(如类目表里那些词),也**不许出现边界说明里的原词**。
   买家不会说「我要咨询退换货类问题」,他只会说「买大了想退」。
3. 像真人在客服窗口打出来的:口语、有省略、可以带错别字。
4. 每条 5–30 个字。

输出一个 JSON 数组,每个元素是对象,含两个字段:
- question:问句本身
- labels:该问句应当打上的标签数组(**标签名必须逐字来自上面的类目表**)

只输出 JSON,不要别的内容。"""


def batch_size(label: str, form: str) -> int:
    """这一批要几条 —— 从 `QUOTA` 与 `FORMS` 推出来的那个算式,**只此一处**。

    单独成函数是为了可单测:「要出去的条数有没有兑现 `FORMS` 的比例」是一件**纯算术**,
    与打不打网络无关。埋在 `run` 里就只能在联网时看结果 —— 而 `FORMS` 本身只是
    一组常量,一个**无视 FORMS**、每批固定要 10 条的生成器能通过全部形态测试。
    """
    return max(2, round(QUOTA[label] * FORMS[form]))


async def run(dry_run: bool) -> None:
    settings = get_settings()
    model = create_extract_model(settings)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    kept, dropped = 0, 0
    with OUT.open("a", encoding="utf-8") as f:      # append:断点续跑
        for label in QUOTA:
            for form in FORMS:
                n = batch_size(label, form)
                prompt = build_prompt(label, form, n)
                if dry_run:
                    print(f"--- {label}/{form} n={n} ---\n{prompt[:400]}…\n")
                    continue
                raw = await _call(model, prompt)
                for item in raw:
                    q, labels = item.get("question", ""), item.get("labels", [])
                    # ⚠️ **三道门,任何一道不过就丢弃**:
                    #   ① 形态合法(非空、不重复、≤3、都是真类目)
                    #   ② 不含禁词(否则学成关键词匹配)
                    #   ③ 主诉求真的是这一类(模型经常跑偏)
                    if not shape_ok(labels):
                        dropped += 1
                        continue
                    if violates_forbidden(q):
                        dropped += 1
                        continue
                    if label not in labels and label != OTHER:
                        dropped += 1
                        continue
                    f.write(json.dumps(
                        {"question": q, "labels": labels, "provenance": "synthetic",
                         "source": "gen", "form": form, "seed_label": label},
                        ensure_ascii=False) + "\n")
                    kept += 1
                f.flush()
    if not dry_run:
        # ⚠️ 这两个数**必须打出来**。丢弃率高得离谱通常意味着**禁词表写宽了**
        #    或模型没守格式 —— 而不是「模型不行」。别跳过这个读数。
        print(f"保留 {kept} 条,丢弃 {dropped} 条(丢弃率 {dropped / max(1, kept + dropped):.1%})")


async def _call(model, prompt: str) -> list[dict]:
    """一次调用 + 解析。解析失败返回空列表(**不抛**)—— 合成可以少几条,不能整轮中断。"""
    from langchain_core.messages import HumanMessage

    try:
        resp = await model.ainvoke([HumanMessage(content=prompt)])
    except Exception as exc:                      # noqa: BLE001 —— 网络抖动重来即可
        print(f"  [warn] 调用失败,跳过这批:{type(exc).__name__}")
        return []
    text = (resp.text or "").strip()
    # 模型常把 JSON 包在 ```json 里 —— 剥掉再解析。
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text[4:] if text.lower().startswith("json") else text
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def main() -> None:
    # ⚠️ 钉死输出编码:本机 `sys.stdout.encoding` 是 **gbk**,而本脚本的输出里
    #    有中文(保留/丢弃那行的读数)。照 `scripts/prepare_topic_data.py` 的房规
    #    把边界钉成 UTF-8,不依赖控制台 codec —— 否则输出被重定向到文件时,
    #    「保留 N 条」那行会按 GBK 落盘,而读的人按 UTF-8 解。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只打印 prompt,不打网络")
    args = ap.parse_args()
    asyncio.run(run(args.dry_run))


if __name__ == "__main__":
    main()
