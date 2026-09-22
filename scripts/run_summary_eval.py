"""摘要标注样例评估。需真实 key(打网络),不属于单测。

用法:
    .venv/Scripts/python.exe scripts/run_summary_eval.py

`SUMMARY_SYSTEM_PROMPT` 是**非可单测产出**(模型措辞在 temperature=0 下依然非确定),
按本项目的工作要求 1,它用**标注样例**验证而不是逐字断言 —— 本脚本即那份验证。

口径**闭式**,四类各自独立:

- `positive`      :`must_contain` 里的词**全部**出现(订单号 / 商品),
                    外加 `must_contain_any` 里**至少一个**(诉求那一路常有多样措辞);
- `negative`      :梗概长度 <= `max_chars`(**应当为空或极短**)。硬凑出「用户咨询了…」
                    这类对话状态就是它要挡的失效模式 —— prompt 明写寒暄/客套/对话状态不留;
- `hallucination` :梗概里**不得**出现 `forbidden_pattern` 形态的数字
                    (默认 `\\d{4,32}`,即订单号 / 手机号那种长度)。
                    ⚠️ 这条的判别力**依赖一个前提**:该用例的对话里本来就没有这个形态的数字
                    —— 否则一段忠实的梗概也会命中,那是**假红**而不是幻觉。
                    本脚本对每条探针**自动核对这个前提**(`用例自检` 那一列)。
- `extraction`    :`target` 是 §4.1 四样提炼物之一(商品 / 标识 / 诉求 / 未解决问题),
                    每样至少一条,分别统计。

**两个「装置自检」,它们才是本脚本最要紧的部分**(本仓的头号风险是假绿,不是错误代码):

1. **探针自检**:`\\d{4,32}` 必须**真的能**匹配它要抓的形态(`20240915` / `13800138000`),
   同时**不能**匹配 `99` —— ch06 的 T1 用 `\\d{4,32}` 去匹配「99」,那是**同义反复**
   (两边长度对不上,任何模型输出都过),零判别力。自检不过 ⇒ 直接退 1,不发布任何数字。
2. **空输出自检**:负例与探针**靠「什么都没有」通过**。若模型把**每一条**都压成空串,
   这两类会全绿而正例全红 —— 那种「绿」是装置坏了,不是结论。故单独统计非空条数,
   一条非空都没有时打 `!!!` 并退 1。

**走的是生产那一段**:`build_summary_messages`(生产 prompt 的唯一来源)+
`result.text.strip()`(与 `summarize_range` 里那两行同源),模型用
`create_extract_model`(温度 0)—— 与 `app/api/chat.py` 起后台摘要任务时传的
`model_factory` 是同一个。**不碰 DB**:本脚本只验 prompt 的产出,落库与锚点推进
由单测(`tests/test_memory_summarize.py`)与端到端验收覆盖。

「一条对不上」不等于「实现坏了」:模型措辞非确定,`MISS` 是**读数**不是判决
—— 全章口径见 spec §10.4。
"""

import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.llm import create_extract_model
from app.memory.summarize import build_summary_messages
from app.schemas import Message

CASES = Path(__file__).resolve().parents[1] / "evals" / "summary_cases.jsonl"

#: 探针自检用的样本。
#: `_MUST_MATCH` 是**真会出现在梗概里**的形态(订单号 8 位、手机号 11 位)——
#: 探针必须能抓住它们,否则它抓不住任何东西;
#: `_MUST_NOT_MATCH` 里 `99` 是 ch06 T1 那条同义反复断言的反面教材,另外两个是
#: 这两组用例里真的会出现的量词(「5 升」「8 小时」),它们**不该**被当成编造的标识。
_MUST_MATCH = ("20240915", "13800138000")
_MUST_NOT_MATCH = ("99", "5 升", "8 小时")


def emit(line: str = "") -> None:
    # 控制台是 cp936;摘要全是中文,一律走字节出口(本仓平台陷阱)。
    stream = getattr(sys.stdout, "buffer", None)
    if stream is None:
        print(line)
        return
    stream.write(line.encode("utf-8") + b"\n")
    stream.flush()


def _turns(row: dict) -> list[Message]:
    """用例里的一条 turn → `schemas.Message`(**带上结构字段**)。

    结构字段必须一起带上,不能只取 `role` / `content`:真实链路里那些行**就是**
    带 `tool_calls` / `tool_call_id` 落库的(`app/agent/nodes.py` 的 `_lc_to_records`),
    而 `role="tool"` 那条**没有 `tool_call_id` 根本构造不出来**(`Message` 的
    validator 直接拒 —— 它拒得有理由:空 `tool_call_id` 发到上游是畸形请求,
    实测回一个无从解释的 400)。
    """
    return [
        Message(
            role=t["role"],
            content=t["content"],
            tool_calls=t.get("tool_calls"),
            tool_call_id=t.get("tool_call_id"),
        )
        for t in row["turns"]
    ]


def _probe_selfcheck(patterns: set[str]) -> bool:
    """证明每个探针正则**有牙**:能抓真形态、不误伤量词。

    这一条是**装置**而不是被测对象 —— 它不过就没有任何数字值得读。
    """
    ok = True
    for pattern in sorted(patterns):
        matched = [s for s in _MUST_MATCH if re.search(pattern, s)]
        not_matched = [s for s in _MUST_NOT_MATCH if not re.search(pattern, s)]
        good = len(matched) == len(_MUST_MATCH) and len(not_matched) == len(_MUST_NOT_MATCH)
        ok = ok and good
        emit(
            f"  探针自检 {pattern}: 抓到 {matched}(应含全部 {len(_MUST_MATCH)} 个);"
            f" 放过 {not_matched}(应含全部 {len(_MUST_NOT_MATCH)} 个) {'✓' if good else '✗'}"
        )
    if not patterns:
        # **响亮失败,不是 ⚠️**:幻觉探针是四类样例之一(spec §10.4),
        # 一条都没有 ⇒ 这一类的判别力为零,而「⚠️ 然后 exit 0」会让人以为跑过了。
        # 与下面那条探针自检一样,装置不成立就不要发布数字。
        emit("  !!! 用例文件里没有任何 forbidden_pattern —— 幻觉探针这一类**没有用例**,本次结果不可读。")
        ok = False
    return ok


def _input_has_pattern(row: dict, pattern: str) -> bool:
    """用例自检:这段对话里本来就有被禁形态的数字吗?

    有 ⇒ 探针**不可能**有意义(一段忠实的梗概也会命中)。这不是模型失败,
    是用例坏了,必须单独报出来,不能混进 MISS 里。
    """
    return any(re.search(pattern, t["content"]) for t in row["turns"])


async def main() -> int:
    settings = get_settings()
    model = create_extract_model(settings)

    rows = [
        json.loads(line)
        for line in CASES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:                      # 空文件不许 ZeroDivisionError
        emit(f"用例文件是空的:{CASES}")
        return 1

    patterns = {r["forbidden_pattern"] for r in rows if r.get("forbidden_pattern")}
    emit(f"模型 {settings.openai_model}(温度 0)/ 用例 {len(rows)} 条 / {CASES.name}")
    emit("装置自检:")
    if not _probe_selfcheck(patterns):
        emit("!!! 探针自检未过 —— 幻觉探针没有判别力,本次结果不可读,直接退 1。")
        return 1
    emit()

    passed = 0
    nonempty = 0
    bad_cases = 0
    per_kind: dict[str, list[int]] = {}
    per_target: dict[str, list[int]] = {}

    for i, row in enumerate(rows, 1):
        kind = row["kind"]
        turns = _turns(row)
        result = await model.ainvoke(build_summary_messages(turns=turns))
        # 与 `summarize_range` 同源的两行:`.text` 取文本,`.strip()` 去掉空白。
        digest = (result.text or "").strip()
        nonempty += bool(digest)

        notes: list[str] = []
        case_ok = True
        pattern = row.get("forbidden_pattern")
        if pattern and _input_has_pattern(row, pattern):
            # 用例坏了:探针的前提不成立。**不计入通过**,单独计数并响亮报出来。
            case_ok = False
            bad_cases += 1
            notes.append(f"!!! 用例自检未过:对话里本来就有 {pattern} 形态的数字,探针恒假")

        if kind in ("positive", "extraction"):
            missing = [k for k in row.get("must_contain") or [] if k not in digest]
            any_of = row.get("must_contain_any") or []
            any_hit = (not any_of) or any(k in digest for k in any_of)
            case_ok = case_ok and not missing and any_hit and bool(digest)
            if missing:
                notes.append(f"缺关键词 {missing}")
            if not any_hit:
                notes.append(f"没覆盖到任一 {'/'.join(any_of)}")
            if not digest:
                notes.append("梗概为空 —— 正例不许为空")
        elif kind == "negative":
            # 「不能硬凑」有**两半**:① 不啰嗦(字数)② **不编内容**(不得凭空冒出标识形态的数字)。
            # 只断字数是**半条断言** —— 一条 15 字的编造订单号(「订单 20240915 已查到」)全过。
            # 两半合起来才是负例要挡的东西。
            limit = row["max_chars"]
            hit = re.search(pattern, digest) if pattern else None
            case_ok = case_ok and len(digest) <= limit and hit is None
            notes.append(f"{len(digest)} 字 / 上限 {limit}{'(空)' if not digest else ''}")
            if hit:
                notes.append(f"硬凑出了 {pattern} 形态的数字:{hit.group(0)!r}")
        elif kind == "hallucination":
            hit = re.search(pattern, digest)
            case_ok = case_ok and hit is None
            if hit:
                notes.append(f"梗概里编出了 {pattern} 形态的数字:{hit.group(0)!r}")
        else:
            case_ok = False
            bad_cases += 1
            notes.append(f"!!! 未知的 kind={kind!r}")

        passed += case_ok
        bucket = per_kind.setdefault(kind, [0, 0])
        bucket[0] += case_ok
        bucket[1] += 1
        if kind == "extraction":
            tb = per_target.setdefault(row["target"], [0, 0])
            tb[0] += case_ok
            tb[1] += 1

        tag = f"[{row['target']}] " if row.get("target") else ""
        emit(f"{'OK  ' if case_ok else 'MISS'} [{i}] {kind:<13}{tag}{'; '.join(notes)}")
        emit(f"        梗概:{digest if digest else '(空)'}")

    emit()
    emit("按类别:")
    for kind in ("positive", "negative", "hallucination", "extraction"):
        got, total = per_kind.get(kind, [0, 0])
        emit(f"  {kind:<14}{got}/{total}")
    emit("四样提炼物(extraction 内部):")
    for target in ("product", "identifier", "request", "unresolved"):
        got, total = per_target.get(target, [0, 0])
        emit(f"  {target:<11}{got}/{total}")

    emit()
    emit(f"非空梗概 {nonempty}/{len(rows)} —— 负例与探针**靠「什么都没有」通过**,")
    emit("  所以这个数必须与总通过数一起读:它接近 0 时那两类的绿是装置坏了,不是结论。")
    emit(f"总通过 {passed}/{len(rows)}")
    # **装置坏了就退非 0** —— 与探针自检同一条规矩:一条用例自检未过、或全部梗概为空,
    # 都意味着上面的数字**不可读**,而「打一行 !!! 然后 exit 0」在脚本/CI 里与通过无异。
    if bad_cases:
        emit(f"!!! 用例自检未过 {bad_cases} 条(见上面的 !!! 行)—— 这些条的结果不可读,退 1。")
        return 1
    if nonempty == 0:
        emit("!!! 所有梗概都是空的 —— 负例/探针的「通过」全部恒真,本次结果不可读,退 1。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
