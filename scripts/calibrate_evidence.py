"""在 evals/测试集.md 上标定 evidence_confidence_threshold。

用法:

    .venv/Scripts/python.exe scripts/calibrate_evidence.py
    .venv/Scripts/python.exe scripts/calibrate_evidence.py --grid 0.05,0.95,0.025

口径:
  「应拒答=是」的桶(D_absent,60 条)**拦下**才算对 ⇒ 拦截率
  其余桶(A/B/C/E,240 条)**放行**才算对 ⇒ 误杀率
  在「拦截率 ≥ 0.6」的阈值里挑**误杀率最低**的;若无解,退化为
  「误杀率 ≤ 0.05 里拦截率最高」并**响亮记账**(spec §4.2)。

⚠️ 前置:Milvus + BGE-M3 + 真实 .env。它走**真实检索链路**
(`KnowledgeRetriever`),不是 ch04 那个脚本的自建复刻 —— 标定的必须是
**线上那条路**。跑一次要加载 2.2GB 权重并跑 300 次检索,前几分钟是它的正常耗时。

⚠️ **本脚本不写配置**:它只打印全表 + 选中的那个,由人写回 `app/config.py`
(选定值属设计授权内的调参,按工作要求第 4 条记账)。
⚠️ 再说一遍口径,**因为它决定别人怎么引用这个脚本的输出**:选中的那个是
**平台内的一次判断**,不是"标定出的最优点" —— 本脚本**证明不了最优**,
它只能证明"这一段里读数相同"。曾有一版 docstring 写了
`--save`,而 argparse 里没有这个开关 —— 那种"文档说有、代码里没有"的开关
比没有更坏,已删。

⚠️ **用例文件是带引号逗号的真 CSV**(300 条里有 125 条的 query 形如
`"满多少钱包邮,不满怎么收运费"`)。所以这里必须用 `csv` 解析,**不能** `split(",")`:
后者会把 query 截断、并且把第 6 列错位到第 7 列 —— 实测「应拒答=是」会被数成
**44 条**(真值 60),也就是**负例凭空少 16 条**、而它们全部混进正常桶里去抬高误杀率。
这个错**不报错、只是数变** —— 正是本仓"看起来在跑、其实量错了"那一类。
"""

import argparse
import asyncio
import csv
import io
import json
import statistics
import sys
from pathlib import Path

# `python scripts/x.py` 时 sys.path[0] 是 **scripts/**、不是仓库根,所以要先补上。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.db.base import get_sessionmaker  # noqa: E402
from app.kb.evidence import evidence_confidence  # noqa: E402
from app.retrieval.search import KnowledgeRetriever  # noqa: E402
from app.tools.registry import build_retriever  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CASES = REPO / "evals" / "测试集.md"

# 表头逐字钉住:列换了位置而脚本按序号读 ⇒ 桶与「应拒答」会错位,
# 而错位后的读数**长得完全正常**(只是数变了)。宁可在这儿响亮地炸。
EXPECTED_HEADER = [
    "id", "桶(bucket)", "问题(query)", "期望章节(expect_section)",
    "标准要点(expect_points)", "应拒答(should_refuse)",
]
REFUSE_BUCKET = "D_absent"


def emit(line: str = "") -> None:
    """把一行以 **UTF-8 字节**写出(cp936 陷阱,同 run_tool_selection_eval)。

    ⚠️ 兜底分支**不能**退化成裸 `print(line)`:本机 locale 是 cp936,而兜底存在的
    意义就是"异常路径也要能出字" —— 一个会炸的兜底等于没有兜底,且它**恰好破坏了
    它自己要防的那件事**(本仓「脚本打印非 ASCII 要钉输出边界」那条)。
    正常 CLI 跑不到这里(`sys.stdout` 总有 `.buffer`),只有换掉 stdout 的调用方看得见。

    ⚠️ **崩的不是"中文"**(实测见下),别把机制记错:
    cp936/GBK **能**编码中文;崩的是 **GBK 之外的码点**。
    而本脚本**真会打的那一行**里就有这么一个 —— 阈值口径那行的 `⇒`(U+21D2):
        `UnicodeEncodeError: 'gbk' codec can't encode character '⇒'`
    (同类还有 `✓` `✗` `⚠`,即 CLAUDE.md 点名的那几个)。
    ⇒ 兜底里 `reconfigure(encoding="utf-8", errors="replace")` 之后才 `print`,
    且**任何**抛出都吞掉:诊断输出不该把主流程带走。
    """
    stream = getattr(sys.stdout, "buffer", None)
    if stream is None:
        reconfigure = getattr(sys.stdout, "reconfigure", None)
        try:
            if callable(reconfigure):
                reconfigure(encoding="utf-8", errors="replace")
            print(line)
        except Exception:
            # 连 print 都出不去时不再抛:诊断输出不该把主流程带走。
            pass
        return
    stream.write(line.encode("utf-8") + b"\n")
    stream.flush()


def load_cases() -> list[dict]:
    """读 300 条用例。**用 csv 模块**,不 split(",")—— 见模块 docstring。"""
    text = CASES.read_text(encoding="utf-8")
    rows = list(csv.reader(io.StringIO(text)))
    header = rows[0]
    if header != EXPECTED_HEADER:
        raise SystemExit(
            "用例文件表头与预期不符,拒绝按序号读列(会把桶和应拒答错位):\n"
            f"  预期 {EXPECTED_HEADER}\n  实际 {header}"
        )
    out = []
    for parts in rows[1:]:
        if len(parts) < 6 or not parts[0].strip():
            continue
        out.append({
            "id": parts[0].strip(),
            "bucket": parts[1].strip(),
            "query": parts[2].strip(),
            "should_refuse": parts[5].strip() == "是",
        })
    # 负例只该来自 D_absent(这是"闭式判据"的前提)。有 `应拒答=是` 落在别的桶,
    # 说明文件被改过而本脚本的口径没跟着改 —— 那时候拦截率的分母是错的。
    stray = sorted({c["bucket"] for c in out
                    if c["should_refuse"] and c["bucket"] != REFUSE_BUCKET})
    if stray:
        raise SystemExit(
            f"「应拒答=是」出现在 {stray} 桶里(预期只有 {REFUSE_BUCKET})"
            "—— 先确认口径,再谈标定。"
        )
    return out


def report_distribution(label: str, values: list[float]) -> None:
    """打印一组置信度的分布 —— 用来判断"这个读数是不是量出来的"。"""
    if not values:
        emit(f"诊断 · {label}:空")
        return
    q = statistics.quantiles(values, n=4) if len(values) > 3 else [values[0]] * 3
    emit(
        f"诊断 · {label}:n={len(values)} min={min(values):.3f} "
        f"p25={q[0]:.3f} p50={statistics.median(values):.3f} p75={q[2]:.3f} "
        f"max={max(values):.3f} 为 0 的={sum(1 for v in values if v == 0.0)}"
    )


async def main() -> None:
    ap = argparse.ArgumentParser(description="置信度阈值标定")
    ap.add_argument("--grid", default="0.05,0.95,0.025", help="起,止,步长")
    # 逐条读数落盘。**不是给运行用的,是给"这张表到底量到了什么"用的**:
    # 缺了它,「阈值不敏感」只能从表上反推(例如"拦截率一整段不动说明那段里
    # 一条用例都没有"),而反推出来的结论正是本仓吃过亏的那种。
    ap.add_argument("--dump", default=None, help="把逐条读数写成 JSON 的路径")
    args = ap.parse_args()

    start, stop, step = (float(x) for x in args.grid.split(","))
    settings = get_settings()
    cases = load_cases()
    n_ref = sum(1 for c in cases if c["should_refuse"])
    n_ok = len(cases) - n_ref
    buckets: dict[str, int] = {}
    for c in cases:
        buckets[c["bucket"]] = buckets.get(c["bucket"], 0) + 1
    emit(f"用例 {len(cases)} 条(应拒答 {n_ref} 条 / 正常 {n_ok} 条)")
    emit("桶分布:" + " ".join(f"{k}={v}" for k, v in sorted(buckets.items())))
    emit("阈值口径:confidence < t ⇒ 拦下(与闸的 `= threshold` 取反一致)")
    emit(f"检索器:top_k={settings.rerank_top_k} "
         f"score_threshold={settings.retrieval_score_threshold} "
         f"evidence_min_score={settings.evidence_min_score}")
    emit("")

    maker = get_sessionmaker()
    scores: list[tuple[bool, float, int]] = []
    async with maker() as session:
        retriever: KnowledgeRetriever = build_retriever(session, settings)
        for i, case in enumerate(cases, 1):
            chunks = await retriever.search(case["query"])
            scores.append((case["should_refuse"],
                           evidence_confidence(chunks, settings=settings),
                           len(chunks)))
            if i % 50 == 0:
                emit(f"  …{i}/{len(cases)}")

    # ---- 先自检:这个读数是不是量出来的 ----------------------------------
    # 检索器整体不出数(例如 Milvus 空了、集合名写错、实体没向量化)时,
    # 每条都是空证据 ⇒ 置信度恒 0 ⇒ 任何阈值都"拦下一切",拦截率恒 1。
    # 那个读数看起来是"完美达标",实际是**量具坏了**。这里显式判死。
    empty = sum(1 for _, _, n in scores if n == 0)
    if empty == len(scores):
        emit("\n!!! 300 条**全部**是空证据 —— 这不是标定结果,是检索链路没通。")
        emit("!!! 先查 Milvus 集合 / 向量化状态,再谈阈值。")
        raise SystemExit(2)

    if args.dump:
        raw = [
            {"id": c["id"], "bucket": c["bucket"], "query": c["query"],
             "should_refuse": c["should_refuse"],
             "chunks": n, "confidence": s}
            for c, (ref, s, n) in zip(cases, scores)
        ]
        Path(args.dump).write_text(
            json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")
        emit(f"逐条读数已写入 {args.dump}")

    emit("")
    report_distribution("正常桶(A/B/C/E)",
                        [s for ref, s, _ in scores if not ref])
    report_distribution("应拒答桶(D_absent)",
                        [s for ref, s, _ in scores if ref])
    emit(f"诊断 · 正常桶里空证据 {sum(1 for r, _, n in scores if not r and n == 0)} 条 / "
         f"应拒答桶里空证据 {sum(1 for r, _, n in scores if r and n == 0)} 条")
    emit("")

    # ---- 扫网格 ---------------------------------------------------------
    emit(f"{'阈值':>8}{'拦截率':>10}{'误杀率':>10}{'拦截数':>8}{'误杀数':>8}")
    emit("-" * 46)
    grid = []
    best = None
    t = start
    while t <= stop + 1e-9:
        blocked = [(ref, s) for ref, s, _ in scores if s < t]
        hit = sum(1 for ref, _ in blocked if ref)
        miss_block = sum(1 for ref, _ in blocked if not ref)
        catch = hit / n_ref if n_ref else 0.0
        kill = miss_block / n_ok if n_ok else 0.0
        grid.append({"t": round(t, 4), "catch": catch, "kill": kill,
                     "hit": hit, "miss_block": miss_block})
        # 先到先得:严格小于 ⇒ 并列时取**低**阈值(拦得少的那个)。
        if catch >= 0.6 and (best is None or kill < best["kill"]):
            best = grid[-1]
        t += step

    for row in grid:
        emit(f"{row['t']:>8.3f}{row['catch']:>10.3f}{row['kill']:>10.3f}"
             f"{row['hit']:>8d}{row['miss_block']:>8d}")

    degenerate = best is None
    if degenerate:
        cand = [r for r in grid if r["kill"] <= 0.05]
        best = max(cand, key=lambda r: r["catch"]) if cand else max(grid, key=lambda r: r["catch"])
        emit("")
        emit('!!! 没有"拦截率≥0.6"的解 —— 退化为"误杀率≤0.05 里拦截率最高"。')
        emit("!!! 这条退化**必须记进 spec §15 与 dev-notes**,不许当成正常结果。")

    # 拦截率在整个网格上纹丝不动,通常是量具坏了而不是"阈值不敏感"。
    if len({round(r["catch"], 6) for r in grid}) == 1:
        emit("")
        emit("!!! 拦截率在整个网格上是常数 —— 先怀疑量具(检索/用例读取),别当结论。")

    emit("")
    emit(f"选中阈值:{best['t']}(拦截率 {best['catch']:.3f},误杀率 {best['kill']:.3f})")
    emit("分支:" + ("退化(没找到满足拦截率≥0.6 的解)" if degenerate else "正常"))
    emit("请人工确认后写回 app/config.py 的 evidence_confidence_threshold。")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
