"""最终修复轮的变异脚本(本机跑一次,产物不回写仓库)。

每条变异:**锚在代码上**(不锚注释)、断言命中数恰好为 1、跑指定用例、
**看到 `N passed|failed` 才判定**,然后原样还原。看不到就打 !!!。
"""

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = str(REPO / ".venv" / "Scripts" / "python.exe")

MUTATIONS = [
    (
        "M1 回捞失败退回 `None`(与零召回同一个值)",
        "app/api/feedback.py",
        '        snapshot = RECALL_FAILED_SNAPSHOT\n',
        '        snapshot = None\n',
        ["tests/test_api_feedback.py::test_retriever_failure_still_pools_the_row_and_logs_loudly",
         "tests/test_api_feedback.py::test_a_hanging_recall_is_bounded_and_lands_on_the_same_sentinel"],
    ),
    (
        "M2 feedback 去掉墙钟上界(裸 await)",
        "app/api/feedback.py",
        "        chunks = await _search_bounded(retriever, body.question, settings=settings)",
        "        chunks = await retriever.search(body.question)",
        ["tests/test_api_feedback.py::test_a_hanging_recall_is_bounded_and_lands_on_the_same_sentinel"],
    ),
    (
        "M3 review 去掉墙钟上界(裸 await)",
        "app/api/review.py",
        """            try:
                await asyncio.wait_for(
                    vectorize_rows(session, store, embedder, rows),
                    timeout=settings.retrieval_timeout_seconds,
                )
            except TimeoutError as exc:""",
        """            try:
                await vectorize_rows(session, store, embedder, rows)
            except TimeoutError as exc:""",
        ["tests/test_api_review.py::test_a_hanging_vectorize_is_bounded_and_becomes_the_same_502"],
    ),
    (
        "M4 nodes._snapshot 的 score 退回原样(不四舍五入)",
        "app/agent/nodes.py",
        '            "score": (\n                None if c.get("score") is None else round(c["score"], 4)\n            ),',
        '            "score": c.get("score"),',
        ["tests/test_api_feedback.py::test_both_snapshot_writers_round_the_score_to_the_same_four_places"],
    ),
]


def run(test_ids: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        [PY, "-m", "pytest", *test_ids],
        cwd=REPO, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    out = proc.stdout + proc.stderr
    line = next(
        (ln for ln in reversed(out.splitlines()) if re.search(r"\d+ (passed|failed)", ln)),
        None,
    )
    if line is None:
        print("!!! 看不到 `N passed|failed` 行,这次结论作废")
        print(out)
        return -1, out
    print(f"    {line.strip()}")
    return proc.returncode, out


def main() -> int:
    bad = 0
    for name, rel, old, new, tests in MUTATIONS:
        path = REPO / rel
        text = path.read_text(encoding="utf-8")
        hits = text.count(old)
        if hits != 1:
            print(f"!!! {name}:锚点命中 {hits} 次(必须恰好 1),变异没生效")
            bad += 1
            continue
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        try:
            code, _ = run(tests)
            verdict = "RED(RED 是这里要的)" if code != 0 else "GREEN(!!! 无判别力)"
            if code == 0:
                bad += 1
            print(f"{name}\n    -> {verdict}")
        finally:
            path.write_text(text, encoding="utf-8")
        # 还原后确认逐字节回到原样
        assert path.read_text(encoding="utf-8") == text, f"{rel} 还原失败"
    # 还原后基线必须全绿
    print("\n还原后基线:")
    code, _ = run([t for _, _, _, _, ts in MUTATIONS for t in ts])
    if code != 0:
        bad += 1
        print("!!! 还原后基线红了")
    print(f"\n结论:{'全部变异都被抓到' if bad == 0 else f'{bad} 处异常'}")
    return bad


sys.exit(main())
