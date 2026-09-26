"""把 17 类表导成 CSV,给人过目(CP-1)。

**为什么要有这一步**:后面所有预标、合成、裁决都照这张表走 ——
它歪了整章数据都歪。而 CSV 能在 Excel 里排着看,比读 Python 源码快得多。

用法:
    .venv/Scripts/python.exe scripts/export_taxonomy_review.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.topic.taxonomy import BOUNDARY, COUNTER, HEAD_LABELS, LABELS, POSITIVE

OUT = Path(__file__).resolve().parents[1] / "evals" / "topic" / "taxonomy_review.csv"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        # utf-8-sig:Excel 在中文 Windows 上按本地编码打开无 BOM 的 CSV 会乱码。
        w = csv.writer(f)
        w.writerow(["#", "类目", "头四类", "边界说明", "正例", "反例(应归哪类)"])
        for i, label in enumerate(LABELS, 1):
            pos = " / ".join(POSITIVE.get(label, ()))
            neg = " / ".join(f"{t}→{tgt}" for t, tgt in COUNTER.get(label, ()))
            w.writerow([i, label, "★" if label in HEAD_LABELS else "", BOUNDARY[label], pos, neg])
    # 钉住输出编码:本机 locale 是 cp936,直接 print 中文时控制台编解码
    # 随调用方而变(与 `build_kb.py` / `mine_qa.py` 同款做法)。
    sys.stdout.buffer.write(f"已写出 {OUT}({len(LABELS)} 行)\n".encode("utf-8"))


if __name__ == "__main__":
    main()
