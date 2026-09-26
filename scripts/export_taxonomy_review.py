"""把 17 类表导成 CSV,给人过目(CP-1)。

**为什么要有这一步**:后面所有预标、合成、裁决都照这张表走 ——
它歪了整章数据都歪。而 CSV 能在 Excel 里排着看,比读 Python 源码快得多。

**行内容只由 `render_rows()` 产出**,写盘与「自检」都走它 ——
这份 CSV 是**与源码同处一地入库的生成物**,所以必须有一条断言钉住
「盘上那份 == 常量算出来的那份」,否则改了 `BOUNDARY` 而忘了重导,
**用户签过字的那份**就会静默地与代码不一致(`tests/test_topic_taxonomy.py`
的 `test_committed_csv_is_in_sync_with_the_constants` 就是那道闸)。

用法:
    .venv/Scripts/python.exe scripts/export_taxonomy_review.py
"""
import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.topic.taxonomy import BOUNDARY, COUNTER, HEAD_LABELS, LABELS, POSITIVE

OUT = Path(__file__).resolve().parents[1] / "evals" / "topic" / "taxonomy_review.csv"

HEADER = ["#", "类目", "头四类", "边界说明", "正例", "反例(应归哪类)"]

#: 尾注 —— 回答 CP-1 的第 ③ 问。spec §4.2 说这件事「必须写进表里」,
#: 而「表」是**用户真正会打开的那份文件**,不是模块 docstring。
FOOTNOTE = (
    "注:「投诉 / 闲聊 / 转人工」是**意图**(用户想干什么),不是**主题**(问题关于什么)"
    "—— 因此上面 17 类里没有它们,它们的主题一律是「其他」。这是刻意的,不是漏写(spec §4.2)。"
)


def render_rows() -> list[list[str]]:
    """表头 + 17 类 + 尾注。**CSV 内容的唯一产出点**(写盘与测试走同一条路)。"""
    rows = [HEADER]
    for i, label in enumerate(LABELS, 1):
        pos = " / ".join(POSITIVE.get(label, ()))
        neg = " / ".join(f"{t}→{tgt}" for t, tgt in COUNTER.get(label, ()))
        rows.append([i, label, "★" if label in HEAD_LABELS else "", BOUNDARY[label], pos, neg])
    rows.append([])  # 空行:把尾注与那 17 行数据隔开,免得被读成第 18 类
    rows.append([FOOTNOTE])
    return rows


def render_csv_text() -> str:
    """与写盘**逐字节相同**的文本(BOM 除外,那一步在 `encode("utf-8-sig")`)。"""
    buf = io.StringIO()
    # csv.writer 默认 lineterminator="\\r\\n";写盘时 newline="" 不做翻译 ⇒ 两边一致。
    csv.writer(buf).writerows(render_rows())
    return buf.getvalue()


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig:Excel 在中文 Windows 上按本地编码打开无 BOM 的 CSV 会乱码。
    OUT.write_bytes(render_csv_text().encode("utf-8-sig"))
    # 钉住输出编码:本机 locale 是 cp936,直接 print 中文时控制台编解码
    # 随调用方而变(与 `build_kb.py` / `mine_qa.py` 同款做法)。
    sys.stdout.buffer.write(f"已写出 {OUT}({len(LABELS)} 行)\n".encode("utf-8"))


if __name__ == "__main__":
    main()
