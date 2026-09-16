"""Markdown 结构感知切分(ch03 spec §6.3)。

纯函数:不碰 IO / DB / LangChain,行为完全由入参决定,可整体单测。
规则要点:
- 按标题层级切,section_path 是根到当前节的标题路径;
- 政策/手册类 questions=所在章节标题,category=上级标题路径(根节回退文档标题);
- 块正文超上限时按「段落 → 句子 → 硬切」递归,块间重叠**只允许从句首开始**
  (起点必须紧跟在句末标点之后),裁不出句界就不重叠 —— 不留半截话;
- 表格按数据行切,每块复制表头行与分隔行,不参与句号重叠;
- `<!--key-->` 标记行本身不进正文,其后的段落或整节置 is_key_clause。
"""

import re
from dataclasses import dataclass, field

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_TABLE_ROW_RE = re.compile(r"^\s*\|.+\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")
_KEY_MARKER = "<!--key-->"

#: 句末标点集:重叠起点只允许出现在这些标点之后,保证块首即句首。
_ENDERS = "。!?;…"


@dataclass
class Chunk:
    """一个待入库的知识块。

    category / questions / answer 三字段按固定顺序拼接后进向量;
    section_path / content_type / is_key_clause 是只存不进向量的元数据。
    """

    category: str
    questions: str
    answer: str
    section_path: str
    content_type: str
    is_key_clause: bool = False


@dataclass
class _Section:
    stack: list[str]
    key: bool
    units: list = field(default_factory=list)  # (kind, payload, unit_key)


def _parse_sections(text: str) -> tuple[list[_Section], str]:
    """行扫描:维护标题栈,把正文攒成段落/表格单元。"""
    sections: list[_Section] = []
    stack: list[str] = []
    doc_title = ""
    pending_key = False
    para: list[str] = []
    table: list[str] = []
    cur: _Section | None = None

    def _flush() -> None:
        nonlocal para, table, pending_key
        if cur is not None:
            if para:
                cur.units.append(("text", "\n".join(para).strip(), pending_key))
                pending_key = False
            if table:
                cur.units.append(("table", list(table), pending_key))
                pending_key = False
        para, table = [], []

    for raw in text.splitlines():
        line = raw.rstrip()
        if line.strip() == _KEY_MARKER:
            _flush()
            pending_key = True
            continue
        m = _HEADING_RE.match(line)
        if m:
            _flush()
            level, title = len(m.group(1)), m.group(2).strip()
            while len(stack) >= level:
                stack.pop()
            stack.append(title)
            if not doc_title:
                doc_title = title
            # 标记直接贴在节标题前 → 整节关键;贴在段前 → 只标记该段(见 _flush)
            cur = _Section(stack=list(stack), key=pending_key)
            sections.append(cur)
            pending_key = False
            continue
        if not line.strip():
            _flush()
            continue
        if _TABLE_ROW_RE.match(line):
            if para:
                _flush()
            table.append(line.strip())
            continue
        if table:
            _flush()
        para.append(line.strip())
    _flush()
    return sections, doc_title


def _split_sentences(text: str) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    for ch in text:
        buf.append(ch)
        if ch in _ENDERS:
            out.append("".join(buf))
            buf = []
    tail = "".join(buf)
    if tail.strip():
        out.append(tail)
    return out


def _split_text(text: str, max_chars: int) -> list[str]:
    """单段超限 → 句子;单句仍超限 → 硬切;再贪心打包成块(不含重叠)。"""
    if len(text) <= max_chars:
        return [text]
    units: list[str] = []
    for s in _split_sentences(text):
        if len(s) <= max_chars:
            units.append(s)
        else:
            for i in range(0, len(s), max_chars):
                units.append(s[i : i + max_chars])
    blocks: list[str] = []
    cur = ""
    for u in units:
        if not cur:
            cur = u
        elif len(cur) + len(u) <= max_chars:
            cur += u
        else:
            blocks.append(cur)
            cur = u
    if cur:
        blocks.append(cur)
    return blocks


def _sentence_overlap(prev: str, budget: int) -> str:
    """prev 尾部预算窗口内找句首作重叠起点;找不到就返回空串(宁可不重叠)。

    j 的下界是 start+1:保证重叠长度 < budget;上界不含 len(prev):prev 以
    句号收尾、且窗口内没有更早句界时(整窗是同一句的后半段)自然落到空串。
    """
    start = max(0, len(prev) - budget)
    for j in range(start + 1, len(prev)):
        if prev[j - 1] in _ENDERS:
            return prev[j:]
    return ""


def _split_table(lines: list[str], max_chars: int) -> list[str]:
    """大表按数据行切,每块复制表头行 + 分隔行。"""
    header = lines[0]
    sep = lines[1] if len(lines) > 1 and _TABLE_SEP_RE.match(lines[1]) else None
    data = lines[2:] if sep else lines[1:]
    prefix = "\n".join([header] + ([sep] if sep else [])) + "\n"
    blocks: list[str] = []
    rows: list[str] = []
    cur_len = len(prefix)
    for row in data:
        need = len(row) + 1
        if rows and cur_len + need > max_chars:
            blocks.append(prefix + "\n".join(rows))
            rows, cur_len = [], len(prefix)
        rows.append(row)
        cur_len += need
    if rows:
        blocks.append(prefix + "\n".join(rows))
    return blocks or [prefix.rstrip("\n")]


def chunk_markdown(
    text: str, *, content_type: str, max_chars: int, overlap_chars: int
) -> list[Chunk]:
    """Markdown 全文 → Chunk 列表。"""
    sections, doc_title = _parse_sections(text)
    chunks: list[Chunk] = []
    for sec in sections:
        units = [(k, p, uk) for k, p, uk in sec.units if p]
        if not units:
            continue
        questions = sec.stack[-1]
        category = (
            " > ".join(sec.stack[:-1]) if len(sec.stack) > 1 else (doc_title or questions)
        )
        section_path = " > ".join(sec.stack)

        rendered = [
            p if k == "text" else "\n".join(p) for k, p, _ in units
        ]
        total = sum(len(r) for r in rendered) + 2 * (len(rendered) - 1)
        if total <= max_chars:
            chunks.append(
                Chunk(
                    category,
                    questions,
                    "\n\n".join(rendered),
                    section_path,
                    content_type,
                    sec.key,
                )
            )
            continue

        prev_text: str | None = None
        for kind, payload, unit_key in units:
            if kind == "text":
                for b in _split_text(payload, max_chars):
                    if prev_text is not None:
                        ov = _sentence_overlap(prev_text, overlap_chars)
                        if ov:
                            b = ov + b
                    prev_text = b
                    chunks.append(
                        Chunk(
                            category,
                            questions,
                            b,
                            section_path,
                            content_type,
                            sec.key or unit_key,
                        )
                    )
            else:
                for b in _split_table(payload, max_chars):
                    chunks.append(
                        Chunk(
                            category,
                            questions,
                            b,
                            section_path,
                            content_type,
                            sec.key or unit_key,
                        )
                    )
                # 表格块不参与句号重叠链:表后的文本不与表前的文本重叠
                prev_text = None
    return chunks
