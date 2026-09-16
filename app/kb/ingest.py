"""语料 → Chunk 行:Markdown 目录解析 + faq 迁移映射 + 批内三元组去重。

对库级去重(同批次撞上已入库行)负责的是 writer.write_chunks —— 查重要
拿数据库现状当参照,只能发生在有会话的那一层。
"""

import re
from pathlib import Path

from app.kb.chunker import Chunk, chunk_markdown

#: 语料文件首行的类型标记,如 `<!--type: policy-->`。
#: 离线脚本宁可崩也不猜类型(ch03 spec §6.4)。
_TYPE_RE = re.compile(r"^<!--\s*type:\s*(\S+)\s*-->\s*$")


def parse_corpus_file(path, *, max_chars: int, overlap_chars: int) -> list[Chunk]:
    """单个 .md → Chunk 列表(不查重,查重归目录入口与 writer)。"""
    md = Path(path)
    text = md.read_text(encoding="utf-8")
    content_type, body = _split_type_marker(text, md.name)
    return chunk_markdown(
        body,
        content_type=content_type,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )


def parse_corpus_dir(
    dirpath, *, max_chars: int, overlap_chars: int
) -> list[Chunk]:
    """目录下全部 .md → Chunk 列表(批内三元组去重,稳定保序)。"""
    chunks: list[Chunk] = []
    for md in sorted(Path(dirpath).glob("*.md")):
        chunks.extend(
            parse_corpus_file(md, max_chars=max_chars, overlap_chars=overlap_chars)
        )
    return _dedupe_by_triple(chunks)


def _split_type_marker(text: str, name: str) -> tuple[str, str]:
    first, _, rest = text.partition("\n")
    m = _TYPE_RE.match(first.strip())
    if not m:
        raise ValueError(f"{name} 首行缺少 <!--type: ...--> 类型标记")
    return m.group(1), rest


def _dedupe_by_triple(chunks: list[Chunk]) -> list[Chunk]:
    seen: set[tuple[str, str, str]] = set()
    out: list[Chunk] = []
    for c in chunks:
        key = (c.category, c.questions, c.answer)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def faq_migration(faq_rows) -> list[Chunk]:
    """既有 Faq 行 → Chunk:questions=真实问法,category 沿用,无章节路径。

    faq_rows 是任何具有 question / answer / category 属性的行(ORM 对象或
    测试替身皆可)。
    """
    return _dedupe_by_triple(
        [
            Chunk(
                category=r.category,
                questions=r.question,
                answer=r.answer,
                section_path=None,
                content_type="faq",
                is_key_clause=False,
            )
            for r in faq_rows
        ]
    )
