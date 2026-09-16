"""chunker 单元测试:结构感知切分的每条规则都要「实现改错就红」。

规约:docs/superpowers/specs/2026-09-16-ecommerce-cs-ch03-kb-design.md §6.3
"""

from app.kb.chunker import Chunk, chunk_markdown

MAX = 100
OVERLAP = 40


def _by_answers(chunks):
    return [c.answer for c in chunks]


# ---- 标题层级与路径 ----


def test_nested_headings_build_path_and_fields():
    doc = """# 售后手册

总则一句话。

## 退换货流程

流程说明。

### 寄回地址

地址是广州市天河区。
"""
    chunks = chunk_markdown(doc, content_type="manual", max_chars=MAX, overlap_chars=OVERLAP)
    paths = {c.section_path: c for c in chunks}
    assert "售后手册 > 退换货流程 > 寄回地址" in paths
    inner = paths["售后手册 > 退换货流程 > 寄回地址"]
    # questions = 最内层标题,category = 上级标题路径(需求 3 的政策类规则)
    assert inner.questions == "寄回地址"
    assert inner.category == "售后手册 > 退换货流程"
    assert inner.answer == "地址是广州市天河区。"
    assert inner.content_type == "manual"
    assert inner.is_key_clause is False


def test_directly_under_h1_category_falls_back_to_doc_title():
    doc = "# 退货政策\n\n七天无理由退货。\n"
    chunks = chunk_markdown(doc, content_type="policy", max_chars=MAX, overlap_chars=OVERLAP)
    assert len(chunks) == 1
    assert chunks[0].section_path == "退货政策"
    assert chunks[0].questions == "退货政策"
    assert chunks[0].category == "退货政策"  # 没有上级了 → 文档标题


def test_heading_lines_do_not_leak_into_answer():
    doc = "## 章节\n\n正文内容。\n"
    chunks = chunk_markdown(doc, content_type="policy", max_chars=MAX, overlap_chars=OVERLAP)
    assert chunks[0].answer == "正文内容。"
    assert "章节" not in chunks[0].answer


def test_empty_section_produces_no_chunk():
    doc = "## 有正文\n\n内容。\n\n## 空章节\n\n### 空的子节\n"
    chunks = chunk_markdown(doc, content_type="policy", max_chars=MAX, overlap_chars=OVERLAP)
    assert {c.section_path for c in chunks} == {"有正文"}


# ---- 超长递归切分 ----


def test_long_section_splits_into_blocks_within_limit():
    para = "甲" * 30 + "。" + "乙" * 30 + "。" + "丙" * 30 + "。"
    doc = "## 长节\n\n" + (para + "\n\n") * 3
    chunks = chunk_markdown(doc, content_type="policy", max_chars=MAX, overlap_chars=OVERLAP)
    assert len(chunks) > 1
    # 上限约束:正文(不含重叠)不超 max_chars,总长不超 max+overlap
    for c in chunks:
        assert len(c.answer) <= MAX + OVERLAP
    # 全部块保持同一路径
    assert {c.section_path for c in chunks} == {"长节"}


def test_run_on_sentence_is_split_by_enders_then_hard_cut():
    # 一整段 250 字、中间只有少量句号:先按句号切,再打包;仍超的单句允许硬切
    para = ("A" * 60 + "。") * 3 + "B" * 250
    doc = "## 节\n\n" + para + "\n"
    chunks = chunk_markdown(doc, content_type="policy", max_chars=MAX, overlap_chars=OVERLAP)
    assert len(chunks) >= 3
    for c in chunks:
        assert len(c.answer) <= MAX + OVERLAP


# ---- 重叠裁到句号 ----


def _last_sentence(text: str) -> str:
    """text 结尾处的最后一句(含收尾句号)。重叠若从整句开始,cur 的开头
    必须逐字等于它;半截句会让 startswith 失败 —— 这是可证伪的方向。"""
    for i in range(len(text) - 2, -1, -1):
        if text[i] in "。!?;…":
            return text[i + 1:]
    return text


def test_overlap_starts_right_after_a_sentence_end():
    # 尾部 40 字落在「乙」句中间:错误实现会把半截「乙」放进下一块开头
    para = "甲" * 30 + "。" + "乙" * 30 + "。" + "丙" * 30 + "。" + "丁" * 30 + "。"
    doc = "## 节\n\n" + para + para + "\n"
    chunks = chunk_markdown(doc, content_type="policy", max_chars=MAX, overlap_chars=OVERLAP)
    assert len(chunks) >= 2
    for prev_c, cur in zip(chunks, chunks[1:]):
        head = cur.answer[:10]
        assert ("。" + head[0]) in prev_c.answer
        # 重叠必须从整句开始:cur 以 prev 的最后一句开头(逐字)。
        # 完全不重叠的实现同样过不了这条(会拿到别的句子开头)。
        assert cur.answer.startswith(_last_sentence(prev_c.answer))


def test_unpunctuated_text_gets_no_overlap():
    # 表格/无标点文本裁不出句号 → 不重叠,绝不留半截话。
    # 刻意用**非周期**文本:「无标点内容」×N 的周期是 5、恰好整除 40,
    # prev[-40:] 会与下一块开头碰巧相同,把正确实现误判成有重叠(假红)。
    body = "".join(f"片段{i:03d}" for i in range(50))
    doc = "## 节\n\n" + body + "\n"
    chunks = chunk_markdown(doc, content_type="policy", max_chars=MAX, overlap_chars=OVERLAP)
    assert len(chunks) >= 2
    for prev_c, cur in zip(chunks, chunks[1:]):
        assert not cur.answer.startswith(prev_c.answer[-OVERLAP:])


# ---- 表格 ----


def _table_doc(rows: int) -> str:
    lines = ["| 商品 | 运费 | 说明 |", "| --- | --- | --- |"]
    lines += [f"| 商品{i} | {i} 元 | 备注备注备注备注{i} |" for i in range(rows)]
    return "## 运费表\n\n" + "\n".join(lines) + "\n"


def test_table_split_copies_header_into_every_block():
    chunks = chunk_markdown(_table_doc(30), content_type="policy", max_chars=MAX, overlap_chars=OVERLAP)
    assert len(chunks) > 1
    for c in chunks:
        assert c.answer.splitlines()[0] == "| 商品 | 运费 | 说明 |"  # 表头每块复制
        assert c.answer.splitlines()[1] == "| --- | --- | --- |"  # 分隔行也复制
        assert len(c.answer) <= MAX + OVERLAP
    # 行数守恒:所有数据行都还在(每行一条商品备注)
    joined = "\n".join(_by_answers(chunks))
    for i in range(30):
        assert f"| 商品{i} " in joined


def test_small_table_stays_one_block():
    chunks = chunk_markdown(_table_doc(2), content_type="policy", max_chars=MAX, overlap_chars=OVERLAP)
    assert len(chunks) == 1
    assert chunks[0].section_path == "运费表"


# ---- 关键条款标记 ----


def test_key_marker_sets_flag_and_is_stripped():
    doc = """# 退货政策

## 运费说明

满 99 包邮,否则收 8 元。

<!--key-->

## 不支持七天无理由的例外

鲜活易腐商品不支持。

## 普通条款

普通内容。
"""
    chunks = chunk_markdown(doc, content_type="policy", max_chars=MAX, overlap_chars=OVERLAP)
    by_path = {c.section_path: c for c in chunks}
    # 标记所在**节**的块全部为关键条款,兄弟节不受影响
    assert by_path["退货政策 > 不支持七天无理由的例外"].is_key_clause is True
    assert by_path["退货政策 > 普通条款"].is_key_clause is False
    for c in chunks:
        assert "<!--key-->" not in c.answer


# ---- Chunk 形状 ----


def test_chunk_has_exactly_the_spec_fields():
    doc = "# 退货政策\n\n七天无理由退货。\n"
    c = chunk_markdown(doc, content_type="policy", max_chars=MAX, overlap_chars=OVERLAP)[0]
    assert isinstance(c, Chunk)
    assert set(Chunk.__dataclass_fields__) == {
        "category",
        "questions",
        "answer",
        "section_path",
        "content_type",
        "is_key_clause",
    }
