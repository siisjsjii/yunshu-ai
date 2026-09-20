"""ingest 单元测试:语料目录解析、类型标记剥离、批内三元组去重。

(原 `faq_migration` 的两条用例随该函数一并删除 —— `faq` 表已废弃。)
"""

import pytest

from app.kb.ingest import parse_corpus_dir, parse_corpus_file

MAX, OVERLAP = 100, 40


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_corpus_file_with_type_marker(tmp_path):
    _write(tmp_path, "doc.md", "<!--type: policy-->\n\n# 退货政策\n\n满 99 包邮。\n")
    chunks = parse_corpus_dir(tmp_path, max_chars=MAX, overlap_chars=OVERLAP)
    assert len(chunks) == 1
    assert chunks[0].content_type == "policy"
    assert "type:" not in chunks[0].answer          # 标记行不进正文
    assert chunks[0].category == "退货政策"
    assert chunks[0].answer == "满 99 包邮。"


def test_corpus_file_missing_marker_fails_loud(tmp_path):
    """离线脚本宁可崩也不猜类型:无标记直接报文件名。"""
    _write(tmp_path, "doc.md", "# 没有标记\n\n正文。\n")
    with pytest.raises(ValueError) as exc:
        parse_corpus_dir(tmp_path, max_chars=MAX, overlap_chars=OVERLAP)
    assert "doc.md" in str(exc.value)


def test_corpus_dir_skips_non_markdown(tmp_path):
    _write(tmp_path, "notes.txt", "不是语料")
    _write(tmp_path, "doc.md", "<!--type: manual-->\n\n# 手册\n\n内容。\n")
    chunks = parse_corpus_dir(tmp_path, max_chars=MAX, overlap_chars=OVERLAP)
    assert len(chunks) == 1


def test_in_batch_triple_dedup_keeps_first(tmp_path):
    body = "<!--type: policy-->\n\n# 政策\n\n同一内容。\n"
    _write(tmp_path, "a.md", body)
    _write(tmp_path, "b.md", body)
    chunks = parse_corpus_dir(tmp_path, max_chars=MAX, overlap_chars=OVERLAP)
    assert len(chunks) == 1


def test_dedup_only_on_full_triple(tmp_path):
    """三元组任一字段不同就不是重复 —— 只比 category 会误杀同名章节。"""
    _write(tmp_path, "a.md", "<!--type: policy-->\n\n# 政策\n\n内容一。\n")
    _write(tmp_path, "b.md", "<!--type: policy-->\n\n# 政策\n\n内容二。\n")
    chunks = parse_corpus_dir(tmp_path, max_chars=MAX, overlap_chars=OVERLAP)
    assert len(chunks) == 2


def test_parse_corpus_file_reads_exactly_one_file(tmp_path):
    """`build_kb --source` 用的入口:只读指定文件,不扫同目录的其它语料。"""
    _write(tmp_path, "a.md", "<!--type: policy-->\n\n# 政策\n\n内容一。\n")
    _write(tmp_path, "b.md", "<!--type: manual-->\n\n# 手册\n\n内容二。\n")
    chunks = parse_corpus_file(tmp_path / "b.md", max_chars=MAX, overlap_chars=OVERLAP)
    assert [c.content_type for c in chunks] == ["manual"]
    assert [c.answer for c in chunks] == ["内容二。"]


