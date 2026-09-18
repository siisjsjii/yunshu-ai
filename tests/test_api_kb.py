"""api/kb 的纯函数 helper 与端点级校验。

纯函数(文档列表/读取、上传校验)直测;任务端点只测 409 忙拒(安全,不碰
真模型/真 Milvus —— start_job 忙时在 spawn 之前就返回 None)。上传+入库+stats
的真实链路由 db 测试与 scripts/acceptance.sh 验收覆盖。
"""

import pytest
from fastapi.testclient import TestClient

from app.api.kb import annotate_chunk, chunk_source_index, list_documents, read_document
from app.retrieval.search import RetrievedChunk
from app.schemas import UploadDocumentRequest


# ---- 检索:chunk → 源文档映射 ----

def _chunk(q="怎么退货", a="七天无理由。", c="退换货") -> RetrievedChunk:
    return RetrievedChunk(question=q, answer=a, category=c)


def test_chunk_source_index_maps_triple_to_file_and_section(tmp_path):
    (tmp_path / "a.md").write_text(
        "<!--type: policy-->\n\n# 退货政策\n\n## 运费说明\n\n满 99 包邮。\n", encoding="utf-8")
    index = chunk_source_index(tmp_path, max_chars=100, overlap_chars=10)
    assert index
    # 索引键是三元组,值带文件名与章节路径
    key, source = next(iter(index.items()))
    assert source["name"] == "a.md"
    assert "运费说明" in source["section_path"]


def test_chunk_source_index_skips_file_without_marker(tmp_path):
    (tmp_path / "stray.md").write_text("# 无标记\n\n正文。\n", encoding="utf-8")
    assert chunk_source_index(tmp_path, max_chars=100, overlap_chars=10) == {}


def test_annotate_chunk_links_to_source_document(tmp_path):
    (tmp_path / "a.md").write_text(
        "<!--type: policy-->\n\n# 退货政策\n\n满 99 包邮。\n", encoding="utf-8")
    index = chunk_source_index(tmp_path, max_chars=100, overlap_chars=10)
    source = next(iter(index.values()))
    chunk = _chunk(q=next(iter(index))[1], a=next(iter(index))[2], c=next(iter(index))[0])
    result = annotate_chunk(chunk, index)
    assert result["document"] == "a.md"
    assert result["section_path"] == source["section_path"]


def test_annotate_chunk_without_source_has_null_document():
    """faq 迁移/挖矿来的块不在任何文件里 → 不附原文链接。"""
    result = annotate_chunk(_chunk(q="挖矿来的问题"), {})
    assert result["document"] is None
    assert result["section_path"] is None
    assert result["question"] == "挖矿来的问题"


def test_list_documents_parses_type_title_and_chunk_count(tmp_path):
    (tmp_path / "a.md").write_text(
        "<!--type: policy-->\n\n# 退货政策\n\n满 99 包邮。\n", encoding="utf-8")
    docs = list_documents(tmp_path, max_chars=100, overlap_chars=10)
    assert len(docs) == 1
    d = docs[0]
    assert d["name"] == "a.md"
    assert d["type"] == "policy"
    assert d["title"] == "退货政策"
    assert d["chunk_count"] >= 1


def test_list_documents_skips_non_markdown(tmp_path):
    (tmp_path / "x.txt").write_text("hi", encoding="utf-8")
    assert list_documents(tmp_path, max_chars=100, overlap_chars=10) == []


def test_list_documents_tolerates_file_without_type_marker(tmp_path):
    (tmp_path / "stray.md").write_text("# 没有标记\n\n正文。\n", encoding="utf-8")
    docs = list_documents(tmp_path, max_chars=100, overlap_chars=10)
    assert len(docs) == 1
    assert docs[0]["type"] == "unknown"
    assert docs[0]["chunk_count"] == 0


def test_read_document_returns_content_and_type(tmp_path):
    (tmp_path / "a.md").write_text("<!--type: faq-->\n\n# 标题\n\n内容。\n", encoding="utf-8")
    d = read_document(tmp_path, "a.md")
    assert d["name"] == "a.md"
    assert d["type"] == "faq"
    assert "内容。" in d["content"]


def test_read_document_blocks_path_traversal(tmp_path):
    assert read_document(tmp_path, "../etc/passwd") is None


def test_upload_schema_rejects_path_traversal():
    with pytest.raises(Exception):
        UploadDocumentRequest(filename="../../etc/passwd", type="policy", content="x")
    with pytest.raises(Exception):
        UploadDocumentRequest(filename="a.txt", type="policy", content="x")


def test_upload_schema_accepts_valid_and_rejects_bad_type():
    ok = UploadDocumentRequest(filename="退货.md", type="faq", content="正文")
    assert ok.filename == "退货.md"
    with pytest.raises(Exception):
        UploadDocumentRequest(filename="a.md", type="video", content="x")


def test_jobs_endpoint_returns_409_when_busy():
    from app.kb.jobs import get_job_store
    from app.main import app

    get_job_store.cache_clear()            # 清 lru_cache(函数上的方法,不是实例的)
    get_job_store().start("mine")          # 占住运行槽
    try:
        client = TestClient(app)            # 不用 with:不进 lifespan,不触发预热
        assert client.post("/api/kb/jobs/vectorize").status_code == 409
        assert client.post("/api/kb/jobs/mine").status_code == 409
    finally:
        get_job_store.cache_clear()
