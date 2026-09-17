"""api/kb 的纯函数 helper 与端点级校验。

纯函数(文档列表/读取、上传校验)直测;任务端点只测 409 忙拒(安全,不碰
真模型/真 Milvus —— start_job 忙时在 spawn 之前就返回 None)。上传+入库+stats
的真实链路由 db 测试与 scripts/acceptance.sh 验收覆盖。
"""

import pytest
from fastapi.testclient import TestClient

from app.api.kb import list_documents, read_document
from app.schemas import UploadDocumentRequest


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
