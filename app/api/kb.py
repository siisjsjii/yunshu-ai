"""知识库管理台(ch04)端点:文档查看/上传、后台任务(vectorize / mine)。

纯函数 helper(文档列表/读取)与薄端点分离 —— helper 可整体单测。
上传只落盘 + 切分入库(pending),**不向量化**;向量化/挖知识是单独的后台
任务(见 `app/kb/orchestrate.py`),前端轮询 `JobStore`。
"""

import re
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select

from app.config import Settings, get_settings
from app.db.models import KnowledgeChunk
from app.db.session import get_session
from app.kb.ingest import parse_corpus_file
from app.kb.jobs import Job, get_job_store
from app.kb.orchestrate import start_job
from app.kb.writer import write_chunks
from app.retrieval.milvus import get_vector_store
from app.retrieval.search import RetrievedChunk
from app.schemas import UploadDocumentRequest
from app.tools.registry import build_retriever

router = APIRouter()

KNOWLEDGE_DIR = Path(__file__).resolve().parents[2] / "knowledge"

_TYPE_RE = re.compile(r"^<!--\s*type:\s*(\S+)\s*-->\s*$")


def _first_h1(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def _doc_type(text: str) -> str:
    m = _TYPE_RE.match(text.partition("\n")[0].strip())
    return m.group(1) if m else "unknown"


def list_documents(dirpath, *, max_chars: int, overlap_chars: int) -> list[dict]:
    """列出目录下全部 .md:文件名、类型、标题(首个 H1)、块数(纯函数 chunker 现算)。

    无类型标记的散文件仍列出,type=unknown、chunk_count=0(不因为一个坏文件
    让整个列表 500)。
    """
    docs = []
    for md in sorted(Path(dirpath).glob("*.md")):
        text = md.read_text(encoding="utf-8")
        try:
            chunks = parse_corpus_file(md, max_chars=max_chars, overlap_chars=overlap_chars)
            count = len(chunks)
        except ValueError:
            count = 0
        docs.append({
            "name": md.name,
            "type": _doc_type(text),
            "title": _first_h1(text) or md.stem,
            "chunk_count": count,
        })
    return docs


def read_document(dirpath, name: str) -> dict | None:
    """读一份文档原文。只允许 `name` 落在 dirpath 内(防路径穿越)。"""
    path = (Path(dirpath) / name).resolve()
    if not str(path).startswith(str(Path(dirpath).resolve())) or not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    return {"name": name, "type": _doc_type(text), "content": text}


def chunk_source_index(dirpath, *, max_chars: int, overlap_chars: int) -> dict:
    """(category, questions, answer) 三元组 → {name, section_path} 的源文档索引。

    `knowledge_chunks` 没有 source 列(ch03 DDL 冻死,不改表),要反查某个
    知识块来自哪个源 `.md`,唯一干净的做法是**重新切分每个文件**、按三元组
    匹配 —— chunker 是纯函数、语料只有几份,每次检索现算毫秒级。
    faq 迁移/挖矿来的块不在此索引里(它们没有源文件)。
    """
    index: dict = {}
    for md in sorted(Path(dirpath).glob("*.md")):
        try:
            chunks = parse_corpus_file(md, max_chars=max_chars, overlap_chars=overlap_chars)
        except ValueError:
            continue  # 无类型标记的散文件跳过
        for c in chunks:
            index[(c.category, c.questions, c.answer)] = {
                "name": md.name, "section_path": c.section_path,
            }
    return index


def annotate_chunk(chunk: RetrievedChunk, source_index: dict) -> dict:
    """检索结果 + 源文档索引 → 带 document/section_path 的返回体。"""
    source = source_index.get((chunk.category, chunk.question, chunk.answer))
    return {
        "question": chunk.question,
        "answer": chunk.answer,
        "category": chunk.category,
        "document": source["name"] if source else None,
        "section_path": source["section_path"] if source else None,
    }


# ---- 端点 ----


@router.get("/api/kb/stats")
async def stats(session=Depends(get_session), settings: Settings = Depends(get_settings)):
    total = (await session.execute(select(func.count(KnowledgeChunk.id)))).scalar_one()
    done = (await session.execute(
        select(func.count(KnowledgeChunk.id)).where(
            KnowledgeChunk.vectorize_status == "done"))).scalar_one()
    try:
        milvus_count = get_vector_store(settings.milvus_uri, settings.milvus_collection).count()
    except Exception:
        milvus_count = None   # Milvus 不在线时降级,不 500 整个管理页
    return {"total": total, "done": done, "pending": total - done, "milvus_count": milvus_count}


@router.get("/api/kb/documents")
async def documents(settings: Settings = Depends(get_settings)):
    return list_documents(
        KNOWLEDGE_DIR,
        max_chars=settings.chunk_max_chars,
        overlap_chars=settings.chunk_overlap_chars,
    )


@router.get("/api/kb/documents/{name}")
async def get_document(name: str):
    doc = read_document(KNOWLEDGE_DIR, name)
    if doc is None:
        raise HTTPException(status_code=404, detail="无此文档")
    return doc


@router.post("/api/kb/documents", status_code=201)
async def upload_document(payload: UploadDocumentRequest,
                          session=Depends(get_session),
                          settings: Settings = Depends(get_settings)):
    path = KNOWLEDGE_DIR / payload.filename
    if path.exists():
        raise HTTPException(status_code=409, detail=f"同名文档已存在:{payload.filename}")
    if not payload.content.strip():
        raise HTTPException(status_code=422, detail="内容不能为空")

    text = f"<!--type: {payload.type}-->\n{payload.content}"
    path.write_text(text, encoding="utf-8")
    try:
        chunks = parse_corpus_file(
            path, max_chars=settings.chunk_max_chars, overlap_chars=settings.chunk_overlap_chars)
    except Exception as exc:
        path.unlink(missing_ok=True)   # 落盘失败回滚
        raise HTTPException(status_code=422, detail=f"文档解析失败:{exc}") from exc

    added = await write_chunks(session, chunks)
    return {
        "name": payload.filename,
        "type": payload.type,
        "title": _first_h1(payload.content) or payload.filename,
        "chunks_added": added,
        "chunks_skipped": len(chunks) - added,
    }


@router.get("/api/kb/search")
async def search_kb(q: str, session=Depends(get_session),
                    settings: Settings = Depends(get_settings)):
    """在线检索:语义检索知识库,返回带源文档链接的知识块列表。"""
    query = q.strip()
    if not query:
        raise HTTPException(status_code=422, detail="查询词不能为空")
    chunks = await build_retriever(session).search(query)
    index = chunk_source_index(
        KNOWLEDGE_DIR, max_chars=settings.chunk_max_chars,
        overlap_chars=settings.chunk_overlap_chars)
    return [annotate_chunk(c, index) for c in chunks]


@router.post("/api/kb/jobs/vectorize", status_code=201)
async def start_vectorize(settings: Settings = Depends(get_settings)):
    job = start_job(get_job_store(), "vectorize", settings)
    if job is None:
        raise HTTPException(status_code=409, detail="已有任务在跑")
    return {"job_id": job.id}


@router.post("/api/kb/jobs/mine", status_code=201)
async def start_mine(settings: Settings = Depends(get_settings)):
    job = start_job(get_job_store(), "mine", settings)
    if job is None:
        raise HTTPException(status_code=409, detail="已有任务在跑")
    return {"job_id": job.id}


def _job_dict(job: Job) -> dict:
    return {
        "id": job.id, "type": job.type, "status": job.status,
        "message": job.message, "result": job.result,
        "created_at": job.created_at, "finished_at": job.finished_at,
    }


@router.get("/api/kb/jobs")
async def list_jobs():
    return [_job_dict(j) for j in get_job_store().list()]


@router.get("/api/kb/jobs/{job_id}")
async def get_job(job_id: str):
    job = get_job_store().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="无此任务")
    return _job_dict(job)
