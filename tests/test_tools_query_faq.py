"""query_faq 的**契约红线**测试:换内部实现,对模型的入参出参一字不变。

检索器是替身,全程不碰 MySQL / Milvus / 模型 —— 契约测试必须能在
`-m "not db"` 的快路径里跑,否则最容易破的接口反而最少被验。

原来钉在 `test_tools_db.py` 里的三条 LIKE 通配符用例(`%` / `_` 按字面匹配)
随关键词查表一起删掉了:`make_query_faq` 内部已无 SQL LIKE,那些断言守的
代码路径不存在了。它们原本防的是「工具返回一堆不相干的答案却报成功」,
这个防线**换成了阈值过滤**(spec §6.6),由 test_retrieval_search 与
检索评估集继续守(见 dev-notes 阶段 7)。
"""

import asyncio
import json

import pytest

from app.retrieval.search import RetrievedChunk
from app.tools.builtin.knowledge import make_query_faq
from app.tools.errors import ToolInfrastructureError, ToolNotFound


class _FakeRetriever:
    def __init__(self, chunks=(), error: Exception | None = None):
        self.calls: list = []
        self.chunks = list(chunks)
        self.error = error

    async def search(self, query: str):
        self.calls.append(query)
        if self.error is not None:
            raise self.error
        return list(self.chunks)


def _call(args: dict, retriever=None):
    tool = make_query_faq(session=None, retriever=retriever or _FakeRetriever())
    return asyncio.run(
        tool.ainvoke({"name": "query_faq", "args": args, "id": "c", "type": "tool_call"})
    )


def _payload(args: dict, retriever=None) -> dict:
    return json.loads(_call(args, retriever).content)


def test_model_facing_schema_is_unchanged():
    """入参 schema 只有 keyword 一个字段 —— 模型看到的东西不许变。"""
    tool = make_query_faq(session=None, retriever=_FakeRetriever())
    schema = tool.args_schema.model_json_schema()
    assert set(schema["properties"]) == {"keyword"}
    assert set(schema.get("required", [])) == {"keyword"}


def test_output_keys_and_types_are_unchanged():
    """出参结构逐 key 断言:键名、类型、count 与 items 的一致性。"""
    retriever = _FakeRetriever(
        [RetrievedChunk("怎么退货\n退货流程", "七天无理由。", "退换货")]
    )
    payload = _payload({"keyword": "退货"}, retriever)
    assert set(payload) == {"keyword", "count", "items"}
    assert payload["keyword"] == "退货"
    assert isinstance(payload["count"], int)
    assert payload["count"] == len(payload["items"]) == 1
    item = payload["items"][0]
    # 旧三字段 + ch04 增补的引用元数据(chunk_id 是 int,section_path 是 str/None)
    assert {"question", "answer", "category"} <= set(item)
    assert isinstance(item["question"], str)
    assert isinstance(item["answer"], str)
    assert isinstance(item["category"], str)
    assert isinstance(item["chunk_id"], int)
    assert item["section_path"] is None or isinstance(item["section_path"], str)


def test_question_keeps_full_text_with_newlines():
    """`question` = 块的 questions 全文(多个问法换行分隔),不截断不合并。"""
    retriever = _FakeRetriever(
        [RetrievedChunk("问法一\n问法二\n问法三", "答案", "分类")]
    )
    assert _payload({"keyword": "q"}, retriever)["items"][0]["question"] == (
        "问法一\n问法二\n问法三"
    )


def test_keyword_is_stripped_before_retrieval_and_echoed_trimmed():
    retriever = _FakeRetriever([RetrievedChunk("q", "a", "c")])
    payload = _payload({"keyword": "  邮费  "}, retriever)
    assert retriever.calls == ["邮费"]
    assert payload["keyword"] == "邮费"


def test_blank_keyword_short_circuits_without_retrieval():
    """空关键词直接落空,**不**发起检索(不该为一次空转去加载 2.2GB 权重)。"""
    retriever = _FakeRetriever([RetrievedChunk("q", "a", "c")])
    with pytest.raises(ToolNotFound):
        _call({"keyword": "   "}, retriever)
    assert retriever.calls == []


def test_no_hit_raises_not_found_with_the_same_wording():
    """全被阈值滤掉 = 检索器返回空 → 仍走 ToolNotFound,防线文案不变。

    「如实告知暂未收录,不要自行编造答案」是 ch02 就立下的防线,换实现
    不该把它丢掉 —— dense 单路没有重排兜底,这条防线比以前更要紧。
    """
    with pytest.raises(ToolNotFound) as exc:
        _call({"keyword": "邮费"}, _FakeRetriever([]))
    message = str(exc.value)
    assert "邮费" in message
    assert "暂未收录" in message
    assert "不要自行编造答案" in message


def test_not_found_message_is_bounded():
    """漏召回的错误文本会回灌进模型上下文,必须截断(关键词长度由模型决定)。"""
    huge = "查无此项" * 1000
    with pytest.raises(ToolNotFound) as exc:
        _call({"keyword": huge}, _FakeRetriever([]))
    assert huge not in str(exc.value)
    assert len(str(exc.value)) < 200


def test_infrastructure_error_is_not_disguised_as_not_found():
    """Milvus/嵌入故障必须原样向上抛(502),绝不伪装成「未收录」。"""
    with pytest.raises(ToolInfrastructureError):
        _call({"keyword": "邮费"}, _FakeRetriever(error=ToolInfrastructureError("检索不可用")))
