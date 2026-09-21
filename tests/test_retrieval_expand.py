"""Query 扩写 + 多路检索合并:**纯函数单测,全程不联网**。

本文件守的是两件容易「看起来对」的事:

1. **合并的并列规则**。同一个块被两路召回、两次分数不同时留哪一份,是一个
   **有实质后果**的选择(检索分数在本项目里当置信度用,见 `expand.py` 的注释),
   而 brief 给的那条用例**测不出它** —— 那里重复块的两次分数都排在最后,
   两种取法给出的顺序完全相同。这里补一条**能区分**的。
2. **基础设施故障不可被吞**。`multi_search` 要容忍单路查询失败,但
   `ToolInfrastructureError`(Milvus/嵌入挂了)必须照抛。一条「宽 except 吞一切」
   的实现返回的是**一次看起来正常的检索**,用户会以为「知识库里没有」 ——
   这正是本仓最怕的静默故障,所以专门有一条用例把 q1 成功、q2 基础设施故障
   的组合钉死。

`expand_queries` 一侧同理:「永远返回 [原问题]」的实现能让**所有失败用例**通过,
所以必须有一条正路用例,期望值与退路值**不同**。
"""

import asyncio
import logging

import pytest
from langchain_core.exceptions import OutputParserException

from app.prompts import EXPAND_SYSTEM_PROMPT, build_expand_messages
from app.retrieval.expand import ExpandQueries, expand_queries, multi_search
from app.retrieval.search import RetrievedChunk
from app.tools.errors import ToolInfrastructureError


def _chunk(chunk_id: int, score: float, question: str = "怎么退货") -> RetrievedChunk:
    """真实的 `RetrievedChunk`(冻结 dataclass,构造不碰任何外部系统)。

    用真类型而不是自造替身:`multi_search` 只读 `.chunk_id` / `.score`,而
    真类型能保证「字段名与检索器实际返回的一致」这件事**不会两边各写一份**。
    """
    return RetrievedChunk(
        question=question, answer="七天无理由", category="退换货",
        chunk_id=chunk_id, section_path="退货政策", score=score,
    )


class _FakeRetriever:
    def __init__(self, by_query: dict[str, list]):
        self.by_query = by_query
        self.calls: list[str] = []

    async def search(self, query: str) -> list:
        self.calls.append(query)
        return self.by_query.get(query, [])


class _FlakyRetriever:
    """`fail_on` 那一路抛错,其余路返回 [chunk 1]。

    非失败路**有返回值**是关键:否则「故障被吞掉」与「故障照抛」两种实现
    都只会看到空结果,用例反而失去判别力。
    """

    def __init__(self, fail_on: str, exc: Exception | None = None):
        self.fail_on = fail_on
        self.exc = exc or RuntimeError("这一路查询挂了")
        self.calls: list[str] = []

    async def search(self, query: str) -> list:
        self.calls.append(query)
        if query == self.fail_on:
            raise self.exc
        return [_chunk(1, 0.9)]


# ---- multi_search:合并与去重 ----


@pytest.mark.anyio
async def test_multi_search_merges_and_dedupes_by_chunk_id():
    """同一个块被两条查询召回 → 只留一份。"""
    r = _FakeRetriever({
        "q1": [_chunk(1, 0.9), _chunk(2, 0.5)],
        "q2": [_chunk(2, 0.7), _chunk(3, 0.8)],
    })
    got = await multi_search(r, ["q1", "q2"])
    assert [c.chunk_id for c in got] == [1, 3, 2]      # 按 score 降序
    assert len([c for c in got if c.chunk_id == 2]) == 1


@pytest.mark.anyio
async def test_same_chunk_keeps_the_higher_score_not_the_first_seen():
    """同一块两路召回、分数不同 → **留分数更高的那一份**。

    ⚠️ 这是本文件对「留哪一份」的**唯一判别点**。上面那条(brief 给的)用例里,
    重复块(2)的两次分数 0.5 / 0.7 **都排在最后**,所以「留首次」与「留高分」
    给出的顺序都是 `[1, 3, 2]` —— 它测不出这个选择。

    这里把重复块的分数抬到**能改变名次**(0.85 > 0.6),于是:
    「留高分」→ `[1, 2, 3]` / 分数 `[0.9, 0.85, 0.6]`;
    「留首次」→ `[1, 3, 2]` / 分数 `[0.9, 0.6, 0.4]`。
    **顺序与分数两处都不同**,换成另一种取法必红。
    """
    r = _FakeRetriever({
        "q1": [_chunk(1, 0.9), _chunk(2, 0.4)],
        "q2": [_chunk(3, 0.6), _chunk(2, 0.85)],
    })
    got = await multi_search(r, ["q1", "q2"])
    assert [c.chunk_id for c in got] == [1, 2, 3]
    assert [c.score for c in got] == [0.9, 0.85, 0.6]


@pytest.mark.anyio
async def test_tied_scores_keep_first_seen_order():
    """分数**相同**时名次按首次出现排(`>` 是严格比较 + Python 稳定排序)。

    只钉顺序,不钉「留的是哪个对象」:同一个 `chunk_id` 必然来自同一行,
    两份内容的 `answer` 逐字相同,选哪一份在真机上**不可观测** —— 为它写
    断言就是为一条不可证伪的差异写测试。

    期望值刻意与「按 chunk_id 升序」相反(`[2, 1]` 不是 `[1, 2]`),
    免得排序键被写成 id 时照样通过。
    """
    r = _FakeRetriever({"q1": [_chunk(2, 0.5)], "q2": [_chunk(1, 0.5)]})
    got = await multi_search(r, ["q1", "q2"])
    assert [c.chunk_id for c in got] == [2, 1]


@pytest.mark.anyio
async def test_multi_search_tolerates_one_query_failing():
    """一条查询挂掉不该让整次检索失败 —— 用剩下的。"""
    r = _FlakyRetriever(fail_on="q2")
    got = await multi_search(r, ["q1", "q2"])
    assert [c.chunk_id for c in got] == [1]


@pytest.mark.anyio
async def test_skipped_query_is_logged_so_the_degradation_is_not_silent(caplog):
    """被跳过的查询必须**留下痕迹**。

    「跳过」本身是正确的降级,但它让这一路的召回悄悄消失 —— 不留日志的话,
    用户少了一半召回、而日志里什么都没有(本仓的静默故障形态)。断言用的是
    查询原文:换一条查询执行失败时,这条日志的 `q2` 不会出现。
    """
    r = _FlakyRetriever(fail_on="q2")
    with caplog.at_level(logging.WARNING):
        got = await multi_search(r, ["q1", "q2"])
    assert [c.chunk_id for c in got] == [1]
    assert "q2" in caplog.text


# ---- multi_search:错误语义(不可吞基础设施故障) ----


@pytest.mark.anyio
async def test_infrastructure_fault_propagates_even_though_other_queries_succeeded():
    """**本文件最重要的一条**:基础设施故障绝不能被降级成「没搜到」。

    q1 成功、q2 抛 `ToolInfrastructureError`。一个「宽 except 吞掉一切」的实现
    会返回 `[chunk 1]` —— 一次**看起来完全正常**的检索(非空、有分数、有 chunk_id),
    用户据此得到「知识库里没这条」;正确行为是**整次抛出**,一路到端点变 502。

    两种实现的输出不同(`[chunk1]` vs 异常),故可判别;它同时钉住 except 子句的
    **顺序** —— 把 `except Exception` 写在前面,这条必红。
    """
    r = _FlakyRetriever(
        fail_on="q2", exc=ToolInfrastructureError("知识检索服务暂时不可用")
    )
    with pytest.raises(ToolInfrastructureError):
        await multi_search(r, ["q1", "q2"])


@pytest.mark.anyio
async def test_infrastructure_fault_on_the_first_query_also_propagates():
    """坏在第一路时同样照抛 —— 免得「第一路成功才抛」这种靠顺序走运的实现溜过。"""
    r = _FlakyRetriever(
        fail_on="q1", exc=ToolInfrastructureError("知识检索服务暂时不可用")
    )
    with pytest.raises(ToolInfrastructureError):
        await multi_search(r, ["q1", "q2"])


# ---- expand_queries:正路(没有它,「永远退回原问题」的实现全绿) ----


class _ExpandedQueries:
    def __init__(self, queries):
        self.queries = queries


class _ExpandModel:
    """替身:结构化调用直接给结果。记录 `(schema, method)`,用于钉 json_mode。"""

    def __init__(self, queries):
        self.queries = queries
        self.calls: list[list] = []
        self.structured: list[tuple] = []

    def with_structured_output(self, schema, method=None):
        self.structured.append((schema, method))
        return self

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        return _ExpandedQueries(self.queries)


class _BoomModel:
    """替身:模型调用**抛错**。默认抛本仓「模型出参不可用」那一族。"""

    def __init__(self, exc: Exception | None = None):
        self.exc = exc or OutputParserException("不是 JSON")
        self.structured: list[tuple] = []

    def with_structured_output(self, schema, method=None):
        self.structured.append((schema, method))
        return self

    async def ainvoke(self, messages):
        raise self.exc


@pytest.mark.anyio
async def test_expand_returns_the_models_queries():
    """正路:模型给的多条查询原样返回。

    期望值(`["退货政策怎么规定的", ...]`)与退路值(`["这个能退吗"]`)**不同** ——
    少了这条,`return [text]` 一行的实现能让本文件其余扩写用例全部通过。
    """
    model = _ExpandModel(["退货政策怎么规定的", "七天无理由的时效", "退货运费谁承担"])
    got = await expand_queries(model, text="这个能退吗", max_queries=3)
    assert got == ["退货政策怎么规定的", "七天无理由的时效", "退货运费谁承担"]


@pytest.mark.anyio
async def test_expand_caps_at_max_queries():
    """条数上限是**结构保证**,不是提示词约定 —— prompt 里写了,也不许模型说了算。"""
    model = _ExpandModel(["a", "b", "c", "d", "e"])
    got = await expand_queries(model, text="能退吗", max_queries=3)
    assert got == ["a", "b", "c"]


@pytest.mark.anyio
async def test_expand_dedupes_before_capping():
    """先去重、再截断 —— 顺序反了会白丢一个角度。

    `["A", "A", "B", "C"]` / max=3:先去重 → `["A", "B", "C"]`(3 个角度);
    先截断 → `["A", "A", "B"]` → 去重 → `["A", "B"]`(**只剩 2 个**)。
    两种实现的输出不同,故可判别。
    """
    model = _ExpandModel(["A", "A", "B", "C"])
    got = await expand_queries(model, text="能退吗", max_queries=3)
    assert got == ["A", "B", "C"]


@pytest.mark.anyio
async def test_expand_strips_and_drops_blank_entries():
    """空白条目要清掉 —— 空串会被当成一路查询发去嵌入,白跑一次且召回为空。"""
    model = _ExpandModel(["  退货政策  ", "", "   ", "运费谁出"])
    got = await expand_queries(model, text="能退吗", max_queries=3)
    assert got == ["退货政策", "运费谁出"]


def test_expand_failure_falls_back_to_original_question():
    """扩写失败 → 退回单路原问题,绝不让检索空转。

    同步用例 + `asyncio.run`(brief 就是这么写的):这条不走 anyio —— 扩写是
    纯函数,不需要事件循环夹具,而 `asyncio.run` 在正在跑的循环里会直接抛。
    """
    got = asyncio.run(expand_queries(_BoomModel(), text="这个能退吗", max_queries=3))
    assert got == ["这个能退吗"]


@pytest.mark.anyio
async def test_expand_empty_model_output_falls_back_to_original_question():
    """模型**空数组**(或全都是空白)→ 同样退回原问题。

    「返回空列表」与「没退回」的代价一样:检索空转,而它看起来和「库里没有」
    一模一样。空数组这条真机可达(模型可能认为原问题已经够好、没什么可泛化)。
    """
    for raw in ([], ["", "   "]):
        got = await expand_queries(_ExpandModel(raw), text="这个能退吗", max_queries=3)
        assert got == ["这个能退吗"], f"queries={raw!r} 时没有退回原问题"


@pytest.mark.anyio
async def test_expand_does_not_swallow_a_transport_fault():
    """**例外里的例外**:只接「模型出参不可用」那一族,上游故障照抛。

    与同一章的 `make_resolve_references_node` / `classify_intent` 一致
    (spec §8:502 归上游故障;「扩写不阻断」降级的是**扩写自己**的失败)。
    宽 `except Exception` 会把上游超时/401 —— 甚至我自己的 `AttributeError` ——
    一律伪装成「这轮没扩写」,而本仓栽在静默降级上太多次。

    这条同时是**可回退标记**:若裁定「超时也该退单路」,把 `except` 那一行放宽后
    这条会红 —— 那正是它存在的意义。
    """
    model = _BoomModel(RuntimeError("上游 429"))
    with pytest.raises(RuntimeError):
        await expand_queries(model, text="这个能退吗", max_queries=3)


# ---- 装配:唯一出口 + 两禁 ----


@pytest.mark.anyio
async def test_expand_uses_json_mode_and_sends_exactly_what_prompts_builds():
    """两件事,各对应一个会静默失效的偏离:

    1. 结构化出参必须走 **`json_mode`** —— 本项目端点上 `function_calling` 与
       `json_schema` 均返回 400(换模型也一样),那是「改了没反应」的形态;
    2. 消息装配留在 `app/prompts.py` 这**唯一出口**。断言用**逐条相等**,
       不是「里有有原问题」—— 后者在漏掉整段 system、或漏渲染 `max_queries`
       时照样通过。
    """
    model = _ExpandModel(["a", "b"])
    await expand_queries(model, text="这个能退吗", max_queries=3)

    schema, method = model.structured[0]
    assert method == "json_mode"
    assert schema is ExpandQueries
    assert model.calls[0] == build_expand_messages(text="这个能退吗", max_queries=3)


def test_build_expand_messages_renders_max_queries_into_the_system_prompt():
    """`{max_queries}` 必须真的渲染进 system(`ChatPromptTemplate` 不渲染就是字面量)。"""
    messages = build_expand_messages(text="这个能退吗", max_queries=2)
    assert messages[0].content == EXPAND_SYSTEM_PROMPT.format(max_queries=2)
    assert "2" in messages[0].content
    assert messages[-1].content == "这个能退吗"


def test_expand_system_prompt_keeps_the_two_hard_rules():
    """提示词的两条硬规矩:

    - 出现**字面 `JSON`** 字样(json_mode 的硬要求);
    - **不得有裸花括号** —— `ChatPromptTemplate` 按 f-string 解析,`{...}` 会在
      装配时抛 `KeyError`,而报错指向别处。
    """
    assert "JSON" in EXPAND_SYSTEM_PROMPT
    assert "queries" in EXPAND_SYSTEM_PROMPT
    assert "{" not in EXPAND_SYSTEM_PROMPT.replace("{max_queries}", "")
