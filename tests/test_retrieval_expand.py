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
from sqlalchemy.exc import OperationalError

from app.prompts import EXPAND_SYSTEM_PROMPT, build_expand_messages
from app.retrieval.expand import ExpandQueries, expand_queries, multi_search
from app.retrieval.search import KnowledgeRetriever, RetrievedChunk
from app.tools.errors import ToolInfrastructureError


def _raw_db_error() -> OperationalError:
    """**未翻译**的 MySQL 故障,与 `_load_rows` 里实际会抛出来的形态一致。"""
    return OperationalError(
        "SELECT knowledge_chunks ...", {}, Exception("Lost connection to MySQL server")
    )


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
    # 每条查询各跑一次、按传入顺序 —— 少了这条,`calls` 属性只写不读,
    # 「跳过某条」或「重复跑某条」都没有断言看着。
    assert r.calls == ["q1", "q2"]


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
async def test_duplicate_appearing_first_with_the_higher_score_also_wins():
    """**镜像用例**:重复块先以高分出现、后又以低分出现 → 仍是高分那份。

    上一条把高分放在**后面**,于是「完全不做比较」的
    `merged[chunk.chunk_id] = chunk`(**last-wins**)恰好也给出正确答案。
    这一条把顺序反过来:last-wins 会塌成 `[1, 3, 2]` / `[0.9, 0.6, 0.4]`,
    正确实现是 `[1, 2, 3]` / `[0.9, 0.85, 0.6]`。

    两条合起来,「留首次 / 留最高 / 留最后」三种写法**两两可分** ——
    单看任何一条都不够。
    """
    r = _FakeRetriever({
        "q1": [_chunk(1, 0.9), _chunk(2, 0.85)],
        "q2": [_chunk(3, 0.6), _chunk(2, 0.4)],
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
async def test_multi_search_keeps_going_after_a_broken_query():
    """**第一路**就挂(非基础设施)时,后面的路照样跑 —— 用剩下的。

    与 brief 那条(`fail_on="q2"`)合起来才完整:那条里坏的是**最后一路**,
    「遇到失败就 `break`」的实现照样绿。这里断言两路**都**跑过。
    """
    r = _FlakyRetriever(fail_on="q1")
    got = await multi_search(r, ["q1", "q2"])
    assert r.calls == ["q1", "q2"]
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
async def test_raw_db_error_from_a_query_is_escalated_not_swallowed():
    """**未翻译的**数据库故障同样必须响亮 —— 上面那两条看不见这个洞。

    上面两条注入的是 `ToolInfrastructureError`(分类表的**产物**)。而
    `KnowledgeRetriever._load_rows` 原先**没有**任何翻译:MySQL 故障时它抛的是
    **裸 `OperationalError`**,`search()` 只回滚再原样上抛。一个「宽 except 吞掉
    一切」的 `multi_search` 会把它当成「这一路查询失败」跳过 → 返回**剩余查询的
    结果**,用户得到「知识库里没有」,而单路 `query_faq` 走 executor 是响亮 502。

    注入已翻译的异常 = **跳过被验的那一步**(本仓定义的假绿,样板见
    `tests/test_api_refund.py:171-186`);所以这里注入**裸** `OperationalError`。

    断言**类型**:`app/api/chat.py` 只按 `ToolInfrastructureError` 给 502,
    原样重抛 `OperationalError` 会变成 500(把「服务端出问题」说成「你的请求有问题」)。
    """
    r = _FlakyRetriever(fail_on="q2", exc=_raw_db_error())
    with pytest.raises(ToolInfrastructureError):
        await multi_search(r, ["q1", "q2"])


@pytest.mark.anyio
async def test_raw_db_error_on_the_first_query_is_also_escalated():
    """坏在第一路时同样升级 —— 免得「第一路成功才升级」这种靠顺序走运的实现溜过。"""
    r = _FlakyRetriever(fail_on="q1", exc=_raw_db_error())
    with pytest.raises(ToolInfrastructureError):
        await multi_search(r, ["q1", "q2"])


class _StubStore:
    def hybrid_search(self, vector, text, top_k, category=None):
        return [("1", 0.9)]


class _StubEmbedder:
    def encode(self, texts):
        return [[0.1, 0.1] for _ in texts]


class _StubReranker:
    def rerank(self, query, candidates):
        return [0.9 for _ in candidates]


class _DeadSession:
    """MySQL 挂了:回查原文时抛**裸** `OperationalError`(与真实现一致)。"""

    async def execute(self, *args, **kwargs):
        raise _raw_db_error()

    async def rollback(self):
        pass


@pytest.mark.anyio
async def test_real_retriever_db_fault_is_not_downgraded_to_empty_by_multi_search():
    """**接缝用例**:真 `KnowledgeRetriever` + 真 `multi_search`,两者合起来仍要响亮。

    上面两条(根修一处、纵深防御一处)各自只验自己那一段;而「合起来对最终调用方
    是什么」没人验 —— 根修漏翻一处、或 `multi_search` 的宽 except 又把它吞回去,
    两段的用例都可能仍然绿。这里让**真的**检索器撞上死 session(不碰真库:
    store/embedder/reranker 都是桩),走**真的** `multi_search`,断言整次检索**上抛**
    `ToolInfrastructureError`,**不是返回空列表** —— 空列表会让
    `make_retrieve_knowledge_node` 报 `0 hits`,用户得到「知识库里没有这条」,
    而库其实只是挂了。这正是 CLAUDE.md 禁止的那种降级。
    """
    retriever = KnowledgeRetriever(
        _DeadSession(), _StubStore(), _StubEmbedder(), _StubReranker(),
        top_k=3, score_threshold=0.5, hybrid_top_k=10,
    )
    with pytest.raises(ToolInfrastructureError):
        await multi_search(retriever, ["能退吗", "退货运费谁出"])


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


@pytest.mark.parametrize("bad", [0, -1])
def test_expand_rejects_non_positive_max_queries(bad):
    """`max_queries < 1` 必须**响亮**,不能静默返回空、更不能静默少一条。

    两种边界都实测过:`0 → []`(正是本模块通篇在防的「检索空转」形状);
    **`-1 → ['a']`** —— Python 负切片是「去掉最后 N 条」,于是负配置**静默少一条**、
    看起来完全正常。所以这里要的是异常,不是「悄悄夹到 1」。
    顺带钉**先校验后调用**:配置错误不该先花一次模型调用。
    """
    model = _ExpandModel(["a", "b"])
    with pytest.raises(ValueError):
        asyncio.run(expand_queries(model, text="能退吗", max_queries=bad))
    assert model.calls == []
    assert model.structured == []


def test_expand_schema_really_validates_the_prompt_contract():
    """`ExpandQueries` 得**真的**能校验 prompt 承诺的那个形状。

    其余用例喂的都是鸭子类型 `_ExpandedQueries`(只有一个 `.queries` 属性),
    所以这个 schema **从没被真跑过** —— 字段名写成别的(如 `query`)时,生产上
    `with_structured_output` 会解析失败,而失败被吞成**合法的** `[原话]` 回退:
    整条链路静默降级成单路,而单测全绿。这里按 prompt 里的字面字段名真校验一次。
    """
    assert set(ExpandQueries.model_fields) == {"queries"}      # spec §4.3:只有它一个
    parsed = ExpandQueries.model_validate({"queries": ["退货政策", "运费谁出"]})
    assert parsed.queries == ["退货政策", "运费谁出"]
    # 缺字段是**正常**的(模型偶尔只吐 `{}`):default_factory 给空数组,
    # 再由 `expand_queries` 走「空 → 退回原问题」那条路,而不是 KeyError。
    assert ExpandQueries().queries == []


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
