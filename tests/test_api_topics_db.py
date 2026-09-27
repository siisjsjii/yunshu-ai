"""分布端点 —— **走真库、直调端点函数**。

⚠️ 三个刻意的选择:

① **不用 `TestClient`**:它会把 `get_engine()` 的 lru_cache 单例绑到 portal 循环上,
   同进程后面的 db 测试会拿到跨循环连接(原因写在 `tests/test_api_ticket.py` 的 docstring 里);
② **不用替身**:本任务的全部价值就是验 `JSON_TABLE` 那段 SQL,替身把它替掉就什么都没测了。
   照 `tests/test_api_conversations_db.py` 直调 `list_conversations` 的先例。
③ **断言一律用「探针前后之差」,不用绝对值**:`topic_classifications` 里躺着 B13 真跑
   写下的 65 行(id 12–76),池子也是共享的 —— 任何「总数 == 65」式的断言都会**被上次
   运行的数据污染**(本仓编目过的那类偶尔红偶尔绿)。差值对「库里此刻有什么」免疫。

探针行**用完即删且 commit** —— `async with session` 退出是 rollback。

## 探针的三个池子行

| id | 池子里有行吗 | 题面 | 用来断什么 |
|---|---|---|---|
| `P1` | 有 | `Q_PROBE` | 多标签展开(两个标签各算一次) |
| `P2` | 有 | **同一条** `Q_PROBE` | 「不同问题数」按**题面**数(计划订正 17-A) |
| `P3` | **没有** | —— | `LEFT JOIN` 不丢行(悬空结果仍进 `total`) |

⚠️ `P3` 这种「悬空结果」是**真实可达**的:`low_confidence_question_id` **刻意不挂外键**
(ORM docstring 逐字写着),所以一个删掉的池子行会留下一行仍被计入的结果行。
`INNER JOIN` 会把它丢掉 —— `total` 的差值当场红。
"""

import pytest
from sqlalchemy import delete, text

from app.api.topics import distribution
from app.db.base import get_engine, get_sessionmaker
from app.db.models import LowConfidenceQuestion, TopicClassification
from app.topic.taxonomy import LABELS

pytestmark = pytest.mark.db

#: 探针用的池子行 id。**高位段,不撞真实数据**(池子今天的 id 上界是 990)。
P1, P2 = 999101, 999102
#: ⚠️ **故意不建池子行**的那一个 —— 悬空结果(见模块 docstring 的表)。
P3 = 999103
PROBE_IDS = [P1, P2, P3]

#: P1 / P2 **共用同一条题面** —— 「不同问题数 < 行数」才可观测(17-A 的全部)。
Q_PROBE = "t14 探针:两条不同的池子行共用这一条题面"

#: P1 的两个标签 + P2 的一个标签 ⇒ 三处标签计数、两行结果。
L_P1 = ["尺码", "退换货"]
L_P2 = ["退换货"]


async def _purge(sm) -> None:
    """删探针行并**提交**。顺序:先结果行、再池子行(结果行是池子行的下游)。

    插入前也要调一次:上一轮若超时 / Ctrl-C / 断言中途红,残留会撞唯一键
    `uk_pool_question`,让这一轮**红在别处**。
    """
    async with sm() as s:
        await s.execute(
            delete(TopicClassification).where(
                TopicClassification.low_confidence_question_id.in_(PROBE_IDS)
            )
        )
        await s.execute(
            delete(LowConfidenceQuestion).where(LowConfidenceQuestion.id.in_([P1, P2]))
        )
        await s.commit()


async def _seed(sm) -> None:
    """三条探针:P1 两个标签、P2 一个标签(**与 P1 同题面**)、P3 悬空。"""
    await _purge(sm)
    async with sm() as s:
        s.add_all([
            # 池子行 id 显式给死:结果行的关联列是探针的一部分,自增 id 拿不稳。
            LowConfidenceQuestion(
                id=P1, question=Q_PROBE, source_conversation_id=None,
                entry_point="置信度闸", reject_reason="t14 探针",
            ),
            LowConfidenceQuestion(
                id=P2, question=Q_PROBE, source_conversation_id=None,
                entry_point="置信度闸", reject_reason="t14 探针",
            ),
            TopicClassification(
                low_confidence_question_id=P1, labels=L_P1, scores={},
                model_version="t14probe",
            ),
            TopicClassification(
                low_confidence_question_id=P2, labels=L_P2, scores={},
                model_version="t14probe",
            ),
            # ⚠️ Q3 **不在池子里** —— 断 `LEFT JOIN` 用。
            TopicClassification(
                low_confidence_question_id=P3, labels=["物流"], scores={},
                model_version="t14probe",
            ),
        ])
        await s.commit()


async def _snapshot() -> dict:
    """**新 session** 读一次端点(身份映射持弱引用 —— 同 session 重读会变成靠 refcount 走运)。"""
    async with get_sessionmaker()() as session:
        return await distribution(session=session)


def _by_label(body: dict) -> dict[str, dict]:
    return {b["label"]: b for b in body["buckets"]}


class _RecordingSession:
    """真 session **外面只套一层记录**:记下发给 MySQL 的每一条 SQL 原文。

    为什么需要它:光断计数,**一个「把 labels 拉回 Python 再数」的实现也照样绿** ——
    它在当前 65 行的规模下与 `JSON_TABLE` 逐位同结果。这条壳不改语义(每条语句
    原样交给真 session),只让「展开发生在 SQL 里」这件事**可观测**。
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.statements: list[str] = []

    async def execute(self, statement, *args, **kwargs):
        self.statements.append(str(statement))
        return await self._inner.execute(statement, *args, **kwargs)


@pytest.mark.anyio
async def test_labels_are_expanded_inside_the_sql_by_json_table():
    """⚠️ **这条断的是「计数发生在 SQL 里」,不是计数对不对**(后者在别的用例里)。

    反面实现:把 `labels` 整列拉回 Python 再 `Counter()` —— 演示规模下**与
    本实现逐位同结果**,唯一会露出来的是数据量上去之后「一次请求拉全表」。
    所以这里断两件事:**执行过的语句里有 `JSON_TABLE`**,且**没有**一条语句
    读了 `labels` 却没展开它。

    `JSON_TABLE` 在本机 MySQL **8.0.46** 上实测可用(`.superpowers/probe_ch10_jsontable.py`),
    **不是照文档推的**。
    """
    engine = get_engine()
    try:
        async with get_sessionmaker()() as session:
            rec = _RecordingSession(session)
            body = await distribution(session=rec)
        assert body["buckets"], "前提:端点得真的返回了桶,否则下面两条在空结果上也绿"

        sql = "\n".join(rec.statements)
        assert "JSON_TABLE" in sql.upper(), (
            "按标签计数必须在 SQL 里展开 JSON 数组 —— 执行过的语句里一条 JSON_TABLE 都没有,"
            f"实际执行了:{rec.statements}"
        )
        # 第二道:被拉回 Python 的那种实现长这样 —— 一条 SELECT 里出现 labels 列,
        # 却没有 JSON_TABLE 展开它。
        raw = [s for s in rec.statements if "labels" in s and "JSON_TABLE" not in s.upper()]
        assert raw == [], f"有语句读了 labels 却没在 SQL 里展开:{raw}"
    finally:
        await engine.dispose()


@pytest.mark.anyio
async def test_a_multi_label_row_counts_in_every_bucket():
    """一行带两个标签,**在两个桶里各算一次** —— 多标签的正确读法。

    断言是**探针前后之差**:标签总次数 **+3**(P1 两个 + P2 一个)、结果行数 **+2**。
    · 「只取 `labels[0]`」的实现 ⇒ 标签总次数 +2 ⇒ 红;
    · 「按行数、不按标签」的实现 ⇒ 同上 ⇒ 红。
    """
    engine = get_engine()
    sm = get_sessionmaker()
    try:
        before = await _snapshot()
        await _seed(sm)
        after = await _snapshot()

        b, a = _by_label(before), _by_label(after)
        assert a["尺码"]["count"] - b["尺码"]["count"] == 1, "P1 的第二个标签没被算到"
        assert a["退换货"]["count"] - b["退换货"]["count"] == 2, (
            "P1(尺码+退换货)与 P2(退换货)各该给「退换货」+1,实际只加了 "
            f"{a['退换货']['count'] - b['退换货']['count']}"
        )
        assert a["物流"]["count"] - b["物流"]["count"] == 1, "悬空结果行(P3)的标签也该算"
        # 「结果行总数的增量」与「标签次数的增量」必须**不同** —— 相等就说明没展开
        delta_labels = sum(x["count"] for x in after["buckets"]) - sum(
            x["count"] for x in before["buckets"]
        )
        delta_total = after["total"] - before["total"]
        assert delta_total == 3, f"三条探针结果行,实际 total 加了 {delta_total}"
        assert delta_labels == 4, (
            f"四处标签(尺码/退换货/退换货/物流)该各算一次,实际标签次数加了 {delta_labels}"
        )
        assert delta_labels > delta_total, "多标签行必须让标签次数多于结果行数"
    finally:
        await _purge(sm)
        await engine.dispose()


@pytest.mark.anyio
async def test_distinct_questions_counts_pool_question_texts_not_result_rows():
    """⚠️ **计划订正 17-A**:两个数**口径不同**,别拿它们对不上当 bug。

    `COUNT(DISTINCT low_confidence_question_id)` 与 `COUNT(*)` 是**恒等**的
    (那一列有唯一键 `uk_pool_question`)—— 真库实测 **65 行 / 65 distinct**,
    而池子里只有 **33 条不同题面** ⇒ 页面上的「不同问题数」曾是一个**没意义的数**。
    改法是回池子按 `q.question` 数。

    判别力全在**探针这两行的差**上:P1 / P2 是**两条不同的池子行**、**同一条题面**
    ⇒ 结果行 **+2**、不同题面 **+1**。按 `low_confidence_question_id` 数会得到 +2
    (与 `total` 的增量相同)—— 那样这条测试对那个实现**不再有判别力**,所以下面
    两个数都断,且桶级的 `distinct` 也断(17-A 要求 summary 与 bucket **都**改)。
    """
    engine = get_engine()
    sm = get_sessionmaker()
    try:
        before = await _snapshot()
        await _seed(sm)
        after = await _snapshot()

        assert after["total"] - before["total"] == 3, "前提:三条探针结果行都该被计入"
        delta_distinct = after["distinct_questions"] - before["distinct_questions"]
        # 探针只贡献 2 条不同题面:Q_PROBE(两条池子行共用)与 P3 的悬空行(没有题面)。
        assert delta_distinct == 1, (
            f"两条**不同的**池子行指向**同一条题面** ⇒ 不同问题数只该 +1,实际 {delta_distinct}"
            "(按 low_confidence_question_id 数会得到 +2 —— 与 total 恒等,那个数没有意义)"
        )

        b, a = _by_label(before), _by_label(after)
        assert a["退换货"]["count"] - b["退换货"]["count"] == 2
        assert a["退换货"]["distinct"] - b["退换货"]["distinct"] == 1, (
            "桶级的 distinct 也要按池子题面数:P1 / P2 同题面 ⇒ 只算 1 条"
        )
    finally:
        await _purge(sm)
        await engine.dispose()


@pytest.mark.anyio
async def test_zero_count_labels_are_still_returned():
    """⚠️ **一个标签都没有的类目也要出现在结果里(count=0)。**

    不补零的话,页面上的条形图会**少几类**,而少的那几类看起来像
    「这一类没问题」—— 恰恰相反,它们是一条样本都没有。

    ⚠️ **判别力不依赖「库里此刻恰好缺某些类」**:哪几类在数据里出现过,由本用例
    **独立读一次原始表**算出来(不是拿端点的输出反推),再要求「没出现过的类目
    必须存在且恰好是 0」。真库今天有 10 类没有样本(商品信息/库存补货/退换货/
    其他/运费/发票/会员积分 这 7 类有),所以补零那一支**今天真的有判别力**。
    """
    engine = get_engine()
    try:
        body = await _snapshot()
        labels = [b["label"] for b in body["buckets"]]
        assert set(labels) == set(LABELS), f"缺这些类目:{set(LABELS) - set(labels)}"
        assert len(labels) == len(LABELS), f"桶不该重复:{labels}"

        async with get_sessionmaker()() as session:
            observed = {
                r.label for r in (await session.execute(text(
                    "SELECT DISTINCT jt.label AS label FROM topic_classifications t, "
                    "JSON_TABLE(t.labels, '$[*]' COLUMNS (label VARCHAR(64) PATH '$')) jt"
                ))).all()
            }
        assert observed, "前提:库里得有样本,否则下面两条在「端点全返 0」时也绿"

        buckets = _by_label(body)
        missing = set(LABELS) - observed
        assert all(buckets[lb]["count"] == 0 for lb in missing), (
            "数据里一条样本都没有的类目必须补成 0,实际:"
            f"{[(lb, buckets[lb]['count']) for lb in missing if buckets[lb]['count'] != 0]}"
        )
        # 反向对照:**有**样本的类目不许被补成 0(否则「全返 0」的实现在上面那条上绿)
        assert all(buckets[lb]["count"] > 0 for lb in observed), (
            f"有样本的类目被算成 0:{[(lb, buckets[lb]) for lb in observed if buckets[lb]['count'] == 0]}"
        )
        assert all(b["distinct"] <= b["count"] for b in body["buckets"]), body["buckets"]
        assert all(b["distinct"] == 0 for b in body["buckets"] if b["count"] == 0)
    finally:
        await engine.dispose()


@pytest.mark.anyio
async def test_buckets_are_sorted_by_count_descending_then_taxonomy_order():
    """页面的条形图顺序必须**确定**:`count` 降序,同分按权威表的类目序。

    ⚠️ 同分不兜底的话,顺序由查询计划决定 —— 页面每次刷新都可能长得不一样,
    而那**不报任何错**(它只让「今天这张图和昨天说的是同一件事」不再成立)。
    判别力:去掉排序键(或把 `count` 那一维删掉)⇒ 第一条红。
    """
    engine = get_engine()
    try:
        body = await _snapshot()
        counts = [b["count"] for b in body["buckets"]]
        assert counts == sorted(counts, reverse=True), f"没按 count 降序:{counts}"
        assert len(body["buckets"]) == len(LABELS), "前提:17 类都在(否则同分组可能是空的)"
        for i in range(1, len(body["buckets"])):
            a, c = body["buckets"][i - 1], body["buckets"][i]
            if a["count"] == c["count"]:
                assert LABELS.index(a["label"]) < LABELS.index(c["label"]), (
                    f"同分的桶该按权威表类目序,实际 {a['label']} 排在 {c['label']} 之前"
                )
    finally:
        await engine.dispose()


@pytest.mark.anyio
async def test_summary_matches_the_table_it_summarizes():
    """summary 三样(`total` / `last_classified_at` / `model_versions`)都**对着真表**核。

    ⚠️ **够不到的地方,如实写在这里**:`total` 与「不同问题行数」在本表上
    **原理上不可区分**(唯一键让 `COUNT(*)` == `COUNT(DISTINCT qid)`),所以这条
    对「total 用错聚合」没有判别力 —— 判别力在别处:探针 `P3` 是**悬空结果**
    (池子里没有那一行),`INNER JOIN` 的实现会让 `total` 少 1,而下面的差值断言
    在 `test_distinct_questions_...` 里断 `total` 加了 3。
    """
    engine = get_engine()
    sm = get_sessionmaker()
    try:
        before = await _snapshot()
        await _seed(sm)
        try:
            async with sm() as session:
                raw = (await session.execute(text(
                    "SELECT COUNT(*) AS total, MAX(classified_at) AS last_at "
                    "FROM topic_classifications"
                ))).one()
                versions = [
                    r[0] for r in (await session.execute(text(
                        "SELECT DISTINCT model_version FROM topic_classifications "
                        "ORDER BY model_version"
                    ))).all()
                ]
                raw_distinct = (await session.execute(text(
                    "SELECT COUNT(DISTINCT q.question) FROM topic_classifications t "
                    "LEFT JOIN low_confidence_questions q "
                    "ON q.id = t.low_confidence_question_id"
                ))).scalar()
            after = await _snapshot()

            # 差值口径:探针加进来之后三个数各自该动多少
            assert after["total"] - before["total"] == 3
            assert after["distinct_questions"] - before["distinct_questions"] == 1
            # 绝对值口径:**对着真表**核(这一半是可被数据污染的,只作补充)
            assert after["total"] == raw.total, (
                f"端点的 total={after['total']} 与真表 COUNT(*)={raw.total} 不符"
            )
            assert after["distinct_questions"] == raw_distinct
            assert after["model_versions"] == versions, (
                f"页面顶部要显示「这个图是谁算的」:实际 {after['model_versions']!r}"
                f",真表里是 {versions!r}"
            )
            assert "t14probe" in after["model_versions"], "前提:探针的版本号该出现在里面"
            assert after["last_classified_at"] == raw.last_at.strftime("%Y-%m-%d %H:%M"), (
                f"最后归类时间取 MAX(classified_at),实际 {after['last_classified_at']!r}"
            )
        finally:
            await _purge(sm)
    finally:
        await engine.dispose()


@pytest.mark.anyio
async def test_the_endpoint_answers_over_http_and_is_not_swallowed_by_static():
    """真打一次 HTTP:路由**在** `mount("/")` 之前,所以它拿到的是 JSON 不是静态 404。

    ⚠️ **不用 `TestClient`**:它跑请求的那个 portal 事件循环会污染
    `get_engine()` 的 lru_cache 单例(见 `tests/test_api_ticket.py` 的 docstring)。
    `httpx.ASGITransport` 在**当前**事件循环里把请求交给 ASGI app,不跑 lifespan,
    不建 portal 线程 —— 与同文件其余用例共用同一个循环,不额外引入一类连接。

    ⚠️ 这条断的是**路由**;返回的数对不对是上面那些用例的事(这里只断 200 +
    结构齐全,免得把「数错了」也判成「路由没挂上」)。
    """
    import httpx

    from app.main import app

    engine = get_engine()
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            resp = await client.get("/api/topics/distribution")
        assert resp.status_code == 200, (
            f"端点没应答({resp.status_code})—— 若静态目录挂在它前面,catch-all 会"
            f"先匹配、把它变成一次静态文件查找。响应前 200 字:{resp.text[:200]!r}"
        )
        body = resp.json()
        assert set(body) == {
            "total", "distinct_questions", "last_classified_at", "model_versions", "buckets"
        }, f"响应键与页面约定不符:{sorted(body)}"
        assert len(body["buckets"]) == len(LABELS)
    finally:
        await engine.dispose()


@pytest.mark.anyio
async def test_empty_table_returns_zeroes_not_an_error():
    """没跑过批处理时返回零值结构,而不是 500 —— 页面第一次打开就是这个状态。

    ⚠️ **这条做不到「真的造出空表」**:`topic_classifications` 是**共享**表,
    里面躺着 B13 真跑写下的 65 行 —— 本用例**不删任何别人的行**(那是别人跑过
    一次的结果,删掉就是把一次真实运行的痕迹抹了)。所以这里断的是**形状**:

    · 零值结构不报错(`total` / `count` / `distinct` 都是非负整数,能直接进模板);
    · 17 个类目**一个不少**(缺类目是页面最容易出的错,且它**不报错**);
    · `last_classified_at` 是字符串或 `None`(`None` 那一支今天在真库上够不到,
      但**形状**必须对 —— 页面拿到一个 datetime 对象会在模板里炸);
    · 补零那一支的判别力在 `test_zero_count_labels_are_still_returned` 里(那条
      用「独立读一次原始表」算出了哪几类没有样本)。
    """
    engine = get_engine()
    try:
        body = await _snapshot()
        assert isinstance(body["total"], int) and body["total"] >= 0
        assert isinstance(body["distinct_questions"], int) and body["distinct_questions"] >= 0
        assert set(b["label"] for b in body["buckets"]) == set(LABELS)
        assert all(isinstance(b["count"], int) and b["count"] >= 0 for b in body["buckets"])
        assert all(isinstance(b["distinct"], int) and b["distinct"] >= 0 for b in body["buckets"])
        assert body["last_classified_at"] is None or isinstance(body["last_classified_at"], str), (
            f"时间要给字符串给页面,实际 {body['last_classified_at']!r}"
        )
        assert isinstance(body["model_versions"], list)
        assert all(isinstance(v, str) for v in body["model_versions"])
    finally:
        await engine.dispose()
