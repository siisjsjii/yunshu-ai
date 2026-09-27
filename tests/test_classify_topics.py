"""批处理:整批原子、服务故障响亮失败、幂等覆盖。用**假服务**(不联网)。

## 这个文件要守的东西,以及每条断言为什么不是同义反复

本仓的头号风险是**假绿**。逐条说清楚「实现改错了,这条的输出会不会不同」:

| 用例 | 一个错误的实现 | 会红在哪 |
|---|---|---|
| `test_service_failure_does_not_write_anything` | 捕获异常、继续写空标签 | `written != []` |
| `test_a_batch_is_all_or_nothing` | 逐条边校验边写 | `written != []` |
| `test_clean_is_applied_before_sending` | 直接发 `row["question"]` | `sent` 里带手机号原文 |
| `test_results_carry_the_question_id` | 丢掉 id(只写 labels) | `KeyError` / 对不回去 |
| `test_a_label_outside_the_taxonomy_is_rejected` | 不查 `taxonomy.LABELS` | 不抛,当场绿成「已校验」 |
| `test_result_count_mismatch_is_rejected` | 只按 `min(len)` 配对 | 不抛,后面按 zip 静默少写 |
| `test_batch_size_is_honoured` | 忽略 `batch_size`(整池一次发) | 只 1 次调用 |
| `test_empty_labels_are_allowed_but_counted` | 空标签抛异常 / 不计数 | 抛 / 计数为 0 |
| `test_real_client_*` | 客户端不拆 `["results"]` / 不校验形状 | 返回值不对 / 不抛 |

⚠️ **`_FakeClient` 逐字照 brief**(它就是「真实 HTTP 客户端」那条接缝的契约):
`predict()` 返回**裸 list**,`{"results": …}` 这个 HTTP 形状**只许**在真客户端里被拆
(`scripts/classify_topics.py` 的 `TopicServiceClient`)—— 漏进纯逻辑层的话,
这个替身与生产走的就不是同一条路了。
"""

import httpx
import pytest
from sqlalchemy import delete, select

from app.db.base import get_engine, get_sessionmaker
from app.db.models import TopicClassification
from app.topic.taxonomy import LABELS
from scripts.classify_topics import (
    TopicServiceClient,
    TopicServiceProtocolError,
    classify_batch,
    load_pool,
    model_version_from_meta,
    run,
    upsert_classifications,
)


class _FakeClient:
    def __init__(self, results=None, error=None):
        self.results = results
        self.error = error
        self.calls = []

    async def predict(self, texts):
        self.calls.append(list(texts))
        if self.error:
            raise self.error
        return self.results


# --------------------------------------------------------------------------
# 纯逻辑层(替身,**不联网**)
# --------------------------------------------------------------------------


@pytest.mark.anyio
async def test_service_failure_does_not_write_anything():
    """⚠️ **基础设施故障绝不伪装成业务结果。**

    连不上服务时写一行空标签的后果:分布页会把「服务挂了」读成
    「这些问题没有主题」—— 正是 ch09 那条 👎 回捞失败被读成
    「知识库缺这块」的同款事故(spec §9.2)。
    """
    written = []
    client = _FakeClient(error=ConnectionError("connect refused"))

    with pytest.raises(Exception):
        await classify_batch([{"id": 1, "question": "x"}], client=client,
                             write=written.append, batch_size=8)
    assert written == [], "服务故障时不许写任何行"


@pytest.mark.anyio
async def test_a_batch_is_all_or_nothing():
    """一批里第 7 条失败 ⇒ **整批不写**。

    部分写入会留下「一半归过类」的池子,而重跑时分不清哪些是这次的、
    哪些是上次的。
    """
    written = []
    client = _FakeClient(results=[{"labels": ["尺码"], "scores": {}}] * 5 +
                                 [_ for _ in [None]] +   # 第 6 条形状不对
                                 [{"labels": ["运费"], "scores": {}}] * 2)
    with pytest.raises(Exception):
        await classify_batch([{"id": i, "question": f"q{i}"} for i in range(8)],
                             client=client, write=written.append, batch_size=8)
    assert written == []


@pytest.mark.anyio
async def test_clean_is_applied_before_sending():
    """⚠️ 发出去的文本必须是**清洗过**的 —— 训练侧也洗过,两侧同源。

    不洗的话模型在真机上看到的是另一种文本(spec §5.1)。
    """
    client = _FakeClient(results=[{"labels": [], "scores": {}}])
    await classify_batch([{"id": 1, "question": " 退货   怎么走  13800138000 "}],
                         client=client, write=lambda r: None, batch_size=8)
    sent = client.calls[0][0]
    assert sent == "退货 怎么走 <手机号>"


@pytest.mark.anyio
async def test_results_carry_the_question_id():
    """结果必须能对回池子行 —— 对不回去的批处理没有意义。"""
    written = []
    client = _FakeClient(results=[{"labels": ["尺码"], "scores": {}}])
    await classify_batch([{"id": 42, "question": "买大了"}],
                         client=client, write=written.append, batch_size=8)
    assert written[0]["low_confidence_question_id"] == 42


@pytest.mark.anyio
async def test_a_label_outside_the_taxonomy_is_rejected():
    """订正 15-D:`labels` 里每个名字都必须在 `taxonomy.LABELS` 里。

    这是「标签顺序 / 类目表两处漂移」这条链上**最靠近数据的那一道**:
    产物里 `labels.json` 与 `app/topic/taxonomy.py` 漂开之后,服务会照它自己那份
    吐出**别的类目名**,而那名字一旦落进 `topic_classifications.labels`(JSON 列,
    数据库对它零约束)就**再也回不来了** —— 分布页会画出一个不存在的类目。

    ⚠️ **判别力**:一个压根不查类目表的实现,拿到 `"不存在的类目"` 会**照写**
    ⇒ 下面 `pytest.raises` 当场红。**别名/近义词不算合法** —— 判据是**逐字** ∈ `LABELS`。
    """
    from scripts.classify_topics import classify_batch as _cb

    written = []
    bad = _FakeClient(results=[{"labels": ["退换货", "不存在的类目"], "scores": {}}])
    with pytest.raises(TopicServiceProtocolError):
        await _cb([{"id": 7, "question": "我要退"}], client=bad,
                  write=written.append, batch_size=8)
    assert written == [], "标签漂移的那一批一行都不许落"

    # 反面:同样的形状、标签全在类目表里 ⇒ **不许**抛(否则上面那条只是「什么都会抛」)
    good = _FakeClient(results=[{"labels": ["退换货", LABELS[-1]], "scores": {}}])
    ok = []
    await _cb([{"id": 7, "question": "我要退"}], client=good,
              write=ok.append, batch_size=8)
    assert ok[0]["labels"] == ["退换货", LABELS[-1]]


@pytest.mark.anyio
async def test_result_count_mismatch_is_rejected():
    """服务契约是「返回条数恒等于输入条数」(§9.2)。

    条数对不上时必须**响亮失败**,不许按 `zip` 配对 —— 那会静默少写/错位:
    「第 3 条的结果配给了第 4 条的问题」在分布页上**完全看不出来**。
    """
    written = []
    client = _FakeClient(results=[{"labels": ["尺码"], "scores": {}}] * 2)  # 只有 2 条
    with pytest.raises(TopicServiceProtocolError):
        await classify_batch([{"id": i, "question": f"q{i}"} for i in range(5)],
                             client=client, write=written.append, batch_size=8)
    assert written == []


@pytest.mark.anyio
async def test_batch_size_is_honoured():
    """`batch_size` 决定**每次 POST 发几条** —— 它是「一次前向的规模」这个旋钮。

    判别力:一个忽略它、把整池一次发出去的实现,调用次数是 1 而不是 2/1。
    (这条不是性能测试:它断的是**分片**这件事真的发生。)
    """
    client = _FakeClient(results=None)
    rows = [{"id": i, "question": f"q{i}"} for i in range(3)]

    async def predict(texts, _c=client):
        _c.calls.append(list(texts))
        return [{"labels": [], "scores": {}} for _ in texts]

    client.predict = predict  # type: ignore[method-assign]
    await classify_batch(rows, client=client, write=lambda r: None, batch_size=2)
    assert [len(c) for c in client.calls] == [2, 1], (
        f"应当切成 2 + 1 两批,实际每批条数 {[len(c) for c in client.calls]}"
    )


@pytest.mark.anyio
async def test_empty_labels_are_allowed_but_counted():
    """订正 15-C:`labels: []` 是**合法的模型输出**(全部低于阈值)⇒ 允许写。

    **但它必须被数出来** —— 否则「服务每次都返回空」会被分布页读成
    「这些问题没有主题」:那是**故障被读成业务结果**(与 `test_service_failure_…`
    要防的是同一件事,**只是那条防的是抛异常,防不了返回空**)。

    判别力:一个不计数(或把空标签当失败抛掉)的实现在这两条断言上各自红一条。
    """
    written = []
    client = _FakeClient(results=[
        {"labels": [], "scores": {}},
        {"labels": ["运费"], "scores": {}},
        {"labels": [], "scores": {}},
    ])
    stats = await classify_batch(
        [{"id": 1, "question": "a"}, {"id": 2, "question": "b"}, {"id": 3, "question": "c"}],
        client=client, write=written.append, batch_size=8,
    )
    # ① 空标签**照写**(不是失败)
    assert [r["low_confidence_question_id"] for r in written] == [1, 2, 3]
    assert written[0]["labels"] == []
    # ② 而且数得出来(计数落空只会在这一条上红)
    assert stats["empty_labels"] == 2, f"空标签应当数出 2 条,实际 {stats}"
    assert stats["empty_ids"] == [1, 3], f"空标签的行号也要留痕,实际 {stats}"


# --------------------------------------------------------------------------
# 真 HTTP 客户端(订正 15-B) —— 用假 transport,**不联网**
# --------------------------------------------------------------------------


def _client_with(handler) -> TopicServiceClient:
    """把一个假 transport 挂进真客户端。

    用 `httpx.MockTransport` 而不是 monkeypatch `post`:被测的是**这个类**
    (它才是生产上唯一发请求的地方),换掉它的 transport 才叫真的走了它一遍。
    """
    return TopicServiceClient("http://svc.test", transport=httpx.MockTransport(handler))


@pytest.mark.anyio
async def test_real_client_unwraps_the_results_key():
    """真客户端负责拆 `{"results": [...]}` —— 拆 **一个**键,别的什么都不猜。

    它今天只在**服务侧**被钉(`tests/test_topic_service.py::test_predict_endpoint_…`),
    客户端侧一个字都没有 ⇒ 两边各写一个键名而**没有任何东西会报错**
    (本仓「同一个词两套行为」家族)。这条把那半个契约钉死在客户端这一侧。
    """

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/predict"
        return httpx.Response(200, json={"results": [{"labels": ["尺码"], "scores": {}}]})

    async with _client_with(handler) as client:
        out = await client.predict(["买大了"])
    assert out == [{"labels": ["尺码"], "scores": {}}]


@pytest.mark.anyio
async def test_real_client_fails_loudly_when_the_key_is_missing():
    """形状漂移(键改名 / 外层不是对象)必须**响亮失败**,不许 `payload.get(...)` 兜底。

    判别力:一个写成 `payload.get("results", [])` 或 `payload` 的实现,
    在下面两条上分别**静默返回空**/**返回 dict**,而调用方会把「空」当成「没有主题」
    —— 那正是本节 15-C 单列出来的那种事故。
    """

    def missing_key(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"predictions": []})

    def not_an_object(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"labels": []}])

    async with _client_with(missing_key) as client:
        with pytest.raises(TopicServiceProtocolError):
            await client.predict(["x"])

    async with _client_with(not_an_object) as client:
        with pytest.raises(TopicServiceProtocolError):
            await client.predict(["x"])


@pytest.mark.anyio
async def test_real_client_raises_on_a_non_2xx():
    """5xx 不许被读成「预测结果为空」—— 基础设施故障一律向上抛(spec §9.2)。"""

    def boom(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="service unavailable")

    async with _client_with(boom) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.predict(["x"])


# --------------------------------------------------------------------------
# `model_version`(订正 15-A) —— 那一列是 NOT NULL,而计划里一个字都没写
# --------------------------------------------------------------------------

_META = {
    "base": "hfl/chinese-roberta-wwm-ext",
    "data_fingerprint": "74bd7d056f68b68c",
    "trained_at_utc": "2026-09-26T17:20:54.214324+00:00",
}


def test_model_version_is_base_plus_run_stamp():
    assert model_version_from_meta(_META) == (
        "hfl/chinese-roberta-wwm-ext@74bd7d056f68b68c::2026-09-26T17:20:54.214324+00:00"
    )


def test_model_version_separates_two_trainings_with_the_same_data_fingerprint():
    """**这一条就是「不能单用 `data_fingerprint`」的全部理由。**

    `data_fingerprint` 是**语料**的、**一次运行**的指纹(T8 复审实测增强产物不可复现,
    见 `train_meta.json` 的 `data_fingerprint_note`)。⇒ 同一份语料训两次
    —— 换 `base`、换超参、换个 `seed`,甚至只是重跑 —— **可以同值**,
    而「这个结果是谁算的」正是这一列要分的东西。

    判别力:一个只返回 `meta["data_fingerprint"]` 的实现,下面第一条断言当场红。
    """
    other = dict(_META, base="hfl/chinese-macbert-base",
                 trained_at_utc="2026-09-27T03:00:00.000000+00:00")
    assert model_version_from_meta(_META) != model_version_from_meta(other)
    # 指纹相同 —— **前提**,否则上面那句可能只是「两个值本来就不一样」
    assert _META["data_fingerprint"] == other["data_fingerprint"]


def test_model_version_fits_the_column():
    """列宽 128。塞整个 dict / 塞原始 meta 文本会**静默截断**(MySQL 非严格模式下)
    或响亮 1406 —— 而这是每一行都要写的一列。"""
    version = model_version_from_meta(_META)
    assert len(version) < 128, f"列宽 128,实际 {len(version)}"


def test_model_version_refuses_a_meta_without_the_three_fields():
    """缺字段 ⇒ **抛**,不许拼出一个 `None@None::None` 那样的值。

    ⚠️ 「拼出 None」比报错坏得多:它是**合法字符串**,会照常落库、照常让分布页
    显示一个假的版本号,而没有任何东西会报错(ch09 「静默无效」家族的同款形状)。
    """
    for missing in ("base", "data_fingerprint", "trained_at_utc"):
        truncated = {k: v for k, v in _META.items() if k != missing}
        with pytest.raises(ValueError):
            model_version_from_meta(truncated)
    with pytest.raises(ValueError):
        model_version_from_meta(None)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# 落库与编排(`db` 标记 —— 真 MySQL)
#
# ⚠️ 本节一律**按 id 过滤**(订正 15-E):`low_confidence_questions` 与
# `topic_classifications` 都是**共享且只追加**的表,「查最近这几条」会被上次运行
# 留下的行污染 ⇒ 偶尔红、偶尔绿(本仓编目过的那一类)。
# --------------------------------------------------------------------------

pytestmark_db = pytest.mark.db

#: 探针 id 用高位段,不撞真实数据(池子今天是**很小**的自增值)。
#: ⚠️ **不挂外键**(ORM docstring 记过),所以这个 id 不必真的在池子里存在。
PROBE_ID = 999002


class _EchoClient:
    """每条输入都给一个「其他」的假服务(`run()` 那一层的替身)。"""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def predict(self, texts):
        self.calls.append(list(texts))
        return [{"labels": ["其他"], "scores": {"其他": 0.9}} for _ in texts]


@pytestmark_db
@pytest.mark.anyio
async def test_dry_run_never_touches_the_writer(monkeypatch):
    """`--dry-run` 只打印,不落库。

    ⚠️ **判据不是「库里条数没变」** —— 那个断言有洞:重跑一条**已经归类过**的池子行
    走的是 upsert,条数本来就不会变(Step 6 跑完之后池子里**每一行**都已归类,
    那条件就恒真了)。真正要断的是**写口压根没被调用**,
    所以这里把写口换成**记录器**(照 `tests/conftest.py` 那三处钩子的做法:
    替身要替得出证据,不能只是 `lambda: None`)。

    池子是**真的读**(`--limit 1` 走真 SQL),只有写口与两个读数被换掉 ——
    这样「`dry_run` 被忽略掉、照写」这件事仍然会在这里红。

    ⚠️ 反面那一半(`dry_run=False` ⇒ 写口必须被调一次)不是凑数:
    没有它的话,「写口没被调」可能只是**装置自己坏了**(spy 压根没接上),
    而那是本仓变异检查里反复抓到的那种假绿。
    """
    from scripts import classify_topics as mod

    calls: list = []

    async def spy_upsert(session, rows, model_version):
        calls.append((list(rows), model_version))
        return len(rows)

    async def fake_counts(session):
        return 0, 0

    monkeypatch.setattr(mod, "upsert_classifications", spy_upsert)
    monkeypatch.setattr(mod, "distribution_numbers", fake_counts)

    summary = await run(client=_EchoClient(), model_version="probe-dry",
                        limit=1, dry_run=True)
    assert summary["classified"] == 1, f"前提:池子里至少要读到 1 行,实际 {summary}"
    assert summary["written"] == 0
    assert calls == [], f"dry-run 不许调用写口,实际被调了 {len(calls)} 次"

    live = await run(client=_EchoClient(), model_version="probe-live",
                     limit=1, dry_run=False)
    assert len(calls) == 1, "非 dry-run 时写口必须恰好被调一次(否则上面那条是无齿的)"
    assert calls[0][1] == "probe-live", f"写口拿到的 model_version 不对:{calls[0][1]!r}"
    assert live["written"] == 1


async def _purge(session) -> None:
    """删掉本文件的探针行并**提交**(不提交的话下一轮会看到残留)。"""
    await session.execute(
        delete(TopicClassification).where(
            TopicClassification.low_confidence_question_id == PROBE_ID
        )
    )
    await session.commit()


async def _probe_rows() -> list[TopicClassification]:
    """**新 session** 读回(同 session 重读是否打到库要靠身份映射的 refcount 走运)。"""
    async with get_sessionmaker()() as session:
        found = await session.execute(
            select(TopicClassification).where(
                TopicClassification.low_confidence_question_id == PROBE_ID
            )
        )
        return list(found.scalars().all())


@pytestmark_db
@pytest.mark.anyio
async def test_upsert_overwrites_instead_of_appending():
    """写口是 `INSERT … ON DUPLICATE KEY UPDATE`:重跑**覆盖**,不是追加。

    T12 在**表**那一层钉过唯一键;这条钉的是**写口用的那条语句** ——
    唯一键在、而写口写成裸 `INSERT`(或 ORM 的 `session.add`)时,
    重跑会**响亮地 1062**,而我们真正要的性质是「静默地覆盖」。
    判别力:去掉 `on_duplicate_key_update` ⇒ 第二次写抛 `IntegrityError` ⇒ 红。

    ⚠️ 一律**按 id 过滤**(15-E):`low_confidence_questions` 与
    `topic_classifications` 都是共享且只追加的表,「查最近这几条」会被上次运行污染。
    """
    engine = get_engine()
    try:
        async with get_sessionmaker()() as session:
            await _purge(session)
            written = await upsert_classifications(
                session,
                [{"low_confidence_question_id": PROBE_ID,
                  "labels": ["尺码"], "scores": {"尺码": 0.9}}],
                "probe-v1",
            )
            await session.commit()
        assert written == 1
        first = await _probe_rows()
        assert len(first) == 1, f"前提:第一行应当落库,实际 {len(first)}"
        assert first[0].labels == ["尺码"]
        assert first[0].model_version == "probe-v1"

        async with get_sessionmaker()() as session:
            await upsert_classifications(
                session,
                [{"low_confidence_question_id": PROBE_ID,
                  "labels": ["运费", "退换货"], "scores": {"运费": 0.7}}],
                "probe-v2",
            )
            await session.commit()

        after = await _probe_rows()
        assert len(after) == 1, (
            f"同一池子行只许有一行,实际 {len(after)} 行 —— "
            "写口不是 upsert(重跑在旁边静静堆了第二份结果,分布页条数会跟着翻倍)"
        )
        assert after[0].labels == ["运费", "退换货"], f"实际 {after[0].labels!r}"
        assert after[0].model_version == "probe-v2", f"实际 {after[0].model_version!r}"
    finally:
        async with get_sessionmaker()() as session:
            await _purge(session)
        await engine.dispose()


def test_the_pool_statement_is_ordered_by_id():
    """`--limit N` 的语义是「最旧的 N 条」,而那**只有**在语句有序时才成立。

    ⚠️ **这条是语句字面断言,不是行为断言 —— 理由必须写清楚**(本仓有先例:
    ch09 那条「`app/` 下有没有人 import langfuse」也是源码扫描,判据同样是
    「有没有人写下这一行」)。

    为什么行为断言在这里是**瞎的**:探测过(M12 变异,见 T13 报告)——
    `load_pool` 去掉 `order_by` 之后,真库上 `test_the_pool_is_read_whole_…`
    **照样绿**,因为 InnoDB 对这条查询恰好按主键返回。
    ⇒ 拿「读回来的 id 是不是升序」当判据,测的是**今天的存储引擎**,
    不是**我们的契约**;而契约会随着加索引 / 换版本**悄悄失效**。

    编译成 MySQL 方言的 SQL 再看 —— 断的是**真的被发出去的那条语句**。
    """
    from sqlalchemy.dialects import mysql

    from scripts.classify_topics import pool_statement

    for limit in (None, 3):
        sql = str(pool_statement(limit).compile(dialect=mysql.dialect()))
        assert "ORDER BY low_confidence_questions.id" in sql, (
            f"读池的语句没有 ORDER BY id:`--limit` 会取到任意 N 条。实际 SQL:{sql}"
        )


@pytestmark_db
@pytest.mark.anyio
async def test_the_pool_is_read_whole_not_just_the_unclassified():
    """`load_pool` 读的是池子**本身**,不是「还没归类过的那些」。

    这条钉一个**设计决定**:批处理每次跑**全池**,幂等交给唯一键 ——
    而不是「挑出还没归类的再算」。后者需要一个「算过没有」的谓词,而本表**没有**
    「状态」列(它是结果表,不是状态表,见 ORM docstring);真去 join 一遍的话,
    `--limit` 的语义会从「最旧的 N 条」悄悄变成「最旧的、且还没算过的 N 条」,
    而**没有任何东西会报错**。
    """
    engine = get_engine()
    async with get_sessionmaker()() as session:
        all_rows = await load_pool(session, None)
        limited = await load_pool(session, 3)
    assert limited == all_rows[:3], f"--limit 断的不是前 3 条:{limited}"
    assert [r["id"] for r in all_rows] == sorted(r["id"] for r in all_rows), (
        "`ORDER BY id` 没了的话,`--limit` 取的就不是最旧的几条"
    )
    await engine.dispose()
