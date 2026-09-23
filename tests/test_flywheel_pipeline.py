"""飞轮编排:幂等、命中累加、逐行失败不拖垮整批。

打**真库**,理由与 `tests/test_api_feedback.py` 同款 —— 本文件要钉的三件事,
替身 session 一件都证不出来:

① **幂等**靠的是库上的谓词 `WHERE matched_review_id IS NULL`。替身 session 会
   **自己把那条 SELECT 答了**(本仓编目过的假绿形态 (g):被测对象把这件事委托给了
   替身),于是「把谓词整个删掉」的实现在替身上照样绿;
② **命中累加**的 `occurrences` 是被 UPDATE 出来的,**读回来**才算数;
③ **逐行失败不拖垮整批**断的是「那一行的 `matched_review_id` 在**库里**仍是 NULL」
   —— 那是个持久化状态,不是内存里的一个变量。

## ⚠️ 池子不是空的:`ORDER BY id` 把本文件的设计钉死了

实测 2026-09-24:`low_confidence_questions` 里有 **30 行**前几章演示/验收留下的旧行
(ids 10–305,全部 `matched_review_id IS NULL`,`entry_point` 是 `置信度闸`/`生成自评`),
而流水线的选择谓词是 `WHERE matched_review_id IS NULL ORDER BY id LIMIT batch_size`
—— **升序,先吃旧的**。⇒ 「往池子里插几行再跑一批」这种最自然的写法会**根本碰不到
自己的行**(批被那 30 行占满);而把批开大到盖住它们,又会**改写不属于本用例的行**
(给别人的行填 `matched_review_id`、替它们建 `review_queue`)。那些行是**别人的数据**,
不许删、也不该被本文件动一下。

**解法:占用 id 1–9 这段空号。** 池子当前的最小 id 是 10 ⇒ 1–9 是空的。**显式指定
主键**插入探针行,它们就是全表最小的 id ⇒ `ORDER BY id LIMIT n` 第一个吃到的就是
它们。整批只碰自己的行。插入走裸 SQL 而不是 ORM:ORM 那条路会给主键留空、由
AUTO_INCREMENT 分配,拿不到「id 最小」这个性质。

⚠️ **这段号只有 9 个,而用例是按 `batch_size` 吃号的** ⇒ 各用例的号段**互相重叠**
(都从 1 开始,吃几行用几个)。重叠是安全的:每个用例**开头**都会先把 `t13probe-`
前缀的行全删掉(上一次崩在断言中间留下的半批数据就是这么清掉的)。谁要把某个用例
扩到一次吃 9 行以上,这条隔离就破了 —— 那时应当先扩这段号,并重新复核 `_cleanup`。

这条机制**自带前置断言**(`_batch_ids`):跑之前先按**生产的同一个谓词**取一批,
确认里面只有本用例的探针行。没有它的话,哪天池子里冒出一行 id 更小的数据,用例会
**静默地**去处理那一行并随机地红/绿 —— 正是本仓「被上次运行的数据污染」那一类。

## 替身为什么长这样

`FakeModel` 替的是**模型边界**(`with_structured_output(...).ainvoke`),不是流水线内部:
异常从**模型那一侧**抛出去,流水线那句 `except` 才真的被行使 —— 本仓的头号假绿形态
是「注入**处理之后**的值」,那样被验的就不是那一句了。它按提示词文本作答(与真模型
同一件事):标准化那一步从提示词里认出**本用例已知的那句原话**(认不出就报错 —— 那
说明流水线没把问题喂给模型),查重那一步把编号条目里**文案与候选完全相同**的那条
认成同义(候选会被原样写进提示词,所以同义命中时那段文本出现**两次**)。

九条用例里**四条打真库**(第二遍不重吃、命中累加、逐行不拖垮、失败留痕),另外**五条
不读库**(标准化的两个退化分支 + 查重的三个退化分支)—— 但那五条也带 db 标记:
本文件的 `pytestmark` 是模块级的(brief 指定的形状)。它们只是**被标记**,不读库。
"""

import logging
import re

import pytest
from langchain_core.exceptions import OutputParserException
from sqlalchemy import text

from app.db.base import get_engine, get_sessionmaker
from app.db.models import ReviewQueue
from app.flywheel.dedupe import DedupeResult, find_duplicate
from app.flywheel.normalize import NormalizeResult, normalize_question
from app.flywheel.pipeline import run_flywheel

pytestmark = pytest.mark.db

#: 探针会话前缀(`t13probe-`)。**结构上**撞不到真实数据:库里的真实会话 id 是
#: `uuid4().hex` 与 `acceptance-*`,别的测试用的是 `t11probe-` / `t12probe-`。
PROBE_PREFIX = "t13probe-"


# --------------------------------------------------------------------------
# 模型替身
# --------------------------------------------------------------------------


_ENTRY_RE = re.compile(r"^\[(\d+)\]\s*(.+)$", re.MULTILINE)


class _Structured:
    """`with_structured_output(...)` 的返回值(真链路上是一个可 await 的链)。"""

    def __init__(self, owner: "FakeModel", schema):
        self._owner = owner
        self._schema = schema

    async def ainvoke(self, messages):
        # 最后一条是 human 消息,提示词就在它的 content 里(与真链路同款)。
        return await self._owner.answer(self._schema, messages[-1].content)


class FakeModel:
    """模型替身。

    - `known`:本用例**已知的原话**。标准化那一步靠它把原话从提示词里认出来;
      一句都认不出 ⇒ **报错**(那说明流水线没把这一行的问题喂给模型 —— 是个真缺陷,
      不该被替身悄悄兜住)。
    - `canon`:原话 → 标准化结果。缺省时原样透传(不关心标准化的用例不必填)。
    - `answers`:原话 → 示例答案(缺省空串)。
    - `exc` + `fail_on`:`fail_on` 出现在提示词里就让**标准化**抛 `exc`;
      `fail_on` 为 `None` 则无条件抛(解析失败那两条用例用)。
    - `dedupe_exc`:让**查重**抛这个异常。
    - `dedupe_index`:查重固定返回的编号;`None` ⇒ 按提示词里的列表真判(见类文档)。
    """

    def __init__(self, *, known=(), canon=None, answers=None, exc=None, fail_on=None,
                 dedupe_exc=None, dedupe_index=None):
        self.known = list(known)
        self.canon = dict(canon or {})
        self.answers = dict(answers or {})
        self.exc = exc
        self.fail_on = fail_on
        self.dedupe_exc = dedupe_exc
        self.dedupe_index = dedupe_index
        #: 每次调用记一笔 —— 「模型到底被叫了几次、叫的是哪句」要能回查。
        self.calls: list[tuple] = []

    def with_structured_output(self, schema, *, method=None):
        # 生产两处都是 `method="json_mode"`(json_mode 是本项目的硬约束,
        # spec §2.3)。替身收窄成**与生产一样**的调用形状:哪天实现改成别的 method,
        # 这里当场炸,而不是让「换掉了结构化输出的模式」悄悄溜过去。
        assert method == "json_mode", f"生产用 json_mode,替身收到 {method!r}"
        return _Structured(self, schema)

    async def answer(self, schema, prompt: str):
        if schema is NormalizeResult:
            return self._normalize(prompt)
        if schema is DedupeResult:
            return self._dedupe(prompt)
        raise AssertionError(f"替身不认识这个 schema:{schema!r}")

    def _normalize(self, prompt: str) -> NormalizeResult:
        raw = next((q for q in self.known if q in prompt), None)
        if raw is None:
            raise AssertionError(
                f"提示词里认不出任何已知原话 —— 流水线没把这一行的问题喂给模型?"
                f"提示词:{prompt!r}"
            )
        self.calls.append(("normalize", raw))
        if self.exc is not None and (self.fail_on is None or self.fail_on in prompt):
            raise self.exc
        return NormalizeResult(
            standard_question=self.canon.get(raw, raw),
            example_answer=self.answers.get(raw, ""),
        )

    def _dedupe(self, prompt: str) -> DedupeResult:
        entries = {int(n): t.strip() for n, t in _ENTRY_RE.findall(prompt)}
        self.calls.append(("dedupe", len(entries)))
        if self.dedupe_exc is not None:
            raise self.dedupe_exc
        if self.dedupe_index is not None:
            return DedupeResult(match_index=self.dedupe_index)
        # 同义的判据:候选问题会被**原样**写进提示词,而列表里那条同义的文案与它
        # 逐字相同 ⇒ 那段文本在提示词里出现**两次**;只是话题相近的那些出现一次。
        for index, entry in entries.items():
            if prompt.count(entry) >= 2:
                return DedupeResult(match_index=index)
        return DedupeResult(match_index=-1)


# --------------------------------------------------------------------------
# 真库的小工具(全部按 `t13probe-` 前缀就地清理)
# --------------------------------------------------------------------------


async def _cleanup() -> None:
    """删本文件写过的**全部**探针行(两张表),并**删后**复查为 0。

    开头也调一次:上一次崩在断言中间会留下半批数据(池子里那几行还在、
    `matched_review_id` 已填),不先清掉的话,下一次跑的第一个断言就红在一个
    与被测行为无关的地方。
    """
    async with get_sessionmaker()() as session:
        for table in ("low_confidence_questions", "review_queue"):
            await session.execute(
                text(f"DELETE FROM {table} WHERE source_conversation_id LIKE :p"),
                {"p": f"{PROBE_PREFIX}%"},
            )
        await session.commit()
        for table in ("low_confidence_questions", "review_queue"):
            left = (
                await session.execute(
                    text(f"SELECT COUNT(*) FROM {table} WHERE source_conversation_id LIKE :p"),
                    {"p": f"{PROBE_PREFIX}%"},
                )
            ).scalar_one()
            assert left == 0, f"{table} 里探针行没删干净,还剩 {left} 行"


async def _insert_probes(rows: list[tuple[int, str, str]]) -> None:
    """按**显式主键**插探针行(文件头解释了为什么必须是这一段低 id)。

    `matched_review_id` 显式写 NULL:它是「尚未处理」的标记,而流水线的选择谓词
    正是 `WHERE matched_review_id IS NULL`(照 ORM 的语义,不靠列默认值)。
    """
    async with get_sessionmaker()() as session:
        for row_id, conversation_id, question in rows:
            await session.execute(
                text(
                    "INSERT INTO low_confidence_questions "
                    "(id, question, source_conversation_id, entry_point, "
                    " reject_reason, evidence_snapshot, matched_review_id) "
                    "VALUES (:id, :q, :c, '置信度闸', 't13 探针', NULL, NULL)"
                ),
                {"id": row_id, "q": question, "c": conversation_id},
            )
        await session.commit()


async def _batch_ids(limit: int) -> list[int]:
    """按**生产的同一个谓词**取一批 id —— 跑之前确认「这批里只有本用例的探针行」。

    这条前置断言守的是本文件赖以隔离的那条性质(探针行 id 最小)。它一旦不成立,
    整批会去处理别人的行,而断言会以「数字对不上」的形式随机地红 —— 报错指向
    被测代码,实际原因是数据。**前置**断言让这件事在跑之前就说清楚。
    """
    async with get_sessionmaker()() as session:
        return list(
            (
                await session.execute(
                    text(
                        "SELECT id FROM low_confidence_questions "
                        "WHERE matched_review_id IS NULL ORDER BY id LIMIT :n"
                    ),
                    {"n": limit},
                )
            ).scalars()
        )


async def _pool_rows(conversation_id: str) -> list[dict]:
    """回查探针池行的**持久化状态**(新 session:不经过身份映射)。"""
    async with get_sessionmaker()() as session:
        rows = (
            await session.execute(
                text(
                    "SELECT id, question, matched_review_id FROM low_confidence_questions "
                    "WHERE source_conversation_id = :c ORDER BY id"
                ),
                {"c": conversation_id},
            )
        ).mappings().all()
    return [dict(r) for r in rows]


async def _review_rows(conversation_id: str) -> list[dict]:
    """回查本用例归并出来的待审行。**按会话过滤** —— `review_queue` 是全库共享的
    (别的任务/别的用例都可能往里写),对总量断言会变成偶尔红偶尔绿。"""
    async with get_sessionmaker()() as session:
        rows = (
            await session.execute(
                text(
                    "SELECT id, standard_question, example_answer, occurrences, status, "
                    "first_raw_question FROM review_queue "
                    "WHERE source_conversation_id = :c ORDER BY id"
                ),
                {"c": conversation_id},
            )
        ).mappings().all()
    return [dict(r) for r in rows]


async def _teardown() -> None:
    """收尾:删探针行 → 复查为 0 → 关连接池(`dispose` 的理由见 test_api_feedback 同款注释:
    跨事件循环的 asyncmy 连接会打出一串无害但很吵的 `ERROR sqlalchemy.pool`)。"""
    await _cleanup()
    await get_engine().dispose()


# --------------------------------------------------------------------------
# 一、编排(真库)
# --------------------------------------------------------------------------


@pytest.mark.anyio
async def test_second_run_does_not_reprocess_already_processed_rows():
    """第二遍**不重吃已经处理过的行** —— 幂等完全由 `matched_review_id IS NULL` 单独保证。

    ⚠️ **为什么这里不是「第二遍 `processed == 0`」**(brief 给的用例名是
    `test_second_run_is_a_noop`):池子是共享的,而 `ORDER BY id LIMIT n` **先吃旧的**
    —— 第一遍吃完本用例的探针行之后,第二遍的批会被那 30 行旧数据填满。真要一个
    「空批」,只能把 `batch_size` 传 0(`LIMIT 0`),而那是个**恒真**断言:批为空时
    任何实现都处理 0 行,连「把谓词整个删掉」的实现也一样。所以这里换一个**同等**
    的观测,而且是**正面**的:

      六行探针(ids 1–6,全表最小)——
        第一遍 `batch_size=3` ⇒ 吃 **1、2、3**;
        第二遍 `batch_size=3` ⇒ **正确的实现**吃 **4、5、6**(下一批最小的未处理行,
                                 还是本用例自己的);
                                 **漏掉 `IS NULL` 谓词的实现**会重吃 **1、2、3**。

    于是「1、2、3 的 `occurrences` 与 `matched_review_id` 有没有被第二遍动过」就是
    那条不变量本身。⚠️ 只断「`review_queue` 行数不变」是**不够的**:重吃的实现会让
    1、2、3 命中它们**自己**已有的那三行(不建新行 ⇒ 行数一个不变),`occurrences`
    却从 1 变成 2 —— 计数虚高是它在库上唯一的痕迹。
    """
    conv = f"{PROBE_PREFIX}one"
    probes = [
        (1, conv, "t13 探针:猫砂盆 Pro 怎么退货"),
        (2, conv, "t13 探针:物流一般几天到"),
        (3, conv, "t13 探针:运费怎么算"),
        (4, conv, "t13 探针:发货时间要多久"),
        (5, conv, "t13 探针:货不对板怎么办"),
        (6, conv, "t13 探针:发票怎么开"),
    ]
    model = FakeModel(known=[q for _, _, q in probes])
    await _cleanup()
    try:
        await _insert_probes(probes)
        assert await _batch_ids(3) == [1, 2, 3], (
            "前置:这批必须只含本用例的探针行(id 1–3 是全表最小)"
        )

        async with get_sessionmaker()() as session:
            first = await run_flywheel(session=session, model=model, batch_size=3)
        assert first == {"processed": 3, "merged": 0, "created": 3, "failed": 0}, (
            f"第一遍应当三行全部新建,实际 {first}"
        )

        queue_after_first = await _review_rows(conv)
        pool_after_first = await _pool_rows(conv)
        assert len(queue_after_first) == 3, f"应当建出三行,实际 {queue_after_first}"
        assert [r["occurrences"] for r in queue_after_first] == [1, 1, 1]
        assert [r["matched_review_id"] for r in pool_after_first] == (
            [r["id"] for r in queue_after_first] + [None, None, None]
        ), (
            f"第一遍跑完前 1、2、3 行各指回自己那行待审行;4、5、6 还没轮到,实际 {pool_after_first}"
        )

        assert await _batch_ids(3) == [4, 5, 6], (
            "前置:第二遍该吃到的下一批**仍然是本用例的探针行**"
            "(否则这一遍会去动别人的行,断言会红在一个与被测行为无关的地方)"
        )
        async with get_sessionmaker()() as session:
            second = await run_flywheel(session=session, model=model, batch_size=3)

        assert second == {"processed": 3, "merged": 0, "created": 3, "failed": 0}, (
            f"第二遍该吃 4、5、6 ⇒ 三行新建;merged 不为 0 说明它**重吃了已处理的行**"
            f"(实际 {second})"
        )
        assert (await _pool_rows(conv))[:3] == pool_after_first[:3], (
            "第二遍不许动 1、2、3 的 matched_review_id"
        )
        assert (await _review_rows(conv))[:3] == queue_after_first, (
            "第二遍不许动已有的待审行(尤其 `occurrences` 不许从 1 涨上去 —— "
            "那正是「重吃」在库上唯一的痕迹)"
        )
    finally:
        await _teardown()


@pytest.mark.anyio
async def test_duplicate_hit_accumulates_occurrences_and_backfills_matched_review_id():
    """查重命中 ⇒ `occurrences` 累加、`matched_review_id` 回填,**不新建行**。

    两行**同义**的探针(模拟「同一个问题被两个入口各收了一次」—— `entry_point`
    不是池子的身份的一部分,所以它在池子里本来就会是两行,spec §7.1):

    - 第一行:**建**一行 `review_queue`(`occurrences=1`);
    - 第二行:命中第一行 ⇒ 累加到 2、回填 `matched_review_id`。

    ⚠️ 第二行能命中,**只因为流水线在循环里**每次重新查 `pending`。把那次 SELECT
    提到循环外(一个看起来很自然的优化)会让第二行看不见刚建的那一行 ⇒ 建出第二行
    ⇒ 这条用例红。这一条是本用例的主要判别力来源。
    """
    conv = f"{PROBE_PREFIX}two"
    q_first = "t13 探针:猫砂盆 Pro 保修多久啊"
    q_second = "t13 探针:那个猫砂盆 Pro 是保修几年"
    canonical = "t13 探针:猫砂盆 Pro 的保修期是多久"
    answer = "整机保修 1 年,非人为损坏可免费维修。"
    model = FakeModel(
        known=[q_first, q_second],
        canon={q_first: canonical, q_second: canonical},
        answers={q_first: answer},
    )
    await _cleanup()
    try:
        await _insert_probes([(1, conv, q_first), (2, conv, q_second)])
        assert await _batch_ids(2) == [1, 2], "前置:这批必须只含本用例的探针行"

        async with get_sessionmaker()() as session:
            stats = await run_flywheel(session=session, model=model, batch_size=2)

        assert stats == {"processed": 2, "merged": 1, "created": 1, "failed": 0}, (
            f"两行同义 ⇒ 一建一并,实际 {stats}"
        )

        rows = await _review_rows(conv)
        assert len(rows) == 1, f"同义的两行只该留下一行,实际 {len(rows)} 行:{rows}"
        assert rows[0]["standard_question"] == canonical, (
            "入队的必须是**标准化后**的问题,不是池子里的原话"
        )
        assert rows[0]["example_answer"] == answer, (
            "示例答案必须从标准化那一步搬过来(漏搬的话审核人拿到的是一行空答案)"
        )
        assert rows[0]["occurrences"] == 2, (
            f"命中必须**累加**到 2(新建一行会让它停在 1,而两行都在原地 —— "
            f"`review_queue` 行数看着也没错),实际 {rows[0]['occurrences']}"
        )
        assert rows[0]["status"] == "pending"
        assert rows[0]["first_raw_question"] == q_first, (
            "`first_raw_question` 记的是**第一次**落池的那句原话"
        )

        pool = await _pool_rows(conv)
        assert [r["matched_review_id"] for r in pool] == [rows[0]["id"]] * 2, (
            f"两行池子行都该指回同一行待审行(第二行靠的是**回填**),实际 {pool}"
        )
    finally:
        await _teardown()


@pytest.mark.anyio
async def test_one_row_failing_does_not_kill_the_batch():
    """模型对第 2 行抛异常 ⇒ 第 1、3 行照常入队,第 2 行的 `matched_review_id` 仍是 NULL。

    三个断言各守一件事:

    ① 第 1、3 行**真的进了队列**(不是「整批都没动」也能满足「第 2 行是 NULL」);
    ② 第 2 行在**库里**仍是 NULL ⇒ 它还是「待处理」,下一次跑会重试它。⚠️ 失败那行
       若是被填上任何值(哪怕是错的),它就**永远不会再被处理**,而没有任何东西报错;
    ③ **日志带 traceback**:失败原因不落日志的话,池子里那行只会「一直不消失」,
       没人知道为什么(spec 里那句「失败原因进日志」就是这个意思)。
    """
    conv = f"{PROBE_PREFIX}three"
    q1 = "t13 探针:第一行能过"
    q2 = "t13 探针:第二行让模型炸"
    q3 = "t13 探针:第三行也能过"
    model = FakeModel(
        known=[q1, q2, q3], fail_on=q2, exc=RuntimeError("模型炸了(替身注入)")
    )
    await _cleanup()
    try:
        await _insert_probes([(1, conv, q1), (2, conv, q2), (3, conv, q3)])
        assert await _batch_ids(3) == [1, 2, 3], "前置:这批必须只含本用例的探针行"

        async with get_sessionmaker()() as session:
            stats = await run_flywheel(session=session, model=model, batch_size=3)

        assert stats == {"processed": 3, "merged": 0, "created": 2, "failed": 1}, (
            f"一行失败不该拖垮整批,实际 {stats}"
        )

        pool = {r["id"]: r for r in await _pool_rows(conv)}
        assert pool[2]["matched_review_id"] is None, (
            "失败那行必须**留在池子里**(matched_review_id 仍是 NULL),否则它再也不会被重试"
        )
        assert pool[1]["matched_review_id"] is not None, "第 1 行应当照常入队"
        assert pool[3]["matched_review_id"] is not None, "第 3 行应当照常入队(失败在它之前)"

        rows = await _review_rows(conv)
        assert len(rows) == 2, f"只有两行该进队列,实际 {len(rows)} 行:{rows}"
        assert q2 not in [r["first_raw_question"] for r in rows], (
            "失败那行一个字都不该进队列"
        )
    finally:
        await _teardown()


@pytest.mark.anyio
async def test_failure_is_logged_with_a_traceback(caplog):
    """失败必须**响亮**留痕 —— 静默的失败会让池子里的行看起来「就是没人处理」。"""
    conv = f"{PROBE_PREFIX}three"
    q = "t13 探针:这一行会让模型炸"
    model = FakeModel(known=[q], exc=RuntimeError("模型炸了(替身注入)"), fail_on=q)
    await _cleanup()
    try:
        await _insert_probes([(1, conv, q)])
        with caplog.at_level(logging.WARNING, logger="app.flywheel.pipeline"):
            async with get_sessionmaker()() as session:
                stats = await run_flywheel(session=session, model=model, batch_size=1)

        assert stats["failed"] == 1
        loud = [
            rec for rec in caplog.records
            if rec.name == "app.flywheel.pipeline" and rec.levelno >= logging.WARNING
        ]
        assert loud, "失败必须留在日志里,否则「处理不了」与「还没轮到」在库上分不开"
        assert any(rec.exc_info for rec in loud), (
            "只写一句「失败了」不够 —— 不带 traceback 就查不出挂在哪一步"
        )
    finally:
        await _teardown()


# --------------------------------------------------------------------------
# 二、两个退化分支(纯逻辑,不读库;随模块被标成 db)
# --------------------------------------------------------------------------


@pytest.mark.anyio
async def test_normalize_falls_back_to_the_raw_question_when_parsing_fails():
    """解析失败 ⇒ **原话入库**,不抛。

    抛出去的话,流水线会把这一行记成 `failed`,而它下次还是同一句原话、还是同一个
    模型 —— 「模型偶尔吐了句废话」于是升级成「这条缺口永远处理不了」。原话没被整理,
    但**信息没丢**(审核人照样看得懂),这是两者里便宜且可逆的那一边。
    """
    raw = "t13 探针:这句原话本身就该被原样兜住"
    model = FakeModel(known=[raw], exc=OutputParserException("模型没吐 JSON"))

    out = await normalize_question(raw, model=model)

    assert out == {"standard_question": raw, "example_answer": ""}, (
        f"解析失败必须退化成原话(而不是抛、也不是空串),实际 {out}"
    )
    assert model.calls == [("normalize", raw)], "退化之前**确实调过模型**"


@pytest.mark.anyio
async def test_normalize_falls_back_when_the_model_returns_a_blank_question():
    """模型返回**空/纯空白** ⇒ 与解析失败同一类,退回原话。

    放行的话 `review_queue.standard_question` 会落一条空串:那一行在审核页上是
    **空白行**,审核人既看不出原话、也判不了(与 ch07「空梗概不许落库」同一条道理
    —— 有个占位的坏值比缺值更坏,因为它看起来是正常的一行)。`example_answer` 也
    必须一起被丢掉:留着它就成了「空问题 + 一个答案」的孤儿行。
    """
    raw = "t13 探针:模型这次一个字都没给"
    model = FakeModel(
        known=[raw], canon={raw: "   "}, answers={raw: "这句答案必须被丢掉"}
    )

    out = await normalize_question(raw, model=model)

    assert out == {"standard_question": raw, "example_answer": ""}, (
        f"空问题必须退化成原话,实际 {out}"
    )


@pytest.mark.anyio
async def test_dedupe_defaults_to_not_a_duplicate_when_parsing_fails():
    """查重解析失败 ⇒ **不匹配**(宁可多建一行,也不要把两个问题并成一个)。

    并错的代价是**永久**的:那两条缺口此后共用一行,`occurrences` 是两条的和,
    审核人删掉其中一条会把另一条一起删掉。多建一行的代价只是审核人多看一眼。
    """
    pending = [ReviewQueue(standard_question="t13 探针:已有问题", example_answer="",
                           first_raw_question="t13 探针:已有问题")]
    model = FakeModel(dedupe_exc=OutputParserException("模型没吐 JSON"))

    hit = await find_duplicate("t13 探针:候选问题", pending, model=model)

    assert hit is None, "解析失败必须按「不匹配」处理"
    assert model.calls == [("dedupe", 1)], "退化之前**确实比过一次**(不是压根没查)"


@pytest.mark.anyio
async def test_dedupe_returns_none_on_an_out_of_range_index():
    """越界编号 ⇒ 不匹配(而不是 `pending[index]` 硬取、也不是并到第 1 条上)。

    模型给出越界编号说明那次判断本来就不可信。硬取会 `IndexError`(被流水线记成
    该行失败 —— 一个「数错了行」升级成「这条缺口处理不了」),而「钳到列表边界」
    更坏:它会**静默**并到一条无辜的问题上。
    """
    pending = [ReviewQueue(standard_question="t13 探针:已有问题", example_answer="",
                           first_raw_question="t13 探针:已有问题")]
    model = FakeModel(dedupe_index=7)

    assert await find_duplicate("t13 探针:候选问题", pending, model=model) is None


@pytest.mark.anyio
async def test_dedupe_does_not_call_the_model_when_there_is_nothing_to_compare():
    """列表为空 ⇒ 直接返回 `None`,**一次模型都不调**。

    不只是省一次往返:空列表交给模型判,它会从一个「什么都没有」的提示词里被要求
    给出一个编号 —— 退化成一次没有依据的猜测。`calls` 断言就为这个闲着(去掉这个
    短路,这条红)。
    """
    model = FakeModel()

    assert await find_duplicate("t13 探针:候选问题", [], model=model) is None
    assert model.calls == [], f"空列表不该调模型,实际 {model.calls}"
