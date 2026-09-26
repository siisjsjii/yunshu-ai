"""标注相关的纯函数:去重、证据串校验、标签漂移、分层抽样。

末条(`test_collect_...`)不测纯函数,测的是 `collect` 的**接线** —— 它没有 DB、
没有网络,所以留在这一份里、也在 `not db` 套件里。
"""

import json

import pytest

from app.topic.labeling import dedupe_questions


def test_dedupe_keeps_first_occurrence_and_source():
    """重复问题只留一条 —— 且**保留先出现的那个来源**(来源优先级由调用方排序决定)。"""
    rows = [
        {"question": "退货怎么走", "source": "pool"},
        {"question": "退货怎么走", "source": "chat"},
    ]
    out = dedupe_questions(rows)
    assert len(out) == 1
    assert out[0]["source"] == "pool"


def test_dedupe_is_on_the_cleaned_text():
    """「退货  怎么走」与「退货　　怎么走」是同一句 —— 按原文去重会漏。

    ⚠️ **这条自证的是「机制对」,不是「今天差多少」。** 实测(2026-09-26):
    按清洗后去重与按原文去重,在当天的 1259 行真实语料上**只差 1 条**,
    且那 1 条的差异来自**脱敏**(两个不同订单号的问题都变成
    `订单 <订单号> 的物流到哪了`)—— **不是**空白/全角。那天库里同一句问题
    在各轮之间是**逐字节相同**的(空白变体一对都没有)。
    ⇒ 设计仍然对(清洗后才是真正要分类的文本),但**别把这个理由当实测事实引用**:
    它是「应该如此」的稳健性论证,不是「测得如此」。
    """
    rows = [
        {"question": "退货 怎么走", "source": "pool"},
        {"question": "退货　　怎么走", "source": "chat"},   # 全角空格
    ]
    assert len(dedupe_questions(rows)) == 1


def test_dedupe_drops_blank_after_cleaning():
    assert dedupe_questions([{"question": "   ", "source": "pool"}]) == []


def test_dedupe_of_empty_list_is_empty():
    assert dedupe_questions([]) == []


def test_collect_keeps_the_pool_row_and_the_cleaned_question(monkeypatch, tmp_path):
    """`collect` 的**接线**:来源优先级、洗后为空不写出、三个字段的形态。

    ⚠️ 上面四条只钉了 `dedupe_questions` 的**机制**(首次出现者胜),没钉 `collect`
    把三个来源按什么顺序喂进去。而顺序是**承重的**:实测(2026-09-26)
    **池子那 33 条问题全部与对话里的重复** ⇒ 顺序写错(或两条 DB 查询的 `source`
    写成同一个值),池子在分布里就**一条都不剩**(实测读数 `{'chat': 163}` /
    `{'pool': 163}`,两种错法各自丢掉另一边),**而单测与实跑都不会报错**。

    造数是三条**洗完之后同一句**的问题(空白不同、来源不同)+ 一条空串:
    ⇒ 只该写出一行,且它是**最先出现**的 `pool` 那条(不是 chat、不是 evalmd);
    ⇒ 空串那条不许写出去。
    全程不碰库、不碰网络:替身替掉 `_from_db`,`OUT` 指向 `tmp_path`。
    """
    import asyncio

    from scripts import prepare_topic_data as p

    async def fake_db():
        return [
            {"question": "退货 怎么走", "source": "pool"},
            {"question": "退货　　怎么走", "source": "chat"},
            {"question": "   ", "source": "pool"},          # 洗后为空 ⇒ 不许写出
        ]

    monkeypatch.setattr(p, "_from_db", fake_db)
    # ⚠️ 这一条的第一版写的是「退货!!!怎么走」—— 它洗完是 `退货!怎么走`(重复标点折成一个),
    #    **与 `退货 怎么走` 不是同一句** ⇒ 根本没走到「撞车」那条分支,断言当场红。
    #    判据是本仓那条:**构造输入前先算一遍它会不会走到那条分支**(制表符才行)。
    monkeypatch.setattr(
        p, "_from_testing_md", lambda: [{"question": "退货\t怎么走", "source": "evalmd"}]
    )
    out = tmp_path / "corpus.jsonl"
    monkeypatch.setattr(p, "OUT", out)

    asyncio.run(p.collect())

    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert lines == [
        {
            "id": "r-0001",
            "question": "退货 怎么走",
            "source": "pool",
            "provenance": "real",
        }
    ]


def test_from_db_gives_the_two_queries_different_sources(monkeypatch):
    """`_from_db` 的两条查询必须带上**不同**的 `source`(第一条 `pool`、第二条 `chat`)。

    ⚠️ 上面那条把 `_from_db` **整个替掉**了 ⇒ 它**看不见**这个缺陷。而本任务最值钱的
    一处发现正是它:brief 的代码块把两条查询都写成 `"pool"`(与同一份 brief 的 Interfaces
    `"pool"|"chat"|"evalmd"`、与「池子 > 对话」那条优先级自相矛盾)。
    实测(2026-09-26,**独立复算**):
      - 两条都记 `pool` ⇒ 分布变成 `{'pool': 163, 'evalmd': 299}`(chat **消失**,pool 虚高);
      - 顺序反过来(chat 先、标签不同)⇒ `{'chat': 163, 'evalmd': 299}`(pool **消失**)。
    两种错法都**不会报错**,而读数会骗人(池子 33 条全部与对话重复,所以它全靠标签才分得开)。

    替身替到**引擎**这一层(不是 `_from_db`),所以真函数体照跑;不改 SQL 文本、只看
    **第几次调用**,免得替身自己依赖表名。
    """
    import asyncio

    from scripts import prepare_topic_data as p

    class _Result:
        def __init__(self, rows):
            self._rows = rows

        def all(self):
            return [(r,) for r in self._rows]

    class _Conn:
        def __init__(self):
            self.calls = 0

        async def execute(self, _query):
            self.calls += 1
            return _Result([f"第{self.calls}条查询的行"])

    class _Ctx:
        def __init__(self, conn):
            self._conn = conn

        async def __aenter__(self):
            return self._conn

        async def __aexit__(self, *_exc):
            return False

    class _Engine:
        # ⚠️ 这里**是同步的** —— SQLAlchemy 的 `AsyncEngine.connect()` 返回的是
        # `AsyncConnection`(本身就是异步上下文管理器),**不是协程** ⇒
        # 真代码写的是 `async with eng.connect()`,不是 `async with await eng.connect()`。
        # 第一版替身写成 `async def connect` ⇒ `TypeError: 'coroutine' object does not
        # support the asynchronous context manager protocol`(替身替错了形状)。
        def connect(self):
            return _Ctx(_Conn())

        async def dispose(self):
            return None

    monkeypatch.setattr(p, "get_engine", lambda: _Engine())

    assert asyncio.run(p._from_db()) == [
        {"question": "第1条查询的行", "source": "pool"},
        {"question": "第2条查询的行", "source": "chat"},
    ]
