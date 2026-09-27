"""主题分布(ch10-B)—— 一个**只读**端点,给管理台的「主题分布」标签页用。

服务的是「先补哪块知识」这个决策,所以页面上的数必须能被追溯:
除了每类的条数,还要给出**不同问题数**与**最后归类时间**(图是旧的还是新的)、
以及**这个图是谁算的**(`model_versions`)。

## 两个数**口径不同**,别拿它们对不上当 bug(计划订正 17-A)

| 字段 | 数的是什么 | 真库读数 |
|---|---|---|
| `total` | **分类结果的条数**(`topic_classifications` 的行数) | 65 |
| `distinct_questions` | **池子里不同题面的条数**(`low_confidence_questions.question`) | 33 |

原稿把 `distinct_questions` 写成 `COUNT(DISTINCT low_confidence_question_id)`,
而那一列上有唯一键 `uk_pool_question` ⇒ 它与 `COUNT(*)` **恒等**:真库实测
**65 行 / 65 distinct**,而池子里只有 **33 条不同题面** ⇒ 页面会显示
「65 / **65**」,那个「不同问题数」是个**没意义的数**(T13 报的,controller 在库里
独立复核过)。改法是**回池子按题面数**(下面的 `LEFT JOIN` + `COUNT(DISTINCT q.question)`),
`summary` 与每个 bucket 的 `distinct` 两处都改。

⚠️ 那 33 是**题面**的不同数,**不是清洗后的**(`clean()` 是 Python 侧的唯一实现,
这一章已经为「清洗两处实现」付过一次账)⇒ 想按清洗后文本数,得在 SQL **之外**
用 `clean()` 再数一层。**别在 SQL 里假装洗过。**

⚠️ 为什么是 `LEFT JOIN` 而不是 `JOIN`:`low_confidence_question_id` **刻意不挂外键**
(见 `app/db/models.py` 的 ORM docstring),所以**悬空结果**(池子行被删了、结果行还在)
是真实可达的。`JOIN` 会把那些行**从 `total` 里悄悄抹掉**,而页面上的数看起来完全正常。

## 标签计数在 SQL 里展开(不是拉回 Python 再数)

`labels` 是 JSON 数组。`JSON_TABLE` 在本机 **MySQL 8.0.46** 上**实测可用**
(`.superpowers/probe_ch10_jsontable.py`,返回 `[('尺码', 1), ('退换货', 1)]`)——
**不是照文档推的**。反面做法是「把 `labels` 列拉回 Python 再 `Counter()`」:
本表今天 65 行,那个实现在演示规模下**与本实现逐位同结果**,区别要到数据量
上去之后才露出来(一次请求拉全表)。`tests/test_api_topics_db.py` 那条
`test_labels_are_expanded_inside_the_sql_by_json_table` 用一层**只记录**的
session 壳钉住「展开发生在 SQL 里」。

## 这个模块**不在实时对话的请求路径上**

它读的是离线批处理的结果表(`scripts/classify_topics.py` 那条旁路写的),
主链路零处调用分类器 —— 由 `tests/test_topics_boundary.py` 的源码扫描守着。
"""

from fastapi import APIRouter, Depends
from sqlalchemy import text

from app.auth import require_admin
from app.db.session import get_session
from app.topic.taxonomy import LABELS

#: 工作台守卫(require_admin)—— 理由见 `app/api/kb.py` 同一行。
router = APIRouter(dependencies=[Depends(require_admin)])

#: 每个类目的**条数**与**不同问题数**。
#:
#: ⚠️ `COUNT(*)` 是「这个标签出现了几次」——多标签行在**每个**它带的标签下各算一次,
#: 所以 `sum(每个桶的 count) >= total`(真库 79 vs 65)。别读成「行数」。
#: ⚠️ `COUNT(DISTINCT q.question)` **忽略 NULL**(悬空结果行没有题面),这正是我们要的。
_DISTRIBUTION = text("""
    SELECT jt.label AS label,
           COUNT(*) AS n,
           COUNT(DISTINCT q.question) AS distinct_n
    FROM topic_classifications t
    JOIN JSON_TABLE(t.labels, '$[*]' COLUMNS (label VARCHAR(64) PATH '$')) jt
    LEFT JOIN low_confidence_questions q ON q.id = t.low_confidence_question_id
    GROUP BY jt.label
    ORDER BY n DESC
""")

#: 总量那条。`LEFT JOIN` 的理由见模块 docstring(悬空结果仍进 `total`)。
_SUMMARY = text("""
    SELECT COUNT(*) AS total,
           COUNT(DISTINCT q.question) AS distinct_questions,
           MAX(t.classified_at) AS last_classified_at
    FROM topic_classifications t
    LEFT JOIN low_confidence_questions q ON q.id = t.low_confidence_question_id
""")

#: 「这个图是谁算的」。**不去重之后取一个**而是全列出来 —— 一次重训之后库里会
#: 同时躺着新旧两批结果,页面该看见「有两版混在一起」,而不是被代表成一个。
_VERSIONS = text(
    "SELECT DISTINCT model_version FROM topic_classifications ORDER BY model_version"
)


@router.get("/api/topics/distribution")
async def distribution(session=Depends(get_session)):
    """`GET /api/topics/distribution` —— 只读,不改任何东西。"""
    summary = (await session.execute(_SUMMARY)).one()
    rows = (await session.execute(_DISTRIBUTION)).all()
    versions = [r[0] for r in (await session.execute(_VERSIONS)).all()]
    total = int(summary.total or 0)

    buckets = [
        {"label": r.label, "count": int(r.n), "distinct": int(r.distinct_n)}
        for r in rows
    ]
    # ⚠️ 把**一个标签都没有的类目**也补上(count=0)。
    #    只返回有数据的类:页面上的条形图会「少几类」,而少的那几类看起来
    #    像「这一类没问题」—— 恰恰相反,它们是**一条样本都没有**。
    #    类目清单**从权威表来**(`app.topic.taxonomy.LABELS`),不是从已有数据来:
    #    另一份手写的清单就是漂移的开始(本仓已有先例)。
    seen = {b["label"] for b in buckets}
    buckets += [{"label": lb, "count": 0, "distinct": 0} for lb in LABELS if lb not in seen]
    # 排序:`count` 降序,同分按权威表的类目序(否则同分的桶顺序随查询计划变,
    # 页面每次刷新都长得不一样)。
    buckets.sort(key=lambda b: (-b["count"], LABELS.index(b["label"])))

    return {
        "total": total,
        "distinct_questions": int(summary.distinct_questions or 0),
        # 给**字符串**给页面(`None` = 一行结果都没有)。给 datetime 对象的话
        # 模板里会炸,而这里离页面很远。
        "last_classified_at": (
            summary.last_classified_at.isoformat(sep=" ", timespec="minutes")
            if summary.last_classified_at else None
        ),
        # 空列表 = 还没跑过批处理。
        "model_versions": versions,
        "buckets": buckets,
    }
