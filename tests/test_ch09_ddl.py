"""db/ch09.sql 的形状 —— 只读文件比对,**全程不连库**。

这些断言守的是 spec §7.1 里**逐条写明**的设计选择;它们变红时先回去读 spec,
别顺手改成"看起来更合理"的样子。

文件下半部分同时**编译 ORM 的 metadata**(仍然不连库,只是把 `CreateTable` 渲染成
MySQL 方言的字符串):两条建库路径(`scripts/init_db.py` 的 create_all / 这份 DDL)
在**列级 DEFAULT** 与**索引覆盖**上必须一致 —— 不一致的后果是「一条省略 status /
occurrences 的裸 INSERT 只在一条路径上成功」,而那种缝**不会自己报错**。
"""

import re
from pathlib import Path

from sqlalchemy.dialects import mysql
from sqlalchemy.schema import CreateTable
from sqlalchemy.sql.schema import Table

from app.db.models import EvalRun, LowConfidenceQuestion, ReviewQueue

SQL = Path(__file__).resolve().parents[1] / "db" / "ch09.sql"


def _sql() -> str:
    return SQL.read_text(encoding="utf-8")


def _sql_without_comments() -> str:
    """剥掉 `--` 注释后的 SQL。

    为什么需要它:本仓的头号风险是**假绿**。下面 `test_two_new_columns` 那条
    断的是「列名出现在整个文件里」,而这份 DDL 的注释**本来就要写**这些列名
    (要解释 matched_review_id 的双语义)⇒ 列名留在注释里、`ADD COLUMN` 被删掉,
    那条断言**照样绿**(实测过一次)。凡是「这个构造真的存在吗」这类断言,
    一律看剥掉注释的原文。

    (本文件的 SQL 里没有任何字符串字面量含 `--`,所以按行裁是安全的。)
    """
    return "\n".join(line.split("--", 1)[0] for line in _sql().splitlines())


def _statements() -> list[str]:
    """按 `;` 切成「真正会执行的一条 SQL」—— **字符串字面量里的 `;` 不算分隔符**。

    为什么不能裸切(`body.split(";")`):这份 DDL 的 COMMENT 里**就有分号**
    (`'归并到的 review_queue.id;NULL = 尚未进流水线'`,照 spec §7.1 逐字抄的)。
    裸切会把 ALTER 从那个分号处劈成两半 ⇒ **正确的 DDL 也会让断言变红**。
    (这个坑我第一轮应用 DDL 时就踩过:报 1064,而那半句 SQL 看起来像编码问题。
    工具比被测对象更容易错 —— 这里按引号切,`''` 转义也处理。)
    """
    body = _sql_without_comments()
    out: list[str] = []
    buf: list[str] = []
    in_str = False
    i = 0
    while i < len(body):
        ch = body[i]
        if in_str:
            if ch == "'":
                if body[i + 1 : i + 2] == "'":
                    buf.append("''")
                    i += 2
                    continue
                in_str = False
            buf.append(ch)
        elif ch == "'":
            in_str = True
            buf.append(ch)
        elif ch == ";":
            out.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
        i += 1
    out.append("".join(buf).strip())
    return [s for s in out if s]


def _only_statement(prefix: str) -> str:
    """取唯一一条以 `prefix` 开头的语句,并断言它**恰好**一条。"""
    hits = [s for s in _statements() if s.upper().startswith(prefix.upper())]
    assert len(hits) == 1, f"{prefix} 应当恰好一条,实际 {len(hits)} 条"
    return hits[0]


def _orm_ddl(model: type) -> str:
    return str(CreateTable(model.__table__).compile(dialect=mysql.dialect()))


def _indexed_columns(table: Table) -> set[str]:
    return {c.name for i in table.indexes for c in i.columns}


# ---------------------------------------------------------------- DDL 文件本身


def test_file_exists_and_is_not_idempotent_by_design():
    s = _sql()
    assert "ALTER TABLE low_confidence_questions" in s
    assert "IF NOT EXISTS" not in s, "不幂等是刻意的(重复执行要响亮地失败)"


def test_two_new_columns():
    s = _sql()
    assert "evidence_snapshot" in s and "JSON" in s
    assert "matched_review_id" in s


def test_two_new_columns_are_really_added_by_the_alter():
    """补上一条的**判别力**:列名必须出现在 `ALTER TABLE` 这条语句的 `ADD COLUMN` 里。

    只断「文件里出现过列名」的话,把 `ADD COLUMN` 整句删掉、注释留着,断言仍绿。
    顺带钉住可空性:`matched_review_id` 的 `NULL` 是**有含义的值**
    (NULL = 尚未进流水线),写成 NOT NULL 就没有「未处理」这个状态了。
    再钉住流水线唯一热路径上的索引(`WHERE matched_review_id IS NULL`)。
    """
    alter = _only_statement("ALTER TABLE LOW_CONFIDENCE_QUESTIONS")
    assert re.search(r"ADD COLUMN\s+evidence_snapshot\s+JSON\s+NULL", alter), alter
    assert re.search(
        r"ADD COLUMN\s+matched_review_id\s+BIGINT\s+UNSIGNED\s+NULL", alter
    ), alter
    assert re.search(r"ADD KEY\s+idx_matched_review\s*\(\s*matched_review_id\s*\)", alter), alter


def test_review_queue_has_no_unique_key_on_standard_question():
    s = _sql()
    assert "CREATE TABLE review_queue" in s
    assert "UNIQUE" not in s.upper(), (
        "查重是语义判断,唯一键管不了 —— 加了会在合理的语义归并上 1062"
    )


def test_no_unique_key_even_after_stripping_comments():
    """同一条规矩换个口径再钉一次:剥掉注释后仍然没有唯一键。

    上面那条是**全文**检查,连注释里出现这个词都会让它变红 —— 所以它其实
    同时约束了注释的写法。这条剥掉注释,守的是**真会被执行的那份 SQL**。
    """
    assert "UNIQUE" not in _sql_without_comments().upper()


def test_eval_runs_shape():
    s = _sql()
    for col in ("trigger_by", "case_count", "metrics", "created_at"):
        assert col in s


def test_eval_runs_columns_are_in_the_create_statement():
    """与上面那条同款硬化:列名必须在 `CREATE TABLE eval_runs` 这条语句里。

    (只断「文件里出现过」的话,把整条 CREATE 删掉、注释里留个 `metrics` 就能骗过去。)
    """
    create = _only_statement("CREATE TABLE EVAL_RUNS")
    for col in ("trigger_by", "case_count", "metrics", "created_at"):
        assert re.search(rf"\b{col}\b", create), f"{col} 不在 CREATE TABLE eval_runs 里"


def test_review_queue_column_defaults_are_in_the_create_statement():
    """`occurrences` / `status` 的**列级 DEFAULT** 必须真的写在 CREATE 里。

    少了它,一条**省略这两列的裸 INSERT** 在严格模式下会失败 —— 而那是 ORM 与 DDL
    两条建库路径之间最容易长出来的一条缝(`default=` 只是 Python 侧默认,不是列级 DEFAULT)。
    """
    create = _only_statement("CREATE TABLE REVIEW_QUEUE")
    assert re.search(r"occurrences\s+INT\s+NOT NULL\s+DEFAULT\s+1", create), create
    assert re.search(r"status\s+VARCHAR\(16\)\s+NOT NULL\s+DEFAULT\s+'pending'", create), create


# ------------------------------------------------- ORM metadata(仍不连库,只编译)


def test_orm_declares_the_same_column_defaults_as_the_ddl():
    """ORM 侧也要有**列级 DEFAULT**(`server_default=`),两侧都要。

    本仓先例:`ToolAuditLog.retry_count` / `duration_ms`。
    最后那条 `'''` 是**反向陷阱**:字符串默认值写成 `server_default="'pending'"`
    会被渲染成 `DEFAULT '''pending'''` —— 数据库里存成「带引号的值」,而它看起来像对的。
    """
    orm = _orm_ddl(ReviewQueue)
    assert "DEFAULT '1'" in orm, orm
    assert "DEFAULT 'pending'" in orm, orm
    assert "'''" not in orm, orm


def test_index_coverage_matches_between_the_two_paths():
    """两条路径的索引**覆盖**必须一致(名字不同是 ch08 记过的已知差异)。

    create_all 自动生成 `ix_*`,DDL 手写 `idx_*` —— 那处差异不影响行为;
    但「一边有、另一边没有」会让查询计划变成「看谁建的库」。
    池子那一列尤其要紧:流水线的选择谓词就是 `WHERE matched_review_id IS NULL`。
    """
    alter = _only_statement("ALTER TABLE LOW_CONFIDENCE_QUESTIONS")
    assert re.findall(r"ADD KEY\s+idx_matched_review\s*\(\s*(\w+)\s*\)", alter) == [
        "matched_review_id"
    ]
    assert _indexed_columns(ReviewQueue.__table__) == {"status"}
    assert _indexed_columns(EvalRun.__table__) == {"created_at"}
    assert "matched_review_id" in _indexed_columns(LowConfidenceQuestion.__table__)
    assert (
        re.search(
            r"KEY\s+idx_status\s*\(\s*status\s*\)", _only_statement("CREATE TABLE REVIEW_QUEUE")
        )
        is not None
    )
    assert (
        re.search(
            r"KEY\s+idx_created\s*\(\s*created_at\s*\)",
            _only_statement("CREATE TABLE EVAL_RUNS"),
        )
        is not None
    )
