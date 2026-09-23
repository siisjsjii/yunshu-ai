"""db/ch09.sql 的形状 —— 只读文件,不连库。

这些断言守的是 spec §7.1 里**逐条写明**的设计选择;它们变红时先回去读 spec,
别顺手改成"看起来更合理"的样子。
"""

import re
from pathlib import Path

SQL = Path(__file__).resolve().parents[1] / "db" / "ch09.sql"


def _sql() -> str:
    return SQL.read_text(encoding="utf-8")


def _sql_without_comments() -> str:
    """剥掉 `--` 注释后的 SQL。

    为什么需要它:本仓的头号风险是**假绿**。下面 `test_two_new_columns` 那条
    断的是「列名出现在整个文件里」,而这份 DDL 的注释**本来就要写**这些列名
    (要解释 matched_review_id 的双语义)⇒ 列名留在注释里、`ADD COLUMN` 被删掉,
    那条断言**照样绿**。凡是「这个构造真的存在吗」这类断言,一律看剥掉注释的原文。

    (本文件的 SQL 里没有任何字符串字面量含 `--`,所以按行裁是安全的。)
    """
    return "\n".join(line.split("--", 1)[0] for line in _sql().splitlines())


def _statements() -> list[str]:
    """按 `;` 切开、去掉空语句 —— 粒度到「真正会执行的一条 SQL」。"""
    return [s.strip() for s in _sql_without_comments().split(";") if s.strip()]


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
    """
    alters = [
        s
        for s in _statements()
        if s.upper().startswith("ALTER TABLE LOW_CONFIDENCE_QUESTIONS")
    ]
    assert len(alters) == 1, f"应当恰好有一条 ALTER,实际 {len(alters)} 条"
    alter = alters[0]
    assert re.search(r"ADD COLUMN\s+evidence_snapshot\s+JSON\s+NULL", alter), alter
    assert re.search(
        r"ADD COLUMN\s+matched_review_id\s+BIGINT\s+UNSIGNED\s+NULL", alter
    ), alter


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
    creates = [
        s for s in _statements() if s.upper().startswith("CREATE TABLE EVAL_RUNS")
    ]
    assert len(creates) == 1, f"应当恰好有一条 CREATE TABLE eval_runs,实际 {len(creates)} 条"
    create = creates[0]
    for col in ("trigger_by", "case_count", "metrics", "created_at"):
        assert re.search(rf"\b{col}\b", create), f"{col} 不在 CREATE TABLE eval_runs 里"
