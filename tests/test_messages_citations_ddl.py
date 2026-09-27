"""`db/followup_messages_citations.sql` 的形状 —— 只读文件比对,**全程不连库**。

这份 DDL 不是一章的产物(它是「历史回载丢了两样东西」这个缺陷的修复),所以
**文件名刻意不带章号**:`db/ch11.sql` 会让人以为有一章 ch11 存在,而后来的
`init_db.py` 顺序说明会照着章号去排它。

文件下半部分同时**编译 ORM 的 metadata**(仍然不连库,只是把 `CreateTable`
渲染成 MySQL 方言的字符串),钉住两条建库路径在**这一列上**必须一致:

- `scripts/init_db.py` 的 `create_all` 会**顺带**把这一列建出来(全新库),
  这与 `db/ch08.sql` 的 `tool_audit_logs`、`db/ch09.sql` 的两列是同一个已知取舍;
- 而**已存在的库**,`create_all` 对已存在的表是**空操作**(本仓硬约束:
  「`init_db.py` 永不加列」)⇒ 只有这份 DDL 加得上。

两侧的可空性也必须一致:写成 `NOT NULL` 的话,「这一轮没有引用」就没有取值
(空数组 `[]` 与「没引用」不可区分,而 `NULL` 才有「这一轮没有」这个语义)。
"""

import re
from pathlib import Path

from sqlalchemy import JSON
from sqlalchemy.dialects import mysql
from sqlalchemy.schema import CreateTable

from app.db.models import MessageRecord

SQL = Path(__file__).resolve().parents[1] / "db" / "followup_messages_citations.sql"
TABLE = "messages"
COLUMN = "citations"


def _sql() -> str:
    return SQL.read_text(encoding="utf-8")


def _body() -> str:
    """剥掉 `--` 注释后的 SQL。

    为什么需要它:本仓的头号风险是**假绿**。若断言写成「列名出现在整个文件里」,
    把 `ADD COLUMN …` 整句删掉、注释里留着 `citations` 两个字,那条断言**照样绿**
    —— 而这份文件的注释**本来就要**写这个列名(要解释它为什么可空)。
    判据一律看**剥掉注释的原文**。

    (本文件的 SQL 里没有任何字符串字面量含 `--`,所以按行裁是安全的。)
    """
    return "\n".join(line.split("--", 1)[0] for line in _sql().splitlines())


def _statements() -> list[str]:
    """按 `;` 切成「真正会执行的一条 SQL」。

    这份 DDL 里**没有**字符串字面量(更不会有含 `;` 的 COMMENT)—— 与
    `db/ch09.sql` 不同,所以裸切是安全的;切成 0 条或 >1 条都会在下面的断言里
    现形(`_only_alter` 断言恰好一条)。
    """
    return [s.strip() for s in _body().split(";") if s.strip()]


def _only_alter() -> str:
    """取唯一一条 `ALTER TABLE` 语句,并断言它**恰好**一条。"""
    hits = [s for s in _statements() if s.upper().startswith("ALTER TABLE")]
    assert len(hits) == 1, f"ALTER TABLE 应当恰好一条,实际 {len(hits)} 条:{hits}"
    return hits[0]


# ---------------------------------------------------------------- DDL 文件本身


def test_file_is_not_a_chapter_numbered_ddl():
    """文件名**不带章号** —— 这不是一章的产物,占 `ch11` 会让后来的顺序说明
    以为有一章 ch11 存在。这条同时是「文件真的在那个路径上」的存在性断言。
    """
    assert SQL.is_file(), f"缺文件:{SQL}"
    assert not re.match(r"ch\d+", SQL.name), (
        f"这份 DDL 不属于任何一章,不该用章号命名:{SQL.name}"
    )


def test_file_is_not_idempotent_by_design():
    """**刻意不幂等**(与 db/ch03 / ch04 / ch06 / ch07 / ch08 / ch09 / ch10 同规矩)。

    重复执行要**响亮地失败**(1060:列已存在),静默跳过会让「列已存在但形状不对」
    永远补不上 —— 那是本仓记过的、只让某些功能悄悄失效却不报错的那一类故障。
    """
    assert "IF NOT EXISTS" not in _body()


def test_the_alter_adds_the_citations_column_with_the_exact_shape():
    """列必须出现在 `ADD COLUMN` 里,且形如 `citations JSON NULL`。

    钉住三件事(每一件都能被一个「看起来对」的写法破坏而又不报错):
    · **`ADD COLUMN` 这个词本身** —— 只断「文件里出现过 citations」的话,
      整句删掉、注释留着,断言仍绿;
    · **`JSON`** —— 写 `TEXT` 时引用往返会退化成字符串(`ctx.citations` 拿到
      `"[{…}]"`,`makeCitesClickable` 的 `citations.find` 直接 TypeError,
      而服务端一切正常);
    · **`NULL`** —— 写成 NOT NULL 就没有「这一轮没有引用」这个取值。
    """
    alter = _only_alter()
    assert re.search(r"ADD COLUMN\s+citations\s+JSON\s+NULL", alter), alter
    assert TABLE in alter, f"ALTER 的目标必须是 {TABLE}:{alter}"


def test_the_file_is_the_charset_preamble_plus_exactly_one_ddl_statement():
    """`SET NAMES utf8mb4`(本仓每份 DDL 都有的一行,理由见文件头那个注释)
    **加**恰好一条 DDL 语句 —— 多出来的语句没人审过(而它会照跑)。
    """
    statements = _statements()
    assert [s.split()[0].upper() for s in statements] == ["SET", "ALTER"], statements
    assert statements[0].lower().replace(" ", "") == "setnamesutf8mb4", statements[0]


# ------------------------------------------------- ORM metadata(仍不连库,只编译)


def test_orm_column_is_nullable_json():
    """ORM 侧的**类型与可空性**。

    类型断的是 `sqlalchemy.JSON`(通用的那个),不是 `mysql.JSON` —— 本表的
    `tool_calls` 用的也是通用 `JSON`(`app/db/models.py`),而 MySQL 方言会把它
    编译成 `JSON`(见下一条),两条路径因此一致。断成 mysql.JSON 会红在一个
    **实现细节**上,而不是红在「这条列是不是 JSON」上。
    """
    col = MessageRecord.__table__.c[COLUMN]
    assert isinstance(col.type, JSON), col.type
    assert col.nullable is True, "可空是这一列的全部意义:NULL = 这一轮没有引用"


def test_orm_renders_the_same_column_as_the_ddl():
    """两条建库路径在**这一列上**必须一致(名字与类型;可空性见上一条)。"""
    orm = str(CreateTable(MessageRecord.__table__).compile(dialect=mysql.dialect()))
    assert re.search(rf"^\s*{COLUMN}\s+JSON", orm, re.M), orm
    # 反向陷阱:把可空性写成 NOT NULL 时,上面那条正则**照样匹配**
    # (它只看列名与类型)—— 所以先在 ORM 自己的编译产物上把 NOT NULL 挡掉。
    assert not re.search(rf"^\s*{COLUMN}\s+JSON\s+NOT NULL", orm, re.M), orm
