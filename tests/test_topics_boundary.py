"""⚠️ 纪律:实时对话主链路**零处**调用主题分类器。

这是本章两条不变量里的第一条(spec §3.3 ① / §9.4)。用**源码扫描**而不是运行时
断言 —— 要抓的是「有没有人写下去」,不是「跑到了没有」
(照 ch09 `test_no_module_outside_observability_imports_langfuse` 的先例)。

**为什么它值得一条测试**:分类器接进主链路的最可能形态是「顺手在 `nodes.py` 里
加一句兜底」—— 而那样做的后果是主链路多一次几百毫秒的旁路调用,且**没有任何断言
会红**;等发现时,「实时对话不调它」这个约定已经消失了,而当初为什么定这条约定
也没人记得。

## 这条守卫覆盖什么、**不**覆盖什么

**覆盖**:请求路径那几处文件里**写下**了下列任何一种 import 语句(行首、允许缩进)。

**不覆盖(如实记账,别把这条读得比它实际宽)**:

- **动态导入**:`importlib.import_module("app.topic.model")`、`__import__(...)`、
  把模块名**拼成字符串**再导 —— **正则一个都看不见**。真正兜住它们的是**别的东西**
  (分类器是**独立进程**里的旁路服务,主链路里根本没有它的地址);
- **进程外调用**:`subprocess` 起来跑 `scripts/classify_topics.py`、或者打一个 HTTP
  端点。源码扫描对这一类同样无话可说。
- **`clean` / `taxonomy` 不在禁令里**,而且**不该在**:`app.topic.taxonomy` 是
  「类目清单的唯一来源」,`app.topic.clean` 是「清洗的唯一实现」—— 禁掉它们等于
  把权威表逼成第二份手写清单。被禁的是**分类器本体**(`model`)与**它的指标**
  (`metrics`)。
"""

import re
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app"

#: 主链路上**不许**出现的名字。加一个就加一条 —— 别写成通配,
#: 通配会把「日志里提了一句 topic」也判成违规,那会诱人去放宽它。
#
# ⚠️⚠️ **计划订正 16(controller,2026-09-27)—— 这一条漏了一种写法** ⚠️⚠️
# 下面那个正则原先只认 `from app.topic.model import …` 与 `import app.topic.model`,
# **漏掉 `from app.topic import model`**(以及 `from app.topic import model, metrics`)。
# 而**本仓另一条同款守卫明确把三种惯用写法都放行过** ——
# `tests/test_topic_clean.py::test_both_sides_use_the_same_clean` 的 docstring 逐字写着:
# 「原来只认 `from app.topic.clean import … clean` **一种写法**,一个**正确**的
# 任务 3 / 任务 13 若写成 `from app.topic import clean` 或 `import app.topic.clean`
# 就会被判红 —— 那会逼它们为了变绿而**改自己的 import**」。
# ⇒ **这里是它的镜像**:那条守卫怕**误红**,这条怕**误绿**。
# **三条惯用写法一个都不许漏** —— 五种写法各有一条元测试
# (`test_every_idiomatic_import_style_is_recognized`)。
#
# ⚠️ `^\s*` 这个锚是**有意的**:扫的是**语句**,不是散文。docstring 或注释里提到
# `app.topic.model` 不该判红(反面教材:B9 那次扫描命中的是 docstring 里那半句,
# **把代码删掉照样绿**)。`test_a_mention_in_prose_is_not_an_import` 钉住这一条。
FORBIDDEN = re.compile(
    r"^\s*(?:import\s+topic_service\b"
    r"|from\s+topic_service\b"
    r"|import\s+app\.topic\.(?:model|metrics)\b"
    r"|from\s+app\.topic\.(?:model|metrics)\b"
    r"|from\s+app\.topic\s+import\s+.*\b(?:model|metrics)\b)",
    re.M,
)

#: 扫描范围:**`app/` 下除 `app/topic/` 之外的每一个 `.py`**。
#
# ⚠️ **这是相对 brief 原稿的一处放宽(brief 写的是 `("agent", "api/chat.py")` ——
# 「请求路径那几处」)**,理由:禁令要守的性质是「**服务进程里**零处调用分类器」,
# 而请求路径不止那两个地方 —— `app/tools/`(`registry` / 内置工具 / 执行器)、
# `app/memory/`、`app/services/`、`app/flywheel/`(它由请求路径 fire-and-forget 起线程)
# 全在同一个进程里。写死成两个名字的后果是:在 `app/tools/builtin/xxx.py` 里
# `from app.topic import model` **一条守卫都不会红**,而那句话读起来像守住了。
# 排除 `app/topic/` 是因为**分类器自己那一包当然 import 自己**。
# (要退回原稿:把这个函数改成只 yield 那两处即可,其余一字不用动。)
EXCLUDED_PREFIX = "topic/"


def _scanned() -> list[tuple[str, object]]:
    """(相对 `app/` 的 posix 路径, Path)的列表 —— **扫描集只有这一个来源**。

    守卫与它的元测试读的都是它,所以「扫了哪些文件」不会在两处各写一份而漂移。
    """
    out = []
    for p in sorted(APP.rglob("*.py")):
        rel = p.relative_to(APP).as_posix()
        if not rel.startswith(EXCLUDED_PREFIX):
            out.append((rel, p))
    return out


#: 五种惯用写法(**三种样式**):`import topic_service` / `from topic_service import …`
#: / `import app.topic.model` / `from app.topic.model import …` / `from app.topic import model`。
IDIOMATIC = (
    "import topic_service",
    "from topic_service import TopicService",
    "import app.topic.model",
    "from app.topic.model import TopicService",
    "from app.topic import model",
    "from app.topic import model, metrics",
    "import app.topic.metrics",
    "from app.topic.metrics import f1_score",
)


def test_main_chain_never_imports_the_topic_classifier():
    offenders = [
        rel for rel, p in _scanned() if FORBIDDEN.search(p.read_text(encoding="utf-8"))
    ]
    assert offenders == [], (
        f"服务进程里出现了主题分类器 —— 本章约定它只在**独立进程**的旁路跑:{offenders}"
    )


def test_the_scan_actually_scans_the_main_chain():
    """⚠️ **元测试**:上一条若因为扫描集写错而扫了个空目录,它会**恒绿**。

    对着一个不存在的路径 `rglob` 返回空迭代器,`offenders` 永远是 `[]` ——
    一条永远通过的守卫,比没有守卫更糟。所以这里除了「扫到足够多文件」,
    还**逐个点名**几个非扫不可的文件(少了任何一个,SCOPE 就是写错了),
    并反过来确认那个**排除**也是真的(排除写错的后果同样是恒绿)。
    """
    rels = {rel for rel, _ in _scanned()}
    assert len(rels) >= 50, f"扫描集只命中 {len(rels)} 个文件 —— 大概写错了"
    for must in ("agent/nodes.py", "agent/graph.py", "api/chat.py", "tools/registry.py"):
        assert must in rels, f"{must} 不在扫描集里 —— 它正是最该被扫的那一类文件"
    assert not [r for r in rels if r.startswith(EXCLUDED_PREFIX)], (
        f"分类器自己那一包不该被扫(它当然 import 自己),实际扫到了 "
        f"{[r for r in rels if r.startswith(EXCLUDED_PREFIX)]}"
    )
    assert (APP / "topic" / "model.py").exists(), (
        "被排除的那个包不存在了 —— 排除成了一条空话(文件改名就要改这里)"
    )


@pytest.mark.parametrize("line", IDIOMATIC)
def test_every_idiomatic_import_style_is_recognized(line):
    """⚠️ **元测试(订正 16 的正面)**:五种写法**逐条**喂进去 ⇒ 必须命中。

    为什么要有它:上面那条守卫的全部判别力都在这个正则上,而它**漏一种写法**
    的表现是**静默**的 —— 扫描照跑、永远绿。手工变异能证明「今天漏没漏」,
    但手工不做就不会重跑;这条每条写法一个参数,漏一种当场红。

    (它**不**替上面的手工变异:正则命中某一行 ≠ 那个文件真的被扫到了 ——
    那是 `test_the_scan_actually_scans_something` 与那次「真的插进主链路文件、
    看它变红」的共同职责。)
    """
    assert FORBIDDEN.search(line), f"这条写法漏了:{line!r}"


@pytest.mark.parametrize(
    "line",
    [
        # ⚠️ 这两条是**有意放行**的:它们是「权威表的唯一来源」与「清洗的唯一实现」,
        # 禁掉就等于逼人再手写一份类目清单(见模块 docstring)。
        "from app.topic.taxonomy import LABELS",
        "from app.topic import clean",
        "from app.topic.clean import clean",
        "import app.topic.taxonomy",
        "from app.topic import taxonomy, clean",
        # ⚠️ 散文里提到,不是语句 —— `^\s*` 锚的用途(*那半句*的教训)
        '"""别 import app.topic.model —— 它会拖慢主链路。"""',
        "# 这里不用 app.topic.metrics,历史包袱在别处",
        "import app.api.conversations",
    ],
)
def test_a_mention_in_prose_or_an_allowed_module_is_not_an_import(line):
    """⚠️ **反向对照**:这条守卫不许宽到「见到 `topic` 就算违规」。

    前几条是**有意放行**的模块(权威表 / 清洗);后三条钉的是**锚**:
    正则扫的是**语句**,不是散文。守卫宽了的代价不是「多报几条」——
    它会诱人**为了变绿而放宽正则**,于是真正的违规也一起漏掉
    (`tests/test_topic_clean.py` 的同款 docstring 记着那次教训的另一面)。
    """
    assert not FORBIDDEN.search(line), f"这条不该判违规:{line!r}"


def test_main_mounts_the_static_dir_after_every_router():
    """⚠️ 本仓硬约束:`mount("/")` 必须在**所有** `include_router` **之后**。

    挂反了的后果不是「新端点 404」那么显眼 —— 静态目录的 catch-all 会**先**匹配,
    于是 `/api/topics/distribution` 变成一次静态文件查找,而**服务照常起、
    别的端点全正常**。

    ⚠️ **这条只读 `main.py` 的源文本,是一层便宜的代理**。真正「挂反了会不会
    真的 404」由真请求断:`tests/test_api_topics_db.py` 的
    `test_the_endpoint_answers_over_http_and_is_not_swallowed_by_static`(它走
    `httpx.ASGITransport` 打真 app)。两条**都**在,是因为这条**不需要 MySQL**
    也跑得起来 —— 只有一条 db 守卫的话,维护者在不带库的环境里跑一遍是全绿的。
    """
    source = (APP.parent / "app" / "main.py").read_text(encoding="utf-8")
    last_include = source.rindex("app.include_router(")
    mount_at = source.index('app.mount("/"')
    assert last_include < mount_at, (
        "`app.mount(\"/\")` 出现在最后一个 `include_router` **之前** —— "
        "静态目录会抢走 /api/*(端点是 404,而服务照常起)"
    )
