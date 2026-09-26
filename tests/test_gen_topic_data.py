"""合成数据的**形态与禁区**检查 —— 全是纯函数,不打网络。"""

import pytest

from app.topic.synth import (
    FORMS,
    QUOTA,
    confusable,
    forbidden_words,
    shape_ok,
    violates_forbidden,
)
from app.topic.taxonomy import COUNTER, HEAD_LABELS, LABELS, OTHER


def test_quota_covers_every_label():
    assert set(QUOTA) == set(LABELS)


def test_head_labels_get_a_bigger_quota():
    """「四大类领头」在**配额上**也要是真的,不能只是术语表里的说法。"""
    others = [q for lb, q in QUOTA.items() if lb not in HEAD_LABELS]
    assert all(QUOTA[lb] >= max(others) for lb in HEAD_LABELS), (
        "头四类的配额必须不低于其余类的最大配额"
    )


def test_other_class_has_a_quota():
    """「其他」**要留配额** —— 语料里它不会少,而合成时最容易忘。"""
    assert QUOTA["其他"] > 0


def test_multi_label_share_is_at_least_thirty_percent():
    """多标签占比 ≥30% 是**方案 A 的全部价值所在**。

    少了它,模型退化成「只报最显眼的那个类」,验收 ③
    (「买大了想退」同时命中多类)从原理上过不了。
    """
    assert FORMS["multi"] >= 0.30


def test_boundary_share_is_at_least_fifteen_percent():
    """近邻边界对 ≥15% —— 少了它,`运费↔物流`、`退换货↔保修维修` 学不开。"""
    assert FORMS["boundary"] >= 0.15


def test_forms_sum_to_one():
    assert abs(sum(FORMS.values()) - 1.0) < 1e-9


@pytest.mark.parametrize(
    "question",
    [
        "保修维修怎么办",  # 类目名
        "我想问退换货的事",  # 类目名
        "关于质量问题",  # 类目名
        "运费管钱物流管货",  # 边界说明的原词
        "价保是补差价",  # 边界说明的原词
    ],
)
def test_forbidden_words_catch_label_names_and_boundary_phrases(question):
    """**禁词**:合成的问句里不许出现类目名或边界说明的原词。

    不设这条,模型学到的就是**关键词匹配** —— F1 会很漂亮,
    而真机上一句不含类目名的真话就废了。这是本章最贵的一条数据纪律。
    """
    assert violates_forbidden(question), f"「{question}」应当命中禁词"


@pytest.mark.parametrize(
    "question",
    ["买大了想退", "快递到哪了", "能开专票吗", "满多少包邮", "用一年了能修吗"],
)
def test_real_questions_pass_the_filter(question):
    """**反向**:真实用户话本来就不含类目名 —— 这条防止禁词表宽到把真话也拦了。

    禁词表写宽了的后果是**合成数据全被丢弃**,而执行者只看到「生成了 0 条」,
    容易误以为是模型没返回,而不是自己的过滤器写坏了。
    """
    assert violates_forbidden(question) == []


def test_forbidden_words_are_derived_from_taxonomy_not_hand_written():
    """禁词表必须**从 `taxonomy` 派生**,不能手写一份。

    手写的会与类目表漂移:加了第 18 类而忘了加进禁词表,
    合成数据里就会出现类目名的原词 —— 静默的,只有真机才暴露。
    """
    words = forbidden_words()
    assert "退换货" in words and "保修维修" in words
    # ⚠️ 上面那两条**手写一张两个词的表就能通过**(它们只能证明「有两个词」)。
    #    这条测试的名字承诺的是「派生」,所以要真去比一遍**覆盖率**:
    #    除「其他」(它的说明是「以上都不是」,不是问句里会出现的东西)之外,
    #    **每一个类目名都必须在禁词表里** —— 少一个就红,这正是「加了第 18 类
    #    而忘了加进禁词表」那条漂移的判别式。
    missing = set(LABELS) - {OTHER} - words
    assert not missing, f"这些类目名没有进禁词表(手写表与 taxonomy 漂移了):{sorted(missing)}"


def test_other_is_not_a_forbidden_word():
    """「其他」**不是禁词** —— 它的边界说明是「以上都不是」,不是问句里会出现的东西。

    `forbidden_words()` 从 `set(LABELS)` 起手,所以「其他」默认**在表里**,
    靠一行 `words.discard(OTHER)` 排掉。少了那一行不会有任何东西红:
    合成出来的问句本来就极少含「其他」两字 —— 也就是说这条是**零可观测后果**的
    静默差别,只能靠这条测试钉住。
    """
    assert OTHER not in forbidden_words()


def test_forbidden_words_do_not_swallow_ordinary_questions():
    """禁词表不许宽到把**真实问句**也拦了。

    这条是上一条的反面。禁词表写宽了的表现是「合成数据全被丢弃」,
    而执行者只会看到「生成了 0 条」,很容易误判成模型没返回。

    ⚠️ 诚实记一笔:这一条**在 `forbidden_words()` 返回空集时恒真** ——
    单看它没有判别力。它守的是「别写宽」这一个方向,另一个方向由上面
    `catch_label_names_and_boundary_phrases` 那条守;两条**合起来**才有判别力。
    """
    for q in (
        "这个多少钱",
        "什么时候能到",
        "能不能开票",
        "我要改成另一个地址",
        "这个颜色还有别的吗",
    ):
        assert violates_forbidden(q) == [], f"「{q}」是正常问句,不该被禁词表拦住"


def test_shape_ok_accepts_single_and_multi():
    assert shape_ok(["尺码"])
    assert shape_ok(["尺码", "退换货"])
    assert shape_ok(["尺码", "退换货", "运费"])


def test_prompt_has_a_literal_JSON_and_no_bare_braces():
    """prompt 的两条本仓硬约束 —— 由**整段核**守住,不是靠看干跑输出。

    硬约束:prompt 里必须出现字面 `JSON`(`json_mode` 要求),
    且**不得有裸花括号**(`ChatPromptTemplate` 按 f-string 解析,一个 `{` 就炸)。

    ⚠️ 为什么不能用 `--dry-run` 代替这条:干跑把每批 prompt 截到 400 字
    (`prompt[:400]`),而完整的类目表有 17 行、远超 400 字 ⇒ **后十几个类目的文本
    在干跑输出里压根看不到**。只靠肉眼的话,「后面的类目里有没有花括号」是
    **没被看过**的 —— 这正是本仓「证据看起来齐了」那一类假绿。
    """
    from scripts.gen_topic_data import batch_size, build_prompt

    for label in QUOTA:
        for form in FORMS:
            prompt = build_prompt(label, form, batch_size(label, form))
            assert "JSON" in prompt, f"{label}/{form} 的 prompt 里没有字面 JSON(json_mode 硬约束)"
            assert "{" not in prompt and "}" not in prompt, (
                f"{label}/{form} 的 prompt 里有裸花括号 —— `ChatPromptTemplate` 那条路会炸"
            )


def test_plan_covers_every_label_and_form_exactly_once():
    """`plan()` 是 `run` 的唯一循环来源 —— 少一格就有一类一类没被造过。"""
    from scripts.gen_topic_data import plan

    batches = plan()
    assert len(batches) == len(QUOTA) * len(FORMS)
    assert [(lb, form) for lb, form, _ in batches] == [
        (lb, form) for lb in QUOTA for form in FORMS
    ]
    assert all(n >= 2 for _, _, n in batches), "每批至少 2 条"


def test_plan_realises_the_quota_and_the_global_form_shares():
    """**要出去的条数**必须兑现 `QUOTA` 与 `FORMS` —— 不然那两组只是没人读的常量。

    ⚠️ 这条订正过两次口径,值得读完:
    ① 起初断言写在**常量**上(`FORMS["multi"] >= 0.30`)—— 无视 `FORMS` 的生成器照样绿;
    ② 改断言 `batch_size` 之后,**接线点**(`n = batch_size(...)`)仍然没人测;
    ③ 现在断言的是 `plan()` —— 它就是 `run` 循环的那一份,所以「计划无视 FORMS」
       会在这里红。
    ⚠️ **按全局判**(controller 2026-09-26 裁定):头四类**永远**到不了 30%
    (`round(90×0.30)=27` / `round(90×0.55)=50` ⇒ 27/91 = 29.7%,是取整效应)。
    把 `FORMS["multi"]` 抬上去凑逐类读数才是真的错 —— 逐类由 `check_rows` **只打印不判**。
    """
    from scripts.gen_topic_data import plan

    batches = plan()
    total = sum(n for _, _, n in batches)
    per_label = {lb: sum(n for l, _, n in batches if l == lb) for lb in QUOTA}
    multi = sum(n for _, form, n in batches if form == "multi")
    boundary = sum(n for _, form, n in batches if form == "boundary")

    for lb, got in per_label.items():
        assert got >= QUOTA[lb], f"{lb} 计划只要 {got} 条,少于配额 {QUOTA[lb]}"
    assert multi / total >= 0.30 - 1e-9, f"计划里 multi 只占 {multi / total:.1%}"
    assert boundary / total >= 0.15 - 1e-9, f"计划里 boundary 只占 {boundary / total:.1%}"
    # 头四类的**逐类** multi 占比确实在 30% 线下 —— 这是**已知且被裁定接受**的取整效应,
    # 写成断言是为了让「以后有人把它当 bug 改」这件事被看见(要改先看 §6 那段裁定)。
    head = [lb for lb in HEAD_LABELS]
    head_share = sum(n for l, form, n in batches if l in head and form == "multi") / sum(
        n for l, _, n in batches if l in head
    )
    assert 0.29 < head_share < 0.30, f"头四类的 multi 占比 {head_share:.1%} 变了,去看裁定"


def test_plan_realises_every_quota():
    """`plan()` 逐类加总 ≥ `QUOTA` —— 上一条已含这一断言,这里留一个**具名**的失败点。"""
    from scripts.gen_topic_data import plan

    per_label = {lb: 0 for lb in QUOTA}
    for lb, _, n in plan():
        per_label[lb] += n
    assert set(per_label) == set(LABELS)
    assert all(per_label[lb] >= QUOTA[lb] for lb in QUOTA), (
        f"达不到配额的类:{[lb for lb in QUOTA if per_label[lb] < QUOTA[lb]]}"
    )


def test_shape_ok_rejects_empty_duplicates_and_overlong():
    assert not shape_ok([]), "没有标签的样本没有训练价值"
    assert not shape_ok(["尺码", "尺码"]), "重复标签是标注错误"
    assert not shape_ok(["尺码", "退换货", "运费", "价保"]), "4 个诉求超出本章的形态假设"
    # ⚠️ 上面三条各只钉住一个拒绝理由(空 / 重复 / 超长),而 `shape_ok` 的**第四个**
    #    理由是「标签必须是真类目」—— 不补这一条,把那个 `all(lb in LABELS …)`
    #    整句删掉,上面三条**全都照样绿**(模型跑偏吐出个自造标签时会静默入库)。
    assert not shape_ok(["尺码", "不存在的类目"]), "表外的标签必须拒掉"


# ─────────────────────── 订正轮 I1:边界形态撞的是**声明的**易混类目 ───────────────────────


@pytest.mark.parametrize("label", sorted(COUNTER))
def test_confusable_comes_from_the_counter_declaration(label):
    """易混邻居必须是 `COUNTER` 声明的那一个 —— **逐类**核,不是抽样核。

    实测(2026-09-26,自由联想版):`物流` 的边界批次撞的是 账号 / 评价 / 会员积分 ——
    那是**共现**,不是**易混**。`COUNTER` 是全章唯一的「谁跟谁易混」权威源。
    """
    assert confusable(label) == COUNTER[label][0][1]


def test_confusable_is_none_exactly_for_the_classes_the_taxonomy_does_not_declare():
    """没声明的类目返回 `None`(走回退话术)—— 而「哪些没声明」必须与 `COUNTER` 一致。"""
    declared = {lb for lb in LABELS if confusable(lb) is not None}
    assert declared == set(COUNTER), "`confusable` 与 `COUNTER` 的覆盖面对不上"
    for lb in ("发票", "支付", "其他"):
        assert confusable(lb) is None, f"{lb} 在 COUNTER 里没有声明,不该凭空给一个邻居"


def test_confusable_never_names_the_label_itself():
    for lb in LABELS:
        assert confusable(lb) != lb, f"{lb} 的「易混邻居」是它自己 —— 那样边界形态没意义"


def test_boundary_prompt_names_the_declared_neighbour_and_demands_two_labels():
    """边界批的 prompt 要同时给出两样:① 撞**哪个**类目;② **两个都要打标**。

    少第 ② 条 = 上一版(147/147 全是单标签)那个缺陷;少第 ① 条 = 自由联想。
    """
    from scripts.gen_topic_data import batch_size, build_prompt

    # 有声明的:点名邻居,并要求 2 个标签
    p = build_prompt("物流", "boundary", batch_size("物流", "boundary"))
    assert "运费" in p, "没有点名 COUNTER 声明的易混邻居"
    assert "两个诉求都要打标" in p
    assert "本批应当是 2 个" in p, "「2 个」没写进输出结构那一句"
    # 没声明的(发票 / 其他):退回「让模型自己挑」,但**打标要求照旧**
    for lb in ("发票", "其他"):
        q = build_prompt(lb, "boundary", 2)
        assert "与它最容易混的那个类目" in q, "回退话术丢了"
        assert "两个诉求都要打标" in q and "本批应当是 2 个" in q, "回退支也必须要求 2 个标签"
    # 另外两个形态**不**该被误伤成多标签要求
    assert "本批应当是 2 个" not in build_prompt("物流", "single", 2)
    assert "两个诉求都要打标" not in build_prompt("物流", "multi", 2)


# ─────────────────────── 订正轮 I3:三道门抽成纯函数,逐道钉住 ───────────────────────


def test_accept_takes_a_good_item():
    from scripts.gen_topic_data import accept

    assert accept({"question": "买大了想退", "labels": ["退换货"]}, "退换货")
    # 多标签(边界形态的样子)也收
    assert accept({"question": "买大了想换小一号", "labels": ["退换货", "尺码"]}, "退换货")


@pytest.mark.parametrize(
    "item,seed,why",
    [
        ({"question": "买大了想退", "labels": []}, "退换货", "① 空标签"),
        ({"question": "买大了想退", "labels": ["尺码", "尺码"]}, "尺码", "① 重复标签"),
        ({"question": "买大了想退", "labels": ["a", "b", "c", "d"]}, "退换货", "① 超长(4 个)"),
        ({"question": "买大了想退", "labels": ["不存在的类目"]}, "退换货", "① 表外标签"),
        # ⚠️ 这两条的标签都是**合法**的(过得了第 ① 道),所以它们只可能死于第 ② 道 ——
        #    用「满多少包邮」会自相矛盾:`forbidden_words` 里没有「包邮」,而它正是
        #    反向用例里那条「真实问句必须放行」。
        ({"question": "我想问退换货的事", "labels": ["退换货"]}, "退换货", "② 含类目名(禁词)"),
        ({"question": "什么时候补差价", "labels": ["价保"]}, "价保", "② 含边界说明原词(禁词)"),
        ({"question": "买大了想退", "labels": ["尺码"]}, "退换货", "③ 主诉求不是这一类"),
        ({"question": "买大了想退"}, "退换货", "缺 labels 键"),
        ({"labels": ["退换货"]}, "退换货", "缺 question 键(会写出一行没有问句的训练数据)"),
        ({"question": "", "labels": ["退换货"]}, "退换货", "空问句"),
        ({"question": "   ", "labels": ["退换货"]}, "退换货", "纯空白问句"),
        ({"question": 123, "labels": ["退换货"]}, "退换货", "question 不是字符串"),
        ("买大了想退", "退换货", "条目整个不是 dict(模型偶尔直接吐字符串数组)"),
        ({"question": "买大了想退", "labels": "退换货"}, "退换货", "labels 不是数组"),
    ],
)
def test_accept_rejects_each_failure_mode(item, seed, why):
    """**三道门逐道钉住** —— 真跑 962 条 `dropped = 0` ⇒ 这三道生产上**一次没开火**。

    不抽出来单测,「禁词表写坏了」与「模型没写禁词」在读数上**逐位相同**(都是
    `dropped = 0`);而禁词纪律的执行点原本只是一个没人测过的 `if`。
    """
    from scripts.gen_topic_data import accept

    assert not accept(item, seed), f"应当被拒:{why}"


def test_accept_lets_the_other_class_through_with_any_labels():
    """「其他」是**共用落点**(转人工 / 投诉 / 闲聊都落它),所以第 ③ 道门对它放行。"""
    from scripts.gen_topic_data import accept

    assert accept({"question": "帮我写首诗", "labels": ["其他"]}, "其他")
    assert accept({"question": "怎么转人工", "labels": ["其他", "退换货"]}, "其他")


# ─────────────────────── 订正轮 I4:产物带 id,前缀与真实语料分得开 ───────────────────────


def test_make_row_carries_the_id_and_the_seven_keys():
    """`id` 是下游的索引键(T5 的 `r["id"] not in done`、T7 的排序、已审合并)。"""
    from scripts.gen_topic_data import make_row

    row = make_row({"question": "买大了想退", "labels": ["退换货"]}, "退换货", "single", 7)
    assert row["id"] == "s-0007"
    assert row["id"].startswith("s-"), "合成数据的 id 必须与真实语料的 `r-` 前缀分得开"
    assert set(row) == {
        "id", "question", "labels", "provenance", "source", "form", "seed_label",
    }
    assert row["question"] == "买大了想退" and row["labels"] == ["退换货"]
    assert row["form"] == "single" and row["seed_label"] == "退换货"


# ────────────────── 订正轮 I2:对**产物**的门槛(不是对常量,也不是对计划) ──────────────────


def _conforming_rows() -> list[dict]:
    """一个**刚好达标**的产物:按同一个 `plan()` 造,逐类 = 配额、多标签 = 30%、边界 = 15%。

    它用 `plan()` 定**行数**,而 `check_rows` 拿的是 `QUOTA` 与两个门槛 ——
    所以 `plan()` 一旦缩水(比如每批只要 10 条),这个夹具会跟着缩,
    而 `check_rows` 会判它**低于配额** ⇒ 仍然红。
    """
    from scripts.gen_topic_data import plan

    rows, i = [], 0
    for label, form, n in plan():
        for _ in range(n):
            i += 1
            second = confusable(label) or "运费"
            labels = [label] if form == "single" else [label, second]
            rows.append({"id": f"s-{i:04d}", "question": f"问句{i}", "labels": labels,
                         "provenance": "synthetic", "source": "gen", "form": form,
                         "seed_label": label})
    return rows


def test_check_rows_passes_on_a_conforming_artifact():
    from scripts.gen_topic_data import check_rows

    ok, report = check_rows(_conforming_rows())
    assert ok, f"刚好达标的产物被误判:\n{report}"
    assert "结论:**通过**" in report
    # 逐类都打印(不是只打全局)—— controller 的裁定是「按全局判、逐类打印」
    assert all(lb in report for lb in LABELS)


def test_check_rows_flags_a_class_below_quota():
    """删掉一类的行 ⇒ 必须**指名**报出来(而不是只在全局占比上晃一下)。"""
    from scripts.gen_topic_data import check_rows

    rows = [r for r in _conforming_rows() if r["seed_label"] != "评价"]
    ok, report = check_rows(rows)
    assert not ok
    assert "评价" in report and "少于配额" in report


def test_check_rows_flags_a_multi_share_below_the_line():
    """把多标签全打成单标签 ⇒ 全局占比塌到 0 ⇒ 红。"""
    from scripts.gen_topic_data import check_rows

    rows = [{**r, "labels": [r["seed_label"]]} for r in _conforming_rows()]
    ok, report = check_rows(rows)
    assert not ok
    assert "全局多标签占比 0.0% < 30%" in report


def test_check_rows_flags_a_boundary_share_below_the_line():
    """把边界形态那一批改名成 single ⇒ 行的总数与逐类配额**都不变**,只有边界占比塌。

    ⚠️ 这正是本测试要的**单一变量**:它证明「边界占比」是一道**独立的门**,
    而不是被配额那条顺手带出来的。
    """
    from scripts.gen_topic_data import check_rows

    rows = [{**r, "form": "single"} if r["form"] == "boundary" else r
            for r in _conforming_rows()]
    ok, report = check_rows(rows)
    assert not ok
    assert "全局边界形态占比 0.0% < 15%" in report


def test_check_rows_flags_missing_and_duplicate_ids():
    """`id` 是 T5 的索引键 ⇒ 自检必须自己看住它(缺一个 = T5 当场 `KeyError`)。"""
    from scripts.gen_topic_data import check_rows

    rows = _conforming_rows()

    missing = [dict(r) for r in rows]
    del missing[0]["id"]
    ok, report = check_rows(missing)
    assert not ok and "没有 id" in report

    dup = [dict(r) for r in rows]
    dup[1]["id"] = dup[0]["id"]
    ok, report = check_rows(dup)
    assert not ok and "id 有重复" in report


def test_check_rows_reports_the_head_class_rounding_effect():
    """头四类那 29.7% 必须**写在报表里**,免得下一个人把它当 bug 去抬 `FORMS`。"""
    from scripts.gen_topic_data import check_rows

    _, report = check_rows(_conforming_rows())
    assert "取整" in report and "29.7%" in report


def test_existing_rows_counts_the_lines_for_resume_numbering(tmp_path):
    """append 续跑时编号**接着编** —— 从 1 重编会让 id 撞号,而下游全靠它索引。

    ⚠️ 这里刻意不解析 JSON:数行数就够(每存活行恰好一行),而且解析会在
    **半行**(上次被中断的写入)上抛,把续跑的门直接关死。
    """
    from scripts.gen_topic_data import _existing_rows

    p = tmp_path / "synthetic.jsonl"
    assert _existing_rows(p) == 0, "文件不存在时应当返回 0(首次运行)"
    p.write_text('{"id": "s-0001"}\n{"id": "s-0002"}\n\n', encoding="utf-8")
    assert _existing_rows(p) == 2, "空行不该被算进去"
