"""标注相关的纯函数:去重、证据串校验、标签漂移、分层抽样。

末两条(`test_collect_...` 与 `test_prelabel_...`)不测纯函数,测的是**接线** ——
`collect`(三源合流)与 `prelabel_topics.run`(三个读数 + 断点续跑)。两处都
**没有 DB、没有网络**(模型替身替在 `ainvoke` 这一层),所以留在这一份里、
也在 `not db` 套件里。
"""

import asyncio
import json

import pytest

from app.topic.labeling import (
    dedupe_questions,
    inject_typo,
    is_unusable_target,
    label_drift,
    pick_review_sample,
    stratified_split,
    train_test_overlap,
    typo_seed,
    validate_evidence,
)


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


# ---- 证据串校验(ch10 spec §6.2)----


def test_evidence_must_be_a_substring_of_the_question():
    """**这是「字面提到」这个口径的结构性保证。**

    没有它,「字面提到」只是 prompt 里的一句话,模型可以凭语义联想打标签,
    而你从输出上看不出来。有了证据串校验,它变成一个**可自动检验**的条件。
    """
    ok, bad = validate_evidence(
        "买大了想退", ["尺码", "退换货"], {"尺码": "买大了", "退换货": "想退"}
    )
    assert ok == ["尺码", "退换货"]
    assert bad == []


def test_fabricated_evidence_is_rejected():
    """模型编了一个原文里没有的片段 —— 这一条必须被挑出来。"""
    ok, bad = validate_evidence(
        "买大了想退", ["尺码", "退换货"],
        {"尺码": "买大了", "退换货": "退款政策"},   # 原文里没有「退款政策」
    )
    assert ok == ["尺码"]
    assert bad == ["退换货"]


def test_label_without_evidence_is_rejected():
    ok, bad = validate_evidence("买大了想退", ["尺码"], {})
    assert ok == []
    assert bad == ["尺码"]


def test_empty_evidence_string_is_rejected_not_accepted():
    """空串是任何字符串的子串 —— **不特判的话这条会假绿**。

    `"" in "任意文本"` 为 True,所以只写 `if ev in question` 的话,
    模型返回 `"尺码": ""` 会被判为「证据合法」。
    """
    ok, bad = validate_evidence("买大了想退", ["尺码"], {"尺码": "   "})
    assert ok == []
    assert bad == ["尺码"]


def test_evidence_is_matched_after_cleaning():
    """证据比对在**清洗后**的文本上做 —— 与预标时喂进去的文本口径一致。"""
    ok, _ = validate_evidence("买大了 想退", ["尺码"], {"尺码": "买大了"})
    assert ok == ["尺码"]


def test_both_sides_of_the_evidence_check_are_cleaned():
    """两侧都要过 `clean()` —— 上面那条用例**两个方向都钉不住**(变异实测,2026-09-26)。

    ⚠️ 实测:把 `clean(ev)` 去掉、或把 `normalized` 换成裸的 `question`,
    上面那条的输入(`"买大了 想退"` / `"买大了"`)**在两种错法下都照绿** ——
    因为那条证据本身不含任何需要清洗的字符(三次独立计算:真实现 / 错法 A / 错法 B
    全是 True)。判据是本仓那条:**构造输入前先算一遍它会不会走到那条分支。**

    下面两条输入各自只钉一侧,且都对着 `validate_evidence` docstring 里那句
    「证据本身也要过一遍清洗,否则全角/空白差异会造成假拒」:

    - 证据侧带**双空格** ⇒ 少一遍 `clean(ev)` 就红(错法 A:False);
    - 问句侧带**全角空格** ⇒ 少一遍 `clean(question)` 就红(错法 B:False)。
    """
    # 证据侧
    ok, _ = validate_evidence("买大了 想退", ["尺码"], {"尺码": "买大了  想退"})
    assert ok == ["尺码"], "证据侧的清洗没了 —— 双空格的证据被误拒"
    # 问句侧
    ok, _ = validate_evidence("买大了　　想退", ["尺码"], {"尺码": "买大了 想退"})
    assert ok == ["尺码"], "问句侧的清洗没了 —— 全角空格的问句匹配不上"


def test_rejected_labels_are_returned_not_silently_dropped():
    """被拒的标签要**返回出来**,不能悄悄吞掉。

    吞掉的后果:一条问题从「3 个标签」变成「1 个标签」而无人知晓,
    而这会**改变训练分布** —— 多标签样本会系统性变少,正是方案 A 要防的。
    """
    _, bad = validate_evidence("买大了想退", ["尺码", "退换货"], {"尺码": "买大了"})
    assert bad == ["退换货"]


# ---- 预标脚本的接线(ch10 spec §6.2/§6.3;不联网、不碰库)----


def test_prelabel_wiring_writes_the_three_readings_and_resumes(monkeypatch, tmp_path, capsys):
    """`prelabel_topics.run` 的三个读数与断点续跑 —— **都是接线,不是纯函数**。

    ⚠️ 为什么必须有这一条:**Step 6 的「被拒 0 条」在「标签本来就都对」与
    「校验器根本没接进来」两种实现下读数完全相同**(本仓编目过的假绿形态)。
    上面六条只钉了 `validate_evidence` 的**机制**,没有一条钉 `run` 真的调它。
    同理「parse_failed 全 false」在「解析失败被写成空标签行」的实现下也读数相同
    (那正是订正 C 的形状)—— 所以这里**造**一条解析失败的输入,并要求它落成
    `parse_failed: true` 这一列。

    替身替在 **`create_extract_model`** 这一层(不是替 `_label`),所以
    `bind(response_format=…)` → `ainvoke` → 去围栏 → `json.loads` → 形状闸 →
    `validate_evidence` → 写盘 → 计数,**整条真跑**;JSON 也真的从文本解出来。

    ⚠️ **订正轮 1 改了三处**(都是评审实测出来的假绿):
    - **F1**:原来「空标签」的两条输入**同时也是 `parse_failed`** ⇒ `zero_label` 这个读数
      **零判别力**(把实现改成「只在解析失败时 +1」照样绿)。⇒ 补 ⑤ 一条
      **解析成功但零标签**的输入——真产物里正是这种行(`r-0049`「你是」)。
    - **F3**:替身 `bind` 原来 `return self`(真 `ChatOpenAI.bind()` 返回的是**新对象**)⇒
      「返回值有没有被用上」**不可观测**:`_bind_json` 写成 `model.bind(…); return model`
      时 `response_format` 在真实链路上静默丢掉,**而断言照样绿**(本仓「替身替被测对象
      完成了语义」形态,同 ch07 的 `FakeSession` 自己 `sorted(...)`)。
    - **Mn1**:非字符串标签原来被**静默丢掉** ⇒ 补 ⑦ 一条 `["尺码", 5]`。
    """
    import asyncio

    from scripts import prelabel_topics as p

    class _Msg:
        def __init__(self, text):
            self.text = text

    bound_instances = []

    class _Model:
        """只实现被测代码真的用到的那两样:`bind` 与 `ainvoke`。

        ⚠️ `bind` **返回一个新对象**(共用同一条文本队列),与真 `ChatOpenAI.bind()` 同形:
        源码逐字「Bind arguments to a `Runnable`, **returning a new `Runnable`**」。
        返回 `self` 的话,「调了 bind 却把返回值丢了」这种写法就**测不出来**。
        """

        def __init__(self, texts, bound=None):
            self._texts = texts          # 故意**共用**同一个 list(新对象也要能取到文本)
            self.bound = bound
            self.used = False

        def bind(self, **kwargs):
            new = _Model(self._texts, bound=kwargs)
            bound_instances.append(new)
            return new

        async def ainvoke(self, _messages):
            self.used = True
            return _Msg(self._texts.pop(0))

    model = _Model([
        # ① 正常:两个标签、证据都是原句子串
        '{"labels": ["尺码", "退换货"], "evidence": {"尺码": "买大了", "退换货": "想退"}}',
        # ② 编造:「退款政策」不在原文里 ⇒ 只该留下「尺码」,且被拒标签要**写出来**
        '{"labels": ["尺码", "退换货"], "evidence": {"尺码": "买大了", "退换货": "退款政策"}}',
        # ③ 散文:JSON 没解出来 ⇒ parse_failed
        "抱歉,我无法判断。",
        # ④ JSON 是合法的,但**形状**不是约定的那样(evidence 是一个字符串)
        '{"labels": ["尺码"], "evidence": "买大了"}',
        # ⑤ **解析成功、零标签**(F1) —— 与 ③④ 不是一回事,而它是 `r-0049` 的形状
        '{"labels": [], "evidence": {}}',
        # ⑥ 带 ```json 围栏(Mn3:那条去围栏分支原来零覆盖)
        '```json\n{"labels": ["退换货"], "evidence": {"退换货": "想退"}}\n```',
        # ⑦ 标签数组里混了一个**非字符串**(Mn1:原来会被静默丢掉)
        '{"labels": ["尺码", 5], "evidence": {"尺码": "买大了"}}',
    ])
    monkeypatch.setattr(p, "create_extract_model", lambda _settings: model)
    monkeypatch.setattr(p, "get_settings", lambda: object())

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [
        {"id": "r-0001", "question": "买大了想退", "source": "pool", "provenance": "real"},
        {"id": "r-0002", "question": "买大了想退", "source": "pool", "provenance": "real"},
        {"id": "r-0003", "question": "在吗", "source": "pool", "provenance": "real"},
        {"id": "r-0004", "question": "买大了想退", "source": "pool", "provenance": "real"},
        {"id": "r-0005", "question": "在吗", "source": "pool", "provenance": "real"},
        {"id": "r-0006", "question": "买大了想退", "source": "pool", "provenance": "real"},
        {"id": "r-0007", "question": "买大了想退", "source": "pool", "provenance": "real"},
    ]) + "\n", encoding="utf-8")
    monkeypatch.setattr(p, "CORPUS", corpus)
    monkeypatch.setattr(p, "SYNTH", tmp_path / "不存在.jsonl")
    out = tmp_path / "prelabeled.jsonl"
    monkeypatch.setattr(p, "OUT", out)

    asyncio.run(p.run(None))

    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in rows] == [
        "r-0001", "r-0002", "r-0003", "r-0004", "r-0005", "r-0006", "r-0007",
    ]
    # ① 原样通过,evidence 保留
    assert rows[0]["labels"] == ["尺码", "退换货"]
    assert rows[0]["rejected_labels"] == []
    assert rows[0]["parse_failed"] is False
    assert rows[0]["evidence"] == {"尺码": "买大了", "退换货": "想退"}
    # ② 被拒的标签**返回出来**、**落进产物**,且它那条 evidence 不留在产物里
    assert rows[1]["labels"] == ["尺码"]
    assert rows[1]["rejected_labels"] == ["退换货"]
    assert rows[1]["evidence"] == {"尺码": "买大了"}
    assert rows[1]["parse_failed"] is False
    # ③ 解析失败必须**在产物里分得开**(订正 C:与「模型真的判了零标签」逐字节相同的那个坑)
    assert rows[2]["labels"] == [] and rows[2]["parse_failed"] is True
    # ④ 形状不对**不是**「JSON 没解出来」,但也不是正常结果 ⇒ 同样归 parse_failed。
    #    ⚠️ 少了那道形状闸,`dict("买大了")` 会直接 ValueError 打断整跑 ——
    #    这条输入探的正是「校验器有没有被接进来」的另一半。
    assert rows[3]["labels"] == [] and rows[3]["parse_failed"] is True
    # ⑤ **解析成功、零标签**(F1):这一条**必须**与 ③④ 分得开 ——
    #    它是 `r-0049`「你是」的形状(模型真的判了「没有主题」),不是故障。
    #    ⚠️ 少了这一条,`zero_label` 这个读数就**零判别力**:
    #    写成「只在 bad 时 +1」照样全绿(评审 M1 实测)。
    assert rows[4]["labels"] == [] and rows[4]["parse_failed"] is False
    # ⑥ 围栏(Mn3)
    assert rows[5]["labels"] == ["退换货"]
    assert rows[5]["parse_failed"] is False
    # ⑦ 非字符串标签**不静默丢**(Mn1):`5` → `"5"`,查不到证据 ⇒ 被拒 ⇒ **可见**
    assert rows[6]["labels"] == ["尺码"]
    assert rows[6]["rejected_labels"] == ["5"]
    assert rows[6]["parse_failed"] is False

    # 三个读数都打出来(只为「人看得到」,但读数本身就是交付物的一部分)
    printed = capsys.readouterr().out
    assert "被证据校验拒掉标签的 2 条" in printed
    assert "JSON 解析失败 2 条" in printed
    assert "空标签 3 条" in printed

    # ---- F3:`bind` 的**返回值**真的被用上了 ----
    # ⚠️ 真 `ChatOpenAI.bind()` 返回的是**新对象**,所以这里断的是三件事:
    #   ① 传出的那个新对象拿到了 kwargs;② **是它**被 `ainvoke` 了;
    #   ③ 原对象**一次都没被用过**。
    #   写成 `model.bind(...); return model`(丢掉返回值)⇒ ②③ 红
    #   —— 而那在真实链路上意味着 `response_format` 静默丢掉、网关侧的保证没了。
    assert len(bound_instances) == 1, "bind 必须被调用恰好一次"
    assert bound_instances[0].bound == {"response_format": {"type": "json_object"}}
    assert bound_instances[0].used is True, "ainvoke 收到的不是 bind 返回的那个对象"
    assert model.bound is None and model.used is False, (
        "原对象被用过了 ⇒ bind 的返回值没有生效"
    )

    # ---- 断点续跑:再跑一次,七条 id 都已在产物里 ⇒ 一条都不处理、一行都不追加 ----
    asyncio.run(p.run(None))
    resume_out = capsys.readouterr().out
    assert "本次处理 0 条" in resume_out
    assert len(out.read_text(encoding="utf-8").splitlines()) == 7
    # Mn2:**空批不许打成 `0.0%`** —— 那个数看着像「质量又被确认了一次」,其实一个样本都没测。
    assert "0.0%" not in resume_out
    assert "本批无读数" in resume_out


def test_main_pins_stdout_encoding(tmp_path):
    """`main()` **自己**必须把 stdout 钉成 UTF-8 —— **本机 locale 是 cp936**。

    `⚠️`(U+26A0)与 `⇒`(U+21D2)编不进 GBK,而它们只出现在「解析失败不为 0」那条
    print 里 ⇒ 不钉的话,**恰恰在最需要它输出的那条路径上**抛 `UnicodeEncodeError`:
    脚本以退出码 1 结束、那行警告消失,而产物(逐行 flush)是好的 ——
    「产物是好的」与「日志里没有警告」叠在一起是最难分辨的一种假绿。

    ⚠️ **子进程故意不加 `-X utf8`**:加的话 stdout 恒是 UTF-8,这条断言就**恒真**
    (本仓那条「平台陷阱的断言要么带可复现证据、要么标未验证」)。
    它的判别力**依赖本机是 cp936** —— 换一台 UTF-8 locale 的机器,这条就退化成
    「函数存在」检查(如实记下来,不装作它到哪都同样有力)。

    ⚠️ **订正轮 1 的 F2:这一跑的是 `main()`,不是 `_pin_stdout_encoding()`。**
    原来写的是 `from … import _pin_stdout_encoding; _pin_stdout_encoding()`,
    测的是**函数体**,不是它命名的那件事(「`main()` **必须**把 stdout 钉住」)——
    评审 M8 实测:把 `main()` 里那一行调用**删掉**(函数留着)照样全绿。
    ⇒ 现在在子进程里直接跑 `main()`,只把 `run` 换成 no-op 桩(不联网、不碰产物)。

    ⚠️⭐ **订正轮 1 的 I4(与 `export_label_review` 那条 F1 完全同构,而且是**先于**本轮
    就存在的)**:上面那个桩只盖住了**网络那一半**,`main()` 里面还有一句
    `asyncio.run(run(args.limit or None))` —— `run` 是**模块全局、调用时求值** ⇒
    **桩一旦静默失效**(有人把 `main()` 换个封装 / 改名 / 把 `run` 提前绑成局部),
    子进程会跑到**真的** prelabel:它**打网络**、并**追加**进仓库里的
    `evals/topic/prelabeled.jsonl`(我的 `split` 的输入!)——
    而这条测试只断 stdout 与 returncode,**产物被改它照样绿**。
    ⇒ 照 F1 的做法:**先把 `OUT` / `CORPUS` / `SYNTH` 指到 `tmp_path`、再调 `main()`**,
    并在父进程里补一句**不依赖 print 措辞**的断言:真产物前后**逐字节相同**。
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    # 真产物:断「它没被这次子进程动过」。⚠️ 它可能**还不存在**(全新 checkout 上没跑过
    # prelabel)—— 那就断「跑完之后仍然不存在」,同样是「没被动过」。
    artifact = root / "evals" / "topic" / "prelabeled.jsonl"
    before = artifact.read_bytes() if artifact.exists() else None
    tmp = tmp_path / "topic"
    code = "\n".join([
        "import sys",
        "from pathlib import Path",
        "from scripts import prelabel_topics as p",
        "async def _noop(_limit):",
        "    return None",
        # ⚠️ **先改路径,再调 main()** —— 顺序反了就白改:真 run 会先落在真目录里。
        f"p.OUT = Path({str(tmp / 'prelabeled.jsonl')!r})",
        f"p.CORPUS = Path({str(tmp / 'corpus.jsonl')!r})",
        f"p.SYNTH = Path({str(tmp / 'synthetic.jsonl')!r})",
        "p.run = _noop",              # 只桩掉网络那一半;`main()` 的其余部分照跑
        "p.main()",
        "print('⚠️ 解析失败不为 0 ⇒ 先停下看产物')",
    ])
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=root, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    assert proc.returncode == 0, f"钉编码没生效(main() 里那一行被删了?):\n{proc.stderr}"
    assert "解析失败不为 0" in proc.stdout
    # ---- I4 的两句:「桩真的拦住了」与「真产物真的没被动」----
    # ⚠️ 第二句**不依赖 print 措辞**(措辞会变,而「有没有真的跑 prelabel」不会)。
    after = artifact.read_bytes() if artifact.exists() else None
    assert after == before, (
        f"{artifact} 在这次子进程里**被改动了** —— 桩失效时真的跑了 prelabel"
        f"(它打网络并**追加**进这份语料;而它正是 `split` 的输入)"
    )


# ---- 人工抽审的分层抽样(ch10 spec §6.3)—— 纯函数,零 IO ----


def _rows():
    rows = []
    for label in ("尺码", "退换货", "运费"):
        for i in range(20):
            rows.append({"id": f"{label}-{i}", "labels": [label], "question": f"{label}问题{i}"})
    return rows


def test_pick_review_sample_takes_from_every_label():
    """**按类分层**抽 —— 不是随机抽。

    随机抽 15 条很可能一条「评价」都抽不到,而人审看不出那个类的系统性错误。
    """
    sample = pick_review_sample(_rows(), per_label=5, seed=0)
    assert len(sample) == 15
    from collections import Counter
    assert Counter(next(iter(r["labels"])) for r in sample) == {
        "尺码": 5, "退换货": 5, "运费": 5
    }


def test_pick_review_sample_is_reproducible_with_a_seed():
    """同一种子两次结果相同 —— 否则「我审的是哪 85 条」说不清。

    ⚠️ **计划订正 6-C(controller,2026-09-26):这条**自己**是**同义反复**,
    光有它**不足以**说明 seed 被用上了** —— 一个**完全忽略 seed** 的实现
    (根本不 shuffle、或每次都按 id 排序返回)同样满足「同一种子两次相同」。
    ⇒ **必须配下面那条 `test_different_seeds_pick_different_samples`**,
    两条合起来才把「随机且可复现」这件事钉住。
    """
    a = pick_review_sample(_rows(), per_label=5, seed=42)
    b = pick_review_sample(_rows(), per_label=5, seed=42)
    assert [r["id"] for r in a] == [r["id"] for r in b]


def test_different_seeds_pick_different_samples():
    """**不同种子给出不同样本** —— 这一条才是「seed 真的被用上了」的判据。

    只有上面那条时,把 `rng.shuffle(shuffled)` 整行删掉,**上面那条照样绿**
    (输入 20 条里取 5 条,不洗牌就是固定取前 5 条 —— 确定,但**不是抽样**)。
    本仓把这种叫「断言在它本该禁止的实现下依然通过」。

    ⚠️ 用 `per_label=5` / 每类 20 条:`C(20,5)` 很大,两个种子撞出**同一集合**
    的概率可忽略;万一将来有人把样本调小,这条会**偶发红**,
    那时该改的是**这条测试的规模**,不是把它删掉。
    """
    a = pick_review_sample(_rows(), per_label=5, seed=0)
    b = pick_review_sample(_rows(), per_label=5, seed=1)
    assert [r["id"] for r in a] != [r["id"] for r in b]


def test_pick_review_sample_takes_what_is_available():
    """某类只有 2 条时,拿 2 条而不是报错 —— 但要**如实少拿**。"""
    rows = [{"id": "a", "labels": ["尺码"], "question": "x"}] * 1 + \
           [{"id": f"b{i}", "labels": ["运费"], "question": f"y{i}"} for i in range(10)]
    sample = pick_review_sample(rows, per_label=5, seed=0)
    from collections import Counter
    got = Counter(next(iter(r["labels"])) for r in sample)
    assert got["尺码"] == 1 and got["运费"] == 5


def test_multi_label_rows_count_toward_every_label():
    """多标签行对**每个**它带的标签都算一个样本。

    只按第一个标签计数的话,多标签样本会集中在某一个类的抽样里,
    而另一个类的「5 条」里一条多标签都没有 —— 人审就看不到那类的多标签错误。
    """
    rows = [{"id": "m1", "labels": ["尺码", "退换货"], "question": "买大了想退"}]
    sample = pick_review_sample(rows, per_label=5, seed=0)
    assert [r["id"] for r in sample] == ["m1"]


def test_multi_label_rows_are_reachable_via_a_label_that_is_not_the_first():
    """⚠️ **实现者补的一条(2026-09-26)——上面那条对它**命名的那件事**零判别力。**

    上面那条 `..._count_toward_every_label` 的输入里 `m1` **每个标签的桶都只有它自己**,
    所以把实现改成「只用 `row["labels"][0]`」(只记第一个标签)**照样绿** ——
    少的那个桶对结果毫无影响。而它 docstring 说的正是「只按第一个标签计数的话……」。
    本仓把这种叫「断言在它本该禁止的实现下依然通过」(与订正 6-C 同族)。

    下面这条让「第二个标签」成为 `a-multi` **唯一**可靠的入口:
    「尺码」桶里有 **30** 条而只抽 3 条 ⇒ 靠第一个标签进来是**碰运气**;
    而「评价」桶只有 `a-multi` 一条 ⇒ 只要那一桶被处理过,它**必然**在样本里。
    ⇒ 正确实现下这条**恒绿**;把 `for label in row["labels"]` 改成只取第一个标签,
    则「评价」桶为空 ⇒ 红(变异 **M2** 实测过,证据在
    `.superpowers/ch10b_t6_mutation_probe.py` 与它的 `.txt` 转录,两份都已入库)。

    ⚠️ **如实记账**:这里的判别力**依赖种子**(种子固定 ⇒ 结果固定,不抖),
    但它不是「构造上必然」的 —— 30 选 3 里选中 `a-multi` 的概率约 10%,
    正好落到那个种子上时这条就**测不出**那个变异了。
    ⇒ 若将来 RNG 的消费顺序变了导致这条**变红**,先跑一遍变异确认,
    再**换一个种子/加大桶的规模**,不要删掉它。
    """
    rows = [{"id": f"z{i:02d}", "labels": ["尺码"], "question": "x"} for i in range(30)]
    rows.append({"id": "a-multi", "labels": ["尺码", "评价"], "question": "买大了想晒单"})
    ids = [r["id"] for r in pick_review_sample(rows, per_label=3, seed=0)]
    assert "a-multi" in ids


# ---- 抽审 CSV 的导出/回收接线(ch10 spec §6.3/§6.4)—— 不联网、不碰真产物 ----
#
# ⚠️ 这几条钉的是 `scripts/export_label_review.py` 的**接线**,不是纯函数。这条链上
# 有四个「错了也看不出来」的地方(①②出在 B6-A,③④出在订正轮 1):
#   ① 表头/列名对不上时,回收**逐行静默回落**成预标标签,而读数看着完全正常;
#   ② 「最终标签」列被顺手填成预标时,**用户改了哪几条就再也分不出来**了(B6-A 的承重不变量);
#   ③ **错误率读数**原本取自「判定」列 —— 用户忘了填判定 ⇒ 读数 0,而更正躺在文件里;
#   ④ 列错位(文本编辑器里打了逗号没加引号)/ id 不存在 / id 重复 ⇒ 无声吞掉用户的更正。
# 全部取 `tmp_path`,所以这几条**不碰** `evals/topic/labels/trainval.csv` 那份真产物
# (它此刻**正被用户改**,是 CP-2 的输入)。


def _write_review_csv(path, rows, header=None):
    import csv
    from scripts import export_label_review as exp

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(exp.HEADER if header is None else header)
        w.writerows(rows)


def _write_raw_csv(path, text):
    """**裸文本**写 CSV —— `csv.writer` 会自动加引号,复现不出「文本编辑器里手打逗号」那种行。"""
    path.write_bytes(text.encode("utf-8-sig"))


def _bind_import(monkeypatch, tmp_path, prelabeled_rows):
    """把回收端的三条路径都指到 `tmp_path`,并写好**配套的** `prelabeled.jsonl`。

    回收端从语料那边取题面、拿它当「改了没有」的基准(`prelabeled` 的 id 集合也是白名单),
    所以每个用例都必须给一份与 CSV **同源**的语料 —— 这也正是真产物的形状。
    """
    from scripts import export_label_review as exp

    pre = tmp_path / "prelabeled.jsonl"
    pre.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in prelabeled_rows) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(exp, "PRELABELED", pre)
    monkeypatch.setattr(exp, "LABELS_DIR", tmp_path)
    monkeypatch.setattr(exp, "REVIEWED", tmp_path / "reviewed.jsonl")
    return exp


def _read_reviewed(tmp_path):
    return [
        json.loads(l) for l in
        (tmp_path / "reviewed.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_export_writes_the_stratified_sample_with_the_judgement_columns_blank(
    monkeypatch, tmp_path, capsys
):
    """`export` 的接线:分层抽样(不是全量、不是随机)、后三列**空着**、BOM 在。

    ⚠️ **「后两列空着」是 B6-A 的承重不变量**(2026-09-26):那份 CSV 是**空着两列入库**的
    —— 用户改完之后 `git diff` 才是「他改了什么」的逐字记录(计划那句「进 git,可追溯」)。
    实现里若顺手把预标标签也写进「最终标签」,表看起来**更完整**,而**改了哪几条永远分不出来**。
    """
    import csv
    import io

    from scripts import export_label_review as exp

    src = tmp_path / "prelabeled.jsonl"
    src.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in (
            [{"id": f"{lb}-{i:02d}", "question": f"{lb}问题{i}", "labels": [lb]}
             for lb in ("尺码", "退换货", "运费") for i in range(20)]
        )) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(exp, "PRELABELED", src)
    monkeypatch.setattr(exp, "LABELS_DIR", tmp_path / "labels")

    exp.export()

    raw = (tmp_path / "labels" / "trainval.csv").read_bytes()
    # utf-8-sig:中文 Windows 的 Excel 打开无 BOM 的 CSV 会乱码。
    assert raw.startswith(b"\xef\xbb\xbf"), "BOM 没了 —— Excel 打开会乱码"
    table = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline="")))
    assert table[0] == exp.HEADER
    body = table[1:]
    assert len(body) == 15, "抽的不是 per_label=5 × 3 类(或没去重)"
    from collections import Counter
    assert Counter(r[2] for r in body) == {"尺码": 5, "退换货": 5, "运费": 5}
    assert all(r[1] for r in body), "「问题」列必须填(人要照着它判)"
    assert {(r[3], r[4], r[5]) for r in body} == {("", "", "")}, (
        "「判定」/「最终标签」/「备注」必须是**空**的 —— 填了就把用户改了什么这件事提前毁掉"
    )
    printed = capsys.readouterr().out
    assert "导出 15 条" in printed
    assert "每类条数" in printed


def test_import_takes_the_final_labels_and_falls_back_to_the_prelabels(
    monkeypatch, tmp_path, capsys
):
    """`import` 的接线:以「最终标签」为准、空了才回落预标、`reviewed` 标上、题面取预标那份。

    分隔符:`|` 与 `,` 都要认,且**两边的空白无所谓**(`尺码, 退换货` 是人最常打的形态)。
    """
    exp = _bind_import(monkeypatch, tmp_path, [
        {"id": "a", "question": "买大了想退", "labels": ["尺码"]},
        {"id": "b", "question": "快递到哪了", "labels": ["物流"]},
        {"id": "c", "question": "能便宜点吗", "labels": ["优惠活动"]},
    ])
    _write_review_csv(tmp_path / "trainval.csv", [
        ["a", "买大了想退", "尺码", "改", "尺码, 退换货", "两个诉求"],
        ["b", "快递到哪了", "物流", "ok", "", ""],
        ["c", "能便宜点吗", "优惠活动", "改", "价保", ""],
    ])

    exp.do_import()

    rows = _read_reviewed(tmp_path)
    assert [(r["id"], r["question"], r["labels"]) for r in rows] == [
        ("a", "买大了想退", ["尺码", "退换货"]),   # 用户填的(逗号 + 空格)
        ("b", "快递到哪了", ["物流"]),              # 没填 ⇒ 回落预标
        ("c", "能便宜点吗", ["价保"]),
    ]
    assert all(r["reviewed"] is True for r in rows)
    printed = capsys.readouterr().out
    assert "回收 3 条" in printed
    assert "标签与预标不同的 2 条" in printed


def test_import_counts_the_error_rate_from_the_labels_not_from_the_verdict(
    monkeypatch, tmp_path, capsys
):
    """★ 订正轮 1 的 **C1**(复审的 S1/S15):读数取自**标签变了没有**,不取自「判定」列。

    用户填了「最终标签」却**忘了填「判定」**(或判定还写着 `ok`)时,旧实现打印的是
    「判『改』的 0 条」—— 一个**看着像「用户什么都没改」的错答案**,而两条更正就躺在文件里;
    spec §6.3 会照着那个 0% 判「≤10% ⇒ 接受、开训」。

    机械定义(文件里 `生效标签 ≠ 预标标签` **就是**「这条被改了」,**不需要用户再报一次**)
    ⇒ 下面三条全算「改」,而「判定」只当交叉校验:少填的那两行**列出来**,但**不拦下整跑**
    (我们对用户说过「没填的按预标算」;而「少算错误率」这个后果由「照算」直接解决)。
    """
    exp = _bind_import(monkeypatch, tmp_path, [
        {"id": "a", "question": "买大了想退", "labels": ["尺码"]},
        {"id": "b", "question": "快递到哪了", "labels": ["物流"]},
        {"id": "c", "question": "能便宜点吗", "labels": ["优惠活动"]},
    ])
    _write_review_csv(tmp_path / "trainval.csv", [
        ["a", "买大了想退", "尺码", "", "尺码, 退换货", "忘了填判定"],
        ["b", "快递到哪了", "物流", "ok", "物流|商品信息", "判定还写着 ok"],
        ["c", "能便宜点吗", "优惠活动", "改", "价保", ""],
    ])

    exp.do_import()

    printed = capsys.readouterr().out
    # ⚠️ 旧实现(以及「把 changed 改回只数判定列」那个变异 **M-C1**)在这里读 **1**,不是 3
    #    —— 变异 M-C1 实测过,证据在 `.superpowers/ch10b_t6_mutation_probe.py`(已入库)。
    assert "标签与预标不同的 3 条" in printed, printed
    assert [r["labels"] for r in _read_reviewed(tmp_path)] == [
        ["尺码", "退换货"], ["物流", "商品信息"], ["价保"],
    ]
    warn = [l for l in printed.splitlines() if "没写「改」" in l]
    assert len(warn) == 1 and "'a', 'b'" in warn[0], printed


def test_import_takes_the_question_from_the_prelabels_not_from_the_csv(
    monkeypatch, tmp_path, capsys
):
    """★ 订正轮 1 的 **I3**(Controller 裁定):题面**以预标为准**,并把不一致的行**打出来**。

    任务 7 的合并是 `{**pre[id], "labels": ...}` ⇒ 训练语料本来就取预标那份;
    照抄 CSV 那份只会让 `reviewed.jsonl` 这个「人工劳动的记录」里题面与**证据串**不同源。
    代价:用户有意订正题面时会被忽略 ⇒ 靠那行打印 + `git diff` 兜底,**不是无声丢弃**。
    """
    exp = _bind_import(monkeypatch, tmp_path, [
        {"id": "a", "question": "买大了想退", "labels": ["尺码"]},
        {"id": "b", "question": "快递到哪了", "labels": ["物流"]},
    ])
    _write_review_csv(tmp_path / "trainval.csv", [
        ["a", "买大了想退货!!!", "尺码", "改", "尺码|退换货", "顺手改了题面"],
        ["b", "快递到哪了", "物流", "ok", "", ""],
    ])

    exp.do_import()

    rows = _read_reviewed(tmp_path)
    assert [r["question"] for r in rows] == ["买大了想退", "快递到哪了"], (
        "题面必须取自 prelabeled.jsonl —— 照抄 CSV 那份会让记录里题面与证据串不同源"
    )
    printed = capsys.readouterr().out
    note = [l for l in printed.splitlines() if "「问题」列与预标不一致" in l]
    assert len(note) == 1 and "'a'" in note[0], printed


def test_import_dedupes_repeated_labels(monkeypatch, tmp_path, capsys):
    """★ 订正轮 1 的 **Mn4**:`退换货|退换货` 不原样入库,也不静默。

    重复类目对训练没有额外含义(标签集本来就当集合用),原样写进产物只是脏数据;
    而去重这件事必须**看得见**(打出来),不然「用户的输入被改过」就是无声的。
    """
    exp = _bind_import(monkeypatch, tmp_path, [
        {"id": "a", "question": "买大了想退", "labels": ["尺码"]},
    ])
    _write_review_csv(tmp_path / "trainval.csv", [
        ["a", "买大了想退", "尺码", "改", "尺码|尺码|退换货", ""],
    ])

    exp.do_import()

    assert [r["labels"] for r in _read_reviewed(tmp_path)] == [["尺码", "退换货"]]
    printed = capsys.readouterr().out
    assert "重复类目" in printed and "'a'" in printed, printed


def test_import_is_not_fooled_by_ordering_or_padding(monkeypatch, tmp_path, capsys):
    """★ 订正轮 2 的 **F4(N1 + N4)**:两个「用户顺手多敲的字符」不该改变读数。

    复审跑了两个变异,**都是全绿**(即没有任何用例钉住它们):

    - **N1:把 `set(labels) != set(pre)` 换成列表比较(顺序敏感)**。
      标签是**集合**(spec §6.5 的「整条完全一致率」按集合比)⇒ 用户把
      `尺码|退换货` 写成 `退换货|尺码` 时,「改了没有」必须还是**没有**;
      按列表比会把它算成一次改动 ⇒ spec §6.3 的错误率**虚高**(而两边的标签一模一样)。
    - **N4:id 不 `strip()`**。文本编辑器里很容易在 id 两端多打空格;而 id 是任务 7 的
      合并键 —— 不洗的话要么在这里**硬失败**、要么把 `" a "` 写进产物(那边 `KeyError`)。
      (就这条断言:产物里的 id 必须是洗过的 `"a"`。)
    """
    exp = _bind_import(monkeypatch, tmp_path, [
        {"id": "a", "question": "买大了想退", "labels": ["尺码", "退换货"]},
        {"id": "b", "question": "快递到哪了", "labels": ["物流"]},
    ])
    # ⚠️ 裸文本:要造出「id 两端带空格」这种格子,csv.writer 写不出来(它会当普通字段)。
    _write_raw_csv(
        tmp_path / "trainval.csv",
        "id,问题,预标标签,判定(ok/改),最终标签,备注\r\n"
        + " a ,买大了想退,尺码|退换货,ok,退换货|尺码,\r\n"   # 只调序 + id 带空格
        + "b,快递到哪了,物流,ok,,\r\n",
    )

    exp.do_import()

    rows = _read_reviewed(tmp_path)
    assert [r["id"] for r in rows] == ["a", "b"], "id 两端没洗掉 ⇒ 写进产物的就是 `\" a \"`"
    # 标签保留**用户写的那个顺序**(我们不去重排),但「改了没有」按集合算 ⇒ 0 条
    assert [r["labels"] for r in rows] == [["退换货", "尺码"], ["物流"]]
    printed = capsys.readouterr().out
    assert "标签与预标不同的 0 条" in printed, (
        "只调序被算成了改动 ⇒ 错误率虚高(变异 N1 就是在这里红)"
    )
    assert "没写「改」" not in printed, "调序不该被当成「漏填判定」"


def test_import_refuses_input_that_would_silently_change_the_reading(monkeypatch, tmp_path):
    """四种「不响亮就会静默给出错读数」的输入 —— 每一种都必须在**写产物之前**停住。

    ① **表头缺列**:`row.get("最终标签")` 恒为 None ⇒ 每行静默回落预标,而读数看着正常;
    ② **判定值认不出**(`改了`):若当成「没改」,错误率被**少算** —— 一个乐观方向的错答案;
    ③ **标签是错字**(`尺碼`):静默丢弃的话,训练集里就少一个标签,而没人会知道;
    ④ ★**S18**:`判定=改` 而标签与预标**一模一样**(最常见成因:更正写进了「备注」列)
       ⇒ 那一行会以**旧标签**入库而记录说「改过」,两份记录互相拆台。

    ⚠️ 每条都断 `产物不存在` —— 「拦住了」与「拦住了但先写了一份错的」是两回事。
    """
    exp = _bind_import(monkeypatch, tmp_path, [
        {"id": "a", "question": "买大了想退", "labels": ["尺码"]},
    ])
    csv_path = tmp_path / "trainval.csv"
    out = tmp_path / "reviewed.jsonl"

    # ① 表头缺列(少了「最终标签」)
    _write_review_csv(
        csv_path, [["a", "买大了想退", "尺码", "改", "备注"]],
        header=["id", "问题", "预标标签", "判定(ok/改)", "备注"],
    )
    with pytest.raises(SystemExit) as e1:
        exp.do_import()
    assert "表头缺列" in str(e1.value), str(e1.value)
    assert not out.exists()

    # ② 判定值认不出
    _write_review_csv(csv_path, [["a", "买大了想退", "尺码", "改了", "退换货", ""]])
    with pytest.raises(SystemExit) as e2:
        exp.do_import()
    assert "认不出的值" in str(e2.value), str(e2.value)
    assert not out.exists()

    # ③ 标签错字
    _write_review_csv(csv_path, [["a", "买大了想退", "尺码", "改", "尺碼", ""]])
    with pytest.raises(SystemExit) as e3:
        exp.do_import()
    assert "不合法类目" in str(e3.value), str(e3.value)
    assert not out.exists()

    # ④ 判定说改过、标签却没变 ⇒ 两边互相拆台(注释里点了「备注」这个成因)
    _write_review_csv(csv_path, [["a", "买大了想退", "尺码", "改", "", "其实该是退换货"]])
    with pytest.raises(SystemExit) as e4:
        exp.do_import()
    assert "一模一样" in str(e4.value) and "备注" in str(e4.value), str(e4.value)
    assert not out.exists()


def test_import_refuses_malformed_rows(monkeypatch, tmp_path):
    """结构坏了要**响亮**停 —— 否则产物里会出现幽灵记录 / 互相矛盾的同 id 记录。

    ① ★**I1**(复审的 S17):在**文本编辑器**里往「最终标签」打了半角逗号却没加引号
       ⇒ 该行字段数比表头多,`DictReader` 把多出来的塞进 `row[None]`,而按列名取值**看不见它**
       ⇒ `最终标签` 只剩 `尺码`,**两个更正被无声吞掉**(条数一致、校验全过、产物照写)。
       Excel 存盘会自动加引号 ⇒ 这条**只在文本编辑器那条路上**炸,而计划推荐的正是那条路;
    ② **I2/S4**:用户自编号的行(`zz-9999`)⇒ 要到任务 7 才炸成 `KeyError`,报错指向别处;
    ③ **I2/S3**:两行同 id ⇒ 产物里两条标签互相矛盾的记录,而任务 7 的按 id 合并**静默**只留后者;
    ④ **Mn2**:纯逗号行 / 纯空格行 ⇒ id 洗成空串的幽灵记录(`{"id": "", "labels": []}`);
    ⑤ **Mn3**:只剩表头 ⇒ 空产物 + 「回收 0 条」,不拒绝;
    ⑥ **(实现者补的护栏)「预标标签」列与 `prelabeled.jsonl` 不同源** ⇒ 「改了没有」就没有基准 ——
       `生效标签 ≠ 预标标签` 那个机械定义会**照着错的基准**算,读数与标签一起错。
       两种形状(写成别的类目 / **少了后半截**)都要拦 —— 后者是 **F4 的 N5** 钉的,
       复审实测「集合相等 → 子集」这个放松**全绿**,而它的后果是错误率读成 ~100% 而不报错。
    """
    exp = _bind_import(monkeypatch, tmp_path, [
        {"id": "a", "question": "买大了想退", "labels": ["尺码"]},
        # ⚠️ `c` 是**两**个标签的 —— 第 ⑥ 条的「子集」形状必须有一个**更大的**语料行
        #    才造得出来(第一版拿单标签的 `a` 去写「截断」,结果那根本不是子集、当场红)。
        {"id": "c", "question": "买大了想退货", "labels": ["尺码", "退换货"]},
    ])
    csv_path = tmp_path / "trainval.csv"
    out = tmp_path / "reviewed.jsonl"
    head = "id,问题,预标标签,判定(ok/改),最终标签,备注\r\n"

    # ① 列错位(7 个字段 > 6 列表头)——
    #    ⚠️ 必须用**裸文本**写:csv.writer 会自动给含逗号的格子加引号,复现不出来。
    _write_raw_csv(csv_path, head + "a,买大了想退,尺码,改,尺码,退换货,物流\r\n")
    with pytest.raises(SystemExit) as e1:
        exp.do_import()
    assert "列错位" in str(e1.value) and "物流" in str(e1.value), str(e1.value)
    assert not out.exists()

    # ② 未知 id
    _write_raw_csv(csv_path, head + "a,买大了想退,尺码,改,退换货,\r\n"
                                + "zz-9999,我编的,尺码,改,退换货,\r\n")
    with pytest.raises(SystemExit) as e2:
        exp.do_import()
    assert "不在预标语料里" in str(e2.value) and "zz-9999" in str(e2.value), str(e2.value)
    assert not out.exists()

    # ③ 同 id 两行
    _write_raw_csv(csv_path, head + "a,买大了想退,尺码,改,退换货,\r\n"
                                + "a,买大了想退,尺码,ok,物流,\r\n")
    with pytest.raises(SystemExit) as e3:
        exp.do_import()
    assert "id 重复" in str(e3.value), str(e3.value)
    assert not out.exists()

    # ④ 幽灵行:纯逗号(5 个逗号 = **恰好** 6 个空字段,不会先撞上 I1 那条)+ 纯空格
    #    ⚠️ 这两条数据行必须**正好 6 个字段**:多一个逗号就会变成「字段比表头多」,
    #    于是 I1 那条先炸、而报的是**上一行**的 id —— 那会让这条用例测到别的东西上。
    #    (第一版就是这么写的,当场红;判据是本仓那句「先算一遍输入会不会走到那条分支」。)
    for ghost in (",,,,,\r\n", "   ,   ,   ,   ,   ,   \r\n"):
        _write_raw_csv(csv_path, head + "a,买大了想退,尺码,ok,,\r\n" + ghost)
        with pytest.raises(SystemExit) as e4:
            exp.do_import()
        assert "不在预标语料里" in str(e4.value), f"{ghost!r} -> {e4.value}"
        assert not out.exists()

    # ⑤ 只剩表头
    _write_raw_csv(csv_path, head)
    with pytest.raises(SystemExit) as e5:
        exp.do_import()
    assert "一行数据都没有" in str(e5.value), str(e5.value)
    assert not out.exists()

    # ⑥ 「预标标签」列与语料不同源 —— 两种形状都要拦:
    #    ① 写成**别的类目**;② **少了后半截**(子集)。它是「改了没有」的比较基准,
    #    基准错了 ⇒ 读数与标签一起错,而且都看不出来。
    #    ⚠️ ② 是**订正轮 2 的 F4(N5)** 钉的那条:复审实测把「集合相等」放松成「子集」后
    #    **全绿** —— 而子集版的后果最重:被截断的格子(CSV 写 `尺码`、语料是 `尺码|退换货`)
    #    会让**每一行未改动的行**都满足 `生效标签 ≠ 预标标签` ⇒ 错误率读成 ~100%,
    #    而**没有任何东西报错**(与 C1 同族的静默错读数)。
    for rid, question, wrong_pre in (
        ("a", "买大了想退", "物流"),        # ① 写成**别的类目**
        ("c", "买大了想退货", "尺码"),      # ② **少了后半截**(语料是 尺码|退换货)
    ):
        _write_review_csv(csv_path, [[rid, question, wrong_pre, "改", "退换货", ""]])
        with pytest.raises(SystemExit) as e6:
            exp.do_import()
        assert "不同源" in str(e6.value), f"预标={wrong_pre!r} -> {e6.value}"
        assert not out.exists()


def test_import_explains_a_file_that_is_not_utf8(monkeypatch, tmp_path):
    """★ 订正轮 1 的 **Mn5**:Excel 存成「CSV(逗号分隔)」⇒ GBK 字节。

    默认报的是 codec 措辞(`'utf-8' codec can't decode byte 0xc2 …`),读起来像代码坏了;
    真实成因只有一个常见解 ⇒ 报错要说人话。(不含糊、也**不写产物**这两点原本就对。)
    """
    exp = _bind_import(monkeypatch, tmp_path, [
        {"id": "a", "question": "买大了想退", "labels": ["尺码"]},
    ])
    csv_path = tmp_path / "trainval.csv"
    out = tmp_path / "reviewed.jsonl"
    csv_path.write_bytes(
        "id,问题,预标标签,判定(ok/改),最终标签,备注\r\na,买大了想退,尺码,改,退换货,\r\n"
        .encode("gbk")
    )

    with pytest.raises(SystemExit) as e:
        exp.do_import()
    assert "不是 UTF-8" in str(e.value) and "CSV(逗号分隔)" in str(e.value), str(e.value)
    assert not out.exists()


def test_export_script_pins_stdout_encoding(tmp_path):
    """`main()` **自己**必须把 stdout 钉成 UTF-8 —— 本机 locale 是 cp936。

    ⚠️ **为什么必须有这一条**:`⚠️`(U+26A0 + VS16)编不进 GBK(实测 2026-09-26),
    而它只出现在「某类一条都没抽到」那条 print 里 ⇒ **用真实数据跑一次 `export` 碰不到它**,
    那样测必然假绿(事实:当天的语料 17 类**每类都抽到了**,那行一次都不打印)。
    这里在**子进程里**跑 `main()`,再由**测试自己**打印一个 `⚠️` —— 钉没钉住当场见分晓。

    ⚠️ 子进程故意**不加 `-X utf8`**:加的话 stdout 恒是 UTF-8,这条断言就**恒真**。
    它的判别力**依赖本机是 cp936**,换一台 UTF-8 locale 的机器就退化成「函数存在」检查
    (如实记下来,不装作它到哪都同样有力)。

    ⚠️⭐ **订正轮 2 的 F1(复审实测的一条悬着的雷)**:这条测试的桩原来是
    `e.export = lambda: None`。**桩一旦静默失效**(有人给 `main()` 的调用路径改名 / 换个封装,
    桩就再也拦不住),子进程会跑到**真的 `export()`** —— 而它写的正是
    `evals/topic/labels/trainval.csv`,**用户此刻正在改的那份、被 git 跟踪的 CP-2 记录**
    ⇒ **用户的改动被覆盖,而这条测试照样绿**(它只断 stdout,不管产物)。
    ⇒ 所以子进程里**先**把 `LABELS_DIR` / `REVIEWED` 指到 `tmp_path`、**再**调 `main()`,
    并在父进程里补两句:**`tmp_path` 里没有 trainval.csv**(真 export 没跑过)+
    **真产物的字节前后相同**。这两句都**不依赖 print 措辞**。
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    # 真产物:断「它没被这次子进程动过」。⚠️ 它可能**还不存在**(全新 checkout 上没跑过 export)
    #  —— 那就断「跑完之后仍然不存在」,同样是「没被动过」。
    artifact = root / "evals" / "topic" / "labels" / "trainval.csv"
    before = artifact.read_bytes() if artifact.exists() else None
    tmp = tmp_path / "labels"
    code = "\n".join([
        "import sys",
        "from pathlib import Path",
        "from scripts import export_label_review as e",
        # ⚠️ **先改路径,再调 main()** —— 顺序反了就白改:真 export 会先落在真目录里。
        f"e.LABELS_DIR = Path({str(tmp)!r})",
        f"e.REVIEWED = Path({str(tmp / 'reviewed.jsonl')!r})",
        "e.export = lambda: None",       # 只桩掉写文件那一半(断言的是 stdout,不是产物)
        'sys.argv = ["export_label_review.py", "export"]',
        "e.main()",
        "print('⚠️ 某类一条都没抽到')",
    ])
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=root, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    assert proc.returncode == 0, f"钉编码没生效(main() 里那一行被删了?):\n{proc.stderr}"
    assert "某类一条都没抽到" in proc.stdout
    # ---- F1 的两句:「桩真的拦住了」与「真产物真的没被动」----
    # ⚠️ 两句都**不依赖 print 措辞**(措辞会变,而「有没有真的跑 export」不会)。
    assert not (tmp / "trainval.csv").exists(), (
        "子进程里**真的跑了 export**(桩失效了:main() 的调用路径被改名 / 换了封装?)"
        " —— 这次它落在 tmp 里、没伤人;但桩失效本身就意味着那条雷又回来了"
    )
    after = artifact.read_bytes() if artifact.exists() else None
    assert after == before, (
        f"{artifact} 在这次子进程里**被改动了** —— 这就是 F1 那条雷(用户的 CP-2 记录被覆盖)"
    )


# ---- 分层抽样与测试集冻结(ch10 spec §5.3 / §5.5)—— 纯函数,零 IO ----
#
# ⚠️ 这一节钉**抽样机制**;脚本侧(`split` 子命令)的接线 —— 合并 `reviewed.jsonl`、
#    两条零标签行的去向、train-vs-test 重复的摘除、三个产物名 —— 在**再下面那一节**。


def _mixed(n_real=100, n_synth=200):
    rows = []
    for i in range(n_real):
        rows.append({"id": f"r{i}", "provenance": "real", "labels": ["尺码"],
                     "question": f"真问题{i}"})
    for i in range(n_synth):
        rows.append({"id": f"s{i}", "provenance": "synthetic", "labels": ["运费"],
                     "question": f"合成问题{i}"})
    return rows


def test_test_set_has_the_prescribed_composition():
    """⚠️ **测试集构成是抽样的硬条件,不是抽完再看结果**(spec §5.3)。

    「真实 80 + 合成 40」是 §8.4 那三层报告能打出来的前提:
    少了它,「只看真实」那一列就没有足够的样本。
    """
    parts = stratified_split(_mixed(), test_real=80, test_synth=40, seed=0)
    test = parts["test"]
    from collections import Counter
    c = Counter(r["provenance"] for r in test)
    assert len(test) == 120
    assert c["real"] == 80 and c["synthetic"] == 40


def test_splits_do_not_overlap():
    """三份**互不相交** —— 有交集就是数据泄漏,而 F1 会因此虚高。"""
    parts = stratified_split(_mixed(), test_real=80, test_synth=40, seed=0)
    ids = [r["id"] for part in parts.values() for r in part]
    assert len(ids) == len(set(ids))


def test_all_rows_are_used():
    parts = stratified_split(_mixed(), test_real=80, test_synth=40, seed=0)
    assert sum(len(p) for p in parts.values()) == 300


def test_split_is_reproducible():
    a = stratified_split(_mixed(), test_real=80, test_synth=40, seed=7)
    b = stratified_split(_mixed(), test_real=80, test_synth=40, seed=7)
    assert [r["id"] for r in a["test"]] == [r["id"] for r in b["test"]]


def test_different_seeds_give_different_test_sets():
    """**不同种子给出不同切分** —— 这一条才把「`seed` 真的被用上」钉住。

    ⚠️ **上面那条「同一种子两次相同」是**同义反复**(计划订正 6-C 已在
    `pick_review_sample` 上记过同一个形状):一个**完全忽略 seed** 的实现
    (根本不 shuffle、每次都按 id 排序返回)同样满足它 —— 确定,但**不是抽样**。
    ⇒ 两条合起来才成立;少了这一条,把 `rng` 换成一个固定顺序的实现照样全绿。

    ⚠️ 判别力**依赖规模**:`_mixed()` 的真实池 100 条里取 80 条,
    `C(100,20)` 极大 ⇒ 两个种子撞出**同一集合**的概率可忽略。
    将来若有人把语料调小,这条会**偶发红**;那时该改的是**这条测试的规模**,
    而不是把它删掉(与 `test_different_seeds_pick_different_samples` 同款记账)。
    """
    a = stratified_split(_mixed(), test_real=80, test_synth=40, seed=0)
    b = stratified_split(_mixed(), test_real=80, test_synth=40, seed=1)
    assert [r["id"] for r in a["test"]] != [r["id"] for r in b["test"]]


def test_head_labels_are_oversampled_into_the_test_set():
    """头四类在测试集里每类 ≥15 条(spec §5.3)—— 否则那条 F1 是噪声。

    这条测试用的语料**故意让头四类样本充足**,好让「配额」这件事可观测;
    真实语料不够时,`stratified_split` 应当**如实少给**并在返回里标注,
    而不是硬凑(硬凑会重复使用同一样本 ⇒ 数据泄漏)。

    ⚠️ **如实记账:这条对「分层 vs 按比例随机」的判别力很弱。** 语料四类**等量**
    (各 60 条)、测试集要 80 条 ⇒ 按比例随机抽也**期望**每类 20 条,
    `>= 8` 在两种实现下都过(无变异可红)。真正钉「轮转取」的是下面
    `test_rare_labels_are_not_squeezed_out_of_the_test_set`(小类必须露面)。
    """
    rows = []
    for label in ("退换货", "物流", "尺码", "发票"):
        for i in range(60):
            rows.append({"id": f"{label}{i}", "provenance": "real",
                         "labels": [label], "question": f"{label}问题{i}"})
    for i in range(60):
        rows.append({"id": f"x{i}", "provenance": "synthetic",
                     "labels": ["评价"], "question": f"评价问题{i}"})
    parts = stratified_split(rows, test_real=80, test_synth=40, seed=0)
    from collections import Counter
    c = Counter(lb for r in parts["test"] for lb in r["labels"])
    for label in ("退换货", "物流", "尺码", "发票"):
        assert c[label] >= 8, f"头四类里的 {label} 在测试集只有 {c[label]} 条"


def test_the_train_val_boundary_comes_from_the_ratios_argument():
    """★ 订正轮 1 的 **I3**:train/val 的边界必须**真的**由 `ratios` 决定。

    ⚠️ **原来这条完全没有**:把公式换成写死的 `len(rest_sorted) * 8 // 9`,
    另外两条测试**全绿**(复审复算过:公式本身是对的 —— `rest = 1303`、
    `round(1303 × 0.8/0.9) = 1158`,与真产物一致 —— **错的是「没有人钉住那个参数」**)。

    两句各钉一半:
    - **`ratios` 影响边界**:`(0.6, 0.2, 0.2)` 与默认下 train 的长度必须不同,
      且**短的那个是长的那个的前缀**(边界挪在同一条序列上,不是重新打乱);
    - **`ratios[2]` 不影响任何东西**:两组的**测试集必须逐条相同**
      —— 测试集是**先按固定条数取走**的(spec §5.3 的硬条件:真实 80 + 合成 40),
      它若跟着 `ratios[2]` 变,§8.4 那三层报告就凑不出来。
      ⇒ 「签名里写着三元组、docstring 说按 8:1:1、而第三个元素一次都没被读」这件事
      **从注释变成了断言**(`stratified_split` 的 docstring 也写明了它不参与计算)。
    """
    default = stratified_split(_mixed(), test_real=80, test_synth=40, seed=0)
    other = stratified_split(_mixed(), test_real=80, test_synth=40,
                             ratios=(0.6, 0.2, 0.2), seed=0)
    n = len(other["train"])
    assert len(default["train"]) != n, "改了 ratios,train/val 的边界一动不动"
    # ⚠️ **这两句钉的是分母**:`ratios[0] / (ratios[0] + ratios[1])`,**不是** `/ sum(ratios)`。
    #    后者会把 `ratios[2]` 偷偷读进分母 —— 而上面那句「边界跟着动」**抓不住它**
    #    (两组一起缩小,相对关系不变;变异 **I3-b** 实测:只有这两句能红)。
    rest = 300 - 120          # `_mixed()` 300 条,测试集先按固定条数取走 120
    assert len(default["train"]) == round(rest * 0.8 / 0.9)
    assert n == round(rest * 0.6 / 0.8)
    assert [r["id"] for r in other["train"]] == [r["id"] for r in default["train"]][:n], (
        "边界不是在同一条序列上挪的 —— 那说明 ratios 顺带把抽样顺序也改了"
    )
    assert {r["id"] for r in other["val"]} == {
        r["id"] for r in default["train"][n:] + default["val"]
    }
    assert [r["id"] for r in other["test"]] == [r["id"] for r in default["test"]], (
        "ratios[2] 被读进去了 —— 测试集必须由 test_real/test_synth 定死(spec §5.3)"
    )


def test_rare_labels_are_not_squeezed_out_of_the_test_set():
    """★ 小类必须**露面** —— 这一条才是「分层」区别于「按比例」的判据。

    语料:两个大类各 100 条 + **10 个小类各 2 条**(共 220 条真实),测试集 80 条。

    - **轮转取**(实现如此):12 个桶轮着弹,80 次里每个桶都被弹到 6–7 次
      ⇒ 那 10 个小类各自的 2 条**全部**进测试集 ⇒ 正确实现下这条**恒绿**;
    - **按比例取**(变异):小类占 20/220 ≈ 9%,期望每类 1.6 条 ——
      某个小类**一条不剩**的概率约 0.4,十个全露面只有 ≈ 0.6% ⇒ 变异约 99% 红。

    ⚠️ **判据**:上面那条 `>= 8` 断的是「头四类够不够多」,而**等量语料下随机抽
    也够多** ⇒ 它对「有没有分层」零判别力。这一条断的是「**小类还在不在**」——
    那正是 `take()` 里那句「轮转取,保证每个类都有代表」在说的事。变异 **M5** 实测过。
    """
    rows = []
    for label in ("尺码", "退换货"):
        for i in range(100):
            rows.append({"id": f"{label}{i}", "provenance": "real",
                         "labels": [label], "question": f"{label}问题{i}"})
    rare = ("发票", "价保", "支付", "账号", "会员积分", "评价", "商品信息", "保修维修",
            "库存补货", "订单修改")
    for label in rare:
        for i in range(2):
            rows.append({"id": f"{label}{i}", "provenance": "real",
                         "labels": [label], "question": f"{label}问题{i}"})
    parts = stratified_split(rows, test_real=80, test_synth=0, seed=0)
    appeared = {lb for r in parts["test"] for lb in r["labels"]}
    missing = [lb for lb in rare if lb not in appeared]
    assert missing == [], f"这些小类在测试集里一条都没有:{missing}"


# ---- 靶子不可信的行(计划订正 9-E)----


def test_unusable_target_predicate():
    """9-E 的判据:零标签**且****有**被证据校验拒掉的标签。

    ⚠️ 两行零标签在产物里**逐字节相同**(都是 `labels: []`)⇒ **只按 `labels` 分不开**
    `r-0049`「你是」(模型真判零诉求)与 `s-0423`「首重多少,超了咋算?」
    (证据校验机械拒到空,`rejected_labels == ['运费']`)—— 后者的真实主题几乎肯定是运费。
    """
    assert is_unusable_target({"labels": [], "rejected_labels": ["运费"]}) is True
    # 模型真判了零诉求 ⇒ 留着(它就是「该判 `其他`」的样本)
    assert is_unusable_target({"labels": [], "rejected_labels": []}) is False
    # 有标签 ⇒ 被拒的那几个不影响这行是个正常样本
    assert is_unusable_target({"labels": ["运费"], "rejected_labels": ["尺码"]}) is False
    assert is_unusable_target({"labels": ["运费"], "rejected_labels": []}) is False
    # 键缺失(别的调用方 / 老产物):**有**被拒的标签才成立,缺键不算
    assert is_unusable_target({"labels": []}) is False


def test_rows_with_an_untrusted_target_are_dropped_out_of_the_split():
    """★ 9-E:靶子不可信的行**哪一份都不进**(排除出 train/val,也不许挪进测试集)。

    以空标签喂进训练是在教模型「这句话没有主题」—— 而那是个**处理产物、不是判断**。

    ⚠️ **反面对照是这条的一半**:模型真判零诉求的那行(`r-0049` 的形状)**要留着**,
    测试集里正缺一条「该判 `其他`」的样本。⇒ 少了下面第二句,把过滤写成
    `labels == []`(两行一起排除)照样绿。
    """
    rows = _mixed(n_real=20, n_synth=20)
    rows.append({"id": "s-bad", "provenance": "synthetic", "labels": [],
                 "rejected_labels": ["运费"], "question": "首重多少,超了咋算?"})
    rows.append({"id": "r-judged", "provenance": "real", "labels": [],
                 "rejected_labels": [], "question": "你是"})
    parts = stratified_split(rows, test_real=8, test_synth=4, seed=0)
    ids = [r["id"] for part in parts.values() for r in part]
    assert "s-bad" not in ids, "靶子不可信的行还是进了某一份"
    assert "r-judged" in ids, "模型真判零诉求的行被一起排除了 —— 过度处置"


# ---- train-vs-test 重复检查(计划订正 9-C)----


def test_train_test_overlap_reports_exact_duplicates_and_near_pairs():
    """★ 9-C:`train_test_overlap` 必须**报得出**训练侧抄了测试侧的句子。

    ⚠️ **为什么要有这一步**(T5 复审发现,见 Task 7 节首):`POSITIVE` 的 34 句正例
    进了 **T4 生成器**的 prompt,而 T4 的禁词表**不覆盖这些例句** ⇒
    重跑 T4 可能把「买大了」「175 穿什么码」整句抄进问句 ⇒ **训练侧抄了测试侧**。
    已发生的一半:冻结测试集里 8 行与渲染块例句字面重叠。

    两档判据、两种处置(混起来处置就错了):
    - **完全相同**(**清洗后**文本相等)⇒ 报进 `exact_train_ids`,`split()` 据此**移出训练侧**;
    - **近重复**(字符二元组 Jaccard ≥ `near`)⇒ **只报不删** ——
      删了就是拿一条**启发式判据**改数据。

    这条用例的两个输入各自钉一件事:
    - `r-exact` 与 `t-exact` **原文不同**(一个是全角空格 U+3000)而**清洗后相同**
      ⇒ 把 `clean()` 换成原文比较,第一条断言就红(变异 **M6**);
    - `r-near` 与 `t-near` 只差最后一个字(「呢」/「啊」)⇒ Jaccard ≈ 0.83 ≥ 0.8,
      但**不是**完全相同 ⇒ 把 `near` 抬到 1.0 就红(变异 **M7**);
    - 同时 `pairs == [...]` 也钉住「完全相同的那对**不重复计入** near」
      (它在 `exact` 里,谁也不会漏看;重复报会让「近重复 N 对」这个读数虚高)。
    """
    train = [
        {"id": "r-exact", "question": "首重　多少", "labels": ["运费"]},       # 全角空格
        {"id": "r-near", "question": "这个订单什么时候能发货呢", "labels": ["物流"]},
        {"id": "r-unrelated", "question": "完全无关的一句", "labels": ["其他"]},
    ]
    test = [
        {"id": "t-exact", "question": "首重 多少", "labels": ["运费"]},
        {"id": "t-near", "question": "这个订单什么时候能发货啊", "labels": ["物流"]},
    ]
    out = train_test_overlap(train, test)          # ← 用**默认** `near`,见变异 M7
    assert out["exact_train_ids"] == ["r-exact"]
    pairs = [(a, b) for a, b, _ in out["near_pairs"]]
    assert pairs == [("r-near", "t-near")], out["near_pairs"]
    sim = out["near_pairs"][0][2]
    assert 0.8 <= sim < 1.0, f"近重复的相似度不该落在 [0.8, 1.0) 之外:{sim}"


def test_train_test_overlap_is_empty_when_nothing_matches():
    """什么都没抄时两个读数都必须是**空的** —— 否则「报出 0 条」这句话没有基准。

    (上面那条只钉了「有重复时报得出来」;一个**恒报**的实现照样能过它。)
    """
    train = [{"id": "r1", "question": "买大了想退", "labels": ["尺码"]}]
    test = [{"id": "t1", "question": "发票多久寄到", "labels": ["发票"]}]
    assert train_test_overlap(train, test) == {"exact_train_ids": [], "near_pairs": []}


def test_train_test_overlap_honours_the_near_threshold():
    """`near` 是**参数**,不是写死的常数 —— 两个方向各钉一次。

    默认 0.8 是「报数」那一档(只报很像的);排查时可以调低看全一些。
    把默认值改成 1.0(等价于「只认完全相同」)或把阈值写死,都会让**其中一个方向**红:

    - `near=0.2` 那一句:写死 0.8 的实现会返回空 ⇒ 红;
    - 默认那一句:默认值抬到 1.0 的实现也会返回空 ⇒ 红(变异 **M7**)。

    这对句子的相似度实测约 **0.5**(「买大了想退」vs「买大了想换货」):
    共享「买大/大了/了想」三个二元组,并集 6 个 —— 硬编码一个能同时满足
    两个方向的常数是做不到的。
    """
    train = [{"id": "r1", "question": "买大了想退", "labels": ["尺码"]}]
    test = [{"id": "t1", "question": "买大了想换货", "labels": ["退换货"]}]
    assert train_test_overlap(train, test)["near_pairs"] == [], "默认阈值不该把 0.5 相似度算进来"
    low = train_test_overlap(train, test, near=0.2)["near_pairs"]
    assert len(low) == 1 and low[0][:2] == ("r1", "t1"), low


# ---- `split` 子命令的接线(ch10 spec §5.3 / §5.5)—— 不联网、不碰库、不碰真产物 ----
#
# ⚠️ 这一节的全部理由是:上面那些纯函数**对了**,接不上也照样全绿。四处接线各自
#    对应一个「看起来做完了、其实没接上」:
#   ① `reviewed.jsonl` 的**覆盖**没做 ⇒ 用户那 84 条的复核成果不进训练集(只影响报表);
#   ② 两行零标签没有显式处置 ⇒ 靶子不可信的那行以空标签喂进训练;
#   ③ train-vs-test 的**完全相同**没有摘除 ⇒ 数据泄漏,而 F1 虚高;
#   ④ 产物写成了 `test.jsonl`(订正 9-B)⇒ 用户那 120 条的复核对指标**零影响**。


def _read_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _bind_split(monkeypatch, tmp_path):
    """`split` 的三个路径全指到 `tmp_path` —— 真产物 `evals/topic/*.jsonl` 此刻不该被写。"""
    from scripts import prepare_topic_data as p

    monkeypatch.setattr(p, "TOPIC_DIR", tmp_path)
    monkeypatch.setattr(p, "PRELABELED", tmp_path / "prelabeled.jsonl")
    monkeypatch.setattr(p, "REVIEWED", tmp_path / "reviewed.jsonl")
    return p


def _split_corpus(tmp_path):
    """给接线用例造语料。⚠️ **两件事都有论证,不是靠运气**(本仓规矩:
    构造输入前先算一遍它会不会走到那条分支)。

    **① `exact_train_ids` 非空**:真实池 **122** 条 = 文本甲 ×**61** + 文本乙 ×**60**
    + 一条零标签;切走 80 条后训练侧(含验证)剩 **42** 条 ⇒ 两个文本的份数(61 / 60)
    **都大于 42** ⇒ 哪个文本都不可能**整组**落在训练侧 ⇒ 两个文本**两侧都有**
    ⇒ 训练侧的**每一条真实行**都是「完全相同」,`exact = 42`。

    **② 验证侧**也**必须**有该被摘的行(订正轮 1 的 **I2**)——
    这一条以前**没有**,而它正是漏洞所在:检测那一步的**入参**写成
    `parts["train"]`(漏掉 val)时,**48 条全绿**(复审实测)。
    论证:`rest` = 42 条真实 + **4** 条合成(合成池 44 − 测试集 40);
    `n_train = round(46 × 8/9) = 41` ⇒ **val 只有 5 条**,而合成的只有 4 条
    ⇒ **val 里至少有 1 条真实行**(≥1 条该被摘的)⇒ 「检测漏喂 val」与
    「摘除只走 train」两种错法都会**少摘**,可被下面第 ③b 条的**独立复算**抓住。
    """
    rows = [{"id": f"r-{i:04d}", "provenance": "real", "source": "chat",
             "labels": ["尺码"], "question": "重复句甲"} for i in range(61)]
    rows += [{"id": f"r-1{i:03d}", "provenance": "real", "source": "chat",
              "labels": ["尺码"], "question": "重复句乙"} for i in range(60)]
    # 9-E 的两行形状:模型真判零诉求 / 证据校验机械拒到空
    rows.append({"id": "r-9001", "provenance": "real", "source": "chat",
                 "labels": [], "rejected_labels": [], "question": "你是"})
    # ⚠️ 合成只留 **4** 条进 rest(44 − 40)—— 上面论证 ② 靠的就是这个数(`< val 的 5 条`)。
    rows += [{"id": f"s-{i:04d}", "provenance": "synthetic", "source": "gen",
              "labels": ["运费"], "question": f"合成问题{i}"} for i in range(44)]
    rows.append({"id": "s-9002", "provenance": "synthetic", "source": "gen",
                 "labels": [], "rejected_labels": ["运费"], "question": "首重多少,超了咋算?"})
    tmp_path.joinpath("prelabeled.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    # 人工改过的那一条:`s-0000` 的标签被改成「发票」—— 覆盖必须生效,
    # 否则用户那 84 条的复核成果进不了语料(只影响报表)。
    tmp_path.joinpath("reviewed.jsonl").write_text(
        json.dumps({"id": "s-0000", "question": "合成问题0", "labels": ["发票"]},
                   ensure_ascii=False) + "\n", encoding="utf-8")
    return rows


def test_split_wires_the_overlay_the_zero_label_rows_and_the_overlap_removal(
    monkeypatch, tmp_path, capsys
):
    """`split` 的四条接线一次钉住(逐条见本节节首那张表)。"""
    import re

    p = _bind_split(monkeypatch, tmp_path)
    rows = _split_corpus(tmp_path)

    p.split()

    out = capsys.readouterr().out
    files = {name: _read_jsonl(tmp_path / f"{name}.jsonl") for name in ("train", "val")}
    files["test"] = _read_jsonl(tmp_path / "topic_test.jsonl")
    # ⚠️ **订正 9-B**:测试集产物叫 `topic_test.jsonl`。写成 `test.jsonl` 的话
    #    下面这行直接红 —— 而那正是「用户那 120 条的复核对指标零影响、报告照常打印」
    #    在测试里的样子。
    assert not (tmp_path / "test.jsonl").exists(), "写出了 `test.jsonl`(订正 9-B 那个名字)"
    written = [(name, r) for name, items in files.items() for r in items]
    ids = [r["id"] for _, r in written]

    # ---- ① `reviewed.jsonl` 的覆盖 ----
    s0 = [r for _, r in written if r["id"] == "s-0000"]
    assert len(s0) == 1 and s0[0]["labels"] == ["发票"], (
        "人工改过的标签没覆盖进语料 ⇒ 用户那 84 条的复核成果只影响报表、不进训练"
    )

    # ---- ② 两行零标签的去向 ----
    assert "零标签行" in out, "没有显式列出零标签行的去向"
    for rid in ("r-9001", "s-9002"):
        assert f"{rid} " in out, f"{rid} 没被列出去向"
    assert "'运费'" in out, "不可信那行必须把 `rejected_labels` 打出来(否则与真判零诉求分不开)"
    # ⭐ **订正轮 1 的 I1**:那一行的**落位必须是从 `parts` 里读出来的**。
    # 原来谓词为真时 `where` 是**硬编码**的「已排除」,从不查实际落位 ⇒ 摘除失效时
    # 打印**照旧**说「已排除」,而那行其实在 `train`(复审实测)。这一行打印是
    # dev-notes / 报告里「零标签两行去向」那个读数的**唯一来源** ⇒ 它撒谎就是读数撒谎。
    bad = [l for l in out.splitlines() if "s-9002" in l]
    assert len(bad) == 1, bad
    assert "**不在任何一份里**" in bad[0], (
        f"零标签行的去向必须**读实际落位**(它是读数),谓词只许给「为什么」:\n{bad[0]}"
    )
    assert "已排除" not in bad[0].split("(")[0], (
        f"去向那一格又被写成硬编码的「已排除」了(它没查 `parts`):\n{bad[0]}"
    )
    assert "s-9002" not in ids, "靶子不可信的行还是进了某一份"
    assert ids.count("r-9001") == 1, (
        "模型真判零诉求的行被一起丢了 —— 过度处置(它该留在某一份里)"
    )

    # ---- ③ train-vs-test 的完全相同被摘除 ----
    m = re.search(r"完全相同 (\d+) 条", out)
    assert m, f"没有打印完全相同条数:\n{out}"
    exact = int(m.group(1))
    assert exact > 0, "这份语料的构造保证了至少一条,读到 0 ⇒ 检查没接上"
    assert len(ids) == len(set(ids)), "三份有交集 —— 数据泄漏"
    total_in = len(rows)
    assert len(ids) == total_in - 1 - exact, (
        f"写出的条数与「输入 {total_in} − 不可信靶子 1 − 完全相同 {exact}」对不上"
        f"(实际 {len(ids)})⇒ 摘除那一步没接上"
    )
    masked = set(r["id"] for r in rows) - set(ids) - {"s-9002"}
    assert not (masked & {r["id"] for r in files["train"] + files["val"] + files["test"]})

    # ---- ③a **用例自检**:验证侧真的有该被摘的行(见 `_split_corpus` 的论证 ②)----
    # ⚠️ 这一句防的是「这条用例**悄悄失去判别力**」:语料若哪天不再保证 val 侧有重复,
    #    下面 ③b 的复算会变得**恒真**,而**不会有任何东西红**。
    m2 = re.search(r"train 摘 (\d+) 条、val 摘 (\d+) 条", out)
    assert m2, f"没有打印逐份摘除数(自检要读它):\n{out}"
    # ⚠️ **这条读数的两种成因必须分开说**(它自己没法分辨,别让它替读者下结论):
    #    (a) 语料漂移:val 里不再有该被摘的行 ⇒ 这条用例对「检测漏喂 val」失去判别力;
    #    (b) **代码漏了 val**(检测只喂 `parts["train"]`,或摘除循环只走 `train`)。
    #    (b) 是复审在 I2 里指的那个洞,而它**正是在这里现形**的(变异 I2-a 实测)。
    assert int(m2.group(2)) >= 1, (
        f"val 那一侧摘了 0 条 —— 两种成因,先分清再修:\n"
        f"  (a) **语料漂移**:val 里不再有该被摘的行(见 `_split_corpus` 的论证 ②)"
        f" ⇒ 这条用例已对「检测漏喂 val」失去判别力;\n"
        f"  (b) **代码漏了 val**:检测那一步只喂了 `parts['train']`,或摘除循环只走 `train`。\n"
        f"train {m2.group(1)} / val {m2.group(2)}"
    )

    # ---- ③b **独立复算**该被摘掉的全集(订正轮 1 的 I2)----
    # ⚠️ 上面那句一致性(`len(ids) == … − exact`)**抓不住**下面这个缺陷:
    #    检测只喂 `parts["train"]`(漏掉 val)时,「检出的」与「摘掉的」用的是**同一个
    #    小集合** ⇒ 两边一起变小,**一致性照样成立**(复审实测:48 条全绿)。
    #    ⇒ 必须在这儿**另算一遍**:训练侧里凡是文本与某个测试侧行逐字相同的,都该被摘。
    #    (这里是**原文**比较,不是 `clean` 后比较 —— 这份语料的重复句本来就连原文都相同,
    #     所以这一句不依赖被测的那条清洗口径。)
    test_texts = {r["question"] for r in files["test"]}
    test_ids = {r["id"] for r in files["test"]}
    pre_trainval = [r for r in rows if r["id"] not in test_ids and r["id"] != "s-9002"]
    expect_drop = {r["id"] for r in pre_trainval if r["question"] in test_texts}
    assert len(expect_drop) == exact, (
        f"独立复算出该摘 {len(expect_drop)} 条、打印说 {exact} 条 —— 检测那一步的"
        f"**入参**漏了训练侧的一半?(`parts['train'] + parts['val']` 里少了哪一半)"
    )
    assert masked == expect_drop, (
        f"实际摘掉的和该摘的对不上:少了 {sorted(expect_drop - masked)}、"
        f"多了 {sorted(masked - expect_drop)}"
    )

    # ---- ④ 测试集构成(与 `split()` 里那条 assert 同源,这里独立复算一遍)----
    from collections import Counter
    c = Counter(r["provenance"] for r in files["test"])
    assert len(files["test"]) == 120 and c["real"] == 80 and c["synthetic"] == 40


def test_main_dispatches_split(monkeypatch):
    """`main()` 的分派:`split` 必须走到 `split()`,不是「还没实现」那条路。

    ⚠️ 守的是**命令行那一半**:上面那条直接调 `p.split()`,所以把 `main()` 里
    `elif args.step == "split"` 那一支删掉(或写错分支名)时它**照样绿** ——
    而 `bash` 里那一行 `.venv/.../prepare_topic_data.py split` 会变成
    `SystemExit: split 还没实现(由后续任务补上)`,读起来像「这功能没做」。
    """
    import sys

    from scripts import prepare_topic_data as p

    called: list[str] = []
    monkeypatch.setattr(p, "split", lambda: called.append("split"))
    monkeypatch.setattr(sys, "argv", ["prepare_topic_data.py", "split"])

    p.main()

    assert called == ["split"]


def test_export_test_writes_every_row_with_the_judgement_columns_blank(
    monkeypatch, tmp_path, capsys
):
    """`export-test` 的接线:全量(不是抽 5 条)、后三列**空着**、BOM 在。

    ⚠️ **「全部 120 条」是 spec §6.4 的原文**(测试集 100% 人工裁决),
    而它与训练/验证那份抽审(`per_label=5`,84 条)**不是同一条路** ——
    照抄 `export()` 会把测试集抽成「5 条一类」,而**报告里那句话会是「已 100% 复核」**。

    ⚠️ **「后三列空着」在这里同样承重**:填了就把「用户改了什么」提前毁掉
    (B6-A 的承重不变量,`git diff` 是唯一的逐字记录)。
    """
    import csv
    import io

    from scripts import export_label_review as exp

    src = tmp_path / "topic_test.jsonl"
    src.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in (
            [{"id": f"t{i:03d}", "question": f"测试问题{i}", "labels": ["尺码"],
              "provenance": "real", "split": "test"} for i in range(7)]
            + [{"id": "t007", "question": "多标签的", "labels": ["尺码", "退换货"],
                "provenance": "synthetic", "split": "test"}]
        )) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(exp, "TOPIC_TEST", src)
    monkeypatch.setattr(exp, "LABELS_DIR", tmp_path / "labels")

    exp.export_test()

    raw = (tmp_path / "labels" / "test.csv").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf"), "BOM 没了 —— Excel 打开会乱码"
    table = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline="")))
    assert table[0] == exp.HEADER, "表头必须与 trainval.csv **同一个**(订正 9-D)"
    body = table[1:]
    assert len(body) == 8, f"不是全量导出(应是 8 条,实际 {len(body)})"
    assert body[0][1] == "测试问题0", "「问题」列必须填(人要照着它判)"
    assert body[0][2] == "尺码"
    assert body[7][2] == "尺码|退换货", "多标签要原样带出来(分隔符与 trainval 一致)"
    assert {(r[3], r[4], r[5]) for r in body} == {("", "", "")}, (
        "「判定」/「最终标签」/「备注」必须是**空**的 —— 填了就把用户改了什么提前毁掉"
    )
    printed = capsys.readouterr().out
    assert "导出 8 条" in printed


def test_the_frozen_test_set_path_points_at_topic_test_jsonl():
    """★ 订正 9-B 的**名字**要有一条直接断言 —— 读端与写端**逐字相同**。

    ⚠️ **为什么不能只靠上一条**:上一条 `monkeypatch` 掉了 `TOPIC_TEST`
    ⇒ `export_test` 里的常量**指错文件它也照样绿**(变异 **M15** 实测:全绿)。
    而它指错时的后果分两种,只有一种响:

    - 指到不存在的文件(`test.jsonl`)⇒ 真实链路上 `FileNotFoundError`,**响**;
    - 指到 `prelabeled.jsonl` 那类**存在**的文件 ⇒ **导出 1424 条给人看**,
      而**没有任何东西会报错** —— 用户会照着那份 CSV 做 100% 复核,
      而评测读的仍是 120 条的那份。

    ⇒ 下面两句:第一句钉**字面**,第二句钉**两侧一致**(切分端写哪里、导出端读哪里)。
    第二句是本任务最值钱的一处 —— 「同一个文件名散落四处」正是订正 9-B 的成因。
    """
    from scripts import export_label_review as exp
    from scripts import prepare_topic_data as p

    assert exp.TOPIC_TEST == exp.ROOT / "evals" / "topic" / "topic_test.jsonl"
    assert exp.TOPIC_TEST == p.TOPIC_DIR / p.TEST_NAME, (
        "切分端写的名字与导出端读的名字不是同一个 —— 用户那 120 条的复核对指标零影响"
    )


def test_main_dispatches_export_test(monkeypatch):
    """`main()` 的命令行分派:`export-test` 必须走到 `export_test()`,不是 `do_import()`。

    ⚠️ **不做这一步的后果是「动作存在但够不着」**:`export_test()` 写好了、单测也绿,
    而命令行打 `export-test` 时**跑的是回收端**(它读 `trainval.csv`,
    在用户那份文件上以「预标标签列不同源」之类的理由停住,或者更糟 —— 真的写产物)。
    这一个位置没有任何别的东西能守住:上一条测的是**函数体**,这条测的是**分派**。
    """
    import sys

    from scripts import export_label_review as exp

    called: list[str] = []
    monkeypatch.setattr(exp, "export", lambda: called.append("export"))
    monkeypatch.setattr(exp, "do_import", lambda: called.append("import"))
    monkeypatch.setattr(exp, "export_test", lambda: called.append("export-test"))
    monkeypatch.setattr(sys, "argv", ["export_label_review.py", "export-test"])

    exp.main()

    assert called == ["export-test"]


# ---- 数据增强:标签漂移自检 + 注入错别字(ch10 spec §5.4;计划订正 11)----
#
# ⚠️ 本节两条线,别混:
#   ① **纯函数**(`label_drift` / `inject_typo` / `typo_seed`)—— 零 IO,可密集断言;
#   ② `scripts/prepare_topic_data.py` 的 `augment` **接线** —— ① 全对了,
#      把它接到 `val.jsonl` 上(**或者干脆接到 `topic_test.jsonl` 上当产物**)
#      **照样全绿**,而那正是 spec §5.4 唯一一条硬约束的反面。
#      ⚠️ 后者是**静默**的破坏:§8 那套指标会变好看,没人会去查测试集被动过。


def test_label_drift_detects_changed_count():
    """标签集合变了就是漂移(「买大了想退」→「不喜欢这个想退」少了一个诉求那种)。"""
    assert label_drift(["尺码", "退换货"], ["退换货"])          # 少了一个
    assert label_drift(["退换货"], ["尺码", "退换货"])          # 多了一个
    assert not label_drift(["尺码", "退换货"], ["退换货", "尺码"])  # **顺序不算变化**


def test_label_drift_is_set_semantics_on_the_other_two_faces():
    """上面那条只钉了「顺序」这一个面 —— 集合语义还有「重复」与「空」两个面,各自对应一种错法。

    - 写成列表比较 ⇒ 上面那条红;
    - 写成 `len(before) != len(after)` ⇒ **重复**那一句把它误判成漂移:
      模型吐 `["尺码","尺码"]` 时丢掉一条**本来没问题**的样本,而 `drifted` 读数虚高;
    - 写成 `bool(after)`(或任何只看向量长度那一侧的东西)⇒ **空**那两句错:
      零标签是一个**合法**结果(`r-0049`「你是」,模型真判零诉求)。
    """
    assert label_drift([], []) is False
    assert label_drift([], ["其他"]) is True
    assert label_drift(["尺码", "尺码"], ["尺码"]) is False, "重复不该被算成漂移"


def test_inject_typo_is_deterministic():
    """同种子同输入 → 同输出。增强产物要**可复现**,否则「这份语料是哪来的」说不清。"""
    assert inject_typo("买大了想退", 1) == inject_typo("买大了想退", 1)


def test_inject_typo_uses_the_seed_to_choose_among_the_candidates():
    """★ **反向断言**:不同种子**能**给出不同结果。

    ⚠️ 只有上面那条时,「seed 被用上了」是**同义反复** —— 一个**完全忽略 seed**
    的实现(每次取 `candidates[0]`、或干脆原样返回)同样满足「同种子同结果」。
    本仓已在 `pick_review_sample` / `stratified_split` 上记过同一个形状
    (`test_different_seeds_pick_different_samples`)。
    """
    text = "退货 尺码 运费 发票 快递 订单 颜色 保修"          # 八个候选**全在**
    seen = {inject_typo(text, s) for s in range(64)}
    assert len(seen) >= 2, f"64 个种子只注出 {len(seen)} 种结果 ⇒ seed 没被用上:{seen}"


def test_inject_typo_actually_changes_something():
    """不是恒等函数 —— 否则「注入了错别字」这句话是空的。"""
    changed = [inject_typo("我要退货", s) for s in range(20)]
    assert any(c != "我要退货" for c in changed)


def test_inject_typo_never_empties_the_text():
    """**11-D:这条原稿零判别力,现在钉两件真的事。**

    ⚠️ 原稿只有 `assert inject_typo("退货", s).strip()`。而实现是
    `text.replace(src, dst, 1)`(dst 非空、src ≠ dst)⇒ 输入非空就**结构上不可能**
    返回空 —— 连「原样返回 `text`」那种错实现**也照样绿**。
    本仓那条判据在此成立:**「注入的错别字不许为空」这句话,没有任何东西守着。**

    ⇒ 现在断的是:① 有可替换词 ⇒ **必定变了**;② 没有可替换词 ⇒ **原样返回**。
    """
    for s in range(50):
        out = inject_typo("退货", s)
        assert out.strip(), "注成空串了 —— 空样本会进训练集而看不出来"
        assert out != "退货", "原样返回了 —— 那等于一条错别字都没注入"
    # ② 反过来那一半:这句话里一个可替换词都没有 ⇒ 原样返回,**不是**「随便塞一个错别字」。
    #    少了这一句,「无脑 append 一个错字」那种错实现照样能过上面那两句。
    assert {inject_typo("在吗", s) for s in range(20)} == {"在吗"}


def test_typo_seed_comes_from_the_row_id():
    """★ **11-C**:种子从**行 id** 派生 —— 同 id 同种子、不同 id 不同种子。

    ⚠️ 「不同 id 不同种子」这一半承重:一个**把 id 丢掉**的实现(返回常数)
    在这里当场红。它在下游的后果是「同一行在不同的跑里注出不同的错别字」——
    产物看起来照常合理,而「这份语料是哪来的」再也说不清。
    """
    ids = [f"r-{i:04d}" for i in range(1, 200)] + [f"s-{i:04d}" for i in range(1, 200)]
    assert typo_seed("r-0001") == typo_seed("r-0001")
    seeds = [typo_seed(i) for i in ids]
    assert len(set(seeds)) == len(ids), "400 个真实 id 撞出了重复种子(id 被丢掉了?)"


def test_typo_seed_is_stable_across_processes():
    """跨进程确定性 —— **这一条只能这么测**。

    ⚠️ 内置 `hash()` 对 str **每进程随机化**(PYTHONHASHSEED)。用上它的后果是
    「同一 id 两次增强得到同一个错别字」在脚本重启之后就没了,而
    **同进程内的任何断言都测不出来**(本仓在 `app/tools/mock_data.py` 上栽过,
    照 `tests/test_tools_random.py::test_seed_is_stable_across_processes` 的先例)。
    另起两个进程:它们默认拿到**不同**的 PYTHONHASHSEED ⇒ 用 `hash()` 的实现两个数不同。
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys; sys.path.insert(0, r'.');"
        "from app.topic.labeling import typo_seed;"
        "sys.stdout.buffer.write(' '.join("
        "str(typo_seed(i)) for i in ('r-0056', 's-0282', 'r-0001')).encode('utf-8'))"
    )
    outs = []
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, "-X", "utf8", "-c", code], cwd=str(root),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        assert proc.returncode == 0, proc.stderr
        outs.append(proc.stdout.strip())
    assert outs[0] == outs[1], (
        f"两个进程算出的种子不同:{outs} —— 用上内置 `hash()` 了?(它对 str 每进程随机化)"
    )
    # 非退化:三个 id 不许算成同一个数(常数也「跨进程稳定」,但那不是 id 派生)
    assert len(set(outs[0].split())) == 3, f"三个不同的 id 算成了同一个种子:{outs[0]!r}"


# ---- `augment` 的接线(ch10 spec §5.4;计划订正 11-A / 11-B / 11-C)----

#: `augment` 的迷你语料。**每行文本里都有两个可替换的同音词对**
#: ⇒ `inject_typo` 的候选不止一个(「行序无关」那条才有判别力,见它的用例自检)。
_AUG_TRAIN = [
    {"id": "r-0001", "question": "买大了要退货,运费谁出", "source": "chat",
     "provenance": "real", "labels": ["尺码", "退换货", "运费"], "split": "train"},
    {"id": "s-0002", "question": "订单里颜色选错了想换货", "source": "gen",
     "provenance": "synthetic", "labels": ["订单修改", "退换货"], "split": "train"},
    {"id": "r-0003", "question": "发票能改成公司抬头吗,保修几年", "source": "pool",
     "provenance": "real", "labels": ["发票", "保修维修"], "split": "train"},
    {"id": "s-0004", "question": "快递太慢了,尺码也不对", "source": "gen",
     "provenance": "synthetic", "labels": ["物流", "尺码"], "split": "train"},
    {"id": "r-0005", "question": "颜色发错了,能换货吗", "source": "chat",
     "provenance": "real", "labels": ["质量问题", "退换货"], "split": "train"},
    {"id": "s-0006", "question": "保修期内坏了,运费谁出", "source": "gen",
     "provenance": "synthetic", "labels": ["保修维修", "运费"], "split": "train"},
    # ⚠️ 后两行是**为了判别力**加的,不是凑数:只有前四行时,
    #    「把种子换成循环下标」那个变异**测不出来**(实测:两跑的产物逐字相同)——
    #    证据是 `.superpowers/ch10b_t8_mutation_probe.py` 的变异 **M4** 与 **M14**
    #    以及它的 `.txt` 转录(两份都已入库);判定判别力的那段断言见
    #    `test_augment_gives_the_same_typo_…` 末尾的用例自检。
    #    ⚠️ 这一行 `r-0005` 的**候选只有一个**(「颜色」) —— 它顺带覆盖
    #    「候选唯一 ⇒ 下标怎么变都不影响它」那条路。
]

#: 替身的回复表:原句 → (改写后的句子, 它自己报的标签)。
#: ⚠️ **改写后的句子也各带两个可替换词** —— 否则「行序无关」那条的用例自检过不去。
_AUG_REPLIES = {
    "买大了要退货,运费谁出": ("我买大了一码想退货,运费得我自己出吗", ["尺码", "退换货", "运费"]),
    "订单里颜色选错了想换货": ("订单里颜色我选错了,能换一个吗", ["订单修改", "退换货"]),
    # ⚠️ 后两条**故意带全角标点**(？ ，):它们让「增强行也过了清洗」这条断言
    #    真的走得通 —— 全是半角标点时,加不加 `clean()` 读数**逐字相同**(假绿)。
    "发票能改成公司抬头吗,保修几年": ("能不能把发票开成公司抬头？保修是几年", ["发票", "保修维修"]),
    "快递太慢了,尺码也不对": ("快递怎么这么慢啊，而且尺码也不合适", ["物流", "尺码"]),
    "颜色发错了,能换货吗": ("收到的颜色不对,能给我换一个吗", ["质量问题", "退换货"]),
    "保修期内坏了,运费谁出": ("还在保修期内就坏了,运费得我自己掏吗", ["保修维修", "运费"]),
}


class _Msg:
    def __init__(self, text):
        self.text = text


class _RewriteStub:
    """改写模型的替身 —— **回复由行内容决定,与调用顺序无关**。

    ⚠️ 这一点是「行序无关」(11-C)那条的**前提**。照 `prelabel` 那份测试的写法
    (**按调用顺序出队**),行序一反过来回复也跟着变 —— 那条断言就只是在测
    「队列按顺序出队」,种子那件事一点也测不到。
    """

    def __init__(self, *, drift_for=(), malformed=None):
        self.replies = dict(_AUG_REPLIES)
        self.drift_for = set(drift_for)
        #: {原句: 那个**坏形状**的 `labels` 值} —— 造「不是 JSON 语法错,而是形状不对」那一档
        self.malformed = dict(malformed or {})
        self.asked: list[str] = []      # 每行的**原句**,按被问到的顺序

    async def ainvoke(self, messages):
        prompt = messages[0].content
        for q, (new_q, labels) in self.replies.items():
            if q in prompt:
                self.asked.append(q)
                if q in self.malformed:
                    return _Msg(json.dumps({"question": new_q, "labels": self.malformed[q]},
                                           ensure_ascii=False))
                # 顺带钉住「prompt 逐条带了原标签」(spec §5.4 的第二条硬约束)——
                # 它的唯一执行点就是 prompt 里那一行字。
                # ⚠️ **不能**写成 `all(lb in prompt)`:`render_taxonomy_for_prompt()`
                #    把 17 个类名全列了一遍 ⇒ 那条断言**恒真**(本仓的假绿形态)。
                joined = " / ".join(labels)
                assert joined in prompt, (
                    f"prompt 里没有这一行的原标签 {joined!r} —— `_rewrite` 没把 "
                    f"`row['labels']` 喂进去(spec §5.4 要求逐条带原标签)"
                )
                if q in self.drift_for:
                    labels = [*labels, "评价"]          # 它自己报的标签多了一个 ⇒ 漂移
                return _Msg(json.dumps({"question": new_q, "labels": labels},
                                       ensure_ascii=False))
        raise AssertionError(
            f"prompt 里没有任何已知原句 —— `_rewrite` 没把这一行的题面喂进去:\n{prompt}"
        )


def _bind_augment(monkeypatch, tmp_path, *, drift_for=(), malformed=None, rows=None):
    """把 `augment` 的路径指到 `tmp_path`,并写出**冻结的** val / topic_test。

    ⚠️ val / topic_test 是**真的写到盘上**的(不是留空、也不是不建):
    11-B 要断的是「跑 `augment` 前后**逐字节相同**」,而对着不存在的文件
    那句恒真 —— 那正是这条守卫最容易变成假绿的地方。

    ⚠️ `TOPIC_DIR` 也一起指过去:凡是用 `TOPIC_DIR` 拼路径的写法(被改变的实现、
    或变异探针)都落在 `tmp_path` 里,**够不到真产物**。
    """
    from scripts import prepare_topic_data as p

    from app.topic.taxonomy import LABELS

    rows = list(_AUG_TRAIN if rows is None else rows)
    bad = {lb for r in rows for lb in r["labels"]} - set(LABELS)
    assert not bad, f"夹具用了不存在的类目 {bad} —— 夹具自己先错了"

    def _write(name, items):
        (tmp_path / name).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in items) + "\n",
            encoding="utf-8")

    _write("train.jsonl", rows)
    _write("val.jsonl", rows[:2])
    _write("topic_test.jsonl", rows[2:])

    stub = _RewriteStub(drift_for=drift_for, malformed=malformed)
    monkeypatch.setattr(p, "TOPIC_DIR", tmp_path)
    monkeypatch.setattr(p, "TRAIN", tmp_path / "train.jsonl")
    monkeypatch.setattr(p, "TRAIN_AUGMENTED", tmp_path / "train_augmented.jsonl")
    monkeypatch.setattr(p, "get_settings", lambda: object())
    monkeypatch.setattr(p, "create_extract_model", lambda _settings: stub)
    return p, rows, stub


def _augmented_rows(tmp_path):
    return [r for r in _read_jsonl(tmp_path / "train_augmented.jsonl") if r.get("augmented")]


def test_augment_appends_augmented_rows_and_leaves_the_frozen_artifacts_byte_identical(
    monkeypatch, tmp_path, capsys
):
    """★ `augment` 的正题:原件照写 + 追加增强行 + **冻结产物一个字节都没动**(11-B)。

    ⚠️ **为什么必须有这条**:`augment()` 的 docstring 写着「验证集与测试集一条都不许动」,
    而那句话此前**只是散文** —— 没有任何东西守着它(本仓已编目:一句看起来成立的
    注释不是守卫)。它一旦被违反,§8 那套指标就不再有任何意义,**而且是静默的**:
    指标会变好看,没人会去查测试集被动过。
    """
    p, rows, stub = _bind_augment(monkeypatch, tmp_path)
    frozen = {name: (tmp_path / name).read_bytes() for name in ("val.jsonl", "topic_test.jsonl")}

    asyncio.run(p.augment())

    out = _read_jsonl(tmp_path / "train_augmented.jsonl")
    n = len(rows)
    # ① 原件照写 —— 增强是**追加**,不是替换
    assert [r["id"] for r in out[:n]] == [r["id"] for r in rows]
    assert [r["question"] for r in out[:n]] == [r["question"] for r in rows]
    # ② 每条原件后面跟着**它自己**的增强版(同序、同 id、题面变了、标签没变)
    aug = out[n:]
    assert [r["id"] for r in aug] == [r["id"] for r in rows]
    assert all(r["augmented"] is True for r in aug)
    assert [r["labels"] for r in aug] == [r["labels"] for r in rows]
    assert all(a["question"] != o["question"] for a, o in zip(aug, rows)), "题面一个字都没改"
    assert stub.asked == [r["question"] for r in rows], "改写次数/顺序与训练集对不上"
    # ③ ⭐ **冻结产物逐字节相同**(11-B 的正题)
    for name, before in frozen.items():
        assert (tmp_path / name).read_bytes() == before, (
            f"{name} 在跑 `augment` 的过程中**被改动了** —— 扩了测试集,"
            f"§8 那套指标就不再有任何意义,而且没人会去查"
        )
    # ④ 增强行的题面**也过了清洗**(spec §5.1:训练侧与推理侧同源)。
    #    ⚠️ 判别力依赖夹具:替身那两条回复里带了**全角标点**(？ ，)——
    #    全是半角时,加不加 `clean()` 读数**逐字相同**(那又是一条假绿)。
    #    而这一条的后果不小:训练语料里两种口径并存,而 `encode_rows` **不再洗**
    #    (`app/topic/model.py` 直接读 `row["question"]`)。
    from app.topic.clean import clean
    for a in aug:
        assert clean(a["question"]) == a["question"], (
            f"增强行的题面没过清洗 ⇒ 训练语料两种口径并存:{a['question']!r}"
        )
    # ⑤ 读数打出来(交付要求的一部分:`kept` / `drifted` / 产物行数)
    printed = capsys.readouterr().out
    assert "增强追加 6 条" in printed, printed
    assert "因标签漂移丢弃 0 条" in printed, printed
    assert "产物 train_augmented.jsonl:12 行(原件 6 + 增强 6)" in printed, printed


def test_augment_refuses_to_read_a_frozen_artifact(monkeypatch, tmp_path):
    """★ 11-B 的另一半:**把 `train_path` 指向 `val.jsonl` ⇒ 那条断言必须红。**

    真正会把验证集吃掉的那种改法(有人决定「train + val 一起增强,验证集样本太少」)
    在这里当场停住,而不是安静地增强一份验证集 —— 而验证集参与早停与阈值选择,
    它被改动与测试集被动过同级。

    ⚠️ 最后那句断的是「产物**不存在**」:「拦住了」与「拦住了但先写了一份错的」
    是两回事 —— 断言若写在打开产物之后,一份空产物已经落在盘上了。
    """
    p, _, _ = _bind_augment(monkeypatch, tmp_path)
    monkeypatch.setattr(p, "TRAIN", tmp_path / "val.jsonl")

    with pytest.raises(AssertionError):
        asyncio.run(p.augment())

    assert not (tmp_path / "train_augmented.jsonl").exists(), (
        "断言写在**打开产物之后** ⇒ 先留了一份空产物再报错"
    )


def test_augment_refuses_to_write_a_frozen_artifact(monkeypatch, tmp_path):
    """★ 11-B:**产物也不许写进冻结产物** —— 这才是**真正**会毁掉测试集的方向。

    ⚠️ 两条断言分得开,不许互相顶替:输入那条守的是「验证集被加强进产物」,
    输出那条守的是「测试集被覆盖」。它们对应两种**不同的**改法。
    """
    p, _, _ = _bind_augment(monkeypatch, tmp_path)
    frozen = (tmp_path / "topic_test.jsonl").read_bytes()
    monkeypatch.setattr(p, "TRAIN_AUGMENTED", tmp_path / "topic_test.jsonl")

    with pytest.raises(AssertionError):
        asyncio.run(p.augment())

    assert (tmp_path / "topic_test.jsonl").read_bytes() == frozen, (
        "冻结的测试集被覆盖了 —— 而那是个静默的破坏(指标会变好看)"
    )


def test_augment_gives_the_same_typo_no_matter_the_row_order(monkeypatch, tmp_path):
    """★ **11-C**:种子从**行 id** 派生 ⇒ 语料重排后每一行注的还是同一个错别字。

    ⚠️ 用**循环下标**当种子时,`train.jsonl` 一旦重排(重跑 `split`、按 id 重排、
    有人手工挪了几行),**每一行**的错别字都变 —— 产物看起来照常合理,
    没有任何东西会报错。这里跑两遍(正序 / 倒序),断**同一 id 的题面逐字相同**。
    """
    seen: dict[str, dict[str, str]] = {}
    for order, rows in (("正序", _AUG_TRAIN), ("倒序", list(reversed(_AUG_TRAIN)))):
        p, _, _ = _bind_augment(monkeypatch, tmp_path, rows=rows)
        asyncio.run(p.augment())
        seen[order] = {r["id"]: r["question"] for r in _augmented_rows(tmp_path)}
    assert set(seen["正序"]) == set(seen["倒序"]), "两跑增强出的 id 集合就不一样"
    assert seen["正序"] == seen["倒序"], (
        "同一行在两份行序里注出了**不同**的错别字 ⇒ 种子跟着行序变了(11-C 那个错法)"
    )

    # ---- 用例自检:这个夹具**真的**能分辨「下标当种子」----
    # ⚠️ 判据是本仓那句「构造输入前先算一遍它会不会走到那条分支」:把两跑的
    #    **循环下标**(实现里是 `enumerate(todo, 1)`,**从 1 起**)当种子算一遍 ——
    #    若两跑的产物**逐字相同**,那上面那条断言对 11-C 那个错法就**零判别力**
    #    (那时该换几个 id / 加几行,不是删掉它)。
    # ⚠️ 两点都吃过亏,别再写回去:
    #    ① 这一段**不引用 `typo_seed`** —— 引用它就成了「用被测的那件事去证明
    #       被测的那件事」;
    #    ② 下标必须**从 1 起**(与实现一致)。第一版写的是 0 起的 `enumerate(…)`
    #       ⇒ 自检报「有判别力」而**变异实测全绿** —— 一段**看起来在守什么、
    #       其实什么都没守**的自检(与本仓那条「一句看起来成立的注释不是守卫」同族)。
    #       `ch10b_t8_mutation_probe.py` 的变异 **M14** 现在钉住这段自检本身:
    #       把夹具截回四行 ⇒ 自检**先红**(而不是悄悄变成绿灯)。
    rewritten = {q: new_q for q, (new_q, _) in _AUG_REPLIES.items()}
    by_index = {
        "正序": {r["id"]: inject_typo(rewritten[r["question"]], i)
                 for i, r in enumerate(_AUG_TRAIN, 1)},
        "倒序": {r["id"]: inject_typo(rewritten[r["question"]], i)
                 for i, r in enumerate(reversed(_AUG_TRAIN), 1)},
    }
    assert by_index["正序"] != by_index["倒序"], (
        "夹具没有判别力:把种子换成**循环下标**时,正序与倒序的产物**逐字相同**"
        " ⇒ 上面那条断言抓不住 11-C 那个错法(换几个 id / 加几行,不是删掉它)"
    )


def test_augment_drops_the_rows_whose_labels_drifted(monkeypatch, tmp_path, capsys):
    """标签漂移的行**整条丢弃**,并如实计数(spec §5.4 的第一条硬约束)。

    ⚠️ **两个方向都要钉**:
    - **该丢的丢了**:漂的那一条**不在**产物里;
    - **不该丢的没丢**:其余三条**在**产物里 —— 把判据写成恒真
      (`label_drift` 直接 `return True`)的实现会在这里红;
    - 原件**照样全写**(丢弃只作用于增强行,不许连原件一起丢)。
    """
    p, rows, _ = _bind_augment(monkeypatch, tmp_path,
                               drift_for=("订单里颜色选错了想换货",))

    asyncio.run(p.augment())

    written = _read_jsonl(tmp_path / "train_augmented.jsonl")
    aug = [r for r in written if r.get("augmented")]
    assert {r["id"] for r in aug} == {r["id"] for r in rows} - {"s-0002"}, (
        "漂移的行还是写进产物了,或者**没漂的行被一起丢了**"
    )
    assert len(written) == len(rows) * 2 - 1, "原件被连带丢了"
    printed = capsys.readouterr().out
    assert "因标签漂移丢弃 1 条" in printed, printed
    assert "增强追加 5 条" in printed, printed


def test_augment_limit_only_samples_the_rewriting(monkeypatch, tmp_path, capsys):
    """★ **11-A**:`--limit N` **只**决定这次改写几行,并且**不许碰冻结的 `train.jsonl`**。

    ⚠️ 本任务原稿写的是「先把 `train.jsonl` 截成 20 条试跑」—— 那是**就地改掉训练集**:
    一份 20 行的训练集会出现在 `git status` 里(` M train.jsonl`),谁在收尾时一
    `git add -A` 就把它提交了,而 Task 9 拿它训练。这条把那个脚枪钉住:
    跑完 `train.jsonl` **逐字节相同**。
    """
    p, rows, stub = _bind_augment(monkeypatch, tmp_path)
    before = (tmp_path / "train.jsonl").read_bytes()

    asyncio.run(p.augment(limit=2))

    assert stub.asked == [r["question"] for r in rows[:2]], "`limit` 取的不是**前 N 行**"
    assert [r["id"] for r in _augmented_rows(tmp_path)] == [r["id"] for r in rows[:2]]
    # 原件仍然**全部**照写(小样只影响「改写几行」,不影响产物的原件那一半)
    assert len(_read_jsonl(tmp_path / "train_augmented.jsonl")) == len(rows) + 2
    assert (tmp_path / "train.jsonl").read_bytes() == before, (
        "`train.jsonl` 被就地改了 —— 那是 11-A 的脚枪(冻结的训练集是 Task 9 的输入)"
    )
    # ⚠️ 小样跑出来的产物**不是**全量 —— 这件事必须打出来,否则一份只增强了 20 条的
    #    产物会安静地冒充全量产物,而 Task 9 照它训练。
    assert "小样" in capsys.readouterr().out


def test_augment_reports_a_malformed_answer_without_counting_it_as_drift(
    monkeypatch, tmp_path, capsys
):
    """改写端没解析出来 ⇒ **按原句写回**,而且**不**记进「漂移」。

    ⚠️ 两个读数混起来的后果(原稿只挡了 JSON 语法那一档,这里钉的是**形状**那一档):
    `drifted` 是**语料质量的读数**(它高说明预标不稳 / prompt 的「诉求个数与类别不变」
    没被遵守),而「模型吐了怪东西」是**故障**。混起来之后,一个「网络抖动 + 标签全对」
    的批次会被读成「标签大面积漂移 ⇒ 回去改 prompt」。

    ⚠️ 两种形状都要造,因为后果不同:
    - `labels` 吐成**字符串** ⇒ `list("退换货")` 按**字符**迭代,每个字符都查无此类目
      ⇒ 原稿照样落进 drift(看起来像模型判错了标签,其实是形状不对);
    - `labels` 吐成 **int** ⇒ `list(5)` 直接 `TypeError` ⇒ **打断整跑**(那是 75 分钟)。
    """
    p, rows, _ = _bind_augment(monkeypatch, tmp_path, malformed={
        "订单里颜色选错了想换货": "退换货",
        "快递太慢了,尺码也不对": 5,
    })

    asyncio.run(p.augment())

    aug = {r["id"]: r for r in _augmented_rows(tmp_path)}
    assert set(aug) == {r["id"] for r in rows}, "解析失败的行被丢了(该按原句写回)"
    for rid, question in (("s-0002", "订单里颜色选错了想换货"), ("s-0004", "快递太慢了,尺码也不对")):
        orig = next(r for r in rows if r["id"] == rid)
        assert aug[rid]["labels"] == orig["labels"], f"{rid} 落回的不是原标签"
        assert aug[rid]["question"] == inject_typo(question, typo_seed(rid)), (
            f"{rid} 的题面该按**原句**写回(那个没解析出来的改写不许进语料)"
        )
    printed = capsys.readouterr().out
    assert "因标签漂移丢弃 0 条" in printed, printed
    assert "解析失败 2 条" in printed, printed


def test_the_augment_paths_point_at_the_right_files():
    """★ 名字要有一条**直接**断言 —— 与订正 9-B(`topic_test.jsonl`)同一个理由。

    ⚠️ 上面那些接线用例全都 `monkeypatch` 掉了 `TRAIN` / `TRAIN_AUGMENTED`
    ⇒ **常量指错文件它们照样绿**(订正 9-B 的 M15 实测过同一形状)。
    指错的后果分两种:`val.jsonl`(验证集被当成训练集加强)最重 ——
    验证集参与早停与阈值选择,它被改动与测试集被动过同级。
    """
    from scripts import export_label_review as exp
    from scripts import prepare_topic_data as p

    assert p.TRAIN == p.TOPIC_DIR / "train.jsonl"
    assert p.TRAIN_AUGMENTED == p.TOPIC_DIR / "train_augmented.jsonl"
    assert p.TRAIN != p.TRAIN_AUGMENTED, "产物指回了输入 —— 那是就地改掉冻结的训练集"
    for frozen in (p.TOPIC_DIR / "val.jsonl", exp.TOPIC_TEST):
        assert p.TRAIN != frozen, "冻结产物被当成增强的输入了"
        assert p.TRAIN_AUGMENTED != frozen, "增强会写进冻结产物 —— 静默的破坏"


def test_main_dispatches_augment_and_forwards_the_limit(monkeypatch):
    """`main()` 的分派:`augment` 必须走到 `augment()`,而且 `--limit` 要**真的传进去**。

    ⚠️ 守的是**命令行那一半**:上面几条直接调 `p.augment(...)`,所以把 `main()`
    里 `elif args.step == "augment"` 那一支删掉、或忘了转发 `--limit` 时它们**照样绿** ——
    而 `bash` 里那一行 `augment --limit 20` 会变成一次**全量**跑
    (1158 条 × 逐条打网络,约 75 分钟)。
    """
    import sys

    from scripts import prepare_topic_data as p

    called: list[tuple[str, int | None]] = []

    async def fake(limit=None):
        called.append(("augment", limit))

    monkeypatch.setattr(p, "augment", fake)

    monkeypatch.setattr(sys, "argv", ["prepare_topic_data.py", "augment", "--limit", "3"])
    p.main()
    assert called == [("augment", 3)]

    monkeypatch.setattr(sys, "argv", ["prepare_topic_data.py", "augment"])
    p.main()
    assert called[-1] == ("augment", None), "不带 `--limit` 时必须传 `None`(= 全量)"


def test_main_refuses_limit_on_the_other_steps(monkeypatch):
    """`--limit` 只对 `augment` 有意义 —— 给 `split` / `collect` 加它必须**响亮地停**。

    ⚠️ 静默忽略是这里唯一危险的走法:有人打 `split --limit 20` 想看小样,
    脚本照跑全量,而**没有任何东西告诉他这个参数被吞了**。

    ⚠️ `split` / `collect` 在这里**被替成计数器**(不是让它真跑):
    真跑 `split()` 会**覆盖 `evals/topic/` 里那三份冻结产物** —— 这条测试自己
    绝不能带那种风险。写成计数器顺带多钉一句:**分派前就停了**(两个都没被调用)。
    """
    import sys

    from scripts import prepare_topic_data as p

    called: list[str] = []
    monkeypatch.setattr(p, "split", lambda: called.append("split"))
    monkeypatch.setattr(p, "collect", lambda: called.append("collect"))

    for step in ("split", "collect"):
        monkeypatch.setattr(sys, "argv", ["prepare_topic_data.py", step, "--limit", "20"])
        with pytest.raises(SystemExit) as e:
            p.main()
        assert "--limit" in str(e.value), str(e.value)

    assert called == [], "`--limit` 没拦住,真跑了别的一步"
