"""标注相关的纯函数:去重、证据串校验、标签漂移、分层抽样。

末两条(`test_collect_...` 与 `test_prelabel_...`)不测纯函数,测的是**接线** ——
`collect`(三源合流)与 `prelabel_topics.run`(三个读数 + 断点续跑)。两处都
**没有 DB、没有网络**(模型替身替在 `ainvoke` 这一层),所以留在这一份里、
也在 `not db` 套件里。
"""

import json

import pytest

from app.topic.labeling import dedupe_questions, pick_review_sample, validate_evidence


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


def test_main_pins_stdout_encoding():
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
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    code = "\n".join([
        "from scripts import prelabel_topics as p",
        "async def _noop(_limit):",
        "    return None",
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
    则「评价」桶为空 ⇒ 红(变异实测过,见 task-6-report)。

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
# ⚠️ 这几条钉的是 `scripts/export_label_review.py` 的**接线**,不是纯函数:
# 那条链上有两个「错了也看不出来」的地方 —— ① 表头/列名对不上时,回收**逐行静默回落**成
# 预标标签,而打印出来的读数是「判『改』的 0 条」(看着像「用户什么都没改」);
# ② 「最终标签」列被顺手填成预标时,**用户改了哪几条就再也分不出来**了(B6-A 的承重不变量)。
# 两者都取 `tmp_path`,所以这几条**不碰** `evals/topic/labels/trainval.csv` 那份真产物。


def _write_review_csv(path, rows, header=None):
    import csv
    from scripts import export_label_review as exp

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(exp.HEADER if header is None else header)
        w.writerows(rows)


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
    """`import` 的接线:以「最终标签」为准、空了才回落预标、「改」的条数是读数、`reviewed` 标上。

    分隔符:`|` 与 `,` 都要认,且**两边的空白无所谓**(`尺码, 退换货` 是人最常打的形态)。
    """
    from scripts import export_label_review as exp

    monkeypatch.setattr(exp, "LABELS_DIR", tmp_path)
    monkeypatch.setattr(exp, "REVIEWED", tmp_path / "reviewed.jsonl")
    _write_review_csv(tmp_path / "trainval.csv", [
        ["a", "买大了想退", "尺码", "改", "尺码, 退换货", "两个诉求"],
        ["b", "快递到哪了", "物流", "ok", "", ""],
        ["c", "能便宜点吗", "优惠活动", "", "价保", "只填了最终标签"],
    ])

    exp.do_import()

    rows = [
        json.loads(l) for l in
        (tmp_path / "reviewed.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [(r["id"], r["question"], r["labels"]) for r in rows] == [
        ("a", "买大了想退", ["尺码", "退换货"]),   # 用户填的(逗号 + 空格)
        ("b", "快递到哪了", ["物流"]),              # 没填 ⇒ 回落预标
        ("c", "能便宜点吗", ["价保"]),              # 只填了最终标签也认(「判定」只影响报表)
    ]
    assert all(r["reviewed"] is True for r in rows)
    printed = capsys.readouterr().out
    assert "回收 3 条" in printed
    assert "判「改」的 1 条" in printed


def test_import_refuses_input_that_would_silently_change_the_reading(monkeypatch, tmp_path):
    """三种「不响亮就会静默给出错读数」的输入 —— 每一种都必须在**写产物之前**停住。

    ① **表头缺列**:`row.get("最终标签")` 恒为 None ⇒ 每行静默回落预标,而读数是「判『改』的 0 条」;
    ② **判定值认不出**(`改了`):若当成「没改」,错误率被**少算** —— 一个乐观方向的错答案;
    ③ **标签是错字**(`尺碼`):静默丢弃的话,训练集里就少一个标签,而没人会知道。

    ⚠️ 每条都断 `产物不存在` —— 「拦住了」与「拦住了但先写了一份错的」是两回事。
    """
    from scripts import export_label_review as exp

    monkeypatch.setattr(exp, "LABELS_DIR", tmp_path)
    monkeypatch.setattr(exp, "REVIEWED", tmp_path / "reviewed.jsonl")
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


def test_export_script_pins_stdout_encoding():
    """`main()` **自己**必须把 stdout 钉成 UTF-8 —— 本机 locale 是 cp936。

    ⚠️ **为什么必须有这一条**:`⚠️`(U+26A0 + VS16)编不进 GBK(实测 2026-09-26),
    而它只出现在「某类一条都没抽到」那条 print 里 ⇒ **用真实数据跑一次 `export` 碰不到它**,
    那样测必然假绿(事实:当天的语料 17 类**每类都抽到了**,那行一次都不打印)。
    这里在**子进程里**跑 `main()`,再由**测试自己**打印一个 `⚠️` —— 钉没钉住当场见分晓。

    ⚠️ 子进程故意**不加 `-X utf8`**:加的话 stdout 恒是 UTF-8,这条断言就**恒真**。
    它的判别力**依赖本机是 cp936**,换一台 UTF-8 locale 的机器就退化成「函数存在」检查
    (如实记下来,不装作它到哪都同样有力)。
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    code = "\n".join([
        "import sys",
        "from scripts import export_label_review as e",
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
