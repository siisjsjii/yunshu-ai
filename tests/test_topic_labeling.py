"""标注相关的纯函数:去重、证据串校验、标签漂移、分层抽样。

末两条(`test_collect_...` 与 `test_prelabel_...`)不测纯函数,测的是**接线** ——
`collect`(三源合流)与 `prelabel_topics.run`(三个读数 + 断点续跑)。两处都
**没有 DB、没有网络**(模型替身替在 `ainvoke` 这一层),所以留在这一份里、
也在 `not db` 套件里。
"""

import json

import pytest

from app.topic.labeling import dedupe_questions, validate_evidence


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
