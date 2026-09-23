"""增量 JSON 解码器 —— 13 条边界 + 穷举切点 fuzz。

判据(本仓血的教训):**构造输入时先算一遍这个输入真的会走到那条分支吗**。
例如"转义跨 chunk"这条,必须把 `\\u4e2d` 切开,不能整段喂进去。
"""

import json

import pytest

from app.agent.json_stream import JsonAnswerDecoder


def _feed_all(dec, fragments):
    out = []
    for f in fragments:
        out.extend(dec.feed(f))
    return out


def _deltas(events):
    return "".join(e.value for e in events if e.kind == "answer_delta")


# ---- 1–6:正常路径与跨 chunk 的转义 ----------------------------------------

def test_1_unicode_escape_split_across_chunks():
    d = JsonAnswerDecoder()
    ev = _feed_all(d, ['{"useful": true, "confidence": 0.9, "answer": "a\\u4e', '2d b"}'])
    assert d.answer == "a中 b"
    assert _deltas(ev) == "a中 b"


def test_2_backslash_split_across_chunks():
    d = JsonAnswerDecoder()
    d.feed('{"useful": true, "confidence": 0.9, "answer": "a\\')
    d.feed('"b"}')
    assert d.answer == 'a"b'


def test_3_structural_chars_inside_answer_are_not_structure():
    d = JsonAnswerDecoder()
    _feed_all(d, ['{"useful": true, "confidence": 0.9, ',
                  '"answer": "含 { } , : 与 \\"引号\\" 的文本"}'])
    assert d.answer == '含 { } , : 与 "引号" 的文本'


def test_4_answer_is_last_key_and_closing_brace_is_separate():
    d = JsonAnswerDecoder()
    ev = _feed_all(d, ['{"useful": true, "confidence": 0.9, "answer": "你好"',
                       '}'])
    assert d.done is True
    assert d.answer == "你好"
    assert [e.kind for e in ev][-1] == "done"


def test_5_float_split_across_chunks():
    d = JsonAnswerDecoder()
    _feed_all(d, ['{"useful": true, "confidence": 0.', '85, "answer": "x"}'])
    assert d.confidence == pytest.approx(0.85)


def test_6_boolean_split_across_chunks():
    d = JsonAnswerDecoder()
    _feed_all(d, ['{"useful": tru', 'e, "answer": "x"}'])
    assert d.useful is True


# ---- 8/13:lead 态的退出 ----------------------------------------------------

def test_8_leading_text_exits_to_plain():
    """首个非空白不是 `{` ⇒ plain(今天的行为),**不是** violation。"""
    d = JsonAnswerDecoder()
    ev = _feed_all(d, ['好的,', '我来回答:退货政策是…'])
    assert d.mode == "plain"
    assert _deltas(ev) == "好的,我来回答:退货政策是…"
    assert not any(e.kind == "violation" for e in ev)


def test_13_whitespace_only_stays_in_lead():
    d = JsonAnswerDecoder()
    ev = _feed_all(d, ["", "  ", "\n"])
    assert d.mode == "lead"
    assert ev == []


def test_12_json_fence_falls_back_to_plain():
    """```json 围栏 ⇒ plain(刻意不剥壳:剥壳要猜模型意图,剥错了更糟)。"""
    d = JsonAnswerDecoder()
    ev = _feed_all(d, ['```json\n{"useful": true, "answer": "x"}```'])
    assert d.mode == "plain"
    assert d.useful is None


# ---- 7:protocol 态下的首键违规 --------------------------------------------

def test_7_wrong_first_key_is_violation_and_nothing_emitted():
    d = JsonAnswerDecoder()
    ev = _feed_all(d, ['{"answer": "先答后判"}'])
    assert d.mode == "protocol"
    assert [e for e in ev if e.kind == "violation"]
    assert _deltas(ev) == "", "违规时一个 answer_delta 都不许出去"


# ---- 9/10/11:终止条件 ------------------------------------------------------

def test_9_incomplete_stream_is_not_done():
    d = JsonAnswerDecoder()
    _feed_all(d, ['{"useful": true, "answer": "半截'])
    assert d.done is False


def test_10_useful_false_with_empty_answer_emits_nothing():
    d = JsonAnswerDecoder()
    ev = _feed_all(d, ['{"useful": false, "confidence": 0.1, "answer": ""}'])
    assert d.useful is False
    assert _deltas(ev) == ""


def test_11_useful_false_with_nonempty_answer_stops_immediately():
    """⚠️ **这条是本任务最重要的用例**。

    协议要求 useful=false 时 answer 为空串,所以"停吐"这条正常路径**没有可吐的东西**
    ⇒ 用正常输入测它**等于没测**。必须构造**违规流**:useful 为 false 但 answer 非空,
    才能真的验到"停吐"这一步。
    """
    d = JsonAnswerDecoder()
    ev = _feed_all(d, [
        '{"useful": false, ',
        '"answer": "这段文本一个字都不该出现在用户面前"}',
    ])
    assert d.useful is False
    assert _deltas(ev) == ""


# ---- fuzz:穷举切点 ---------------------------------------------------------

FULL = json.dumps(
    {"useful": True, "confidence": 0.75,
     "answer": '中文答案:含 "引号"、\\反斜杠、\n换行、中文与 { } 结构符'},
    ensure_ascii=False,
)


def _run(fragments):
    d = JsonAnswerDecoder()
    ev = []
    for f in fragments:
        ev.extend(d.feed(f))
    return d, ev


def test_fuzz_every_two_way_split_matches_unsplit():
    """把完整 JSON 按**每一个可能的切点**切成两段 —— 结论必须与不切时逐字节相同。"""
    base_d, base_ev = _run([FULL])
    for i in range(1, len(FULL)):
        d, ev = _run([FULL[:i], FULL[i:]])
        assert d.answer == base_d.answer, f"切点 {i} 的 answer 不一致"
        assert d.useful == base_d.useful
        assert d.confidence == base_d.confidence
        assert _deltas(ev) == _deltas(base_ev), f"切点 {i} 的增量流不一致"
        assert d.done is True


def test_fuzz_escaped_unicode_split_at_every_point():
    """专钉 `\\uXXXX` 与 `\\` 的**每一个**内部切点。"""
    payload = '{"useful": true, "confidence": 0.5, "answer": "中\\u4e2d\\"x\\\\y"}'
    base_d, _ = _run([payload])
    for i in range(1, len(payload)):
        d, _ = _run([payload[:i], payload[i:]])
        assert d.answer == base_d.answer, f"切点 {i}: {d.answer!r} != {base_d.answer!r}"


# ---- 额外一条(不在 brief 的清单里,**加**的,不是改的)--------------------
#
# 上面两条 fuzz 的**两边都由同一个解码器产出** —— 一个「两边一起错」的实现能让
# `d.answer == base_d.answer` 恒真,于是**只**验了「切分不改变结果」,没验「结果是
# 对的」。实测(MUTATION-4:`_emit_text` 不再吐增量)时两条 fuzz **全绿**,红的只有
# test_1 那条绝对断言 —— 洞是真的。这条用一个**外部参照系**(`json.loads`)把这个洞
# 堵上:转义译错、少译一个字符、多吐一个字,都会在这里当场红。

ESCAPED = '{"useful": true, "confidence": 0.5, "answer": "中\\u4e2d\\"x\\\\y"}'
#: **非 BMP**(代理对)—— `ensure_ascii=True` 的网关正是这么发 `😀` 的
SURROGATE = '{"useful": true, "confidence": 0.5, "answer": "\\ud83d\\ude00 ok \\u4e2d"}'

ORACLE_PAYLOADS = (FULL, ESCAPED, SURROGATE)


def test_oracle_answer_matches_json_loads_at_every_split():
    """独立口径:`answer` 必须逐字等于 `json.loads` 的结果(不切 / 每一个切点)。

    ⚠️ 这是 brief 那两条 fuzz 的**外部参照系**。它们两边都由同一个解码器产出,
    「两边一起错」时全绿(实测:MUTATION-4 / MUTATION-5);只有 `json.loads` 能判对。
    **代理对那条载荷是它的另一半边**:`\\ud83d\\ude00` 若不拼回去,`answer` 里就是
    两个孤立代理,而两条 fuzz 依然全绿 —— 它们只比「切与不切一致」。
    """
    for text in ORACLE_PAYLOADS:
        expected = json.loads(text)["answer"]
        assert _run([text])[0].answer == expected, f"不切就对不上 json.loads: {text}"
        for i in range(1, len(text)):
            d, _ = _run([text[:i], text[i:]])
            assert d.answer == expected, f"切点 {i}: {d.answer!r} != {expected!r}"


def test_fuzz_random_three_to_eight_way_splits():
    """spec §12.1 要的第二组切分:**随机切成 3–8 段**,结论仍须与不切时逐字节相同。

    种子写死 ⇒ 完全可复现;段数**断言**在 3–8 之间(不然就是「输入小到触发不了
    被测行为」那一类假绿)。另加一条自检:必须有若干轮的切点**落在转义序列内部**,
    否则这组用例对「转义跨 chunk」零覆盖 —— 白跑。
    """
    import random

    payloads = [
        FULL,
        '{"useful": true, "confidence": 0.5, "answer": "中\\u4e2d\\"x\\\\y"}',
    ]
    rng = random.Random(20260923)
    escape_hits = 0
    for _ in range(200):
        for text in payloads:
            want = json.loads(text)["answer"]
            cuts = sorted(rng.sample(range(1, len(text)), rng.randint(2, 7)))
            parts = [text[a:b] for a, b in zip([0] + cuts, cuts + [len(text)])]
            assert 3 <= len(parts) <= 8, f"段数 {len(parts)} 不在 3–8"
            d, _ = _run(parts)
            assert d.answer == want, f"切点 {cuts}: {d.answer!r} != {want!r}"
            assert d.useful is True and d.done is True
            # 自检:这一轮的切点是否真的落在某个 `\` 与它后面的字符之间
            if any(text[c - 1] == "\\" for c in cuts):
                escape_hits += 1
    assert escape_hits > 0, "一轮都没切在转义内部 ⇒ 这组用例对该分支零覆盖"


BAD_USEFUL = [
    '{"useful": 1, "answer": "这段不许出去"}',
    '{"useful": 0, "answer": "这段不许出去"}',
    '{"useful": nul, "answer": "这段不许出去"}',
    '{"useful": TRUE, "answer": "这段不许出去"}',      # 大小写变体
    '{"useful": True, "answer": "这段不许出去"}',
    '{"useful": tru, "answer": "这段不许出去"}',       # 半截字面量
    '{"useful": "true", "answer": "这段不许出去"}',    # 字符串不是布尔
    '{"useful": , "answer": "这段不许出去"}',          # 值整个缺失
]


@pytest.mark.parametrize("stream", BAD_USEFUL)
def test_extra_malformed_useful_is_violation_not_false(stream):
    """`useful` 的值**不是字面 `true`/`false`** ⇒ violation,**不许当成 `useful=false`**。

    spec §5.6 那一类走的是 **fail-open**(理由原文:按 `useful=false` 处理等于
    **把一段可能完全正确的回答扔掉**并落一条 spec 点名过的**假池记录**)。
    今天这个值畸形,是**协议不合**,不是**证据不足**。

    三件事必须同时成立:①`violation` 非空且**说明是「值」不是「键」**;
    ②`useful` **保持 `None`**(由调用方走它既有的降级路,解码器不造第三种结局);
    ③**一个 `answer_delta` 都不出去**,且此后 `feed` 冻结。
    """
    d = JsonAnswerDecoder()
    ev = []
    # 再切一刀:跨 chunk 的畸形值必须同样被逮住(不能只在整段喂时才对)
    for fragment in (stream[:12], stream[12:]):
        ev.extend(d.feed(fragment))
    assert d.violation, f"{stream}: 应当 violation"
    assert "first_key" not in d.violation, f"{stream}: 这是**值**的违规,不是**键**的"
    assert d.useful is None, f"{stream}: useful 必须保持 None(不许当 False)"
    assert _deltas(ev) == "", f"{stream}: 违规时一个 answer_delta 都不许出去"
    # ⚠️ 这两条是评审补的:原先这组用例**从没看 `done`**,于是把违规分支改成
    # `_done = True` + append `Event("done")` 之后**8 条全绿** —— 那句
    # 「畸形 useful 不吐 done」等于没有牙。
    assert d.done is False, f"{stream}: 违规不是收尾,done 必须为 False"
    assert [e.kind for e in ev] == ["violation"], \
        f"{stream}: 事件序列必须**恰好**是 [violation](不许夹一个 done)"
    assert d.feed('" 更多"') == [], f"{stream}: 违规后 feed 必须冻结返回 []"
    assert _deltas(ev) == "", f"{stream}: 冻结之后增量仍然为空"


@pytest.mark.parametrize("stream", [
    '{"useful": true , "answer": "x"}',      # 值**外侧**的空白是合法 JSON
    '{"useful":\n true, "answer": "x"}',     # 值**前面**的空白同理
    '{"useful":true, "answer": "x"}',        # 冒号后无空白
])
def test_extra_useful_surrounding_whitespace_is_accepted(stream):
    """口径边界:`.`strip()` 只吃**值外侧**的空白 —— 那是 JSON 语法允许的,照常接受。

    与上一条合起来才是「严格只认字面 `true`/`false`」的完整口径:
    **周围空白不算异常,大小写变体算异常。**
    """
    d = JsonAnswerDecoder()
    ev = d.feed(stream)
    assert d.violation is None, f"{stream}: 不应当 violation"
    assert d.useful is True, f"{stream}: 应当解出 True"
    assert _deltas(ev) == "x"


# ---- 嵌套值:只有深度 0 的 `}` 才收尾 -------------------------------------
#
# 评审逮到的 Critical。原先标量后遇到的 `}` 会被当成**根对象的收尾括号** ⇒
# 嵌套对象的值(`"f0": {}`)**静默吞掉整段回答**,而调用方按 §7 的契约拿到的是
# `useful=True + done=True + answer=""` ⇒ **什么都不发、也不走兜底**。
# 一条**静默空回复**比任何一种兜底都糟,而且调用方测不出来。

@pytest.mark.parametrize("stream", [
    '{"useful": true, "f0": {}, "answer": "hi"}',            # 空对象
    '{"useful": true, "f0": {"a": 1}, "answer": "hi"}',      # 非空对象
    '{"useful": true, "f0": {"a": {"b": null}}, "answer": "hi"}',   # 两层嵌套
    '{"useful": true, "tags": [1, 2], "answer": "hi"}',      # 数组(防回归)
])
def test_extra_nested_value_does_not_swallow_the_answer(stream):
    """嵌套的 `{...}` / `[...]` 不得把根对象的收尾提前 ⇒ `answer` 必须完整拿到。"""
    d, ev = _run([stream])
    assert d.answer == "hi", f"{stream}: answer 被吞了"
    assert _deltas(ev) == "hi"
    assert d.violation is None, f"{stream}: 这是合法协议,不该违规"
    assert d.done is True, f"{stream}: 流应当正常收尾"
    assert d.useful is True
    assert [e.kind for e in ev].count("done") == 1


def test_extra_string_valued_confidence_emits_no_confidence_event():
    """⚠️ **钉住实际行为**:`confidence` 的值是**字符串**时,压根不会有 confidence 事件。

    走的不是「解析失败 ⇒ `None`」那条路 —— 字符串值由 `_finish_string` 收尾,
    而那个分支**只把状态推回 `_EXPECT_KEY`、不解析值**(只有 `useful` 与 `answer`
    两个键在那里有特判)。所以 **`confidence is None` 且事件表里没有 confidence**。

    **行为不改**(§5.4:`confidence` 只记录,不做第二个阈值),但报告 §7-3 给 T10 的
    说明原先写成「解析失败仍会吐 `Event("confidence", None)`」—— 那句只对标量成立。
    这条用例把真实的形状钉住,免得 T10 照着一句错描述写。**标量路径**那条仍然成立:
    `{"confidence": abc}` 会吐一个 `value=None` 的 confidence 事件。
    """
    stream = '{"useful": true, "confidence": "0.9", "answer": "x"}'
    d, ev = _run([stream])
    assert d.confidence is None
    assert [e.kind for e in ev] == ["useful", "answer_delta", "done"]
    assert not any(e.kind == "confidence" for e in ev)
    # 对照组:**标量**解析失败时,那个事件是会吐的(值与形状都不同)
    d2, ev2 = _run(['{"useful": true, "confidence": abc, "answer": "x"}'])
    assert d2.confidence is None
    assert [e.kind for e in ev2] == ["useful", "confidence", "answer_delta", "done"]
    assert [e.value for e in ev2 if e.kind == "confidence"] == [None]


def test_extra_nested_value_split_across_chunks():
    """同上,但把那个嵌套对象的 `{` / `}` 切在两个 chunk 里。"""
    stream = '{"useful": true, "f0": {"a": 1}, "answer": "hi"}'
    at = stream.index("{", 1)          # 嵌套对象的开括号
    d, ev = _run([stream[:at + 1], stream[at + 1:]])
    assert d.answer == "hi"
    assert d.done is True and d.violation is None
    assert [e.kind for e in ev].count("done") == 1


def test_extra_answer_value_that_is_an_object_yields_no_answer():
    """⚠️ **记账**:`answer` 的值是**对象**时,`answer` 为空且**不违规**。

    这是裁定里明确给的期望(`violation is None` / `done is True`)。但要说清楚:
    它与上面那条 Critical 的**形态相同** —— `useful=True + answer=""` 到了调用方
    就是一条静默空回复。区别在于这次是**模型真的没给字符串答案**(协议违规),
    而不是我们把已经收到的答案弄丢了。

    **本用例只钉「当前行为」,不是「已认可的行为」**;要不要按裁定 ④ 的同一条
    fail-open 逻辑(非字符串 `answer` ⇒ `violation` ⇒ 调用方降级成纯文本)处理,
    见报告 §10 的 concern,需要控制器拍板 —— 届时改的是这个断言,那是**有意**的改动。
    """
    d, ev = _run(['{"useful": true, "answer": {"text": "hi"}}'])
    assert d.answer == ""
    assert d.violation is None
    assert d.done is True
    assert _deltas(ev) == ""


# ---- 代理对:非 BMP 字符必须拼回一个码点 ----------------------------------

def test_extra_surrogate_pair_is_combined_and_is_utf8_encodable():
    """`\\ud83d\\ude00`(`😀`,`ensure_ascii=True` 的网关**正是这么发**)必须拼回一个码点。

    不拼的话 `answer` 里是两个**孤立代理**,而 `answer.encode("utf-8")` 直接抛
    `UnicodeEncodeError: surrogates not allowed` —— T10 要把 `answer_delta.value`
    放进 **UTF-8 的 SSE `token` 帧**、并进 `log_turn` 的 JSON
    ⇒ 那会变成**流中途一条 `error` 帧**,而不是一句回答。
    """
    wire = '{"useful": true, "confidence": 0.5, "answer": "\\ud83d\\ude00 ok"}'
    want = json.loads(wire)["answer"]
    assert want == "\U0001f600 ok"
    first = wire.index("\\ud83d")
    cases = {
        "整段喂": [wire],
        "切开第一个 \\u 的十六进制位": [wire[:first + 2], wire[first + 2:]],
        "切在两个 \\u 之间": [wire[:first + 6], wire[first + 6:]],
        "逐字符": list(wire),
    }
    for label, parts in cases.items():
        d, ev = _run(parts)
        assert d.answer == want, f"{label}: {d.answer!r} != {want!r}"
        assert _deltas(ev) == want, f"{label}: 增量流对不上"
        d.answer.encode("utf-8")            # ← 关键:绝不抛 UnicodeEncodeError
        assert d.done is True and d.violation is None


@pytest.mark.parametrize("stream,label", [
    ('{"useful": true, "answer": "\\ud83d"}', "孤立高代理在串尾"),
    ('{"useful": true, "answer": "\\ud83dX"}', "高代理后面不是低代理"),
    ('{"useful": true, "answer": "\\ude00"}', "孤立低代理"),
    ('{"useful": true, "answer": "\\ud83d\\ud83d\\ude00"}', "两个高代理后跟一个低代理"),
    ('{"useful": true, "answer": "\\ud83d", "confidence": 0.5}', "后面还有别的键"),
])
def test_extra_lone_surrogate_becomes_replacement_char(stream, label):
    """**孤立代理**(不配对的)⇒ U+FFFD。口径写死在这里,别让它漂移。

    **刻意与 `json.loads` 不同**:后者原样给回一个孤立代理,而那正是上面那条
    `UnicodeEncodeError` 的形态。选 U+FFFD 的理由是**可预测 + 一定编得出来** ——
    宁可让用户看见一个 `�`,也不能让 SSE 在流中途吐 `error` 帧。
    """
    d, _ = _run([stream])
    d.answer.encode("utf-8")                       # ← 绝不抛
    assert "�" in d.answer, f"{label}: 应当有替换字符"
    assert not any("\ud800" <= c <= "\udfff" for c in d.answer), f"{label}: 不许留孤立代理"


def test_extra_done_is_emitted_at_most_once():
    """`done` 事件**至多一次** —— 读代码时逮到的真缺陷,补上用例钉住。

    `{"useful": false}` 的标量**直接在 `}` 上结束**,于是 `_finish_scalar`
    收一次尾、同一个字符接着又被 `_EXPECT_KEY` 收一次尾 ⇒ 两个 `done`。
    brief 的清单里没有这一条(它的 `useful=false` 用例后面都跟着 `,`)。
    """
    for stream in (
        ['{"useful": false}'],
        ['{"useful": false, "answer": ""}'],
        ['{"useful": true, "answer": "x"}'],
    ):
        ev = _feed_all(JsonAnswerDecoder(), stream)
        assert [e.kind for e in ev].count("done") == 1, f"{stream} 的 done 不是恰好一个"
