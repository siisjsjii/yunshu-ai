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

def test_oracle_answer_matches_json_loads_at_every_split():
    """独立口径:`answer` 必须逐字等于 `json.loads` 的结果(不切 / 每一个切点)。"""
    payload = '{"useful": true, "confidence": 0.5, "answer": "中\\u4e2d\\"x\\\\y"}'
    for text in (FULL, payload):
        expected = json.loads(text)["answer"]
        assert _run([text])[0].answer == expected, "不切就对不上 json.loads"
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
