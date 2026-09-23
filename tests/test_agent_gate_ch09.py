"""ch09 闸:判据换成 `evidence_confidence` 之后的通过/不通过两支。

⚠️ **证据一律用 dict 形状**(`retrieve_knowledge` 写进 `state["evidence"]` 的那一种),
**不用 `RetrievedChunk`**:闸里读的是 `app/kb/evidence.py`(属性访问 `c.score`),
而通道里装的是**键值 dict** —— 拿对象当入参会绕过「dict → 对象」那一步转换,
于是测试全绿、而生产上**每一次知识问答**都炸在同一个属性上。
判据还是本仓那条:**替身的形状必须等于生产的形状**(ch07 形态 ⑦「替身替被测对象
完成了语义」)。

⚠️ 通过/不通过两支**共用同一个 `_settings()`** —— 两个结论必须来自**数据**,
不能来自配置。配置那一侧靠变异验证(把阈值拉到 1.0 / 压到 0.0),见报告。
"""

import pytest

from app.agent.nodes import make_confidence_gate_node
from app.config import Settings


def _settings(**over):
    base = {
        "openai_base_url": "http://x", "openai_api_key": "k",
        "openai_model": "m", "database_url": "mysql://x",
        # 0.5 落在"弱证据算出来 0.2267"与"强证据算出来 0.74"之间 ⇒ 同一份配置下
        # 两支分开。**这个数是用例的局部阈值,不是生产默认值**
        # (生产默认值由 `tests/test_config_ch09.py` 钉)。
        "evidence_confidence_threshold": 0.5,
    }
    return Settings(**{**base, **over}, _env_file=None)


class _FakeSession:
    def __init__(self):
        self.added = []
        self.commits = 0

    def add(self, row):
        self.added.append(row)

    async def commit(self):
        self.commits += 1


def _gate(session, settings):
    return make_confidence_gate_node(
        settings=settings, session=session, conversation_id="c1"
    )


def _ev(score, chunk_id=1):
    """一条证据 —— 键与 `retrieve_knowledge` 写进通道的那六个**逐字相同**。"""
    return {
        "chunk_id": chunk_id, "section_path": "退换货 > 退货政策",
        "question": "怎么退货", "answer": "七天无理由退货",
        "category": "退换货", "score": score,
    }


@pytest.mark.anyio
async def test_weak_evidence_is_blocked_and_reason_names_all_three_signals():
    """单条 0.2:`0.6*0.2 + 0.2*(1/3) + 0.2*0.2 = 0.2267 < 0.5` ⇒ 拦。

    拦下之后落池的那条 `reject_reason` 必须**写全三个信号** —— 审核页旁边
    就是这段文字,它要回答的是"为什么这条被判成答不了",不是一个分数。
    """
    s = _settings()
    sess = _FakeSession()
    out = await _gate(sess, s)({"user_input": "猫砂盆多少钱", "evidence": [_ev(0.2)]})

    assert out["gate_passed"] is False
    assert out["trace"] == ["confidence_gate:fail"]
    assert sess.commits == 1 and len(sess.added) == 1
    assert sess.added[0].entry_point == "置信度闸"

    r = sess.added[0].reject_reason
    assert "置信度" in r
    assert "低于阈值" in r
    assert "top1=" in r
    assert "条数=" in r
    assert "分差=" in r


@pytest.mark.anyio
async def test_strong_evidence_passes_and_writes_nothing():
    """三条 0.9:`0.6*0.9 + 0.2*1.0 + 0.2*0 = 0.74 ≥ 0.5` ⇒ 过,且不落池。

    ⚠️ 三条**同分** ⇒ 分差信号是 0,所以这条过闸靠的是 top1 与条数两项,
    不是「分差大」。换一组不同分的强证据会得到更高的分,但那样就同时证明了
    三件事,红了反而看不出坏在哪一项。
    """
    s = _settings()
    sess = _FakeSession()
    ev = [_ev(0.9, chunk_id=i) for i in range(1, 4)]
    out = await _gate(sess, s)({"user_input": "退货政策", "evidence": ev})

    assert out["gate_passed"] is True
    assert out["trace"] == ["confidence_gate:pass"]
    assert sess.commits == 0 and sess.added == []


@pytest.mark.anyio
async def test_empty_evidence_is_blocked_with_readable_reason():
    s = _settings()
    sess = _FakeSession()
    out = await _gate(sess, s)({"user_input": "x", "evidence": []})

    assert out["gate_passed"] is False
    assert sess.added[0].reject_reason == "检索为空"


@pytest.mark.anyio
async def test_empty_evidence_is_blocked_even_with_a_zero_threshold():
    """**fail-closed**:阈值压到 0(合法下界)时,空证据仍必须拦。

    `evidence_detail([])` 给的 `confidence` 就是 `0.0`,而 `0.0 >= 0.0` 为真 ——
    判据里**少了 `bool(evidence)` 这一项**,空证据在阈值为 0 时就会放行,进 Agent
    带着一段空的知识去作答。用例专门把阈值压到 0 才看得见这个差别。
    """
    s = _settings(evidence_confidence_threshold=0.0)
    sess = _FakeSession()
    out = await _gate(sess, s)({"user_input": "x", "evidence": []})

    assert out["gate_passed"] is False
    assert sess.added[0].reject_reason == "检索为空"


@pytest.mark.anyio
async def test_single_mid_score_that_the_old_criterion_passed_is_now_blocked():
    """**判据真的换了** —— 这条数据在旧判据下通过、在新判据下被拦。

    单条 0.3:`0.6*0.3 + 0.2*(1/3) + 0.2*0.3 = 0.3067`,低于 0.5 ⇒ 拦;
    而旧判据 `max(分数) >= retrieval_score_threshold`(0.3 ≥ 0.25,生产默认值)
    ⇒ **放行**。所以「回退成旧判据」这类改动会让这条用例红 —— 它是本章
    「判据换掉了」这条事实的**唯一**守卫(其余用例在旧判据下同样成立)。
    """
    s = _settings()
    # 前提断言:旧判据的阈值确实在这条分数**下方**(高了的话本用例就失去意义,
    # 而它红起来会指向"用例过期"而不是"实现错了")。
    assert 0.3 >= s.retrieval_score_threshold

    sess = _FakeSession()
    out = await _gate(sess, s)({"user_input": "猫砂盆多少钱", "evidence": [_ev(0.3)]})

    assert out["gate_passed"] is False
    assert sess.added[0].entry_point == "置信度闸"
