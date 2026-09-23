"""证据置信度 —— 把"检索 + 精排的结果"压成一个 0–1 的分(ch09 spec §4.1)。

**纯函数、零 IO、不依赖 LangChain。** 与 `app/kb/assess.py` 同层。

三个信号(**都在 `evidence_min_score` 过滤后的集合上取**):
  top1  精排最高分(`RetrievedChunk.score` **已经是重排 sigmoid 分**)
  count 有效证据条数(分数 ≥ `evidence_min_score` 的那些,封顶 `evidence_max_count`)
  gap   top1 − top2(过滤后只剩一条时 top2 记 0 ⇒ gap 就是 top1)

为什么要有 count 与 gap:单条高分可能是巧合;**够多条中高分**才叫"知识库覆盖了";
分差大说明那条明确对口,分差小说明几条都不对口。

⚠️ **与 spec §4.1 的一处偏离(记账)**:spec 那张表把 top1 写作 `max(c.score)`、
未提过滤。实现改成**先按 `evidence_min_score` 过滤再取 top1/top2**,理由见
`evidence_detail` 里的注释与 `tests/test_kb_evidence.py` 那条低分用例的实测
(0.4647 vs 0.4667)。影响面:全部块都低于 min_score 时返回全 0(不过滤则会返回
一个由噪声分算出的非零置信度)。一句话回退:把 `valid` 换回 `sorted(所有 score)`。

⚠️ **阈值不是拍出来的**:`evidence_confidence_threshold` 由
`scripts/calibrate_evidence.py` 在 `evals/测试集.md` 上标定(spec §4.2)。
本模块的**权重初值是拍的**,标定只负责阈值 —— 这两件事别混。
"""

from collections.abc import Sequence

from app.config import Settings
from app.retrieval.search import RetrievedChunk


def evidence_detail(
    chunks: Sequence[RetrievedChunk], *, settings: Settings
) -> dict:
    """三个信号 + 合成分。空证据返回全 0。"""
    if not chunks:
        return {"top1": 0.0, "top2": 0.0, "count": 0, "gap": 0.0, "confidence": 0.0}

    # ⚠️ **先按 evidence_min_score 过滤,再取 top1/top2** —— 不是"先分高低、再只
    # 拿过线的块数数"。两种写法在"条数"上一样,在 **gap** 上不一样,而差别是错的
    # 那一侧:不过滤时一条 0.01 的噪声会占住 top2 的位子,把 gap 从 0.50 压到 0.49
    # (`tests/test_kb_evidence.py::test_low_scores_do_not_count_toward_the_count_signal`
    # 实测 0.4647 vs 0.4667)—— 即"噪声**压低**置信度",与"噪声不该进判据"相反。
    # 更糟的是:全部块都低于 min_score 时,不过滤会让 top1 仍是那条噪声分,于是一个
    # 知识库根本没覆盖的问题拿到非零置信度。min_score 的语义只有一个 ——
    # 「**什么算一条证据**」—— 三个信号都该在它过滤后的集合上算。
    valid = sorted(
        (c.score for c in chunks if c.score >= settings.evidence_min_score),
        reverse=True,
    )
    if not valid:
        # 一条证据都没有 == 空证据。与上面的 `if not chunks` 同一个出口。
        return {"top1": 0.0, "top2": 0.0, "count": 0, "gap": 0.0, "confidence": 0.0}

    top1 = float(valid[0])
    top2 = float(valid[1]) if len(valid) > 1 else 0.0
    # clamp 到 [0,1]:spec §4.1 的合成式写的就是 clamp(top1-top2, 0, 1)。
    # 两个 sigmoid 分本身就在 [0,1] 内,gap 天然不会越界;这里留着是防"上游哪天
    # 换了不打 sigmoid 的重排器"——那时这条 clamp 是唯一挡住 confidence > 1 的东西。
    gap = max(0.0, min(top1 - top2, 1.0))
    count = len(valid)

    count_signal = min(count / settings.evidence_max_count, 1.0)
    confidence = (
        settings.w_evidence_top1 * top1
        + settings.w_evidence_count * count_signal
        + settings.w_evidence_gap * gap
    )
    return {
        "top1": round(top1, 4),
        "top2": round(top2, 4),
        "count": count,
        "gap": round(gap, 4),
        "confidence": round(min(max(confidence, 0.0), 1.0), 4),
    }


def evidence_confidence(
    chunks: Sequence[RetrievedChunk], *, settings: Settings
) -> float:
    return evidence_detail(chunks, settings=settings)["confidence"]
