"""ch09 新增配置项:默认值 + 下界。"""

import pytest
from pydantic import ValidationError

from app.config import Settings


def _settings(**over):
    base = {
        "openai_base_url": "http://x", "openai_api_key": "k",
        "openai_model": "m", "database_url": "mysql://x",
    }
    return Settings(**{**base, **over}, _env_file=None)   # 硬约束:必须传 _env_file=None


def test_langfuse_defaults_are_empty_so_tests_never_go_online():
    s = _settings()
    assert s.langfuse_public_key == ""
    assert s.langfuse_secret_key == ""
    assert s.langfuse_base_url == "https://us.cloud.langfuse.com"


def test_evidence_weights_are_bounded():
    s = _settings()
    assert s.w_evidence_top1 + s.w_evidence_count + s.w_evidence_gap == pytest.approx(1.0)
    assert 0.0 <= s.evidence_confidence_threshold <= 1.0
    assert s.evidence_max_count >= 1


def test_snapshot_and_flywheel_bounds():
    s = _settings()
    assert s.snapshot_top_n >= 1
    assert s.snapshot_answer_chars >= 1
    assert s.flywheel_batch_size >= 1


def test_out_of_range_is_rejected():
    with pytest.raises(Exception):
        _settings(evidence_confidence_threshold=2.0)
    with pytest.raises(Exception):
        _settings(snapshot_top_n=0)


# ---- 以下为 T1 修复轮追加:上面四条断言强度不够,这里补齐 ----
#
# 为什么必须补:`pydantic_settings.BaseSettings` 的 `validate_default=True`
# (实测:一个 `default=2.0, ge=0.0, le=1.0` 的探针类**构造时就抛**,
#  而同样的探针作为普通 `BaseModel` 正常构造)⇒ **凡是构造得出来的 `Settings`,
# 其默认值本就落在自己声明的范围内**。于是上面那几条 `0.0 <= x <= 1.0` /
# `x >= 1` 对**一个完全没有声明任何边界的实现**同样全真(已实测:无边界版
# 三个断言全 True)—— 它们测的是「默认值的范围」,不是「字段上钉了 ge/le」。


def test_ch09_fields_have_defaults():
    """只钉默认值不被静默改掉。

    与 `test_config.py` 的 `test_ch03_fields_have_defaults` 同一个目的:
    这些数是**规格的一部分**(spec §4 的加权公式直接吃三个权重),改一个数
    就是规格符合性失败,而它**不会有任何运行时症状** —— 置信度分数悄悄变了,
    阈值却还是老的,表现为拦截率/误杀率整体漂移。
    ⚠️ 权重之和那条(`test_evidence_weights_are_bounded`)只钉住了「和」:
    `0.5/0.3/0.2` 也满足它,而 §4 的分数会跟着变。所以逐个 `==` 是必须的。
    """
    s = _settings()
    assert s.langfuse_public_key == ""
    assert s.langfuse_secret_key == ""
    assert s.langfuse_base_url == "https://us.cloud.langfuse.com"
    # T8 已标定,占位值 0.42 换成 0.2(spec §15.9)—— 这条 `==` 按它**自己**
    # 上面那句「届时这个 `==` 要一起改」跟着改。**断言形式一个字没动**:
    # 仍然是"逐字钉住那个数",所以它守的"默认值不被静默改掉"这个性质没有变弱;
    # 变的只是被钉的那个值从"占位"变成"标定值"。
    # 标定读数:阈值 0.2 ⇒ 拦截率 0.967 / 误杀率 0.175(evals/测试集.md 300 条)。
    assert s.evidence_confidence_threshold == 0.2
    assert s.evidence_min_score == 0.15
    assert s.evidence_max_count == 3
    assert s.w_evidence_top1 == 0.6
    assert s.w_evidence_count == 0.2
    assert s.w_evidence_gap == 0.2
    assert s.snapshot_top_n == 5
    assert s.snapshot_answer_chars == 400
    assert s.flywheel_batch_size == 10


# [0,1] 上的比例/阈值:两个方向都要拒(越界一个方向等于永远全滤空,
# 另一个方向等于没有阈值)。
_PROPORTION_FIELDS = [
    "evidence_confidence_threshold",
    "evidence_min_score",
    "w_evidence_top1",
    "w_evidence_count",
    "w_evidence_gap",
]
# 条数/字数:<=0 会让对应的功能**静默空转**(见下面每条的理由)。
_COUNT_FIELDS = [
    "evidence_max_count",
    "snapshot_top_n",
    "snapshot_answer_chars",
    "flywheel_batch_size",
]


@pytest.mark.parametrize("bad", [-0.1, 1.1])
@pytest.mark.parametrize("field", _PROPORTION_FIELDS)
def test_out_of_range_proportions_are_rejected(field, bad):
    """九个带边界字段里,这五个的两个方向都必须被拒 —— 断言**出错信息里
    出现该字段名**,所以它测的是「**哪个**字段被拒」,不是「有没有抛」。"""
    with pytest.raises(ValidationError) as exc:
        _settings(**{field: bad})
    assert field in str(exc.value)


@pytest.mark.parametrize("bad", [0, -1])
@pytest.mark.parametrize("field", _COUNT_FIELDS)
def test_non_positive_counts_are_rejected(field, bad):
    """0 与负数两种越界**都不报错、只静默变坏**,所以必须在启动时拒:

    - `evidence_max_count <= 0` → 加权公式里的 `min(条数/max_count, 1)` 除零,
      或被钳成恒 1(条数那一项从此没有区分力);
    - `snapshot_top_n <= 0` → 快照存 0 条召回片段,飞轮的训练数据**永远是空的**,
      而流水线每一步都"成功";
    - `snapshot_answer_chars <= 0` → 落库的答案恒为空串,同上;
    - `flywheel_batch_size <= 0` → `range()` 空转,一轮飞轮"跑成功"但一条没处理
      —— 与 ch03 的 `mine_batch_conversations` 是同一个形状。

    仓库先例:`test_config.py:216-222`(rerank_top_k)与 `:243-249`
    (mine_batch_conversations)都是这个写法。
    """
    with pytest.raises(ValidationError) as exc:
        _settings(**{field: bad})
    assert field in str(exc.value)
