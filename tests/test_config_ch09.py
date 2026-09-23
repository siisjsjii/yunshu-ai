"""ch09 新增配置项:默认值 + 下界。"""

import pytest

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
