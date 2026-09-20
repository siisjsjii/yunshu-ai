import pytest

from app.refund.categories import REFUND_REASON_CATEGORIES, is_valid_category


def test_categories_are_a_closed_nonempty_set():
    assert len(REFUND_REASON_CATEGORIES) >= 3
    assert len(set(REFUND_REASON_CATEGORIES)) == len(REFUND_REASON_CATEGORIES)


def test_valid_category_accepts_only_exact_members():
    first = REFUND_REASON_CATEGORIES[0]
    assert is_valid_category(first) is True


def test_valid_category_accepts_every_member():
    """**每一个**成员都必须通过,不能只认第一个。

    只断 `is_valid_category(REFUND_REASON_CATEGORIES[0]) is True` 时,
    一个「只认第一个类目」的实现(哪怕写成 `value == REFUND_REASON_CATEGORIES[0]`)
    照样全绿 —— 上面那条是必要但不充分的。
    """
    assert [c for c in REFUND_REASON_CATEGORIES if not is_valid_category(c)] == []


@pytest.mark.parametrize("bad", ["", "不存在的类目", " 商品质量问题 "])
def test_valid_category_rejects_non_members_without_trimming(bad):
    """不做 strip 归一:带空格的输入更可能是一次真实的传参错误,不是同一个类目。"""
    assert is_valid_category(bad) is False
