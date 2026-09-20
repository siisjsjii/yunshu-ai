import pytest

from app.refund.categories import REFUND_REASON_CATEGORIES, is_valid_category


def test_categories_are_a_closed_nonempty_set():
    assert len(REFUND_REASON_CATEGORIES) >= 3
    assert len(set(REFUND_REASON_CATEGORIES)) == len(REFUND_REASON_CATEGORIES)


def test_valid_category_accepts_only_exact_members():
    first = REFUND_REASON_CATEGORIES[0]
    assert is_valid_category(first) is True


def test_categories_are_exactly_the_agreed_five():
    """类目**内容**是前后端契约,不是可以随手增删的集合。

    上面的用例只断「>= 3 项且互不相同」,于是从元组里删掉「与描述不符」
    (或换成任何一个新词)**照样全绿** —— 而前端仍会把它作为选项发上来,
    `/api/refund` 的校验会把用户的正常选择判成 422。所以内容要逐个钉死。
    """
    assert set(REFUND_REASON_CATEGORIES) == {
        "商品质量问题",
        "不想要了",
        "发错货",
        "少发/漏发",
        "与描述不符",
    }


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
