"""标注相关的纯函数:去重、证据串校验、标签漂移、分层抽样。"""

import pytest

from app.topic.labeling import dedupe_questions


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
    """「退货  怎么走」与「退货怎么走」是同一句 —— 按原文去重会漏。

    池子里的问题来自不同轮次,空白/全角差异很常见;按原文去重会
    让同一条问题在合成配额里被算两次,分布跟着偏。
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
