"""对话前置校验测试。全部用替身,不联网、不碰 DB。

ch05 Task 8:`stream_turn` 连同它的 18 条用例一起删掉了 —— 单轮编排改由图
(`app/agent/nodes.py` 的 agent 节点 + `app/agent/graph.py`)负责。

ch07 T10:`prepare_turn` 从「校验 + 裁剪并返回历史」缩成**纯校验** ——
历史改由 `layers` 分层派生(见 `app/services/chat.py` 的 docstring),
`trim.select_history` 连同它的三条用例一起删除(那三条里守着的**不变量**
搬到了 `tests/test_trim.py` 的 `to_rounds` 一族上)。

本文件因此只剩**两条 400 判据**,每条都要有一条**能区分**的用例:
把判据删掉或放宽 ⇒ 对应那条变红(判别力实测见 task-10-report.md)。
"""

import pytest

from app.config import Settings
from app.memory import budget
from app.memory.trim import ContextBudgetUnavailable, ContextOverflowError
from app.prompts import render_system_prompt
from app.services.chat import prepare_turn

REQUIRED = {
    "openai_base_url": "https://example.invalid/v1",
    "openai_api_key": "sk-test",
    "openai_model": "test-model",
    "database_url": "mysql+asyncmy://u:p@h:3306/db",
}


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **REQUIRED, **overrides)


def _budget(settings: Settings):
    return budget.derive(
        settings=settings, system_prompt=render_system_prompt(settings.brand_name)
    )


def test_prepare_turn_accepts_a_turn_that_fits_and_returns_nothing():
    """装得下就**什么都不抛**,而且**不返回历史**(它是纯校验)。

    两个方向都有判别力:整段实现写成「恒定抛异常」时这里红;
    而「又把历史返回回来了」的实现会让 `is None` 红 —— 那正是本任务要钉的契约
    (返回历史意味着调用方可能拿它去分层,而裁剪过的历史会让层 2 少算)。
    """
    settings = _settings()
    assert prepare_turn(
        settings=settings, budget=_budget(settings), user_input="在吗"
    ) is None


def test_prepare_turn_rejects_input_over_the_per_message_cap():
    """**本轮输入本身**超 `max_user_input_tokens` ⇒ 抛(端点据此在流前 400)。

    这条在 T2 到 T10 之间是**空档**:旧口径把 `count_tokens(user_input)` 算进
    「已用」,新口径把它归进单轮峰值的 `max_user_input_tokens`,于是实际输入
    长度**再也没有人比过** —— 50k token 的一句话会一路送到上游。

    判据取「同一段文本在松上限下**不抛**」作对照:少了它,一条「恒定抛异常」
    的实现也能让上面那半条通过。
    """
    text = "这是一句明显超过五个 token 的话"
    with pytest.raises(ContextOverflowError):
        prepare_turn(
            settings=_settings(max_user_input_tokens=5),
            budget=_budget(_settings(max_user_input_tokens=5)),
            user_input=text,
        )
    # 对照:同一个输入在足够大的上限下照常返回(None)。
    settings = _settings(max_user_input_tokens=2000)
    assert prepare_turn(
        settings=settings, budget=_budget(settings), user_input=text
    ) is None


def test_prepare_turn_rejects_a_budget_that_cannot_fit_one_round():
    """**spec §8 的判据是「装不下一轮」,不是「预算为负」**。

    两者在「预算为正、但小于 `per_round_steady`」时结论相反:比如
    `history_budget = 4537` 而 `per_round_steady = 10000`,按 spec 该 400,
    按 `history_budget < 0` 却**静默带着一小段历史往下走** —— 用户拿到一个
    上下文被悄悄截到几乎没有的回答,而没有任何东西报错。
    这条用例就是那个中间区间(实测:`keep_rounds=20 × 10000` 那一支打不过窗口,
    `history_budget` 落在 4537 > 0 上)。

    判别力:判据删掉(不检查)或放宽回 `< 0` ⇒ 这条红(实测见报告)。

    断的是**子类** `ContextBudgetUnavailable`,不是父类(终审 Minor 12):本章
    此前的修复只钉住了「两个类的文本不同」,没有任何东西断言**这条分支真的
    走到子类上** —— 实现退回 `raise ContextOverflowError(...)` 时,因为
    `ContextBudgetUnavailable` 是它的子类,凡是 `pytest.raises(父类)` 的断言
    都照样绿,而用户看到的是「你的话太长」那种与本故障无关的文案。
    """
    settings = _settings(per_round_steady=10000)
    b = _budget(settings)
    assert 0 < b.history_budget < settings.per_round_steady   # ← 前提:那个中间区间

    with pytest.raises(ContextBudgetUnavailable):
        prepare_turn(settings=settings, budget=b, user_input="在吗")
