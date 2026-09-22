"""ch07:两个上下文日志。**只做组装与落盘 —— 无 IO、无副作用、不依赖 LangChain。**

这是本章唯一的观测面。上下文组装在外面完全看不见:中段有没有被截短、
梗概有没有被注入、三层边界选在哪 —— 全都发生在没人看得见的那一串消息里。
本章之前,「截短生效」与「截短静默失效」在外部**长得一模一样**。

**为什么是 JSON 而不是自由文本**:spec §10.3 明确「不测日志格式 —— 它是给人
看的」,而 §10.2 又要求「层 2 的计数确实按截短后」。两条要同时成立的唯一办法
就是把**契约**(键与值)与**文案**(措辞)分开:JSON 让单测断前者、人读后者。
改一句措辞不该让任何测试变红;少一个键或取错版本必须变红。

`ensure_ascii=False`:中文原样写进 `log/app.log`(handler 的 utf-8 由
`app/logging_setup.py` 钉住)。转义成 `\\uXXXX` 的话日志对人就没用了。

**四条硬要求**(spec §7.6)。注意:验收 4b 判的是 **`history_ctx` 那一行**里
看得见截短后的形态,而那条线**要等分层接上(T10)才成立** —— 详见
`history_ctx` 的 docstring,别把 4b 挪到 `model_ctx` 上:
1. `tokens` **必须分段** —— 只给一个总计的话,层 2 那段的数字再也不会随截短
   而变,「按截短后计数」失效与生效无从区分;
2. `sliding` 装的是**截短后**的消息,它要能一眼看出 `…` 与 `[工具结果] `;
3. `model_ctx` 的 `summary` 是**全文**,不是条数;
4. **不记 prompt / response 原文**(密钥泄漏面 + 日志膨胀源,spec §8)。
   这里记的是**组装后的上下文窗口**(`sliding`)与梗概 —— 那两样正是要观测的
   东西;而 system prompt 全文与模型的原始响应一个字都不进日志。**异常**这条
   路径在本模块不存在(它不捕获任何东西),将来若加,文本必须先过
   `app.sanitize.redact_api_key`。
"""

import json
import logging
from collections.abc import Sequence

from app.memory import trim
from app.memory.budget import ContextBudget
from app.memory.layers import Layers
from app.schemas import Message

logger = logging.getLogger(__name__)


def _sliding(messages: Sequence[Message]) -> list[dict]:
    """逐条消息的**角色 + 内容**,内容取**传进来的那一份**(即截短后的形态)。

    本函数**不做任何截短** —— 截短是 `layers.truncate` 在分层时做的,这里只是
    把它如实搬到日志里。在这里再截一次就是两把尺子:日志会显示一个与模型
    实际收到的不一样的窗口,而两者看起来都合理。
    """
    return [{"role": message.role, "content": message.content} for message in messages]


def _summary_rows(summaries: Sequence[tuple]) -> list[dict]:
    """把摘要行归一成 `{"seq", "upto_msg_id", "content"}`。

    接受两种形态:
    - `(seq, content)` —— `app/services/history.load_summaries` 今天的返回;
    - `(seq, upto_msg_id, content)` —— spec §7.6 的 payload 形状,调用方手里
      有锚点时用。

    `upto_msg_id` 是**能把梗概与锚点对上**的唯一字段:「这一段压到哪条」。
    拿不到时记 `None`(而不是省略这个键)—— 键恒在,读日志的人与断言都不用
    先判断形态;而把「不知道」和「压到第 0 条」混为一谈才是错的。
    """
    rows = []
    for item in summaries:
        seq, content = item[0], item[-1]
        upto = item[1] if len(item) == 3 else None
        rows.append({"seq": seq, "upto_msg_id": upto, "content": content})
    return rows


def _emit(prefix: str, payload: dict) -> None:
    """一行 JSON,前缀固定。**前缀就是契约的一部分** —— 断言与 `grep` 都认它。"""
    logger.info("%s %s", prefix, json.dumps(payload, ensure_ascii=False))


def model_ctx(
    *,
    conversation_id: str,
    summary: str,
    layers: Layers,
    evidence_tokens: int,
    budget: ContextBudget,
) -> None:
    """主力 Agent 每次调模型前一行。

    `sliding` 是**组装顺序**(层 2 截短段在前、层 1 原文段在后,spec §7.3),
    也就是模型实际看到的那个窗口。

    `budget` 是 `memory.budget.derive` 的产物。日志里同时给**用量与预算**,
    是因为只给用量的话「这段快满了」要靠人记住窗口有多大 —— 而那正是本章
    调参(`per_round_steady` 等未实测值)要做的事。

    `bounds` 取自 `layers` 自己带的两个锚点,**不接受调用方另行传入**:`bounds`
    是把这一行日志与**某一段具体历史**对上的唯一字段,而它一旦由调用方传、
    又带个 `0` 的默认值,拿不到锚点的调用方(agent 节点里根本没有锚点来源)
    就会报出 `{0, 0}` 而切分用的是真锚点 —— 一个长得像真值、却什么也没说的
    观测面。跟着 `Layers` 走,两者就不可能不一致。
    """
    sliding = _sliding([*layers.layer2, *layers.layer1])
    summary_tokens = trim.count_tokens(summary) if summary else 0
    _emit(
        "model_ctx",
        {
            "conversation_id": conversation_id,
            "summary": summary,          # 全文,不是条数
            "sliding": sliding,          # 截短后的实际形态
            "rounds": len(sliding),
            "tokens": {
                "layer1": layers.layer1_tokens,
                "layer2": layers.layer2_tokens,
                "summary": summary_tokens,
                "evidence": evidence_tokens,
                "total": (
                    layers.layer1_tokens
                    + layers.layer2_tokens
                    + summary_tokens
                    + evidence_tokens
                ),
            },
            "budgets": {
                "layer1": budget.layer1_budget,
                "layer2": budget.layer2_budget,
            },
            "bounds": {
                "summary_upto_msg_id": layers.summary_upto_msg_id,
                "layer1_from_msg_id": layers.layer1_from_msg_id,
            },
        },
    )


def history_ctx(
    *,
    conversation_id: str,
    summaries: Sequence[tuple],
    history: Sequence[Message],
    budget: ContextBudget,
) -> None:
    """指代消解 / 意图识别共用的那份上下文,**每轮必打**。

    **「每轮必打」包括不进 Agent 的那几轮**(闲聊 / 投诉 / 兜底 / 退款子流程):
    那几轮恰恰最容易「看起来正常、其实上下文是错的」—— ch06 的一整个交付物
    就是那么丢的(节点写了值、通道根本不存在、单测全绿而生产恒为 `None`),
    而它当时没有任何观测面。

    ⚠️ **这一份不分层**:入参 `history` 是端点已经裁好的**单一滑窗**
    (指代消解与意图识别都只看最近的一段)。所以:

    - `tokens.layer2` 与 `tokens.evidence` 结构性为 `0`,**不是因为截短/检索
      没生效** —— 键保留只是为了与 `model_ctx` 同形(读日志的人用同一套键读
      两行),而不是在声称这里有分层;
    - **没有 `bounds`**。这份上下文不是由「两个锚点切出来的」,给它补一对锚点
      只能靠调用方另行传入,而那正是「长得像真值、其实什么也没说」的形状
      (`0` 在本章是有含义的值)。spec §7.6 给 `history_ctx` 的字段表里本来
      也只有 `summary` / `sliding` / `tokens`。

    ⚠️ **当下它还承载不了验收 4b。** 4b 判的是 `history_ctx` 里看得见**截短后的
    形态**(客服答复带 `…`、工具结果是一行 `[工具结果] `),而端点今天传进来的
    `history` 是 `trim.select_history` 的输出 —— 那个函数**只整轮丢弃、从不标注
    内容**,所以那两个标记**不可能出现**。产出它们的是 `layers.truncate`,
    分层要到 T10 才接上。`model_ctx` 的 `sliding` 确实已经带截短形态,但**那不是
    4b 指定的那条线** —— 4b 指定的是本函数这一行,别把验收标准挪过去迎合实现。

    `tokens.summary` 是**摘要段**的估算:多段摘要按 `"\\n\\n"` 相拼后计数
    (与 `prompts.py` 里段间连接符同源)。真注入时用的是
    `memory.summarize.join_summaries` 的拼法 —— 两者若日后不一致,这个数会
    偏一点(而**日志与模型实际收到的不是同一段文本**,正是本模块最该避免的
    那种「看起来正常」)。**改 `join_summaries` 的拼法时把这里一起改。**
    """
    sliding = _sliding(history)
    rows = _summary_rows(summaries)
    summary_tokens = (
        trim.count_tokens("\n\n".join(row["content"] for row in rows)) if rows else 0
    )
    layer1_tokens = sum(trim.count_tokens(message.content) for message in history)
    _emit(
        "history_ctx",
        {
            "conversation_id": conversation_id,
            "summary": rows,             # 摘要行,不是全文
            "sliding": sliding,
            "rounds": len(sliding),
            "tokens": {
                "layer1": layer1_tokens,
                "layer2": 0,             # 见 docstring:这一份不分层
                "summary": summary_tokens,
                "evidence": 0,           # 消解/意图不注入检索证据
                "total": layer1_tokens + summary_tokens,
            },
            "budgets": {
                "layer1": budget.layer1_budget,
                "layer2": budget.layer2_budget,
            },
            # **没有 bounds**:这份上下文不是用两个锚点切出来的(见 docstring)。
        },
    )
