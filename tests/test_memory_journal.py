"""ch07 T7:上下文观测面 —— `journal` 的两行 JSON + 日志落盘。

本章的上下文组装在外面完全看不见:中段有没有被截短、梗概有没有被注入、
三层边界选在哪 —— 全都发生在没人看得见的 prompt 里。**本文件验的就是
把这几件事变得看得见的那一层。**

**断言的是 JSON 契约(键在不在、值对不对),不是文案。** spec §10.3 明确
「不测日志格式 —— 它是给人看的」,而 §10.2 又要求「层 2 的计数确实按截短后」。
两条要同时成立的唯一办法就是把契约与文案分开:JSON 让单测断前者、人读后者。
改一句措辞不该让本文件变红;而**少一个键、值取错版本**必须变红。

`client_factory` 从 `tests/test_api_chat.py` import 进来(它不是 conftest 的
fixture —— 本仓的 conftest 只有 `anyio_backend`)。`tests/test_api_ticket.py`
已有同款先例(`from test_api_chat import FakeSession`)。
"""

import json
import logging
from logging.handlers import RotatingFileHandler

from test_api_chat import FakeChunk, FakeSession, client_factory

from app import main as main_module
from app.config import Settings
from app.db.models import Conversation, ConversationSummary, MessageRecord
from app.logging_setup import setup_logging
from app.memory import budget, journal, layers, summarize, trim
from app.schemas import Message

REQUIRED = dict(
    _env_file=None,
    openai_base_url="https://example.invalid/v1",
    openai_api_key="sk-test",
    openai_model="m",
    database_url="mysql+asyncmy://u:p@127.0.0.1:3306/x",
)


def _settings(**over):
    return Settings(**{**REQUIRED, **over})


def _last_payload(caplog, prefix: str) -> dict:
    """从日志里取出最后一条 `"<prefix> <json>"` 的 JSON 体。

    取不到**直接抛** —— 退化成「返回 {}」会让上面的 `<=` 断言恒真,
    而这条用例的全部价值就在于「那一行到底有没有」。
    """
    for record in reversed(caplog.records):
        if record.message.startswith(f"{prefix} "):
            return json.loads(record.message.split(" ", 1)[1])
    raise AssertionError(f"日志里没有 {prefix} 行")


# ---------------------------------------------------------------- model_ctx


def test_model_ctx_logs_segmented_tokens_not_only_a_total(caplog):
    """**必须能在日志里分开读到各段的 token 数。**

    只打一个总计的话,「层 2 按截短后计数」**失效与生效长得一模一样** ——
    这是 spec §7.6 专门为那处洞加的观测面,也是验收 4b 的判据。
    """
    s = _settings()
    b = budget.derive(settings=s, system_prompt="你是客服。")
    got = layers.split(
        [Message(id=1, role="user", content="你好")],
        summary_upto_msg_id=0, layer1_from_msg_id=0, settings=s,
    )
    with caplog.at_level(logging.INFO):
        journal.model_ctx(
            conversation_id="c1", summary="用户问过订单 1002",
            layers=got, evidence_tokens=0, budget=b,
        )

    payload = _last_payload(caplog, "model_ctx")
    assert {"layer1", "layer2", "summary", "evidence", "total"} <= set(payload["tokens"])
    assert payload["tokens"]["layer1"] == got.layer1_tokens     # ← 真的按截短后
    assert payload["bounds"]["layer1_from_msg_id"] == 0


def test_model_ctx_segments_carry_distinct_values_not_one_total(caplog):
    """**分段的值必须真的是分段的**,不能各段都填同一个总数。

    上一条只钉住「键在」与 `layer1` 那一段。一个把所有段都填成 `total` 的
    实现能让它**照样通过** —— 而那种日志与「只打一个总计」在观测上完全等价:
    层 2 那段的数字再也不会随截短而变。这条用两个**互不相等且都非零**的段
    把它区分开。
    """
    s = _settings()
    b = budget.derive(settings=s, system_prompt="你是客服。")
    got = layers.split(
        [
            Message(id=1, role="user", content="你好"),
            Message(id=2, role="assistant", content="客" * 200),
            Message(id=3, role="user", content="现在这句"),
        ],
        summary_upto_msg_id=0, layer1_from_msg_id=3, settings=s,
    )
    assert got.layer2_tokens > 0 and got.layer1_tokens > 0
    with caplog.at_level(logging.INFO):
        journal.model_ctx(
            conversation_id="c1", summary="用户问过订单 1002",
            layers=got, evidence_tokens=17, budget=b,
        )

    tokens = _last_payload(caplog, "model_ctx")["tokens"]
    assert tokens["layer2"] == got.layer2_tokens
    assert tokens["layer1"] == got.layer1_tokens
    assert tokens["evidence"] == 17
    assert tokens["total"] == (
        tokens["layer1"] + tokens["layer2"] + tokens["summary"] + tokens["evidence"]
    )
    # 段与段之间必须真的不同 —— 否则「分段」没有带来任何观测面。
    assert tokens["layer2"] != tokens["layer1"] != tokens["evidence"]


def test_model_ctx_sliding_is_the_truncated_form_plus_the_raw_tail(caplog):
    """`sliding` 装的是**截短后**的消息,不是别的什么东西。

    若 `sliding` 只装层 1(或装了原文),「截短真的生效了吗」在 `model_ctx`
    这一行里**看不出来**。

    ⚠️ **这不是验收 4b。** 4b 判的是 **`history_ctx` 那一行**里看得见截短后的
    形态(spec §10.5),而那条线要等 T10 把分层接上才成立 —— 端点今天传进去的
    `history` 是 `trim.select_history` 的输出,那个函数**只整轮丢弃、从不标注
    内容**,`…` 与 `[工具结果] ` 不可能出现。这里能看见形态,是因为本用例
    自己拿 `layers.split` 造了一份**分了层**的 `Layers`;生产上 `model_ctx`
    的调用点(T10 的 agent 节点)同样会拿到分了层的 `Layers`,所以那边成立。
    **不要把 4b 挪到这一行来** —— 那是改验收标准去迎合实现。
    """
    s = _settings(layer2_assistant_chars=10)
    b = budget.derive(settings=s, system_prompt="你是客服。")
    got = layers.split(
        [
            Message(id=1, role="user", content="你好"),
            Message(id=2, role="assistant", content="客" * 200),
            Message(id=3, role="tool", content="工" * 200, tool_call_id="c1"),
            Message(id=4, role="user", content="现在这句"),
        ],
        summary_upto_msg_id=0, layer1_from_msg_id=4, settings=s,
    )
    with caplog.at_level(logging.INFO):
        journal.model_ctx(
            conversation_id="c1", summary="", layers=got, evidence_tokens=0, budget=b,
        )

    sliding = _last_payload(caplog, "model_ctx")["sliding"]
    # 顺序 = 组装顺序:层 2(截短)在前、层 1(原文)在后。
    assert [m["role"] for m in sliding] == ["user", "assistant", "tool", "user"]
    assert sliding[-1]["content"] == "现在这句"          # 层 1 是原文,一个字不动
    assert sliding[0]["content"] == "你好"               # user 消息永不截短
    truncated = sliding[1]
    assert truncated["content"].endswith("…")            # 省略号来自 layers.ELLIPSIS
    assert len(truncated["content"]) < 200               # 真的是截短版,不是原文
    assert sliding[2]["content"].startswith("[工具结果] ")
    assert len(sliding[2]["content"]) < 200


def test_model_ctx_bounds_come_from_the_layers_it_was_handed(caplog):
    """`bounds` 取自 `Layers` 自己带的锚点 —— 调用方**传不进来**,也就伪造不了。

    锚点若做成 `model_ctx` 的参数、又带一个 `0` 的默认值,拿不到锚点的调用方
    (agent 节点里根本没有锚点来源)就会打出 `{0, 0}`,而 `layers.split` 用的是
    **真锚点** —— 日志与切分不一致,而两边都不报错。`bounds` 是这一行日志与
    某一段具体历史对上的唯一字段,恒为 `{0, 0}` 等于让它说不出「被截的是哪一段」;
    更糟的是 `0` 在本章**是个有含义的值**(尚无梗概 / 层 1 起于最早),
    读日志的人分不出它是真值还是占位。

    喂进去的 `Layers` 由 `split` 产出、锚点是**非默认**的 4 / 7 ——
    把 `Layers` 的这两个字段写死成 0 的实现会立刻变红。
    (简报那条 `bounds ... == 0` 的断言做不到这件事:它的期望值恰好等于那个默认值。)
    """
    s = _settings()
    b = budget.derive(settings=s, system_prompt="你是客服。")
    got = layers.split(
        [Message(id=7, role="user", content="现在这句")],
        summary_upto_msg_id=4, layer1_from_msg_id=7, settings=s,
    )
    # 替身自检:锚点真的**跟着 Layers 走到了这里**,而不是被 split 丢掉。
    assert (got.summary_upto_msg_id, got.layer1_from_msg_id) == (4, 7)

    with caplog.at_level(logging.INFO):
        journal.model_ctx(
            conversation_id="c1", summary="", layers=got, evidence_tokens=0, budget=b,
        )

    payload = _last_payload(caplog, "model_ctx")
    assert payload["bounds"] == {"summary_upto_msg_id": 4, "layer1_from_msg_id": 7}
    assert payload["conversation_id"] == "c1"
    assert payload["rounds"] == len(payload["sliding"])


def test_model_ctx_summary_is_the_text_not_a_count(caplog):
    """`summary` 是**全文**,不是条数、也不是长度。

    只给个数的话,「梗概真的注入了吗」只能靠推理;而梗概正是验收 2 的落点。
    """
    s = _settings()
    b = budget.derive(settings=s, system_prompt="你是客服。")
    got = layers.split([], summary_upto_msg_id=0, layer1_from_msg_id=0, settings=s)
    text = "用户问过订单 1002,要求退货。"
    with caplog.at_level(logging.INFO):
        journal.model_ctx(
            conversation_id="c1", summary=text, layers=got,
            evidence_tokens=0, budget=b,
        )

    assert _last_payload(caplog, "model_ctx")["summary"] == text


def test_model_ctx_counts_the_summary_with_the_shared_counter(caplog):
    """梗概那一段的 token 数走**同一个 counter**(`trim.count_tokens`)。

    另立口径(如 `len(text)`)就是两把尺子 —— 而它与真计数在中英混排上给出
    不同的数,却两边看起来都合理。这里用一个 `len` 与真计数**不相等**的文本
    把两者区分开。
    """
    text = "订单 1002 已取消,用户要求退款并补偿运费。"
    assert trim.count_tokens(text) != len(text)     # ← 否则这条断言区分不出 len 实现

    s = _settings()
    b = budget.derive(settings=s, system_prompt="你是客服。")
    got = layers.split([], summary_upto_msg_id=0, layer1_from_msg_id=0, settings=s)
    with caplog.at_level(logging.INFO):
        journal.model_ctx(
            conversation_id="c1", summary=text, layers=got,
            evidence_tokens=0, budget=b,
        )

    assert _last_payload(caplog, "model_ctx")["tokens"]["summary"] == trim.count_tokens(text)


# -------------------------------------------------------------- history_ctx


def test_history_ctx_logs_summary_lines_and_segmented_tokens(caplog):
    """`history_ctx` 的 `summary` 是**摘要行列表**,`tokens` 同样分段。

    **没有 `bounds`** —— 这份上下文不是用两个锚点切出来的,给它补一对锚点只能
    靠调用方另行传入(spec §7.6 给它的字段表里也只有 summary/sliding/tokens)。
    """
    s = _settings()
    b = budget.derive(settings=s, system_prompt="你是客服。")
    history = [
        Message(id=5, role="user", content="现在这句"),
    ]
    with caplog.at_level(logging.INFO):
        journal.history_ctx(
            conversation_id="c9",
            summaries=[(1, "第一段:用户问过订单 1002"), (2, "第二段:要求退款")],
            history=history, budget=b,
        )

    payload = _last_payload(caplog, "history_ctx")
    assert payload["conversation_id"] == "c9"
    assert "bounds" not in payload
    assert [row["seq"] for row in payload["summary"]] == [1, 2]
    assert [row["content"] for row in payload["summary"]] == [
        "第一段:用户问过订单 1002", "第二段:要求退款",
    ]
    assert {"layer1", "layer2", "summary", "evidence", "total"} <= set(payload["tokens"])
    assert payload["tokens"]["layer1"] == trim.count_tokens("现在这句")
    assert payload["tokens"]["summary"] == trim.count_tokens(
        "第一段:用户问过订单 1002\n\n第二段:要求退款"
    )
    assert payload["tokens"]["total"] == sum(
        payload["tokens"][k] for k in ("layer1", "layer2", "summary", "evidence")
    )
    assert [m["content"] for m in payload["sliding"]] == ["现在这句"]


def test_history_ctx_accepts_summary_rows_carrying_their_upto_id(caplog):
    """三元的摘要行(带 `upto_msg_id`)也要收 —— 那正是 §7.6 的 payload 形状。

    `load_summaries` 今天只给 `(seq, content)`,所以两元形态是端点实际会传的;
    但「这一段压到哪条 id」是日志里唯一能把梗概与锚点对上的字段,能拿到时
    必须能记下来。两种形态都收,而不是让调用方各自拼一遍。
    """
    s = _settings()
    b = budget.derive(settings=s, system_prompt="你是客服。")
    with caplog.at_level(logging.INFO):
        journal.history_ctx(
            conversation_id="c9",
            summaries=[(1, 12, "压到第 12 条")],
            history=[], budget=b,
        )

    rows = _last_payload(caplog, "history_ctx")["summary"]
    assert rows == [{"seq": 1, "upto_msg_id": 12, "content": "压到第 12 条"}]


def test_history_ctx_is_logged_even_for_the_fallback_branch(client_factory, caplog):
    """每轮必打,**包括不进 Agent 的那几轮**(闲聊/投诉/兜底/退款子流程)。

    那几轮恰恰最容易「看起来正常、其实上下文是错的」—— ch06 的 T4 就是这么丢的
    (节点写了 `confidence`、通道根本不存在、**单测全绿而生产恒为 `None`**),
    而它当时**没有任何观测面**。

    这条走**端点**、用 `intent="闲聊"`(路由不进 Agent),断言 `history_ctx`
    照样出现。**不走端点的话**,「每轮必打」这件事其实没被验到 ——
    直接调 `journal.history_ctx` 只能证明函数本身能打。
    """
    client, _ = client_factory(batches=[[FakeChunk("你好呀")]], intent="闲聊")
    with caplog.at_level(logging.INFO):
        with client as c:
            c.post("/api/chat/stream", json={"message": "你好"})
    _last_payload(caplog, "history_ctx")      # 取不到就抛 ⇒ 红


def test_history_ctx_from_the_endpoint_carries_the_real_session_and_window(
    client_factory, caplog
):
    """端点上那一行带的是**这个会话**与**它当时真的读进上下文的那段历史**。

    上一条只验「有没有那一行」。conversation_id 写错、`sliding` 装成空列表
    都照样绿 —— 而这一行存在的全部意义就是让人能把它与某个会话对上。

    ⚠️ 两个锚点取 **(1, 3)**,而且必须是**轮的起点**(3 是那条 user)。
    T10 之前这里写的是 (2, 4):`layer1_from=4` 落在一条 **assistant** 上,而
    端点的层 1 选择走 `trim_messages(start_on="human")` —— **一段没有 user 的
    层 1 会被整轮丢掉**(实测 `sliding` 里只剩层 2)。锚点在**生产**上恒是轮的
    起点(`0` 或 `degrade` 挪出来的那个,它只落在 user 上;摘要任务推进
    `summary_upto` 时用的也是 `layer1_from`),所以 (2, 4) 那种取值不是本章的
    语义 —— 拿它当输入,验的就成了另一件事。
    """
    db = FakeSession()
    db.conversations["c-anchored"] = Conversation(
        id="c-anchored", user="demo-user", status="active",
        summary_upto_msg_id=1, layer1_from_msg_id=3,
    )
    for role, content in [
        ("user", "你好"), ("assistant", "你好呀"),
        ("user", "订单 1002"), ("assistant", "已登记"),
    ]:
        db.add(MessageRecord(conversation_id="c-anchored", role=role, content=content))
    client, _ = client_factory(batches=[[FakeChunk("好的")]], session=db, intent="闲聊")

    with caplog.at_level(logging.INFO):
        with client as c:
            c.post(
                "/api/chat/stream",
                json={"session_id": "c-anchored", "message": "你好"},
            )

    payload = _last_payload(caplog, "history_ctx")
    assert payload["conversation_id"] == "c-anchored"
    assert [m["content"] for m in payload["sliding"]][-1] == "已登记"


def test_history_ctx_reads_summary_rows_from_the_database(client_factory, caplog):
    """摘要行来自**库**,不是硬编码的空列表。

    这条同时是替身自检:`load_summaries` 走的是 `select(ConversationSummary)`,
    替身不认识这个实体 —— 不补那一支的话,端点里所有用例都会红在一个
    「替身不支持的实体」上,而那指向脚手架、不指向实现。
    """
    db = FakeSession()
    db.conversations["c-sum"] = Conversation(
        id="c-sum", user="demo-user", status="active",
        summary_upto_msg_id=2, layer1_from_msg_id=4,
    )
    db.summaries.append(
        ConversationSummary(
            conversation_id="c-sum", seq=1, upto_msg_id=2, content="压到第 2 条",
        )
    )
    client, _ = client_factory(batches=[[FakeChunk("好的")]], session=db, intent="闲聊")

    with caplog.at_level(logging.INFO):
        with client as c:
            c.post("/api/chat/stream", json={"session_id": "c-sum", "message": "你好"})

    rows = _last_payload(caplog, "history_ctx")["summary"]
    assert [row["content"] for row in rows] == ["压到第 2 条"]


# ------------------------------------------------------------ 日志落盘与自检


def test_setup_logging_pins_utf8_and_does_not_stack_handlers(tmp_path):
    """`encoding="utf-8"` 是硬要求;重复调用**不得**叠加 handler。

    本机 locale 是 cp936。不给 encoding 时 Python 用
    `locale.getpreferredencoding()`,中文日志行会在**写日志的时候**抛
    `UnicodeEncodeError` —— 那个异常的栈指向与业务毫无关系的地方。
    断言 encoding 属性而不是「在 cp936 机器上真的炸没炸」,是因为后者随运行
    环境变化(在 UTF-8 机器上恒绿),而前者是这条要求本身。

    幂等那一半同样承重:lifespan 每次 `TestClient(app)` 都会跑一遍,
    不挡的话每开一次客户端就多一个文件句柄 + 多一行重复日志。
    """
    root = logging.getLogger()
    # **先把 root 清空**:同一进程里别的用例(任何一次 `TestClient(app)`)已经
    # 跑过 lifespan、装过一个指向仓库 `log/` 的 handler;不清掉的话
    # `setup_logging` 会按幂等守卫直接返回,而这条用例就变成**在别人的
    # handler 上**做断言 —— 红法指向测试自己。
    saved = list(root.handlers)
    level = root.level
    for handler in saved:
        root.removeHandler(handler)
    try:
        setup_logging(log_dir=str(tmp_path))
        setup_logging(log_dir=str(tmp_path))

        added = [h for h in root.handlers if isinstance(h, RotatingFileHandler)]
        assert len(added) == 1
        assert added[0].encoding == "utf-8"

        logging.getLogger("app.probe").info("中文日志:订单 1002 已取消")
        for handler in added:
            handler.flush()
        assert "中文日志:订单 1002 已取消" in (tmp_path / "app.log").read_text(
            encoding="utf-8"
        )
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()
        for handler in saved:
            root.addHandler(handler)
        root.setLevel(level)


def test_startup_budget_self_check_is_skipped_under_pytest(monkeypatch, caplog):
    """pytest 下**连配置都不读** —— 单测不该因为一个算术配置去打 error 日志。

    `TestClient` 会跑 lifespan,而 `TestClient` 在本项目里到处都是。断言
    「`get_settings` 一次都没被调用」而不是「没有 error 记录」:后者的红法
    取决于当前配置恰好装不装得下一轮(一个**随环境变化**的期望值),
    而前者直接钉住那条守卫本身。
    """
    called: list[int] = []
    monkeypatch.setattr(
        main_module, "get_settings", lambda: called.append(1) or _settings()
    )
    with caplog.at_level(logging.INFO):
        main_module._startup_budget_self_check()

    assert called == []
    assert [r for r in caplog.records if r.name.startswith("app.")] == []


def test_lifespan_does_not_configure_file_logging_under_pytest(
    client_factory, monkeypatch
):
    """pytest 下 lifespan **不落盘** —— 一次全量套件不该往仓库根写一个真实文件。

    这条同时护住一个更隐蔽的东西:验收 4b 判的是 `grep log/app.log`,而单测里
    的端点用例**同样会打 `history_ctx`**。不挡住的话,那个文件里躺着上一次单测
    的产物,验收的 grep 会因为**旧行**而恒真 —— 假绿里的假绿。

    断言的是「`setup_logging` 一次都没被调用」(直接钉住那条守卫),
    而不是「文件没出现」:后者在别的开发者机器上可能因为残留文件而恒假,
    与运行环境有关,而前者只取决于代码。
    """
    called: list[int] = []
    monkeypatch.setattr(main_module, "setup_logging", lambda *a, **kw: called.append(1))

    client, _ = client_factory(batches=[[FakeChunk("你好呀")]], intent="闲聊")
    with client as c:
        c.post("/api/chat/stream", json={"message": "你好"})

    assert called == []


def test_history_ctx_counts_the_summary_with_the_shared_joiner(caplog):
    """`tokens.summary` 必须与**真正注入时**的拼法同源。

    `journal` 原先自己复制了一份 `"\n\n"`,而权威是
    `summarize.join_summaries`(T8)。两处不一致时,这一行报的 token 数
    就不是模型实际收到的那段文本的数 —— 一个「看起来正常、其实什么也没说」的
    观测面。

    ⚠️ 这条断言的**强度是有限的**:分隔符从 `"\n\n"` 改成 `"\n"` 时
    tiktoken 给出的数**恰好相等**,所以它抓不到那一种漂移。真正堵住它的是
    `journal` 现在**直接调用** `join_summaries`(不再留第二份实现)——
    这条用例是跨模块的交叉核对,不是唯一防线。
    """
    rows = [(1, "第一段:用户问过订单 1002"), (2, "第二段:要求退款并补偿运费")]
    s = _settings()
    b = budget.derive(settings=s, system_prompt="你是客服。")
    with caplog.at_level(logging.INFO):
        journal.history_ctx(
            conversation_id="c9", summaries=rows,
            history=[Message(id=5, role="user", content="现在这句")], budget=b,
        )

    payload = _last_payload(caplog, "history_ctx")
    assert payload["tokens"]["summary"] > 0
    assert payload["tokens"]["summary"] == trim.count_tokens(summarize.join_summaries(rows))
