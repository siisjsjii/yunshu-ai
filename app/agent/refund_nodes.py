"""ch06 退款子流程的六个节点(退款退货 / 售后两条意图的出口)。

```
refund_pick_order ──缺订单号── interrupt({"frame": "order_choice", ...})
      │                              ↑ 前端渲染订单卡片
      │                              └── resume(选中的订单号)──┐
      │ 已有订单号                                            │
      ↓                                                      │
refund_fetch_order ←──────────────────────────────────────────┘
      ↓  调 query_order 取这一单(工具,不新写业务能力)
refund_expand_retrieve    Query 扩写 → 多路检索 → 按 chunk_id 去重合并
      ↓
refund_judge       同一个主力模型,**一次** ainvoke,不绑工具、不进 ReAct
      ├ 能退   → refund_offer      token 话术 + {frame:"refund_offer"} → 前端表单
      └ 不能退 → refund_explain    token 话术(说明原因 / 如实说判不了)
```

三件事决定了本文件的形状,每一件都是**本机实测**得到的,不是推理:

**① `interrupt()` 会把节点从头重跑。** 实测(langgraph 1.2.11):resume 时
`interrupt()` **之前**的代码会再执行一遍,`interrupt()` 之后的部分只跑一次。
所以 `refund_pick_order` 里**不许有任何有副作用的动作** —— 取订单数据放在它
**后面**的节点。同理,弹卡片所需的候选订单只能由**纯函数**算出
(`candidate_orders` + `_order_record`,两者都是确定性、无 IO),重跑一遍
不会产生第二次查询、也不会多扣一次钱。

**② 节点返回的每个 key 都必须在 `ChatState` 里声明过。** LangGraph 对未声明
通道的写入是**静默丢弃**的(`wrote to unknown channel ..., ignoring it`,
只 warning 不抛)。T4 的 `confidence` 就是这样在生产里恒为 null 而单测全绿的。

**③ 发帧的只有 `refund_offer` / `refund_explain` 两处。** `refund_judge` 只写
state —— 它若顺手把话术发出去,offer/explain 会**再发一遍**,用户看到两段
同样的话。卡片帧同理:它由**端点**从 `__interrupt__` 载荷转出(spec §5.2),
节点自己发就会渲染出两组卡片。

**本文件不 import `app/agent/nodes.py` 的任何东西**:那是 ch05 的节点工厂,
这里的六个节点各自独立(唯一的共用件是 `app/prompts.py` 的消息装配与
`app/tools/executor.py` 的工具语义)。
"""

import json
import logging

from langchain_core.exceptions import OutputParserException
from langgraph.types import interrupt
from pydantic import ValidationError

from app.agent.state import RefundJudgement
from app.prompts import build_refund_judge_messages
from app.refund.categories import REFUND_REASON_CATEGORIES
from app.refund.orders import candidate_orders
from app.retrieval.expand import expand_queries, multi_search
from app.schemas import Message
from app.tools.business import _order_record
from app.tools.errors import ToolInfrastructureError
from app.tools.executor import (
    ERROR_NOT_FOUND,
    ERROR_TIMEOUT,
    execute_tool,
)

logger = logging.getLogger(__name__)

# Query 扩写的条数上限现在是 `Settings.query_expansion_max_queries`(spec §9,
# 默认 3,`Field(ge=1)`)—— T8 接线落地。**这里不再留模块常量**:留一份就是
# 「改了 settings 没反应」的那种配置项(它看起来在、其实没人读)。
# 取值边界与理由见 `app/config.py` 该字段旁边的实测记录。

#: 取数的 tool_call id。固定值即可 —— 这一轮里每次调用只有一次,且它**不出站**
#: (不像 Agent 那轮会把 tool_call 帧发给前端)。
FETCH_TOOL_CALL_ID = "refund-fetch-order"

#: 查不到单时给用户的话术。**刻意不回显工具的原文**:`query_order` 的
#: `ToolNotFound` 文案是写给**模型**的(`app/tools/business.py` 里同族的
#: `query_logistics` 那条甚至写着「请如实告知用户,不要自行编造物流信息」),
#: 原样吐给用户就是把提示词漏出去。
_NOT_FOUND_TEMPLATE = (
    "没能查到订单 {no},请核对订单号后再试一次。"
    "如果号码没错,我可以帮您转人工客服核实。"
)

#: 取数**超时**时的话术。**一个字都不许提订单号对不对**:超时是服务端没在
#: 时限内返回,与用户报的号码无关 —— 说成「请核对订单号」就是拿我们自己的
#: 故障指责用户输入(审查实测指出过这条)。
_TIMEOUT_TEMPLATE = "订单查询暂时没有返回,请稍后再试一次;也可以让我帮您转人工客服。"

#: 取数失败、但**不是**业务性未找到(工具名不在注册表 / 参数不合 schema)时
#: 上抛的固定文案。这是接线 bug,对用户只能交代「服务端出问题」,不能交代成
#: 「你的订单号查不到」。文案与 `app/tools/executor.py` 的服务端故障同一族。
_FETCH_WIRING_FAILURE = "订单查询服务暂时不可用"

#: 判定出参解析不出来时的话术。**如实说判不了** —— 不许猜。
#: (`refund_decision` 同时留 `None`:它既不是「能退」也不是「不能退」。)
JUDGE_FAILED_REPLY = "抱歉,我暂时判断不了这一单能不能退,建议联系人工客服核实。"

#: 判定给的话术是空串时的兜底(真机可达:模型可能只填 can_refund)。
OFFER_FALLBACK = "这一单可以申请退款,请选择退款原因后提交。"
EXPLAIN_FALLBACK = "这一单暂时不能直接退款,建议联系人工客服进一步核实。"


# ---- 纯函数:槽位与卡片 ------------------------------------------------


def _corpus(state) -> list[Message]:
    """订单号扫描用的语料:**历史 + 本轮两段文本**。

    本轮**原话排在最后** = 被 `_scan` 先扫到(`candidate_orders` 的
    `reversed(history)` 语义)。它比 `resolved_input` 权威:改写理论上可能把
    订单号改掉或整段吃掉,而用户的原话不会。
    """
    return [
        *(state.get("history") or []),
        Message(role="user", content=state.get("resolved_input") or ""),
        Message(role="user", content=state.get("user_input") or ""),
    ]


def _candidates(state) -> list[str]:
    """候选订单号 —— **槽位与卡片共用这一个入口**。

    正则不在本文件重写(T1 的 `candidate_orders` 是那份规则的唯一实现:
    两份正则迟早漂移,而漂移的表现是「卡片上出现一个查无此单的号」)。
    槽位与卡片**共用**同一份候选,是为了不让「候选从哪来」这件事出现第二种说法。
    """
    return candidate_orders(_corpus(state), state.get("conversation_id") or "")


def _slot_from_context(state) -> str:
    """订单号槽位来源①②(spec §3.3):**本轮问题里**含合法订单号 → 直接用;
    历史里出现过 → 用最近一个(`candidate_orders` 已按由近及远取)。

    **`candidate_orders` 在什么都没找到时会回落到演示订单池**,所以判据不能是
    「返回非空」—— 那会让每一个缺号的退款请求都跳过卡片。这里逐条回语料里
    **对一遍**:演示池里的号码不在语料里,自然被排除。
    """
    messages = _corpus(state)
    for no in _candidates(state):
        if any(no in (m.content or "") for m in messages):
            return no
    return ""


def _card_options(state) -> list[dict]:
    """卡片候选:订单号 + 状态/商品/金额(spec §5.2 的形状)。

    走到这里时**候选必然来自演示池**:只要语料里给出过任何号码,`_slot_from_context`
    已经填了槽位、根本不会弹卡片。所以「历史那一支」在**今天**不可达 ——
    但仍然照常传完整语料(见 `_candidates`):把它改成只传空历史,就等于让
    「候选从哪来」在槽位与卡片两处各有一份说法,而那份说法只在**将来**
    槽位的判据变动时才分叉(那时卡片会悄悄变成另一批号码)。

    详情借 `app/tools/business.py` 的 `_order_record`(**私有**,同模块的
    `query_order` 也用它):它是**纯函数、无 IO**,所以放在 `interrupt()` 之前
    也安全 —— resume 重跑一遍只是把同样的数算第二遍。

    为什么不调 `query_order` 工具:那会在**每一次 resume 之前**先查一遍
    (节点从头重跑),四条候选就是八次查询;而且返回的是 JSON 文本,
    这里要的是卡片字段。
    """
    return [
        {"order_no": no, **_pick_record(_order_record(no))}
        for no in _candidates(state)
    ]


def _pick_record(record: dict) -> dict:
    """从订单记录里挑卡片要展示的三项(不把整条记录塞进帧)。"""
    return {k: record[k] for k in ("status", "product", "amount")}


def _evidence_of(chunks) -> list[dict]:
    """检索块 → state 里的证据(与 `retrieve_knowledge` 同一形状)。

    `render_evidence`(判定 prompt 用的那个)按这几个键渲染,少一个键就是
    KeyError —— 而"少一个键"这种错**只在真机上炸**,所以形状照抄不省。
    """
    return [
        {
            "chunk_id": c.chunk_id,
            "section_path": c.section_path,
            "question": c.question,
            "answer": c.answer,
            "category": c.category,
            "score": c.score,
        }
        for c in chunks
    ]


def _citations_of(evidence: list[dict]) -> list[dict]:
    """证据 → `citations` 帧的条目(编号 `n` 即 prompt 里的 `[n]`)。"""
    return [
        {"n": i + 1, **{k: e[k] for k in
                        ("chunk_id", "section_path", "question", "answer", "category")}}
        for i, e in enumerate(evidence)
    ]


# ---- ① 槽位闸:除了 interrupt 什么都不干(F3)------------------------------


def _picked_order_no(picked) -> str:
    """resume 载荷 → 订单号字符串。

    **三种形状都收**,因为它们在文档里各出现过一次、而端点(T8)只会挑一种:
    裸串(`Command(resume="20240915")`,本任务的 brief)、
    `{"order_no": …}`(spec §5.1 的请求体 `{"resume": {"order_no": "1002"}}`)、
    以及那个请求体的 pydantic 实例(端点直接 `Command(resume=body.resume)` 时)。
    端点递错形状的表现是**用户点完卡片后收到「没能查到订单 {'order_no': …}」**
    —— 这句话会把内部的载荷形状漏给用户,而且它出现在**跨任务**的接缝上
    (本文件的单测传的是裸串,端点测试若没走真实 resume 路径就看不见)。

    **只取值,不清洗**:号码合不合法由 `query_order` 的 `_require_order_no` 判
    (它才是有权说「查无此单」的地方)。这里截断/纠正格式,等于把
    「这个号我处理不了」伪装成「用户没选号」。
    """
    if picked is None:
        return ""
    if isinstance(picked, str):
        return picked.strip()
    if isinstance(picked, dict):
        value = picked.get("order_no")
    else:
        value = getattr(picked, "order_no", None)
    return str(value or "").strip()


def make_refund_pick_order_node():
    """订单号槽位闸。**`interrupt()` 之外不干任何事。**

    实测(F3):resume 时节点**从头重跑**。把「取订单数据」放进本节点、且摆在
    `interrupt()` **之前**,那笔查询就会执行两遍
    (`tests/test_agent_refund.py::test_fetch_runs_exactly_once_across_resume`
    钉着这条:挂起后 `calls == []`、续跑后 `calls == [...]`)。
    (摆在 `interrupt()` **之后**的取数**不会**执行两遍 —— 那种摆法不是 F3
    那个故障;但本条的要求是「这个节点里除了 interrupt 什么都不干」,
    它比「别执行两遍」更严:一个节点只该有一件事,别让人去分辨自己这句代码
    站在 interrupt 的哪一侧。)

    resume 的返回值就是选中的订单号:本节点在挂起时**还没**写过 state,
    所以重跑时 `order_no` 仍是空 → 再次走到 `interrupt()`,而这一次它
    **返回** resume 值而不是抛(`Command(resume=...)`,见 langgraph 文档)。

    **没有 `emit` 参数**:本节点结构上不可能发帧。用户看到的卡片来自端点对
    `__interrupt__` 载荷的转帧(spec §5.2),两条路都发就是两组卡片 ——
    签名里不给 emit,比"记得别发"更硬。
    """

    async def refund_pick_order(state) -> dict:
        order_no = (state.get("order_no") or "").strip()
        if order_no:
            return {"trace": [f"refund:已有订单号 {order_no}"]}

        slot = _slot_from_context(state)
        if slot:
            return {"order_no": slot, "trace": [f"refund:问题/历史里取到订单号 {slot}"]}

        picked = interrupt({"frame": "order_choice", "options": _card_options(state)})
        return {
            "order_no": _picked_order_no(picked),
            "trace": ["refund:resume 回填订单号"],
        }

    return refund_pick_order


# ---- ② 取这一单 -----------------------------------------------------------


def make_refund_fetch_order_node(*, registry, settings):
    """用 `query_order` 取这一单。

    **走 `execute_tool` 而不是直接 `tool.ainvoke`**:超时、重试白名单
    (`query_order` 幂等,可重试)、`ToolNotFound` 不重试、SQLAlchemy 故障翻成
    `ToolInfrastructureError` —— 这些语义全都只在执行器里。直接 `ainvoke`
    等于把「数据库挂了」和「这个单号查不到」混成一件事。

    **失败要分三种说**(`outcome.error_kind`,执行器带出来的):

    | 种类 | 这是谁的问题 | 这里怎么办 |
    |---|---|---|
    | `ERROR_NOT_FOUND` | 用户给的号码(或那单真的不存在) | 如实告知 + 请核对(`refund_explain`) |
    | `ERROR_TIMEOUT` | **服务端**:查询没在时限内返回 | 说「稍后再试」,**不指责用户** |
    | 其余(工具名不在注册表 / 参数不合 schema / 种类未知) | **我们自己**(接线或代码 bug) | `ToolInfrastructureError` 上抛,绝不产出面向用户的「查无此单」 |

    最后一行是本节点最容易犯的错,也是它存在的理由:`ok=False` 只说明「没成功」,
    而**超时与接线 bug 都不等于「你要的东西不存在」**。把 MySQL 卡住说成
    「请核对订单号」,就是拿服务端的故障指责用户输入 —— 与
    `app/tools/errors.py` 里 `ToolInfrastructureError` 那条注释同一件事。

    真正的业务性未找到(以及超时)都**不是**异常:`order_data` 留空 dict +
    `refund_decision=False`,路由据此走 `refund_explain`,**不进判定**
    (「查不到这一单」不是「这一单不能退」)。而基础设施异常
    (`SQLAlchemyError`/意外异常)在 `execute_tool` 里就已经上抛了。
    """

    async def refund_fetch_order(state) -> dict:
        order_no = (state.get("order_no") or "").strip()
        outcome = await execute_tool(
            tool_call={
                "type": "tool_call",  # 缺这个键 `BaseTool.ainvoke` 会把整个 dict
                                      # 当**参数**去校验 schema(CLAUDE.md 硬约束)
                "name": "query_order",
                "args": {"order_id": order_no},
                "id": FETCH_TOOL_CALL_ID,
            },
            registry=registry,
            settings=settings,
        )
        if not outcome.ok:
            # 回显截断:`order_no` 来自客户端(resume 载荷),长度不受我们控制。
            no = order_no[:32]
            if outcome.error_kind == ERROR_TIMEOUT:
                logger.warning("退款子流程:取订单 %s 超时", no)
                return {
                    "order_data": {},
                    "refund_decision": False,
                    "reply": _TIMEOUT_TEMPLATE,
                    "trace": ["refund:fetch timeout"],
                }
            if outcome.error_kind != ERROR_NOT_FOUND:
                # 注册表里没有 `query_order`、或参数不合 schema、或执行器将来
                # 加了新的失败种类:都是**我们的** bug,不是用户输入的问题。
                # 上抛(端点变 502 + 固定文案),而不是编一句「查无此单」——
                # 后者会让一次接线错误伪装成用户的订单号打错了,而**谁都不会
                # 去查**。日志里带上注册表,省得下次还要猜。
                logger.error(
                    "退款子流程:取数失败且非业务性未找到(kind=%s):%s;注册表=%s",
                    outcome.error_kind, outcome.summary, sorted(registry),
                )
                raise ToolInfrastructureError(_FETCH_WIRING_FAILURE)
            logger.info("退款子流程:订单 %s 查不到(%s)", no, outcome.summary)
            return {
                "order_data": {},
                "refund_decision": False,
                "reply": _NOT_FOUND_TEMPLATE.format(no=no),
                "trace": ["refund:fetch not_found"],
            }
        # 工具契约是 JSON(`query_order` 的既定出参)。**不兜底**:解析不出来
        # 是我自己或工具的缺陷,必须响 —— 塞进 except 就变成「查不到单」,
        # 而那是一条会安静地骗用户的降级路。
        return {
            "order_data": json.loads(outcome.content),
            "trace": ["refund:fetch ok"],
        }

    return refund_fetch_order


def route_after_fetch(state) -> str:
    """取到单才去查条款;没取到直接去解释。

    判据是 `order_data` 的**空 dict**(取数失败时唯一的写法),不是
    `refund_decision` —— 后者在判定之后才会被写,拿它当判据会把「还没判」
    误读成「判完了且不能退」。
    """
    return "refund_expand_retrieve" if state.get("order_data") else "refund_explain"


# ---- ③ 扩写 → 多路检索 ----------------------------------------------------


def make_refund_expand_retrieve_node(*, model, retriever, emit, settings):
    """Query 扩写 → 多路检索 → 去重合并。

    扩写输入按 spec §4.3 是「已消解的问题 + 这一单的上下文」:带上商品与状态,
    模型才可能扩出「定制商品能不能退」这种贴着这一单的角度。

    `multi_search` 的故障语义原样透传:`ToolInfrastructureError` 绝不能降级成
    「没搜到」(`app/retrieval/expand.py` 里两个 except 的顺序就是全部要害)。

    发 `citations` 帧与 `retrieve_knowledge` 同形(键名必须是 `items`,ch04 的
    前端读的是 `payload.items || []`)。命中的编号必须与判定 prompt 里的 [n]
    同源,所以这里只是把同一份 evidence 换个形状。
    """

    async def refund_expand_retrieve(state) -> dict:
        order = state.get("order_data") or {}
        text = state["resolved_input"]
        context = "、".join(
            str(order[k]) for k in ("product", "status") if order.get(k)
        )
        queries = await expand_queries(
            model,
            text=f"{text}(这一单:{context})" if context else text,
            max_queries=settings.query_expansion_max_queries,
        )
        chunks = await multi_search(retriever, queries)

        evidence = _evidence_of(chunks)
        citations = _citations_of(evidence)
        if citations:
            emit({"frame": "citations", "items": citations})
        top = f" top={evidence[0]['score']:.2f}" if evidence else ""
        return {
            "evidence": evidence,
            "citations": citations,
            "trace": [f"refund_expand_retrieve:{len(queries)} 路 {len(evidence)} 命中{top}"],
        }

    return refund_expand_retrieve


# ---- ④ 判一次 -------------------------------------------------------------


def make_refund_judge_node(*, model):
    """「这一单能不能退」—— 同一个主力模型,**一次**结构化调用,不绑工具、不进 ReAct。

    **判据的形态(实现者裁定)**:走 `with_structured_output(RefundJudgement,
    method="json_mode")`,而不是「在自由文本里找『不能退』三个字」。理由:

    - 文本启发式把判定权交给措辞。模型写「这一单**不**符合退款条件」时不含
      「不能退」,于是被判成「能退」并弹出退款入口 —— 而这类误判**不出现在
      任何日志里**;
    - `json_mode` 是本项目端点上唯一可用且已在用的结构化出参方式(抽取 / 意图 /
      扩写三处同源),校验由 pydantic 兜底,不需要我写关键词表。

    **判不出来时如实说判不出来**:`(OutputParserException, ValidationError)`
    → `refund_decision=None` + `JUDGE_FAILED_REPLY`,路由去 `refund_explain`。
    接的是「模型出参不可用」这一族(同 `classify_intent` / `expand_queries`);
    **不接**上游故障(超时/401/限流)—— 那些该一路抛到端点的 error 帧,
    宽 `except Exception` 会把它们变成静默降级。

    **本节点不发帧**:话术写在 state 里,由 offer / explain 出口发一次。
    这里顺手 `emit` 的话,用户会看到两段一模一样的话。

    结构化出参这条链**在节点里建**,不在工厂里:工厂是 `build_graph` 每请求都
    要跑一遍的,而 `with_structured_output` 一旦放在那儿,就要求**每一个**模型
    替身(包括整章里跟退款毫无关系的那些)**都**实现它 —— 否则图根本编不出来。
    只有真的走到判定这条路才会碰它,和「不在 import 时做 IO」同一条理由。
    """
    async def refund_judge(state) -> dict:
        messages = build_refund_judge_messages(
            order=state.get("order_data") or {},
            evidence=state.get("evidence") or [],
            user_input=state["resolved_input"],
        )
        chain = model.with_structured_output(RefundJudgement, method="json_mode")
        try:
            result = await chain.ainvoke(messages)
        except (OutputParserException, ValidationError) as exc:
            logger.warning("退款判定出参不可用,如实说明判不了:%s", exc)
            return {
                "refund_decision": None,
                "reply": JUDGE_FAILED_REPLY,
                "trace": ["refund:judge 判不了(出参不可用)"],
            }

        can_refund = bool(result.can_refund)     # 字段名由 pydantic 保证,不 getattr 兜底
        # 空话术**不在这里兜底**:两个出口节点各自知道自己该说什么(能退 / 不能退),
        # 兜底文案归它们。这里再兜一次 = 同一条规则写两遍,改一处就分叉 ——
        # 而变异实测显示这一份**根本不可达**(去掉它没有任何用例变红,因为出口
        # 那层原样接住了)。测试钉在出口那层:`test_empty_judge_reply_...` 两条。
        reply = (result.reply or "").strip()
        return {
            "refund_decision": can_refund,
            "reply": reply,
            "trace": [f"refund:judge {'能退' if can_refund else '不能退'}"],
        }

    return refund_judge


def route_after_judge(state) -> str:
    """**只有明确的 `True` 才给退款入口。**

    `None`(判不了)与 `False`(不能退)都去同一个出口:对用户来说两者都要
    「说明原因 + 给出去处」,区别只在话术里(判不了时话术自己会说出来),
    但两者都**绝不能**弹退款表单 —— 用真值判断(`if state.get(...)`)会把
    别的实现漏洞(比如写进来一个非空的字符串)也放行。
    """
    return "refund_offer" if state.get("refund_decision") is True else "refund_explain"


# ---- ⑤ 两个出口 -----------------------------------------------------------


def make_refund_offer_node(*, emit):
    """能退:话术(token 帧)+ 退款表单(refund_offer 帧)。**不写库** ——
    用户点了提交才走 `POST /api/refund`(spec §3.3)。

    顺序是先话术后表单:前端把两者都塞进**同一个气泡**,反过来会让表单出现在
    说明文字上面。

    类目从 `REFUND_REASON_CATEGORIES` 下发,**前端不硬编码**这份清单
    (两处各写一份的话,漂移的表现是「表单能选、提交回 422」)。
    """

    async def refund_offer(state) -> dict:
        order_no = state.get("order_no") or ""
        reply = (state.get("reply") or "").strip() or OFFER_FALLBACK
        emit({"frame": "token", "text": reply})
        emit({
            "frame": "refund_offer",
            "order_no": order_no,
            "categories": list(REFUND_REASON_CATEGORIES),
        })
        return {"reply": reply, "choices": [], "trace": [f"refund:offer {order_no}"]}

    return refund_offer


def make_refund_explain_node(*, emit):
    """不能退 / 查不到单 / 判不了:把 state 里那句话术如实说出去。

    三个来路共用这一个出口,话术由上游节点写进 `reply`(各写各的原因),
    本节点只负责把它变成 token 帧并落库 —— 空话术才用固定文案兜底。
    """

    async def refund_explain(state) -> dict:
        reply = (state.get("reply") or "").strip() or EXPLAIN_FALLBACK
        emit({"frame": "token", "text": reply})
        return {"reply": reply, "choices": [], "trace": ["refund:explain"]}

    return refund_explain
