"""转人工工具(ch10-A)—— **模拟**接人工,不接真人系统。

设计要点(都对应一次讨论,别静默改掉):

1. **它是只读的。** `app/tools/policy.py` 里显式声明成 `read`,`kind_of` 因此
   放行直调,**不过 ch08 的确认流**。用户 2026-09-25 拍板:它是模拟接口、
   无真实副作用,而要求用户对「转人工」再点一次确认是坏体验。
   注意「未声明 = 只读」意味着这行声明**在行为上是空操作** ——
   它的价值是可 grep、可 review;真正的守卫是 `tests/test_handoff_tool.py`
   里那条「将来有人改成 write 会被拦下」的断言。

2. **不加 `session` 依赖。** 它不落库,而且**转人工这一轮本来就没有业务行**:
   `conversations.status` 至今只有**一个**写方 —— `app/tools/builtin/tickets.py`
   的 `create_ticket`(置 `pending_human`),而它只在用户**另外**要建工单时才跑;
   用户只说「转人工」时那一列**一动不动**。这轮唯一的痕是**执行器为这次调用落的
   `tool_audit_logs` 行**(`status=success`,见 `app/tools/executor.py`)——
   它是痕,但不是业务留痕。
   ⚠️ 这里原先写的是「转人工的留痕由 `conversations.status` 与既有的建单路径负责」
   —— **是错的**(审查 M3):那是**建工单**的路径,不是转人工的;
   验收脚本自己的局限行当时也写着「它今天确实不留痕」,两处互相矛盾。
   本工具刻意**不做**那件事:用户要的是转人工,不是建工单。

3. **`build()` 的签名与其他 builtin 模块一致**,包内自动发现按这个签名调用;
   不用的参数照样要收,否则发现会失败。
"""

import json

from langchain.tools import tool

from app.tools.errors import ToolNotFound
from app.tools.mock_data import ECHO_LIMIT, handoff_record


def build(*, session, conversation_id, retriever):
    @tool
    async def transfer_to_human(reason: str) -> str:
        """把用户转接给人工客服。用户明确要求转人工、找真人、要人工处理时使用。"""
        cleaned = reason.strip()
        if not cleaned:
            raise ToolNotFound("请说明需要人工处理的什么问题")
        return json.dumps(handoff_record(cleaned[:ECHO_LIMIT]), ensure_ascii=False)

    return [transfer_to_human]
