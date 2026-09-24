"""T10 §15.11 的取证脚本:首键违规那一支**只跑一轮模型**,没有重试。

`_Model.calls` 数的是 `astream` 被调了几次 —— 重试若存在,它会是 2。
跑完即弃(不提交、不进测试套),只为给 spec §15.11 那两句话一个可复现的读数。
"""

import asyncio
import os
import sys

# `python .superpowers/x.py` 时 `sys.path[0]` 是 `.superpowers/`,不是仓库根
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agent.nodes import make_agent_node
from app.config import Settings
from app.memory import budget
from app.prompts import render_system_prompt

REQUIRED = {
    "openai_base_url": "https://example.invalid/v1", "openai_api_key": "sk-test",
    "openai_model": "test-model", "database_url": "mysql+asyncmy://u:p@h:3306/db",
}


class _Chunk:
    def __init__(self, text=""):
        self.text = text
        self.content = text
        self.tool_calls = []
        self.usage_metadata = None

    def __add__(self, other):
        return self


class _Model:
    def __init__(self, rounds):
        self._rounds = [list(r) for r in rounds]
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def astream(self, msgs, **kw):
        self.calls += 1
        for text in self._rounds.pop(0):
            yield _Chunk(text=text)


class _Session:
    def add(self, obj):
        pass

    async def commit(self):
        pass


async def main() -> None:
    for name, script in [
        ("首键违规(first_key=answer)", ['{"answer": ', '"您好", "useful": true}']),
        ("首字符不是 { (plain)", ["好的,我来查一下", "退货政策是 7 天"]),
        ("合规协议", ['{"useful": true, "confidence": 0.9, "answer": "七天。"}']),
    ]:
        settings = Settings(_env_file=None, **REQUIRED)
        model = _Model([script])
        frames = []
        node = make_agent_node(
            model=model, tools=[], registry={}, settings=settings,
            emit=frames.append, session=_Session(),
            context_budget=budget.derive(
                settings=settings,
                system_prompt=render_system_prompt(settings.brand_name)),
        )
        out = await node({"conversation_id": "c1", "user_input": "退货政策",
                          "resolved_input": "退货政策", "intent": "商品咨询",
                          "history": [], "evidence": []})
        print(f"{name}: astream 调用 {model.calls} 次;trace={out['trace']}")


asyncio.run(main())
