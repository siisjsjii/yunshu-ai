"""工具选择评估集。

用法:.venv/Scripts/python.exe evals/run_tool_selection_eval.py
(需真实 key + MySQL;要覆盖物流那 3 条还需起物流 MCP Server)

评分口径:闭式精确匹配 —— 工具名是枚举,不存在"措辞不同"的模糊地带。
与 ch01 的 expected_solution(自由文本、关键词口径最终被证明是样本拟合)
形成对比:本口径的分数可以直接引用。

只跑「工具选择」这一段,不执行工具 —— 故不产生任何 DB 写入。

⚠️ **测的必须是生产那套工具**(ch08 T7 改)。原先这里用 `build_tools`,那是
注册表的**内置那一半**的投影;而 T7 把 `query_logistics` 从内置下线、改由
**独立进程**的物流 MCP Server 提供 ⇒ **那 3 条物流用例在结构上不可能通过**
(模型看不到那个工具)。一句「口径变了」描述不了「有 3 条根本跑不了」。
现在改成与 `app/api/chat.py` **同款**:`await discover_mcp_specs` → `build_registry(extra=...)`
→ 从注册表投影出绑给模型的那批。

**降级行为与生产一致**:某一台 Server 没起 ⇒ 它的工具不出现,脚本照跑
(`discover_mcp_specs` 内部跳过 + 一条 warn)。所以两个 Server 都没起时
脚本**仍然能跑完**,只是物流/售后那几条必然 MISS —— 故开头会打印一份
**工具清单与来源**,让读结果的人一眼看出「这次少了什么」而不是把它
读成「模型退步了」。

输出编码:全部输出经 emit() 以 UTF-8 字节直接写 stdout,不经过控制台的
locale 编解码 —— 本机 locale 是 cp936,而结果行要打印 ✓/✗,直接 print
会抛 UnicodeEncodeError,裸跑即崩。这与 ch01 验收脚本是同一个坑(cp936
在本章属复发型陷阱),修法同源:**钉死输出边界的字节流,而不是指望
控制台用对编码**。因此裸跑 `.venv/Scripts/python.exe evals/run_tool_selection_eval.py`
即可,不需要 `-X utf8`,也不需要设 PYTHONIOENCODING。
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.db.base import get_sessionmaker
from app.llm import create_chat_model
from app.mcp.client import discover_mcp_specs
from app.tools.registry import build_registry

CASES = Path(__file__).with_name("tool_selection_cases.jsonl")


def emit(line: str = "") -> None:
    """按 UTF-8 字节写一行输出,绕过控制台编码。

    逐行 flush:本脚本是给人看的进度输出,不是吞吐路径,15 行不值得攒批。
    没有 .buffer 时(如被 StringIO 接管 stdout)退回 print —— 那种环境里
    文本层自己就是 UTF-8,不会重现本机的 cp936 问题。
    """
    stream = getattr(sys.stdout, "buffer", None)
    if stream is None:
        print(line)
        return
    stream.write(line.encode("utf-8") + b"\n")
    stream.flush()


def load_cases() -> list[dict]:
    lines = CASES.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


async def select_tool(model, tools, text: str) -> str | None:
    """跑一轮工具选择,返回模型选中的工具名;没调工具则返回 None。"""
    bound = model.bind_tools(tools)
    accumulated = None
    async for chunk in bound.astream([{"role": "user", "content": text}]):
        accumulated = chunk if accumulated is None else accumulated + chunk
    tool_calls = list(getattr(accumulated, "tool_calls", None) or [])
    return tool_calls[0]["name"] if tool_calls else None


async def main() -> int:
    settings = get_settings()
    model = create_chat_model(settings)
    cases = load_cases()

    async with get_sessionmaker()() as session:
        # 与 `app/api/chat.py` 同款的两步:先现问现拿 MCP 工具清单,再纯组装注册表。
        # **不在这里 try/except** —— 降级是 `discover_mcp_specs` 自己的职责
        # (单台跳过 + warn,全挂返回 []),这里再包一层就等于同一条规则两处实现。
        mcp_specs = await discover_mcp_specs(settings=settings)
        registry = build_registry(
            session=session,
            conversation_id="_eval",
            settings=settings,
            extra=mcp_specs,
        )
        tools = [spec.tool for spec in registry.values()]

        emit(f"模型:{settings.openai_model}  用例数:{len(cases)}")
        # **工具清单必须先打**:少了哪几个是「这次环境没起 Server」还是
        # 「模型退步了」,只有这张表分得开。按 source 分组,并点名 MCP 那几台。
        emit(f"注册表:{len(registry)} 个工具")
        for spec in registry.values():
            emit(f"    {spec.name:<24} source={spec.source}")
        mcp_sources = sorted({s.source for s in registry.values() if s.source != "builtin"})
        emit(
            "MCP 来源:" + ("、".join(mcp_sources) if mcp_sources else "(无 —— 两个 Server 都没起)")
        )
        emit()
        hits = 0
        misses: list[tuple[dict, str | None]] = []

        for case in cases:
            actual = await select_tool(model, tools, case["text"])
            ok = actual == case["expected"]
            hits += ok
            emit(
                f"  [{'✓' if ok else '✗'}] {case['text'][:30]:<32}"
                f" 期望={case['expected']}  实际={actual}"
            )
            if not ok:
                misses.append((case, actual))

    total = len(cases)
    emit()
    emit(f"工具选择准确率:{hits}/{total} = {hits / total:.1%}")

    if misses:
        emit()
        emit(f"未命中 {len(misses)} 条(附备注,便于判断是标注问题还是模型问题):")
        for case, actual in misses:
            emit(f"  - {case['text']}")
            emit(f"      期望={case['expected']}  实际={actual}")
            if case.get("note"):
                emit(f"      备注:{case['note']}")

    return 1 if misses else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
