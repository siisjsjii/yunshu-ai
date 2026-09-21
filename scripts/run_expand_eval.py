"""Query 扩写标注样例评估。需真实 key(打网络),不属于单测。

用法:
    .venv/Scripts/python.exe scripts/run_expand_eval.py

口径**闭式**,两条各自独立:
- `min_queries`:至少产出几条查询(少于它说明模型没真的「泛化」);
- `must_cover_any`:几条查询里**至少一条**含这些关键词中的**任意一个**
  (只看有没有覆盖该角度,不做字符串全等 —— deepseek 在 temperature=0 下依然
  非确定,本仓已记账)。

**JSON 可解析率必须在「被吞之前」量**。`expand_queries` 的契约就是把解析失败
退化成 `[原问题]`(见 `app/retrieval/expand.py`),所以在它外面看,一次失败的
扩写与「模型确实只返回了原问题」长得一模一样。`_ParseProbe` 因此包在**真模型**
外面:同一次调用里,解析成败被记下来,而 `expand_queries` 照常工作 —— 于是
「可解析率」与「扩写效果」来自**同一批样本**,不会因为非确定性而互相打架。

每条另打一个 `退回原问题` 标记(`queries == [原话]`):那是 `expand_queries`
的**退路出口**,不是模型输出。没有这个标记,「全部退回原问题 + 2 条只覆盖了
原话里本来就有的词」也能拿到好看的通过数。
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.llm import create_extract_model
from app.retrieval.expand import expand_queries

CASES = Path(__file__).resolve().parents[1] / "evals" / "expand_cases.jsonl"

#: spec §9 的 `query_expansion_max_queries` 默认值。**这里先写死**:那个配置项
#: 连同它的越界校验在接线任务里落地(`app/config.py` 不在本任务的文件清单内),
#: 而 `expand_queries` 本来就把它当参数收,不依赖配置项存在。
MAX_QUERIES = 3


def emit(line: str = "") -> None:
    # 控制台是 cp936,扩写结果是中文 —— 一律走字节出口(本仓平台陷阱)。
    stream = getattr(sys.stdout, "buffer", None)
    if stream is None:
        print(line)
        return
    stream.write(line.encode("utf-8") + b"\n")
    stream.flush()


class _ParseProbe:
    """包住真模型,记录每次**结构化调用有没有解析成功**。

    存在的唯一理由见模块 docstring:`expand_queries` 会把解析失败吞成 `[原话]`,
    可解析率在它外面量不出来。这里只做记录,不改变任何行为 —— `ainvoke` 照原样
    把结果或异常透给调用方。
    """

    def __init__(self, model):
        self._model = model
        self.ok = 0
        self.failed = 0
        self.last_error: str | None = None

    def with_structured_output(self, schema, method=None):
        inner = self._model.with_structured_output(schema, method=method)
        probe = self

        class _CountingChain:
            async def ainvoke(self, messages):
                try:
                    result = await inner.ainvoke(messages)
                except Exception as exc:          # 只计数,照原样再抛
                    probe.failed += 1
                    probe.last_error = f"{type(exc).__name__}: {exc}"
                    raise
                probe.ok += 1
                return result

        return _CountingChain()


async def main() -> int:
    settings = get_settings()
    probe = _ParseProbe(create_extract_model(settings))

    rows = [
        json.loads(line)
        for line in CASES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:                      # 空文件不许 ZeroDivisionError
        emit(f"用例文件是空的:{CASES}")
        return 1

    emit(f"模型 {settings.openai_model}(温度 0)/ 用例 {len(rows)} 条 / max_queries={MAX_QUERIES}")
    emit()

    passed = 0
    for i, row in enumerate(rows, 1):
        queries = await expand_queries(
            probe, text=row["text"], max_queries=MAX_QUERIES
        )
        enough = len(queries) >= row["min_queries"]
        covered = any(
            kw in q for q in queries for kw in row["must_cover_any"]
        )
        # `expand_queries` 的**退路出口**:原话原样返回。标出来,免得把退路
        # 当成模型输出读。
        fallback = queries == [row["text"]]
        ok = enough and covered
        passed += ok

        emit(
            f"{'OK ' if ok else 'MISS'} [{i}] {row['text']!r} → {len(queries)} 条"
            f"{'  ← 退回原问题(扩写没产出一条新查询)' if fallback else ''}"
        )
        for q in queries:
            emit(f"        - {q}")
        emit(
            f"        条数 ≥ {row['min_queries']}: {'✓' if enough else '✗'}"
            f"   覆盖任一 {'/'.join(row['must_cover_any'])}: {'✓' if covered else '✗'}"
        )

    calls = probe.ok + probe.failed
    emit()
    emit(f"JSON 可解析率 {probe.ok}/{calls} = {probe.ok / calls:.1%}(失败 {probe.failed})")
    if probe.last_error:
        emit(f"  最后一次解析失败:{probe.last_error[:200]}")
    emit(f"用例通过 {passed}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
