"""把 `get_stream_writer()` 包一层,让节点能脱离图被单测。

真机验证(langgraph 1.2.11):图运行之外调 `get_stream_writer()` 抛
`RuntimeError: Called get_config outside of a runnable context`。

**关键:必须在「每次发帧时」才去取 writer,不能在 `make_emitter()` 里取一次。**
`make_emitter()` 是在**端点里**调的 —— 那时图还没开始跑,上下文里没有 writer,
一次性的取法会永远拿到 no-op 分支:前端**一帧都收不到**,而所有单测仍然全绿
(单测把 collector 直接注入节点,根本不经过这里)。这正是本项目最怕的那类
「假绿 + 静默故障」。

约定:发出的 payload 形如 `{"frame": <名>, ...字段}`,由 `app/api/chat.py`
逐条翻成 SSE 帧。**节点的出站协议只有这一个形状。**
"""

from collections.abc import Callable

from langgraph.config import get_stream_writer


def make_emitter(collector: Callable[[dict], None] | None = None) -> Callable[[dict], None]:
    """返回一个 emit 函数:图运行中发真帧,图外退化成 collector(或静默丢弃)。

    每次调用都重新取一次 writer —— 这样同一个 emitter 既能被节点在图里用,
    又能在图外被单测直接调,不需要调用方关心自己在不在图里。
    """

    def emit(payload: dict) -> None:
        try:
            writer = get_stream_writer()
        except RuntimeError:
            # 图外:单测路径。没给 collector 就丢弃。
            if collector is not None:
                collector(payload)
            return
        writer(payload)

    return emit
