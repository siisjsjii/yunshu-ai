"""转人工工具(ch10-A):确定性 + 权限声明 + 入参回显截断。"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.tools.builtin.handoff import build
from app.tools.errors import ToolNotFound
from app.tools.mock_data import ECHO_LIMIT, handoff_record
from app.tools.policy import kind_of

REPO_ROOT = Path(__file__).resolve().parents[1]


def _tool():
    """`build()` 的签名与其他 builtin 模块一致;转人工不碰会话,故传 None。"""
    return build(session=None, conversation_id="conv-test", retriever=None)[0]


@pytest.mark.anyio
async def test_same_reason_always_gives_the_same_agent():
    """同一入参必须永远得到同样的工号 —— **跨进程**也要稳定。

    这条守的是「伪随机不能用内置 hash()」那条硬约束:`hash()` 对 str 每进程
    随机化(PYTHONHASHSEED),同进程内的测试**完全测不出来**,只有重启后
    用户会发现「同一个问题每次转给不同的人」。

    ⚠️ **所以这条测试本身抓不到那个陷阱** —— 内置 `hash()` 在同一个进程里
    也是确定的,两次调用照样相等(这正是本仓点名的「假绿」形态:实现改错
    它也不会红)。**跨进程那一半由本文件末尾的
    `test_handoff_record_is_stable_across_processes` 单独守**(写法照抄
    `tests/test_mock_data.py::test_order_record_is_stable_across_processes`,
    全仓只该有**一个**跨进程约定)。这条守的是本工具**每次都走同一条派生**,
    是那条跨进程断言的前提(它要是随机了,跨进程比对也无从谈起)。
    """
    tool = _tool()
    a = json.loads(await tool.ainvoke({"reason": "我要找真人"}))
    b = json.loads(await tool.ainvoke({"reason": "我要找真人"}))
    assert a == b


@pytest.mark.anyio
async def test_different_reasons_can_give_different_agents():
    """不是恒等函数 —— 否则上面那条测试对一个常量返回也成立。"""
    tool = _tool()
    reasons = ["我要找真人", "转人工", "帮我接客服", "人工客服在哪", "我要投诉"]
    seen = {json.loads(await tool.ainvoke({"reason": r}))["agent_no"] for r in reasons}
    assert len(seen) > 1


@pytest.mark.anyio
async def test_result_carries_what_the_user_needs():
    """回复里必须**真的有**工号与等待时长 —— 任务 3 的端到端断言就断这两个键。"""
    payload = json.loads(await _tool().ainvoke({"reason": "转人工"}))
    assert payload["agent_no"].startswith("A")
    assert isinstance(payload["queue_position"], int)
    assert payload["eta_minutes"] >= 0


@pytest.mark.anyio
async def test_long_reason_is_truncated_not_rejected():
    """超长入参夹到上限,而不是抛错。

    理由同 `create_ticket` 的 `ticket_type`:这是把用户的诉求交给人工,
    一个被模型撑爆的字段不该让整次转人工失败。`ECHO_LIMIT` 是
    `mock_data` 里已有的常量(32),复用而非新写一个。

    ⚠️ **`assert payload["agent_no"]` 一条是不够的(实测)**:把 `[:ECHO_LIMIT]`
    整个删掉,那句话照样绿 —— 工号还是工号,只是从另一个入参派生出来的。
    判别力在**逐值比对**:本工具的结果必须**就是**「截短后的入参」推导出来的
    那一条记录。下面第二条 `!=` 是**夹具自检**(防这个输入恰好让两个派生撞上
    —— 撞上时第一条也会退化成无判别力,而那会静默发生)。
    """
    long = "转" * 500
    payload = json.loads(await _tool().ainvoke({"reason": long}))
    assert payload == handoff_record(long[:ECHO_LIMIT])
    # 夹具自检:实测这两个派生**确实不同**(截短 → A118/1/5,不截短 → A205/2/5),
    # 所以上一条真的能区分「夹了」与「没夹」。
    assert handoff_record(long[:ECHO_LIMIT]) != handoff_record(long)


@pytest.mark.anyio
async def test_blank_reason_is_rejected():
    """空入参必须**抛**,而且必须抛 `ToolNotFound`(而不是任意异常)。

    类型是承重的:`ToolNotFound` 是可恢复的 —— 执行器把它回灌给模型、流照常
    `done`;**换成 `ToolInfrastructureError` 就是 502 + error 帧**,用户会因为
    模型传了个空字段而看到对话中断。`pytest.raises(Exception)` 对这两种实现
    一视同仁,所以这里钉具体的类。
    """
    tool = _tool()
    with pytest.raises(ToolNotFound):
        await tool.ainvoke({"reason": "   "})


def test_transfer_to_human_is_declared_read():
    """⚠️ 这条测试的**判别力**要说清楚。

    `policy.kind_of` 的规矩是「未声明 = 只读」,所以这条断言对一个
    **漏声明**的实现同样会通过 —— 它守的不是「有没有声明」,而是
    「**将来有人把它改成 write 时会被拦下来**」。声明本身在行为上是空操作,
    它的价值是让这个决定可被 grep 到、可被 review。
    """
    assert kind_of("transfer_to_human") == "read"


def test_transfer_to_human_is_not_a_write_tool():
    """它**不许**进 `WRITE_TOOLS` —— 进了就会被确认流拦住,而用户说的是
    「转人工」不是「请再点一次确认」(用户 2026-09-25 拍板:算只读、直调不确认)。
    """
    from app.tools.policy import WRITE_TOOLS

    assert "transfer_to_human" not in WRITE_TOOLS


def test_handoff_record_is_stable_across_processes():
    """**跨进程**:子进程里算一遍,与本进程比对。

    为什么要单独一条:上面那条 `test_same_reason_always_gives_the_same_agent`
    **抓不到这条硬约束** —— 内置 `hash()` 对 str 的随机化只在**换进程**时才
    显形(`PYTHONHASHSEED`),同进程内它一样是确定的。按本仓那条判据反问
    「实现改错了这条断言的输出会不会不同」:把 `rng(...)` 换成 `hash(...)`
    派种子,那条测试**照样绿**(T2 亲自跑过这个变异,确认它绿 —— 见报告 M1),
    所以判别力只能由本文件这条来给。

    这条检查是 ch02 spec §「种子必须用 `hashlib.sha256`」那一节点名要的
    (`docs/superpowers/specs/2026-09-16-ecommerce-cs-ch02-tools-design.md`:
    「把 `sha256` 换成内置 `hash()`,必须红(同进程内测不出来)」)。

    ⚠️ **既有那两条跨进程用例够不到这里**:`tests/test_mock_data.py` 那条钉的是
    `order_record`、`tests/test_tools_random.py` 那条钉的是 `logistics_record`
    —— 两条都只验各自那条随机流,`handoff_record` 用没用 sha256 它们一概不知。

    必须加 `-X utf8` —— 本机 locale 是 cp936,管道上的 stdout 按 GBK 编码
    而父进程按 UTF-8 解码,报错会表现为 `proc.stdout is None`。
    """
    reason = "我要找真人"
    code = (
        "import json, sys;"
        "sys.path.insert(0, '.');"
        "from app.tools.mock_data import handoff_record;"
        "sys.stdout.buffer.write("
        f"json.dumps(handoff_record({reason!r}), ensure_ascii=False).encode('utf-8'))"
    )
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == handoff_record(reason)
