"""三个确定性伪随机工具的测试。不联网、不碰 DB。"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.tools.business import query_logistics, query_order, query_product
from app.tools.errors import ToolNotFound

REPO_ROOT = Path(__file__).resolve().parents[1]


def _call(tool, args: dict) -> str:
    tool_call = {"name": tool.name, "args": args, "id": "call_1", "type": "tool_call"}
    return asyncio.run(tool.ainvoke(tool_call)).content


def test_same_order_id_gives_same_result():
    """同一订单号永远返回同样数据。"""
    assert _call(query_logistics, {"order_id": "1001"}) == _call(
        query_logistics, {"order_id": "1001"}
    )


def test_different_order_ids_differ():
    """不同订单号应有不同数据,否则工具等于常量。"""
    assert _call(query_logistics, {"order_id": "1001"}) != _call(
        query_logistics, {"order_id": "1002"}
    )


def test_result_is_json_with_chinese_not_escaped():
    """返回 JSON 字符串,且中文不被转义成 \\uXXXX(白烧 token)。"""
    raw = _call(query_logistics, {"order_id": "1001"})
    payload = json.loads(raw)
    assert set(payload) >= {"order_id", "status", "location"}
    assert "\\u" not in raw
    assert any("一" <= ch <= "鿿" for ch in raw)


def test_seed_is_stable_across_processes():
    """跨进程确定性。

    这条是本章最容易写错的断言 —— 内置 hash() 对 str 每进程随机化
    (PYTHONHASHSEED),用它会让同一订单号在重启后返回不同数据,
    而同进程内的任何测试都测不出来。故必须另起两个进程比对。
    """
    code = (
        "import asyncio, sys; sys.path.insert(0, '.'); "
        "from app.tools.business import query_logistics; "
        "print(asyncio.run(query_logistics.ainvoke("
        "{'name': 'query_logistics', 'args': {'order_id': '1001'}, "
        "'id': 'c', 'type': 'tool_call'})).content)"
    )
    # 子进程的输出编码必须显式钉成 UTF-8。本机 locale 是 cp936,Python 会把管道
    # 上的 stdout 按 GBK 编码,而下面按 UTF-8 解码 —— 不钉的话子进程输的中文会
    # 在解码时炸掉,表现为 proc.stdout 为 None。这与工具逻辑无关,纯属平台差异。
    outs = []
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, "-X", "utf8", "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(REPO_ROOT),
        )
        assert proc.returncode == 0, proc.stderr
        outs.append(proc.stdout)
    assert outs[0] == outs[1]
    assert outs[0].strip()  # 非空,防止两边都空而"通过"


def test_malformed_order_id_raises_tool_not_found():
    """不像订单号的输入 → ToolNotFound(可恢复),不是随机编一个结果。"""
    with pytest.raises(ToolNotFound):
        _call(query_logistics, {"order_id": "abc"})


def test_query_product_uses_keyword():
    a = _call(query_product, {"keyword": "无线耳机"})
    b = _call(query_product, {"keyword": "保温杯"})
    assert a != b
    assert "无线耳机" in a
