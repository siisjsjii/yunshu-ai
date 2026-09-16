#!/usr/bin/env bash
# ch01 端到端验收。前置:另开一个终端启动服务
#   .venv/Scripts/python.exe -m uvicorn app.main:app --port 8000
# 需要真实 API key(.env)。
set -uo pipefail

# ── 本仓库运行在 Windows + Git Bash 下,有两处平台陷阱,已在脚本内规避 ──
#
# 1) 中文请求体不能走 argv。MSYS2 把 UTF-8 参数交给原生 curl.exe 之前会按
#    当前 ANSI 代码页(CP936)重新编码,"你好" 会变成 4 个 GBK 字节,服务端
#    json 解析失败,只回 {"detail":"There was an error parsing the body"}。
#    而断言里的 grep 全是 ASCII,这种全盘失败还好,怕的是它被掩盖成"通过"。
#    故所有含中文的请求体一律走 stdin(heredoc)—— 管道是字节流,不做转换。
#
# 2) 断言必须比对"拼回后的回复",不能直接 grep 原始 SSE 流。接口是逐 token
#    推送的,模型输出的订单号 20240915 会被切成 "202" / "409" / "15" 三个
#    独立帧,原始流里根本不存在连续的 "20240915" 子串 —— 直接 grep 原始流
#    是抽奖(取决于分词器怎么切),不是验收。join_tokens 负责还原。
BASE="${BASE:-http://localhost:8000}"
PYTHON="${PYTHON:-.venv/Scripts/python.exe}"
PASS=0
FAIL=0

pass() { echo "  ✅ $1"; PASS=$((PASS + 1)); }
fail() { echo "  ❌ $1"; FAIL=$((FAIL + 1)); }

# 把 SSE 里的 token 帧拼回完整回复文本。
# 按字节读写 stdin/stdout —— Windows 上 Python 对管道默认用 ANSI 代码页,
# 直接用 sys.stdout.write 会把中文写成乱码,那样"通过"也证明不了什么。
join_tokens() {
  "$PYTHON" -c '
import json, sys

texts = []
for raw in sys.stdin.buffer:
    line = raw.decode("utf-8", "replace").rstrip("\r\n")
    if not line.startswith("data: "):
        continue
    try:
        payload = json.loads(line[len("data: "):])
    except json.JSONDecodeError:
        continue
    if isinstance(payload, dict) and isinstance(payload.get("text"), str):
        texts.append(payload["text"])
sys.stdout.buffer.write("".join(texts).encode("utf-8"))
'
}

# 断言一段文本"非空且含真实 CJK 字符"。这是编码完好性的唯一自动化证据 ——
# 否则整份脚本的断言全是 ASCII,一份彻底乱码的回复照样能刷出"通过 N 项"。
#
# 不能用 grep '[一-龥]'。实测(本机 MSYS2):对 real UTF-8 和三种 mojibake
# 样本它**全部 MATCH** —— LC_ALL=C 下 bracket expression 逐字节比较,
# '[一-龥]' 退化成字节区间 0x80-0xe9(U+FFFD 的字节序列是 ef bf bd:
# 首字节 ef 不在区间内,但后两个**续字节** bf / bd 落在里面),于是任何
# 含 U+FFFD 的文本都会命中,这个 grep 永远通过,是个假断言。
# Python 按码点判断,与 locale 无关。
has_cjk() {
  "$PYTHON" -c '
import sys

text = sys.stdin.buffer.read().decode("utf-8", "replace")
if not text.strip():
    reason = "为空"
elif not any("一" <= c <= "鿿" or "㐀" <= c <= "䶿" for c in text):
    reason = "不含任何 CJK 字符"
elif "�" in text:
    reason = "含 U+FFFD 替换字符"
else:
    reason = None

if reason is not None:
    sys.stdout.buffer.write(reason.encode("utf-8"))
    raise SystemExit(1)

n = sum(1 for c in text if "一" <= c <= "鿿" or "㐀" <= c <= "䶿")
sys.stdout.buffer.write(f"{n} 个 CJK 字符".encode("utf-8"))
'
}

# 取出本轮实际调用了哪些工具(按出现顺序,逗号分隔)。
# 直接对原始 SSE 流 grep 工具名是不行的 —— 事件名与 data 分处两行,
# 且工具名只出现在 JSON 里。按 SSE 块解析才可靠。
called_tools() {
  "$PYTHON" -c '
import json, sys

names = []
for block in sys.stdin.buffer.read().decode("utf-8", "replace").split("\n\n"):
    event = data = None
    for line in block.splitlines():
        if line.startswith("event: "):
            event = line[len("event: "):]
        elif line.startswith("data: "):
            data = line[len("data: "):]
    if event == "tool_call" and data:
        try:
            names.append(json.loads(data)["name"])
        except (json.JSONDecodeError, KeyError):
            pass
print(",".join(names))
'
}

# 取出每个 tool_result 的成败(ok/fail,逗号分隔)。
tool_result_states() {
  "$PYTHON" -c '
import json, sys

states = []
for block in sys.stdin.buffer.read().decode("utf-8", "replace").split("\n\n"):
    event = data = None
    for line in block.splitlines():
        if line.startswith("event: "):
            event = line[len("event: "):]
        elif line.startswith("data: "):
            data = line[len("data: "):]
    if event == "tool_result" and data:
        try:
            states.append("ok" if json.loads(data).get("ok") else "fail")
        except json.JSONDecodeError:
            pass
print(",".join(states))
'
}

# 从工具实现里取订单 1001 的确定性物流状态。
# 动态取值而非写死 —— 工具改了种子函数也不必改脚本;
# 而"工具到底返回什么"由 Tier 1 的跨进程确定性测试守护。
expected_logistics_status() {
  "$PYTHON" -c '
import asyncio, json, sys

from app.tools.business import query_logistics

tool_call = {
    "name": "query_logistics",
    "args": {"order_id": "1001"},
    "id": "probe",
    "type": "tool_call",
}
payload = json.loads(asyncio.run(query_logistics.ainvoke(tool_call)).content)
sys.stdout.buffer.write(payload["status"].encode("utf-8"))
'
}

if [ ! -x "$PYTHON" ]; then
  echo "找不到 $PYTHON —— 请在项目根目录运行本脚本。" >&2
  exit 2
fi

echo "=== 验收 1:流式回复 ==="
OUT1=$(curl -sN -X POST "$BASE/api/chat/stream" \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{"message":"你好，我想咨询退货"}
JSON
)

echo "$OUT1" | head -c 400
echo

if echo "$OUT1" | grep -q "event: meta"; then
  pass "收到 meta 首帧"
else
  fail "没有 meta 首帧"
fi

TOKENS=$(echo "$OUT1" | grep -c "^event: token" || true)
if [ "$TOKENS" -gt 3 ]; then
  pass "收到 $TOKENS 个 token 帧(逐 token 推送)"
else
  fail "token 帧只有 $TOKENS 个,不是逐 token 推送"
fi

if echo "$OUT1" | grep -q "event: done"; then
  pass "收到 done 帧"
else
  fail "没有 done 帧"
fi

REPLY1=$(echo "$OUT1" | join_tokens)
echo "  ── 拼回后的完整回复(中文完整性证据):$REPLY1"

if REASON=$(echo "$REPLY1" | has_cjk); then
  pass "第一轮回复非空且含 $REASON(编码完好)"
else
  fail "第一轮回复$REASON —— 编码损坏"
fi
echo

echo
echo "=== 验收 2:两轮上下文 ==="
SID="acceptance-$$"
OUT2A=$(curl -sN -X POST "$BASE/api/chat/stream" \
  -H 'Content-Type: application/json' \
  --data-binary @- <<JSON
{"session_id":"$SID","message":"我的订单 20240915 还没发货"}
JSON
)

if echo "$OUT2A" | grep -q "event: error"; then
  echo "  ⚠️  第一轮就返回了 error 帧,下面的上下文断言无意义:"
  echo "$OUT2A" | grep "^data: .*message" | head -3
fi

OUT2=$(curl -sN -X POST "$BASE/api/chat/stream" \
  -H 'Content-Type: application/json' \
  --data-binary @- <<JSON
{"session_id":"$SID","message":"我刚才说的订单号是多少？"}
JSON
)

REPLY2=$(echo "$OUT2" | join_tokens)
echo "  第一轮:$(echo "$OUT2A" | join_tokens)"
echo "  第二轮:$REPLY2"
echo

# 第二轮问的是上一轮说过的信息,模型无法靠猜 —— 必须真的拿到历史。
# 比对拼回后的文本:逐 token 推送会把订单号切成多个帧,原始流里没有连续子串。
if echo "$REPLY2" | grep -q "20240915"; then
  pass "第二轮回复中出现了第一轮的订单号 20240915(上下文接通)"
else
  fail "第二轮回复中没有 20240915 —— 上下文没接住"
fi

echo
echo "=== 验收 3:结构化抽取 ==="
OUT3=$(curl -s -X POST "$BASE/api/extract" \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{"text":"订单 20240915 的鞋码不对，我想换大一码"}
JSON
)

echo "$OUT3"
echo

if echo "$OUT3" | grep -q '"order_id"' && echo "$OUT3" | grep -q "20240915"; then
  pass "抽出 order_id"
else
  fail "order_id 没抽出来"
fi

if echo "$OUT3" | grep -q '"request_type"'; then
  pass "抽出 request_type"
else
  fail "request_type 没抽出来"
fi

if echo "$OUT3" | grep -q '"expected_solution"'; then
  pass "抽出 expected_solution"
else
  fail "expected_solution 没抽出来"
fi

echo
echo "=== 验收 4:工具调用链路(需求 4) ==="
EXPECTED_STATUS=$(expected_logistics_status)
echo "  订单 1001 的确定性物流状态:$EXPECTED_STATUS"

# 工具取不到值时上面那条命令会**静默返回空**。空串会让 `grep -qF ""` 匹配
# 任何文本 —— 于是"回复复述了工具结果"这条断言永远通过,而它恰恰是本章
# 唯一证明"模型真的读到了工具返回"的证据。宁可在此处硬失败。
if [ -z "$EXPECTED_STATUS" ]; then
  fail "取不到订单 1001 的确定性物流状态 —— 下面的复述断言会退化成恒真,不再可信"
fi

SID_TOOL="acceptance-tool-$$"
OUT4=$(curl -sN -X POST "$BASE/api/chat/stream" \
  -H 'Content-Type: application/json' \
  --data-binary @- <<JSON
{"session_id":"$SID_TOOL","message":"订单 1001 的物流到哪了"}
JSON
)

TOOLS4=$(echo "$OUT4" | called_tools)
STATES4=$(echo "$OUT4" | tool_result_states)
REPLY4=$(echo "$OUT4" | join_tokens)
echo "  调用的工具:[$TOOLS4]  结果:[$STATES4]"
echo "  回复:$REPLY4"

if [ "$TOOLS4" = "query_logistics" ]; then
  pass "模型选中 query_logistics(验收标准 1 的后端一半)"
else
  fail "期望选中 query_logistics,实际调用了 [$TOOLS4]"
fi

if [ "$STATES4" = "ok" ]; then
  pass "工具执行成功"
else
  fail "工具执行未成功,结果状态为 [$STATES4]"
fi

# 断言模型真的**读懂了工具结果**而不是自己编。
# 比的是工具返回的确定性状态词 —— 模型没拿到结果就不可能说对。
# -F:状态词是数据不是正则(虽然当前取值全是 CJK、不含任何 ASCII 元字符,
# 但"当前取值恰好安全"不该是这条断言成立的前提)。
#
# 前半句 `[ -n ... ]` 是防"恒真":空模式会让 grep 匹配任何文本(上面已单独
# 记一次失败),但若不在这里挡住,这条断言还会**额外**刷出一个通过 —— 一条
# 永远不会失败的断言正是本章要防的东西。取不到值时必须走 else。
if [ -n "$EXPECTED_STATUS" ] && echo "$REPLY4" | grep -qF "$EXPECTED_STATUS"; then
  pass "回复中复述了工具返回的状态「$EXPECTED_STATUS」(工具结果真的被用上了)"
else
  fail "回复中没有「$EXPECTED_STATUS」—— 模型没有用上工具结果"
fi

echo
echo "=== 验收 5:FAQ 查表(验收标准 2) ==="
OUT5=$(curl -sN -X POST "$BASE/api/chat/stream" \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{"message":"退货政策是什么"}
JSON
)

TOOLS5=$(echo "$OUT5" | called_tools)
STATES5=$(echo "$OUT5" | tool_result_states)
REPLY5=$(echo "$OUT5" | join_tokens)
echo "  调用的工具:[$TOOLS5]  结果:[$STATES5]"
echo "  回复:$REPLY5"

if [ "$TOOLS5" = "query_faq" ] && [ "$STATES5" = "ok" ]; then
  pass "query_faq 查到了退货政策"
else
  fail "期望 query_faq 命中,实际工具=[$TOOLS5] 结果=[$STATES5]"
fi

if REASON=$(echo "$REPLY5" | has_cjk); then
  pass "回复非空且含 $REASON"
else
  fail "回复$REASON"
fi

echo
echo "=== 验收 6:邮费漏召回(验收标准 3,预期失败) ==="
OUT6=$(curl -sN -X POST "$BASE/api/chat/stream" \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{"message":"邮费是多少"}
JSON
)

TOOLS6=$(echo "$OUT6" | called_tools)
STATES6=$(echo "$OUT6" | tool_result_states)
REPLY6=$(echo "$OUT6" | join_tokens)
echo "  调用的工具:[$TOOLS6]  结果:[$STATES6]"
echo "  回复:$REPLY6"

# 反向断言:这一条**期望查不到**。若它竟然查到了,说明 faq 种子里混进了
# 「邮费」条目,验收标准 3 的前提被破坏 —— 那才是失败。
if [ "$TOOLS6" = "query_faq" ] && [ "$STATES6" = "ok" ]; then
  fail "「邮费」竟然从 faq 查到了 —— 种子数据污染了,验收标准 3 失去意义"
else
  pass "「邮费」未能查到(预期漏召回),工具=[$TOOLS6] 结果=[$STATES6]"
fi

echo
echo "================================"
echo "通过 $PASS 项，失败 $FAIL 项"
[ "$FAIL" -eq 0 ] || exit 1
