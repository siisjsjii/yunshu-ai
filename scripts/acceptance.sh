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

echo "  ── 拼回后的完整回复(中文完整性证据):"
echo "$OUT1" | join_tokens
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
echo "================================"
echo "通过 $PASS 项，失败 $FAIL 项"
[ "$FAIL" -eq 0 ] || exit 1
