#!/usr/bin/env bash
# ch07 验收 1–5(端到端)。前置:MySQL 在跑、Milvus 在跑、`.env` 里有真实 key。
#
# **本脚本自己起服务**(8000 = 演示配置,8001 = 默认配置),不要求你先起好。两个理由
# 都不是讲究:
#   ① 验收 2 要求的是一组**显式覆盖的窗口参数**(spec §9.4),而 `.env` 里那套是默认值。
#      让操作者手工带 env 起服务 = 把「这一轮验的是哪套配置」交给记忆与手速;
#   ② `log/app.log` **必须在服务启动前截断**。不截断的话,断言会命中上一次运行(或单测)
#      留下的旧行 —— 于是**无论被测代码对不对,它们都是真的**(本仓最怕的那种假绿)。
#      这件事只有脚本自己做得准:文件句柄一开,截断就变成「它在写、我在砍」。
#
# 断言依据一律是**确定性的东西**:SSE 帧的名字与载荷、`log/app.log` 里那几行 JSON、
# 以及**由生产代码现算出来的**预算数字。不靠模型自由文本(deepseek 在 temperature=0 下
# 依然非确定),也不 grep 原始 SSE 流(逐 token 推送会把 20240915 切成多帧)。
# 唯一一处读模型自由文本的地方是验收 2 的落点,它走 `warn()` 档(spec §10.5)。
#
# 验收 3 要的是**默认**配置,与验收 1/2/4 的演示配置互斥,所以它排在最后(换一个服务)。
#
# 平台陷阱(本仓在 ch02/ch05/ch06 各栽过,这里逐条防):
#   1) **含中文的请求体不走 curl 的 argv**:MSYS2 会按 CP936 重编码,服务端只回
#      `error parsing the body`。所有请求体一律走 stdin heredoc。
#   2) **`python -c` 的片段源码里不许出现中文**:那段源码走 argv,同样会被重编码。
#      凡是需要中文(省略号 / `[工具结果] `)一律写 `\uXXXX` 转义,源码保持纯 ASCII。
#   3) **python 侧不接 `/tmp/...` 这类路径**:Windows 的 Python 把 `/tmp/x` 当成
#      `C:\tmp\x`(与 bash 的 /tmp 不是同一个地方)。临时文件一律用**相对路径**。
#   4) **不起管道截断**(`| tail`/`| head`):ch06 因为加了它,后台只剩尾巴,
#      11 条失败的内容全丢了,只能重跑一次。要落盘就整段落盘。
set -uo pipefail

cd "$(dirname "$0")/.." || exit 2
[ -f app/main.py ] || { echo "请在项目根目录运行本脚本(找不到 app/main.py)" >&2; exit 2; }

PYTHON="${PYTHON:-.venv/Scripts/python.exe}"
PORT_A="${PORT_A:-8000}"
PORT_B="${PORT_B:-8001}"
BASE_A="http://localhost:$PORT_A"
BASE_B="http://localhost:$PORT_B"
ASK_BASE="$BASE_A"          # ask_timed 打哪一个服务(验收 3 会切到 B)
LOG="log/app.log"
WORK=".ch07_acceptance"

PASS=0; FAIL=0; WARN=0
ok()   { echo "  PASS  $1"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL  $1"; FAIL=$((FAIL+1)); }
# 未复现(不是失败,也不是通过):只在「判定模型这一轮给出的结论」决定某条断言判什么时用。
# **必须显式计数并打印**,否则它就退化成一条悄悄跳过的检查 —— 而「悄悄跳过」正是本项目
# 最怕的假绿形态(spec §10.5 沿用 ch06 写死的这条理由)。
warn() { echo "  WARN  $1"; WARN=$((WARN+1)); }
# 装置坏了(该出现的东西没出现、日志读不了、配置对不上)。它比 FAIL 更严重:
# 此时**所有**「没有命中」的断言都恒真,必须响亮地说出来,而不是让它们静静地绿着。
boom() { echo "  !!!!  $1"; FAIL=$((FAIL+1)); }

if [ ! -x "$PYTHON" ]; then
  echo "找不到 $PYTHON —— 请在项目根目录运行本脚本。" >&2
  exit 2
fi

rm -rf "$WORK"; mkdir -p "$WORK"
SERVER_PID=""
KEEP=0
cleanup() {
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null
    sleep 1
    kill -9 "$SERVER_PID" 2>/dev/null
  fi
  # 失败时**保留**证据目录(SSE 原文、timing、两个服务的控制台输出),通过时才清掉。
  # ⚠️ 判据是 `FAIL -gt 0` **或** `KEEP -eq 1`,**不能只看 FAIL**:几条 preflight 分支
  # 在**一条断言都还没打**的时候就 `exit 1`(那时 FAIL 仍是 0),只认 FAIL 的话
  # `cleanup()` 会把工作目录连同**唯一的**那份控制台日志一起删掉 ——
  # 而那几条恰恰是最可能真触发的路径(Milvus 没起 / 模型加载失败 / 端口被占),
  # 也就是**最需要证据的时候**。这正是 ch06「`tail` 把错误切掉、11 条失败没留下」的形状。
  if [ "$FAIL" -eq 0 ] && [ "$KEEP" -eq 0 ]; then
    rm -rf "$WORK"
  else
    echo "  (证据留在 $(pwd)/$WORK/ —— SSE 原文、timing、两个服务的控制台输出)"
  fi
}
trap cleanup EXIT
# ⚠️ **外部打断(Ctrl-C / `timeout` 发的 TERM)必须自己 `exit`**:装了 trap 之后
# bash **不会**因为收到信号就退出 —— 信号处理函数返回后,脚本**继续往下跑**。
# 实测(2026-09-22,故意用错 DB 端口触发失败路径时撞上):TERM 到了 → `cleanup`
# 以 `FAIL=0 && KEEP=0` 判定「这是一次成功」⇒ **把工作目录删掉**;然后脚本
# **接着往下走**到失败分支,那里再 `cat "$WORK/server_a.log"` 就只剩
# 「No such file or directory」—— 证据在它被需要的前一刻被自己删了。
# 打断时一律 `KEEP=1`(这正是最需要证据的时候)并真的退出。
on_signal() {
  KEEP=1
  exit 130
}
trap on_signal INT TERM

# preflight 失败的统一出口:**必须用它,不要直接 `exit 1`**(理由见 `cleanup`)。
fail_exit() {
  KEEP=1
  exit 1
}

# ── 预检:两个端口必须是空的 ────────────────────────────────────────────────
# 本脚本**自己起服务**,所以 8000/8001 上不能有别的进程:占用时 uvicorn 起不来,
# 而下面的 wait_ready 会连到**那个旧进程**上并「通过」—— 于是整套验收测的是别的代码。
# 这正是本仓记过的「起服务前先查端口」(僵尸进程会让你 curl 到旧代码,得出假红/假绿)。
for p in "$PORT_A" "$PORT_B"; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://localhost:$p/api/conversations")
  if [ "$code" != "000" ]; then
    echo "预检失败:端口 $p 上已经有服务在监听(返回 $code)。本脚本要自己起服务,请先清掉:"
    echo "  netstat -ano | grep ':$p'                                    # 记下 PID"
    echo "  powershell -NoProfile -Command \"Stop-Process -Id <PID> -Force\""
    # 这一支用普通 exit:**此刻工作目录还是空的**(一条断言都没打、一个文件都没写),
    # 没有证据可留。其余几支一律走 `fail_exit`。
    exit 1
  fi
done

# ── 日志:先截断,再起服务(顺序反了就是「它在写、我在砍」)──────────────────
mkdir -p log
: > "$LOG"

start_server() {   # $1=端口 $2=控制台输出文件;其余参数是给这一次启动的 env 覆盖
  local port="$1" out="$2"; shift 2
  env "$@" "$PYTHON" -m uvicorn app.main:app --port "$port" > "$out" 2>&1 &
  SERVER_PID=$!
}
stop_server() {
  [ -n "$SERVER_PID" ] || return 0
  kill "$SERVER_PID" 2>/dev/null
  sleep 1
  kill -9 "$SERVER_PID" 2>/dev/null
  SERVER_PID=""
}
# 就绪轮询。**按挂钟计时(`SECONDS`),不是按次数**,而且 curl 自带 `--max-time`:
# 按次数循环时,每次 curl 若因为服务半死而卡上几秒,「60 秒」会变成好几分钟
# (实测:后端连不上库时,一轮 60 次跑了近 200 秒,把外面的 `timeout` 先等来了)。
wait_ready() {     # $1=base $2=秒数
  local deadline=$((SECONDS + $2))
  while [ "$SECONDS" -lt "$deadline" ]; do
    [ "$(curl -s --max-time 5 -o /dev/null -w '%{http_code}' "$1/api/conversations")" = "200" ] && return 0
    sleep 1
  done
  return 1
}
# 等 BGE-M3 预热完(它跑在后台线程里,要十几秒)。不等的话,第一次碰到检索的轮次会
# 白等一次现场加载。
#
# ⚠️ **不能拿 `BGE-M3` 这个 ASCII 子串当判据**:同一条日志流里还有一句
# `BGE-M3 预热失败,首次检索会退化为现场加载`(app/main.py),它**也含**这个子串 ——
# 于是预热**失败**时脚本会打印「BGE-M3 预热完成」。这是**断言级的错**(消息说谎),
# 不只是措辞问题;而下面二十轮的读数就没人敢信了。
#
# 判据句(「预热完成」/「预热失败」)**只能以十六进制码点传参**(纯 ASCII 走 argv),
# 由 python 侧 `chr()` 拼出来:直接把中文当参数传同样会走 argv、同样被 CP936 重编码,
# 而那个后果是**沉默地等满 90 秒**(永远匹配不上)。路径走 argv 没事(纯 ASCII)。
_warmup_line() {   # $1 = 空格分隔的码点,例:"9884 70ed"
  "$PYTHON" -c '
import sys
try:
    text = open(sys.argv[1], "rb").read().decode("utf-8")
except (OSError, UnicodeDecodeError):
    raise SystemExit(1)
needle = "".join(chr(int(h, 16)) for h in sys.argv[2].split())
raise SystemExit(0 if needle in text else 1)' "$LOG" "$1"
}
wait_warmup() {    # 判据 = 「预热完成」(9884 70ed 5b8c 6210)
  local i
  for i in $(seq 1 90); do
    _warmup_line "9884 70ed 5b8c 6210" && return 0
    sleep 1
  done
  return 1
}
warmup_failed() { _warmup_line "9884 70ed 5931 8d25"; }   # 判据 = 「预热失败」

# ── SSE 收发 ────────────────────────────────────────────────────────────────
# 发一条消息,并把**首帧到达时刻**记进 timing 文件(验收 4 的时序判据靠它)。
#   `start` = 请求发出前;`frame` = 第一帧(meta)到达;`end` = 流结束。
# ⚠️ 这里**不截断输出**:SSE 原文整段落进 $3,断言与人工复核都读它。
ask_timed() {   # $1=sid $2=消息 $3=SSE 输出文件 $4=timing 文件
  printf 'start\t%s\n' "$(date +%s%3N)" >> "$4"
  curl -s -N --max-time 300 -X POST "$ASK_BASE/api/chat/stream" \
    -H "Content-Type: application/json" \
    --data-binary @- <<JSON | "$PYTHON" -c '
import sys, time
ts = sys.argv[1]
out = sys.stdout.buffer
seen = False
for line in sys.stdin.buffer:
    if not seen:
        seen = True
        open(ts, "a", encoding="utf-8").write("frame\t%d\n" % (time.time() * 1000))
    out.write(line)
out.flush()
' "$4" > "$3"
{"session_id":"$1","message":"$2"}
JSON
  printf 'end\t%s\n' "$(date +%s%3N)" >> "$4"
}

has_done()  { grep -q '^event: done' "$1"; }
error_n()   { grep -c '^event: error' "$1"; }
done_n()    { grep -c '^event: done' "$1"; }
# error 帧的文案(诊断用;已经过 redact_api_key)。
error_text() {
  "$PYTHON" -c '
import json, sys
lines = sys.stdin.buffer.read().decode("utf-8", "replace").splitlines()
for i, l in enumerate(lines):
    if l.strip() == "event: error" and i + 1 < len(lines) and lines[i+1].startswith("data: "):
        sys.stdout.buffer.write(json.loads(lines[i+1][6:]).get("message", "").encode("utf-8"))
        break'
}
# 预热那一轮**到底有没有真的检索**,输出 `hits=N faq_ok=M`(两个出口都认)。
#
# 为什么要两个出口:检索在本仓有两条路 ——
#   * **知识类**(`商品咨询`)走 `retrieve_knowledge` 节点里的检索,`trace` 里留
#     `retrieve_knowledge:N hits`;
#   * **业务类**(`物流`/`订单`)由 Agent 调 `query_faq` 那个**工具**。
# 「运费怎么算」这种题面实测被判成 **物流** ⇒ 走的是第二条路。只认第一条会让
# 预热**假失败**(第一次跑就这么失败了三次,而三次**都真的检索了**:
# 24 秒那次就是冷加载重排器,`faq_ok=1`)。
# `hits=-1` 表示那一行没出现(-1 不是 0:0 与「节点没跑」在数值上不可区分)。
warm_evidence() {
  "$PYTHON" -c '
import json, re, sys
lines = sys.stdin.buffer.read().decode("utf-8", "replace").splitlines()
hits = -1
faq_ok = 0
last_call = None
for i, l in enumerate(lines):
    if i + 1 >= len(lines) or not lines[i+1].startswith("data: "):
        continue
    if l.strip() == "event: tool_call":
        try:
            last_call = json.loads(lines[i+1][6:]).get("name")
        except json.JSONDecodeError:
            last_call = None
    elif l.strip() == "event: tool_result":
        try:
            obj = json.loads(lines[i+1][6:])
        except json.JSONDecodeError:
            continue
        if obj.get("ok") and last_call == "query_faq":
            faq_ok = 1
data = [l[6:] for l in lines if l.startswith("data: ")]
trace = []
try:
    trace = json.loads(data[-1]).get("trace") or []
except (IndexError, json.JSONDecodeError):
    trace = []
for t in trace:
    m = re.search(r"retrieve_knowledge:(\d+)", t)
    if m:
        hits = int(m.group(1))
        break
sys.stdout.buffer.write(("hits=%d faq_ok=%d\n" % (hits, faq_ok)).encode("ascii"))'
}
join_tokens() {
  "$PYTHON" -c '
import json, sys
lines = sys.stdin.buffer.read().decode("utf-8", "replace").splitlines()
parts = []
for i, l in enumerate(lines):
    if l.strip() == "event: token" and i + 1 < len(lines) and lines[i+1].startswith("data: "):
        parts.append(json.loads(lines[i+1][6:]).get("text", ""))
sys.stdout.buffer.write("".join(parts).encode("utf-8"))'
}

# ── 读日志 ──────────────────────────────────────────────────────────────────
# 一行 JSON、前缀固定(`app/memory/journal.py` 与 `app/memory/tasks.py` 是仅有的两个
# 生产者)。**按 JSON 的键认事件,不按中文文案认** —— 文案是给人读的,改一句措辞
# 不该让断言变红或变绿:
#   model_ctx 有 `bounds`;history_ctx 有 `sliding` 而没有 `bounds`;
#   summary done 有 `covered_from`;trigger 有 `layer2_budget`;降级有 `layer1_budget`;
#   skip 有 `reason`(且无 `turns` 之外的歧义,故判在 start 之前);fail 有 `anchors_advanced`。
# 输出是一串 ASCII 的 `key=value`(与 ch06 的 db_state 同款),供 bash 侧取值。
log_state() {   # $1=sid $2=要在梗概里找的 ASCII 串(可空)
  "$PYTHON" -c '
import json, sys
from datetime import datetime

path, sid, needle = sys.argv[1], sys.argv[2], sys.argv[3]
ELL = chr(0x2026)
TOOL_PREFIX = "[\u5de5\u5177\u7ed3\u679c] "

try:
    text = open(path, "rb").read().decode("utf-8")
except UnicodeDecodeError:
    sys.stdout.buffer.write(b"utf8=bad\n")
    raise SystemExit(3)

def ms(line):
    return int(datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S,%f").timestamp() * 1000)

counts = dict(model_ctx=0, history_ctx=0, degrade=0, trigger=0, sum_start=0,
              sum_done=0, sum_skip=0, sum_fail=0, other=0)
first = {}
done_ts = []
mc_layer1_over = mc_layer2_over = mc_seen = 0
last_budget = None
any_summary_has_needle = 0
hc_trunc_assistant = hc_trunc_tool = hc_lines_with_trunc = hc_lines_with_tool = 0
hc_window_max = 0

for line in text.splitlines():
    i = line.find("{")
    if i < 0:
        continue
    try:
        obj = json.loads(line[i:])
    except json.JSONDecodeError:
        continue
    if not isinstance(obj, dict) or obj.get("conversation_id") != sid:
        continue
    keys = set(obj)
    if "bounds" in keys:
        kind = "model_ctx"
    elif "sliding" in keys:
        kind = "history_ctx"
    elif "covered_from" in keys:
        kind = "sum_done"
    elif "layer2_budget" in keys:
        kind = "trigger"
    elif "layer1_budget" in keys:
        kind = "degrade"
    elif "anchors_advanced" in keys:
        kind = "sum_fail"
    elif "reason" in keys:
        kind = "sum_skip"
    elif "turns" in keys:
        kind = "sum_start"
    else:
        kind = "other"
    counts[kind] += 1
    first.setdefault(kind, obj)

    if kind == "model_ctx":
        mc_seen += 1
        tok, bud = obj["tokens"], obj["budgets"]
        mc_layer1_over += tok["layer1"] > bud["layer1"]
        mc_layer2_over += tok["layer2"] > bud["layer2"]
        last_budget = (bud["layer1"], bud["layer2"])
        if needle and needle in (obj.get("summary") or ""):
            any_summary_has_needle = 1
    elif kind == "history_ctx":
        last_budget = (obj["budgets"]["layer1"], obj["budgets"]["layer2"])
        # 名字里的 "layer1" 指的是**那一行的 `tokens.layer1` 键**,而它在 `history_ctx` 里
        # 数的是**整个扁平窗口**(见 journal 的 docstring)⇒ 它报的是**窗口**的上界,
        # 不是「层 1」的。键名与打印文案都按这个来(本章刚写进规则 (f) 的那一条)。
        hc_window_max = max(hc_window_max, obj["tokens"]["layer1"])
        ent = obj["sliding"]
        if any(e["role"] == "tool" for e in ent):
            hc_lines_with_tool += 1
        n_asst = sum(1 for e in ent if e["role"] == "assistant" and e["content"].endswith(ELL))
        n_tool = sum(1 for e in ent if e["role"] == "tool" and e["content"].startswith(TOOL_PREFIX))
        hc_trunc_assistant += n_asst
        hc_trunc_tool += n_tool
        if n_asst or n_tool:
            hc_lines_with_trunc += 1
    elif kind == "sum_done":
        done_ts.append(ms(line))

out = ["utf8=ok"]
for k in ("model_ctx", "history_ctx", "degrade", "trigger", "sum_start",
          "sum_done", "sum_skip", "sum_fail", "other"):
    out.append("%s=%d" % (k, counts[k]))
for kind, pre in (("degrade", "degrade_first"), ("trigger", "trigger_first"),
                  ("sum_done", "done_first")):
    obj = first.get(kind) or {}
    for key, val in obj.items():
        if key != "conversation_id" and isinstance(val, (int, str)):
            out.append("%s_%s=%s" % (pre, key, val))
deg, trg, don = first.get("degrade"), first.get("trigger"), first.get("sum_done")
out.append("degrade_worst_over=%d" % (deg["layer1_tokens"] - deg["layer1_budget"] if deg else -1))
out.append("trigger_worst_over=%d" % (trg["layer2_tokens"] - trg["layer2_budget"] if trg else -1))
out.append("done_min_elapsed_ms=%d" % (don["elapsed_ms"] if don else -1))
out.append("mc_layer1_over=%d" % mc_layer1_over)
out.append("mc_layer2_over=%d" % mc_layer2_over)
out.append("mc_seen=%d" % mc_seen)
out.append("any_summary_has_needle=%d" % any_summary_has_needle)
out.append("hc_trunc_assistant=%d" % hc_trunc_assistant)
out.append("hc_trunc_tool=%d" % hc_trunc_tool)
out.append("hc_lines_with_trunc=%d" % hc_lines_with_trunc)
out.append("hc_lines_with_tool=%d" % hc_lines_with_tool)
out.append("hc_window_max_tokens=%d" % hc_window_max)
if last_budget:
    out.append("budget_layer1_logged=%d" % last_budget[0])
    out.append("budget_layer2_logged=%d" % last_budget[1])
out.append("done_ts_ms_list=%s" % ",".join(str(x) for x in done_ts))
sys.stdout.buffer.write(("\n".join(out) + "\n").encode("utf-8"))
' "$LOG" "$1" "$2"
}

# 从 log_state 的输出里取一个 key 的值(不接 head/tail —— 那是截断管道)。
lget() { printf '%s\n' "$1" | sed -n "s/^$2=//p"; }

# 生产代码**现算**一遍预算(spec §9.4 的口径:断言的是实际算出来的数,不是预定的数)。
# 与端点用的是同一个 `budget.derive`、同一份 settings(env 覆盖一起传进来)。
expected_budget() {   # $1..=env 覆盖
  env "$@" "$PYTHON" -c '
import sys
from app.config import get_settings
from app.memory import budget
from app.prompts import render_system_prompt
s = get_settings()
b = budget.derive(settings=s, system_prompt=render_system_prompt(s.brand_name))
sys.stdout.buffer.write((
    "layer1=%d layer2=%d history=%d window=%d fixed=%d peak=%d\n"
    % (b.layer1_budget, b.layer2_budget, b.history_budget, b.window,
       b.fixed_overhead, b.peak)).encode("ascii"))'
}
eb_get() { printf '%s\n' "$1" | sed -n "s/.*$2=\([0-9]*\).*/\1/p"; }

# 验收 4 的时序判据:把每个 `summary done` 配到**它触发时那一轮**,再看那一轮的**首帧**
# (meta)是不是在它之前到达。「不阻塞」唯一可观测的差异就在这里:
#   * 阻塞实现:端点在流开始前 `await` 摘要 ⇒ `summary done` 必然早于**本轮**首帧;
#   * 不阻塞实现:首帧在摘要还在跑的时候就发出去了 ⇒ 至少有一轮「首帧 < done」。
# **必须逐对配**,不能写成「存在某一对首帧 < 某一对 done」:后者在阻塞实现下也成立
# (上一轮的首帧当然早于下一轮的 done)—— 那就是一条恒真的假断言。
#
# ⚠️ **配对基准取「这个任务是什么时候开始的」,不是「done 什么时候到」**(2026-09-22 修)。
# 原实现按「最后一个 `start <= done` 的轮次」配,而这两件事不是一回事:一次摘要要跑
# 6~11 秒,完全可以**落在两轮之间的空隙里**。实测(终审后第 2 次验收,`turn13_done_gap_ms=-345`,
# 一个**假红**):那次 done 属于**第 11 轮**触发的任务(它的 `elapsed_ms=6825` ⇒ 任务始于
# `done-6825`,落在第 11 轮窗口内),却在第 13 轮请求发出后 48 ms、首帧之前 345 ms 落地,
# 于是被配到了第 13 轮 —— 按正确配对它是 **+6768 ms(PASS)**。
# 任务起点在**日志行自己身上**(`done` 行带 `elapsed_ms`,它就是 `_run` 的计时起点),
# 不需要另立时钟:任务起点必然落在**触发它的那一轮**的窗口里(端点在流开始前的守卫段
# 同步触发它),用它配对才等于 docstring 说的「配到它触发时那一轮」。
# **判别力不变**:阻塞实现里 done 早于本轮的 meta 首帧 ⇒ gap 仍为负 ⇒ 照样红。
nonblock_pairs() {   # $1=sid $2=timing 文件
  "$PYTHON" -c '
import json, sys
from datetime import datetime

log, sid, tsv = sys.argv[1], sys.argv[2], sys.argv[3]
turns = []
cur = None
for line in open(tsv, encoding="utf-8"):
    if "\t" not in line:
        continue
    part, _, val = line.rstrip("\n").partition("\t")
    if not val:
        continue
    t = int(val)
    if part == "start":
        cur = {"start": t, "frame": None}
        turns.append(cur)
    elif cur is not None and part == "frame":
        cur["frame"] = t
dones = []
for line in open(log, "rb").read().decode("utf-8").splitlines():
    i = line.find("{")
    if i < 0:
        continue
    try:
        obj = json.loads(line[i:])
    except json.JSONDecodeError:
        continue
    if not isinstance(obj, dict) or obj.get("conversation_id") != sid:
        continue
    if "covered_from" in obj:
        at = int(datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S,%f").timestamp() * 1000)
        # 任务**起点** = done 那一刻减去它自报的耗时(`_run` 的计时区间)。
        # 缺 `elapsed_ms`(旧格式/手工拼的行)时退回 done 本身 —— 那是修之前的行为,
        # 会重新长出上面那条假红,所以它只是保底,不是等价物。
        dones.append((at - int(obj.get("elapsed_ms") or 0), at))
pairs_ok = skipped = 0
gaps = []
for began, d in dones:
    k = None
    for idx, tn in enumerate(turns):
        if tn["start"] <= began:
            k = idx
    if k is None or turns[k]["frame"] is None:
        skipped += 1
        continue
    gap = d - turns[k]["frame"]
    gaps.append(gap)
    pairs_ok += gap > 0
    sys.stdout.buffer.write(("turn%d_done_gap_ms=%d\n" % (k + 1, gap)).encode("ascii"))
sys.stdout.buffer.write((
    "turns=%d\ndones=%d\npairs_ok=%d\npairs_skipped=%d\nmin_gap_ms=%d\n"
    % (len(turns), len(dones), pairs_ok, skipped, min(gaps) if gaps else -1)).encode("ascii"))' \
    "$LOG" "$1" "$2"
}

# ══════════════════════════════════════════════════════════════════════════════
# 场景素材。**长消息是刻意的**:层 1 的预算有 4853 token(演示配置),短消息凑不出级联,
# 而级联正是验收 2 的对象。长消息里都带**演示订单池**里的号码(20240915 / 20240901 /
# 20240818 / 20240808),这样退款子流程不会因为缺号而挂起(挂起轮没有 done 帧,
# 会把验收 1 的红引到别处)。**20240915 只出现在第 1 轮** —— 验收 2 的落点问的是
# 「最开始那个订单」,它若在后面的轮次里也出现,那条断言就证明不了「梗概记住了它」。
LONG_A='你好,我想麻烦你帮我查一下订单 20240915 的物流现在到底是什么情况。我在你们家下单买了那个熊本熊保温杯,当时页面上写的是当天发货,结果到现在我这边看到的还是已发货的状态,既没有具体的城市,也没有下一站的预计时间,我自己在「我的订单」里翻来覆去看了好几遍也看不明白。我不太会操作这些,所以想请你帮我直接查一下最新的物流轨迹,告诉我现在到哪个城市了、下一站大概去哪、以及最可能送到的日期。如果遇到分拨中心积压或者天气原因造成的延误,也请如实告诉我,不要只说一句耐心等待。另外我还想问一下,这个包裹是哪家快递公司承运的,如果我一直收不到,能不能联系快递的网点自己去找,或者改成在驿站自提?我的收件地址是一个老小区,门牌号码写得比较绕,快递员有几次都送到隔壁单元去了,这次能不能麻烦你备注一下让快递员提前打电话给我。还有就是,我白天要上班,家里没人签收,如果今天下午送到楼下没人,会不会被直接退回去?退回去的话我再想拿回来要重新走什么流程、大概要等几天?麻烦你按顺序一条一条告诉我,辛苦了,谢谢。'
LONG_B='你好,我又来麻烦你了,想请你帮我看看订单 20240818 的物流。这次买的是那个静音无线鼠标,下单之后我一直在等,可是页面上显示的物流信息好像停在一个中转站好几天都没有更新了,我不确定是不是丢件了,还是只是扫描没及时同步。你能不能帮我查一下它最后一次扫描是什么时候、在哪个城市,以及如果确实卡住了,我这边可以做什么,是不是需要你们去联系快递公司发起查询。另外我想确认一下,如果包裹在途中损坏了,签收之后还能不能申请处理,需要保留哪些照片或者凭证。还有,我这个订单是帮同事一起买的,收货人写的是他的名字,如果需要本人签收或者出示取件码,我能不能拿着他的手机号去代领?这一单我比较着急,麻烦你尽量帮我查得细一点,谢谢。'
LONG_C='你好,我想咨询一下订单 20240808 的物流情况。这一单是那个折叠台灯,我看到物流信息显示已经到达本市了,但是一直停在站点没有安排派送,已经过去两天了,我不清楚是不是需要我自己去网点取件,还是说快递员会再安排一次派送。我想请你帮我确认几件事:第一,现在包裹到底在哪个网点,能不能查到网点的联系电话和营业时间;第二,如果我今天自己去网点取件,需要带什么凭证,报手机号可以吗,要不要出示下单时的付款记录;第三,如果我一直等不到派送,能不能申请改为送货上门或者放在楼下的驿站,需要额外收费吗;第四,万一最后包裹被退回仓库了,我这一单会不会有额外的手续,需要我这边做什么。顺便我还想问几个和这一单相关的细节:一是签收的时候能不能要求当面开箱验货,如果快递员说必须先签字才给开箱,这种情况我应该怎么办,有没有你们这边的规定可以引用;二是如果外包装明显有凹陷,我是直接拒收更省事,还是先签收再走售后流程,两种做法对后续处理有什么区别;三是我在备注里写「工作日晚上六点以后派送」,这条备注快递员到底看不看得见,如果看不见我应该通过什么渠道告诉他;四是我这一单的发票能不能随包裹一起寄过来,还是只能开电子发票发到邮箱。麻烦你帮我一条一条看一下,谢谢。'
LONG_D='你好,我还有几个和订单 20240901 有关的问题想一起问清楚,麻烦你帮我看一下。这一单买的是那个大容量运动水壶,下单之后物流显示已经揽收,但是两天都没有新的扫描记录,我有点担心。我想知道:第一,从揽收到发出一般需要多久,超过多久算异常;第二,如果快递公司一直没有更新,应该由你们去催还是我自己打电话催,哪种更快;第三,包裹如果在路上耽误了,能不能申请补偿或者优惠券;第四,我能不能临时把收货地址改到公司,改地址要不要收费,会不会影响原来的时效承诺。另外还有几个细节:一是这一单我是用花呗分期付的,如果最后要退款,分期的利息会不会退给我,退款是原路返回吗;二是我的收货手机号前阵子换过了,现在页面上显示的号码是旧的,需要怎么改才不会影响派送;三是如果快递员把包裹放在了快递柜,而我没有及时取,超时产生的保管费应该由谁承担;四是我看到你们页面上写着「七天无理由」,这个七天是从签收当天算还是从发货当天算,遇到周末顺延吗。这些问题我攒了好几天了,麻烦你按顺序一条一条帮我确认一下,说清楚一点,谢谢。'

SID1=$("$PYTHON" -c "import uuid;print(uuid.uuid4().hex)")
SID3=$("$PYTHON" -c "import uuid;print(uuid.uuid4().hex)")
case "$SID1$SID3" in
  *[!0-9a-f]*|"") echo "生成的会话 id 非法 —— 检查 $PYTHON 能不能跑。" >&2; exit 2;;
esac

# 验收 1 的二十轮。**前十轮是长消息**(每条约 500–760 token,把级联推起来 ——
# 层 1 的预算 4853 靠短消息凑不出来,而级联正是验收 2 的对象),后十轮短。
# 第 15 轮是验收 2 的落点(问「最开始那个订单」)。
MSG1=(
  "$LONG_A"
  "$LONG_B"
  "$LONG_C"
  "$LONG_D"
  "$LONG_B"
  "$LONG_C"
  "$LONG_D"
  "$LONG_B"
  "$LONG_C"
  "$LONG_D"
  "订单 20240901 的物流到哪了"
  "好的,谢谢"
  "那订单 20240808 显示派送了吗"
  "嗯嗯,我知道了"
  "我最开始问的那个订单,现在物流到底是什么情况?"
  "嗯,那我再等等"
  "好"
  "谢谢"
  "明白了"
  "那我先这样,有问题再问你"
)

DEMO_ENV=(
  MODEL_CONTEXT_WINDOW=18000 MAX_OUTPUT_TOKENS=2000 MAX_USER_INPUT_TOKENS=2000
  MAX_AGENT_STEPS=3 TOOL_RESULT_MAX_TOKENS=1200 RERANK_TOP_K=5
)

echo "== 起服务(演示配置:spec §9.4 那一组显式覆盖)=="
start_server "$PORT_A" "$WORK/server_a.log" "${DEMO_ENV[@]}"
if ! wait_ready "$BASE_A" 60; then
  echo "预检失败:$BASE_A 60 秒内没起来。控制台输出(前 60 行 + 末 15 行):"
  # **文件整份留着、路径打出来,终端只印头尾** —— 这两个要求不冲突,而且是必须分开的:
  #   * ch06 的教训是「`tail` 让**证据**没了」⇒ 这里证据**在文件里**(失败不删,见 cleanup),
  #     而且**行数与路径都打出来**,谁都能去看全文;
  #   * 但把整份 `cat` 出来同样是错的:后端连不上库时,一轮就重试出 **7354 行 / 467 KB**
  #     (60 次同样的 traceback),断言输出会被冲得看不见 —— 那才是真正丢证据的方式。
  # 头尾各印一段:头部有启动横幅,尾部有**真正的错因**(SQLAlchemy 的异常文本在最后一行)。
  "$PYTHON" -c '
import sys
path = sys.argv[1]
lines = open(path, "rb").read().decode("utf-8", "replace").splitlines()
head, tail = 60, 15
out = list(lines[:head])
if len(lines) > head + tail:
    out.append("... [中间省略 %d 行,全文见文件] ..." % (len(lines) - head - tail))
out += lines[-tail:] if len(lines) > tail else []
out.append("(共 %d 行)" % len(lines))
sys.stdout.buffer.write(("\n".join(out)).encode("utf-8") + b"\n")' "$WORK/server_a.log"
  echo "  (上面是节选;**完整控制台输出在 $WORK/server_a.log,失败不删**) "
  fail_exit
fi
echo "  服务已就绪(pid $SERVER_PID)"
if wait_warmup; then
  echo "  BGE-M3 预热完成(碰到检索的轮次可以跑了)"
elif warmup_failed; then
  # **分开报**:这句话与上面那句差一个词,含义却相反。拿 `BGE-M3` 子串去 grep 的话,
  # 预热**失败**会被打印成「预热完成」—— 消息说谎,而下面二十轮的读数就没人信了。
  warn "BGE-M3 **预热失败**(日志里是「预热失败,首次检索会退化为现场加载」)——
        碰到检索的轮次会白等一次现场加载,读数仍有效,但要记着这一条"
else
  warn "90 秒内既没看到「预热完成」也没看到「预热失败」—— 预热线程可能还没跑完"
fi
EXP_A=$(expected_budget "${DEMO_ENV[@]}")
echo "  生产代码现算的预算:$EXP_A"

# ── 预检 3:检索链路真的能用(顺带把重排器的懒加载逼出来)──────────────────
# 三件事,都不是形式主义:
#   ① **环境预检**:Milvus 不可用时,凡是走到检索的轮次都会 502「工具执行失败」
#      (实测:一次 `panic: etcdserver: leader changed` 让容器内的进程死掉,
#      而 `docker ps` 里它照样显示「Up」)。那种红与上下文管理毫无关系,不该混进本脚本;
#   ② **把重排器的冷加载先付掉**。`app/main.py` 的预热只覆盖**嵌入模型**(BGE-M3),
#      `bge-reranker-v2-m3` 是首次 `rerank()` 才加载的 —— **实测 21.0 秒**
#      (独立进程直测),而真机上第一次 `query_faq` 实测耗时 **24 秒**(2026-09-22 记录)。
#      **注意别把它说成「超时 502」**:`execute_tool` 的
#      `asyncio.wait_for(..., timeout=10)` **拦不住**它 —— 那次重排是**同步调用、
#      阻塞事件循环**的,而 `wait_for` 的定时器回调在循环被占住时根本没机会跑
#      (本机 python 3.13.14 实测:同样 3 秒的阻塞,套 `to_thread` 会超时、
#       直接阻塞循环则不会)。所以用户看到的是**一次 24 秒的静默停顿**,
#      而不是 502 —— 代价换了个样子,仍然要付。这是 ch04 的欠账(预热只做了一半),
#      记账在 T13 报告里,**不在本章修**。
#   ③ 顺便把「检索真的通了」钉住:三条判据(done 帧 + 无 error + **检索路径真的跑了**)。
#      判据取 `hits >= 0`,**不是** `hits >= 1`:命中数为 0 有两种成因 ——
#      `_hybrid_hits` 空 / 重排后被阈值滤空(阈值 0.25 是 ch04 定的,**真实长问句
#      只高出 0.02~0.09**)。后者**已经过了重排器**(顺序:hybrid → rerank → 阈值过滤),
#      所以「0 命中且不报错」同样是环境可用 + 重排器热了的证据。
#      **第 2 次跑就是被 `hits >= 1` 卡掉的**:三次都 `hits=0`、`done=1`、`error=0`,
#      而第一次那 16 秒里重排器明明加载完了(控制台里两次 `Loading weights`)。
#      —— 「判据比现象更严」也是假红的一种,和假绿一样要防。
#      题面挑「邮费是多少」:`evals/retrieval_cases.jsonl` 里的正例(它**本该**命中),
#      命中数因此更能说明环境;而 `hits=0` 时上面的口径照样成立。
SIDW=$("$PYTHON" -c "import uuid;print(uuid.uuid4().hex)")
WARM_OK=no
for attempt in 1 2 3; do
  T0=$(date +%s)
  ask_timed "$SIDW" "你们这里的邮费是多少" "$WORK/warm_$attempt.sse" "$WORK/warm.tsv"
  T1=$(date +%s)
  EV=$(warm_evidence < "$WORK/warm_$attempt.sse")
  HITS=$(printf '%s\n' "$EV" | sed -n 's/.*hits=\([-0-9]*\).*/\1/p')
  FAQ=$(printf '%s\n' "$EV" | sed -n 's/.*faq_ok=\([0-9]*\).*/\1/p')
  if has_done "$WORK/warm_$attempt.sse" && [ "$(error_n "$WORK/warm_$attempt.sse")" -eq 0 ] \
     && { [ "${HITS:-(-1)}" -ge 0 ] || [ "${FAQ:-0}" -eq 1 ]; }; then
    WARM_OK="第 $attempt 次,$((T1-T0)) 秒(retrieve_knowledge 命中=$HITS / query_faq ok=$FAQ)"
    break
  fi
  echo "  预热第 $attempt 次没成($((T1-T0)) 秒):done=$(done_n "$WORK/warm_$attempt.sse") error=$(error_n "$WORK/warm_$attempt.sse") $EV $(error_text < "$WORK/warm_$attempt.sse")"
  sleep 5
done
# 重排器到底加载了没有:控制台里 `Loading weights` 这个进度条被刷了多少次
# (每加载一个模型刷很多次,所以数字很大 —— 它只能回答「加载过」,回答不了「加载了几个模型」)。
# **只作证据打印,不作判据** —— 那是 transformers 进度条的输出形状,不是契约。
LOADS=$(grep -o 'Loading weights' "$WORK/server_a.log" 2>/dev/null | grep -c . )
if [ "$WARM_OK" = "no" ]; then
  echo "预检失败:检索链路不可用(三次预热都没拿到「真的检索过且没有 error 帧」的一轮)。"
  echo "  本脚本的二十轮里,模型随时可能碰到检索 —— 它挂了的话那些轮次会 502,"
  echo "  而那不是本章的缺陷。先修环境再跑:"
  echo "    docker ps | grep milvus                 # 「Up」不代表进程活着(见过 panic 之后仍显示 Up)"
  echo "    docker logs milvus-standalone --since 10m | grep -n 'panic:'"
  echo "    curl -s http://127.0.0.1:19530/healthz   # 正常应回 OK"
  echo "    docker restart milvus-standalone        # panic 后重启通常即可(实测一次)"
  echo "  (三次预热的 SSE 原文在 $WORK/warm_*.sse —— 它们保留了失败现场)"
  fail_exit
fi
echo "  检索预热通过:$WARM_OK"
echo "  控制台里 \"Loading weights\" 进度条刷了 ${LOADS:-0} 次(只能说明「确实加载过模型」;只作证据)"
echo

# ══════════════════════════════════════════════════════════════════════════
echo "== 验收 1:连聊二十轮(token 不爆、不崩)=="
TS1="$WORK/timing1.tsv"; : > "$TS1"
DONE_OK=0; ERR_N=0; SUSPECT=""
for i in $(seq 1 20); do
  ask_timed "$SID1" "${MSG1[$((i-1))]}" "$WORK/s1_$i.sse" "$TS1"
  d=$(done_n "$WORK/s1_$i.sse")
  e=$(error_n "$WORK/s1_$i.sse")
  if [ "$d" -ge 1 ] && [ "$e" -eq 0 ]; then
    DONE_OK=$((DONE_OK+1))
  else
    SUSPECT="$SUSPECT [$i:done=$d err=$e]"
    if [ "$e" -ge 1 ]; then
      ERR_N=$((ERR_N+1))
      echo "  第 $i 轮 error 帧:$(error_text < "$WORK/s1_$i.sse")"
    fi
  fi
  printf '  第 %2d 轮:done=%s error=%s\n' "$i" "$d" "$e"
done
if [ "$DONE_OK" -eq 20 ]; then
  ok "二十轮全部有 done 帧、且没有 error 帧"
else
  bad "二十轮里有 $((20-DONE_OK)) 轮不正常(共 $ERR_N 轮带 error 帧):$SUSPECT"
fi

ST1=$(log_state "$SID1" "20240915")
if [ "$(lget "$ST1" utf8)" != "ok" ]; then
  boom "$LOG 不是 UTF-8 —— 日志 handler 的 encoding 没钉住(cp936 下中文日志会炸)"
fi
# 正向对照:下面所有「没有命中」的断言(验收 3)都靠它证明**日志是活的、筛选是通的**。
# 没有它,「零条降级」既可能是真的,也可能是日志根本没写、或者会话 id 筛错了。
MC1=$(lget "$ST1" model_ctx); HC1=$(lget "$ST1" history_ctx)
if [ "${MC1:-0}" -ge 1 ] && [ "${HC1:-0}" -ge 20 ]; then
  ok "日志是活的:该会话 model_ctx=$MC1 行 / history_ctx=$HC1 行"
else
  boom "日志里该会话的 model_ctx=${MC1:-0} / history_ctx=${HC1:-0} 行 —— 观测面没接上,下面所有「没有命中」的断言都不可信"
fi
# 「token 不爆」的落点。两段的界限**不一样**,分开断:
#   * 层 1 是**硬**约束(`select_layer1` 用 trim_messages 卡死),任何一行超了都是缺陷;
#   * 层 2 允许**瞬时**超预算 —— 触发摘要那一轮的后台任务还没跑完,这一轮的窗口里
#     仍然装着那段超出预算的旧历史(设计如此,spec §3.3)。所以它只要求
#     「超预算的行数 <= 触发次数」,不能要求 0(那是把正常的异步语义当成缺陷)。
TRG=$(lget "$ST1" trigger)
MAXW=$(lget "$ST1" hc_window_max_tokens)
OVER1=$(lget "$ST1" mc_layer1_over); OVER2=$(lget "$ST1" mc_layer2_over)
if [ "${MC1:-0}" -ge 1 ] && [ "${OVER1:-1}" = "0" ]; then
  ok "层 1 从没超过它自己的预算(model_ctx $MC1 行全部合规;扁平窗口最大 ${MAXW:-0} token)"
else
  bad "有 model_ctx 行层 1 超预算($OVER1 行)—— 硬约束被破了"
fi
if [ "${OVER2:-1}" -le "${TRG:-0}" ]; then
  ok "层 2 只有触发摘要的那几轮瞬时超预算($OVER2 行 <= 触发 $TRG 次 —— 异步摘要的应有形状)"
else
  bad "层 2 超预算 $OVER2 行,多于触发次数 $TRG —— 有超预算却没有起摘要的轮次"
fi
# 服务真的是按 §9.4 那组参数起的吗?**读日志里实际算出来的数**去比生产代码现算的数。
LOG_B1=$(lget "$ST1" budget_layer1_logged); LOG_B2=$(lget "$ST1" budget_layer2_logged)
EXP_B1=$(eb_get "$EXP_A" layer1); EXP_B2=$(eb_get "$EXP_A" layer2)
if [ "$LOG_B1" = "$EXP_B1" ] && [ "$LOG_B2" = "$EXP_B2" ]; then
  ok "窗口参数与演示配置一致:日志报 layer1=$LOG_B1 / layer2=$LOG_B2(= 生产代码现算值)"
else
  bad "日志报的预算 ($LOG_B1/$LOG_B2) 与演示配置现算的 ($EXP_B1/$EXP_B2) 不一致 —— 这一轮验的不是 §9.4 那组配置"
fi
echo

# ══════════════════════════════════════════════════════════════════════════
echo "== 验收 2:演示配置下的完整级联(降级 → 触发 → 压缩)=="
DEG=$(lget "$ST1" degrade); DON=$(lget "$ST1" sum_done)
SKP=$(lget "$ST1" sum_skip); FLR=$(lget "$ST1" sum_fail)
if [ "${DEG:-0}" -ge 1 ]; then
  ok "级联第一环:层 1 超预算 → 降级 ${DEG} 次(第一次 $(lget "$ST1" degrade_first_from) → $(lget "$ST1" degrade_first_to),挪后层1=$(lget "$ST1" degrade_first_layer1_tokens) <= 预算 $(lget "$ST1" degrade_first_layer1_budget))"
else
  bad "一次降级都没有 —— 层 1 从没超过 $(eb_get "$EXP_A" layer1) token(二十轮里前八轮是长消息,不该超不过)"
fi
if [ "${TRG:-0}" -ge 1 ]; then
  ok "级联第二环:层 2 超预算 → 触发摘要 ${TRG} 次(第一次 layer2_tokens=$(lget "$ST1" trigger_first_layer2_tokens) > budget=$(lget "$ST1" trigger_first_layer2_budget))"
else
  bad "一次摘要触发都没有 —— 层 2 从没超过 $(eb_get "$EXP_A" layer2) token"
fi
# 触发**条件本身**在日志里成不成立。只断「有一行 trigger」是弱判据:把判据改成恒真
# (每次都触发)照样有一行 trigger。`trigger_worst_over` 是「超过预算多少」:
# **-1 表示压根没有 trigger 行**,必须与 0 分开读 —— 而 0 意味着「刚好等于预算就触发」,
# 那也是与实现不符的(代码用的是严格 >)。
OVR=$(lget "$ST1" trigger_worst_over)
if [ "${TRG:-0}" -ge 1 ] && [ "${OVR:-0}" -ge 1 ]; then
  ok "触发判据成立:层 2 的**截短后**用量确实超出预算 $OVR token(不是「恒真触发」)"
else
  bad "触发判据不成立(trigger=$TRG 行、超预算量=$OVR)—— 要么没触发,要么触发是恒真的"
fi
if [ "${DON:-0}" -ge 1 ]; then
  ok "级联第三环:摘要落库 ${DON} 次(第一次第 $(lget "$ST1" done_first_seq) 段,覆盖 ($(lget "$ST1" done_first_covered_from) → $(lget "$ST1" done_first_upto)],耗时 $(lget "$ST1" done_first_elapsed_ms) ms)"
else
  bad "没有 summary done —— 摘要没落成库(触发 $TRG 次,skip ${SKP:-0} 次,fail ${FLR:-0} 次)"
fi
[ "${SKP:-0}" -ge 1 ] && echo "  ── 另有 summary skip $SKP 次(区间为空 / 模型吐空,两个成因在日志里分开记)"
[ "${FLR:-0}" -ge 1 ] && echo "  ── 另有 summary fail $FLR 次(异常摘要,文本已过 redact_api_key,边界未推进)"

# 落点:靠梗概里的订单号答「最开始那个订单」。**唯一依赖模型判定的一条** ⇒ warn 档
# (spec §10.5:必须显式计数并打印,否则它就退化成一条悄悄跳过的检查)。
echo "  ── 落点(模型判定,warn 档):第 15 轮问「我最开始问的那个订单,现在物流到底是什么情况?」"
PROBE_REPLY=$(join_tokens < "$WORK/s1_15.sse")
HAS_ID=no; case "$PROBE_REPLY" in *20240915*) HAS_ID=yes;; esac
HAS_DIGEST=$(lget "$ST1" any_summary_has_needle)   # 0/1(来自 log_state;**任一** model_ctx 行置的,不是「最后一段」)
echo "     回复里出现订单号 20240915:$HAS_ID / 日志里**任一** model_ctx 的梗概含 20240915:$HAS_DIGEST(1=含)"
echo "     回复原文:$PROBE_REPLY"
# ⚠️ 比较的是 `1`,不是 `yes` —— 第 4 次跑时这里写错过(`any_summary_has_needle` 是 0/1),
# 于是「梗概里有、回复里没有」被打印成了「梗概与回复里都没有」:
# **判据写错,证据的读法就跟着错**,而两句话看起来都合理。
if [ "$HAS_ID" = "yes" ] && [ "$HAS_DIGEST" = "1" ]; then
  warn "落点**成立**(梗概与回复里都有 20240915)。**仍然不计为通过** —— 它是模型判定,不是代码判定(§10.5:必须显式计数并打印)。"
elif [ "$HAS_DIGEST" = "1" ]; then
  warn "落点**一半成立**:梗概里记着 20240915(日志直接可见),但第 15 轮的回复没有引用它。**模型行为**,不是代码缺陷;§11.6 明说摘要质量只能靠标注样例 + 人工抽查。"
else
  warn "落点**未复现**:梗概与回复里都没有 20240915。模型没把订单号抄进梗概 —— 同样是模型行为,必须显式记下来而不是悄悄跳过。"
fi
echo

# ══════════════════════════════════════════════════════════════════════════
echo "== 验收 4:摘要不阻塞该轮回复 + 两个上下文日志看得见 =="
NB=$(nonblock_pairs "$SID1" "$TS1")
PAIRS=$(lget "$NB" pairs_ok); MIN_GAP=$(lget "$NB" min_gap_ms)
echo "$NB" | sed -n 's/^/     /p'
if [ "${DON:-0}" -ge 1 ] && [ "${PAIRS:-0}" -ge 1 ]; then
  ok "回复不等摘要:有 $PAIRS 次「首帧先于该轮的 summary done」(最小间隔 ${MIN_GAP} ms)"
else
  bad "每一次 summary done 都早于它那一轮的首帧(done=${DON:-0} pairs_ok=${PAIRS:-0})—— 这一轮的回复被摘要挡住了"
fi
if [ "${MC1:-0}" -ge 1 ] && [ "${HC1:-0}" -ge 1 ]; then
  # `model_ctx` 是**每轮一行**(在 Agent 的 ReAct 循环之前打一次),不是「每次调模型
  # 前一行」—— 多步的那几轮里第 2、3 步模型调用没有独立的一行(spec §7.6 已订正)。
  # 所以这里的下界是 `>= 1 行/轮` 而不是相等,`$MC1` 因此会**小于**轮数。
  ok "两个上下文日志都在打:model_ctx=$MC1 行(每轮一行,见 spec §7.6 的订正)/ history_ctx=$HC1 行(每轮必打)"
else
  bad "上下文日志缺失(model_ctx=${MC1:-0} / history_ctx=${HC1:-0})"
fi
# 分段 token 的观测面。**注意别把两行的 `tokens.layer1` 拿来比** —— 同名不同义:
# `history_ctx` 数的是**整个扁平窗口**,`model_ctx` 只数层 1(journal 的 docstring 记着)。
# 只有 `model_ctx` 那一行带分段预算与锚点,所以分段观测面按它断。
DEG_T=$(lget "$ST1" degrade_first_layer1_tokens)
TRG_T=$(lget "$ST1" trigger_first_layer2_tokens)
if [ "${MC1:-0}" -ge 1 ] && [ -n "$DEG_T" ] && [ -n "$TRG_T" ]; then
  ok "日志里有分段口径的观测面:降级行报了挪后的层1 用量($DEG_T)与预算、触发行报了层2 用量($TRG_T)与预算"
else
  echo "  ── 注:分段观测面不全(降级行层1=$DEG_T 触发行层2=$TRG_T)"
fi
echo

# ══════════════════════════════════════════════════════════════════════════
echo "== 验收 4b:「history_ctx」那一行里看得见截短后的形态 =="
# ⚠️ **判在 `history_ctx` 这一行上,不是 grep 整个日志文件**:`model_ctx` 的 `sliding`
# 也带同一批截短形态,整文件 grep 会命中它,而 4b 指定的那条线(每轮必打、含不进
# Agent 的那几轮)可能一个字都没说 —— 那就成了一条**假通过**的断言。
# 另外判在 `sliding` 数组的**条目**上,不是整行文本:梗概正文里出现「…」不算数。
TA=$(lget "$ST1" hc_trunc_assistant); TT=$(lget "$ST1" hc_trunc_tool)
TL=$(lget "$ST1" hc_lines_with_trunc)
HT=$(lget "$ST1" hc_lines_with_tool)
if [ "${TA:-0}" -ge 1 ] && [ "${TT:-0}" -ge 1 ]; then
  ok "history_ctx 的 sliding 里有截短后的形态:客服答复带 … 的条目 $TA 个、工具结果一行化的 $TT 个(分布在 $TL 行里)"
else
  bad "history_ctx 的 sliding 里看不到截短形态(答复带 …:${TA:-0} / 工具结果一行化:${TT:-0})—— 截短在这条线上没生效,或分层没接上(T7 记过这个形态)"
fi
if [ "${TT:-0}" -ge 1 ] && [ "${HT:-0}" -ge 1 ]; then
  ok "带工具往返的 history_ctx 行有 $HT 行 —— 上面的「工具结果一行化」不是无源之水"
else
  bad "没有任何 history_ctx 行带 role=tool(工具结果没进过窗口)—— 「工具结果一行化」这一半验不到"
fi
echo

# ══════════════════════════════════════════════════════════════════════════
echo "== 验收 3:默认窗口下纯聊天二十轮,不触发任何降级与摘要(反向断言)=="
# 它防的是「保守起见每次都压一点」—— 那种实现能让验收 1/2/4 全绿。
# **必须换一个默认配置的服务**:层 1 预算从 4853 变成 3173,是更严的那一侧;
# 在演示配置下跑,这条断言会因为预算更宽而更容易成立(等于放宽了口径)。
echo "  (停掉演示配置的服务,用默认配置起 $PORT_B)"
stop_server
start_server "$PORT_B" "$WORK/server_b.log"
if ! wait_ready "$BASE_B" 60; then
  echo "预检失败:$BASE_B 60 秒内没起来。控制台输出(前 60 行 + 末 15 行):"
  "$PYTHON" -c '
import sys
path = sys.argv[1]
lines = open(path, "rb").read().decode("utf-8", "replace").splitlines()
head, tail = 60, 15
out = list(lines[:head])
if len(lines) > head + tail:
    out.append("... [中间省略 %d 行,全文见文件] ..." % (len(lines) - head - tail))
out += lines[-tail:] if len(lines) > tail else []
out.append("(共 %d 行)" % len(lines))
sys.stdout.buffer.write(("\n".join(out)).encode("utf-8") + b"\n")' "$WORK/server_b.log"
  echo "  (上面是节选;**完整控制台输出在 $WORK/server_b.log,失败不删**)"
  fail_exit
fi
echo "  服务已就绪(pid $SERVER_PID,默认窗口)"
ASK_BASE="$BASE_B"
TS3="$WORK/timing3.tsv"; : > "$TS3"
CHAT=("你好" "谢谢" "嗯嗯" "好的" "在吗" "辛苦了" "哈哈" "我先看看" "明白" "真好"
      "谢谢啦" "嗯" "知道了" "再见" "好嘞" "收到" "谢谢,麻烦你了" "嗯嗯好"
      "那我先这样" "好的,回头聊")
D3=0
for i in $(seq 1 20); do
  ask_timed "$SID3" "${CHAT[$((i-1))]}" "$WORK/s3_$i.sse" "$TS3"
  d=$(done_n "$WORK/s3_$i.sse")
  [ "$d" -ge 1 ] && D3=$((D3+1))
done
if [ "$D3" -eq 20 ]; then
  ok "纯聊天二十轮全部有 done 帧"
else
  bad "纯聊天二十轮里只有 $D3 轮有 done 帧"
fi
ST3=$(log_state "$SID3" "")
HC3=$(lget "$ST3" history_ctx)
# 正向对照:**先证明这一轮的日志是活的**,否则下面那串 0 全是恒真的。
# 判据用 history_ctx 而不是 model_ctx:纯聊天不进 Agent,`model_ctx` 结构性为 0 ——
# 拿它当对照就会把「这一类不走 Agent」读成「日志没写」。
if [ "${HC3:-0}" -ge 20 ]; then
  ok "对照:该会话 history_ctx=$HC3 行(二十轮都打了日志)"
else
  boom "该会话 history_ctx 只有 ${HC3:-0} 行 —— 日志没接上,下面「一次都没触发」的断言恒真,不可读"
fi
B3_1=$(lget "$ST3" budget_layer1_logged); B3_2=$(lget "$ST3" budget_layer2_logged)
E3=$(expected_budget)
E3_1=$(eb_get "$E3" layer1); E3_2=$(eb_get "$E3" layer2)
echo "  默认配置下生产代码现算的预算:$E3"
if [ "$B3_1" = "$E3_1" ] && [ "$B3_2" = "$E3_2" ]; then
  ok "8001 跑的确实是**默认**配置:日志报 layer1=$B3_1 / layer2=$B3_2(= 现算值)"
else
  bad "8001 报的预算 ($B3_1/$B3_2) 与默认配置现算的 ($E3_1/$E3_2) 不一致 —— 这一轮的反向断言不是在默认窗口下做的"
fi
D3DEG=$(lget "$ST3" degrade); D3TRG=$(lget "$ST3" trigger)
D3DONE=$(lget "$ST3" sum_done); D3START=$(lget "$ST3" sum_start)
D3SKIP=$(lget "$ST3" sum_skip); D3FAIL=$(lget "$ST3" sum_fail)
ZERO_ALL=yes
for v in "$D3DEG" "$D3TRG" "$D3DONE" "$D3START" "$D3SKIP" "$D3FAIL"; do
  [ "${v:-1}" = "0" ] || ZERO_ALL=no
done
if [ "$ZERO_ALL" = "yes" ]; then
  ok "装得下就不压:降级 0 次、触发 0 次、摘要 start/done/skip/fail 全 0(扁平窗口最大 $(lget "$ST3" hc_window_max_tokens) token < 层1 预算 $E3_1)"
else
  bad "纯聊天也动了上下文:降级=$D3DEG 触发=$D3TRG start=$D3START done=$D3DONE skip=$D3SKIP fail=$D3FAIL —— 「保守起见每次都压一点」"
fi
echo
stop_server

# ══════════════════════════════════════════════════════════════════════════
echo "════════════════════════════════════════════════════════════════════════"
echo "⚠️  验收 5(侧栏多会话 / 切回旧会话 / 接着聊)**只能人工验**,本脚本覆盖不到。"
echo "    脚本能验的边界到「服务端给了正确的数据」为止 —— 前端是否画出来、点击是否发对"
echo "    请求,由 T12 的浏览器探针(headless Chrome + 真库 324 条会话)覆盖过,不在本脚本里。"
echo "    人工步骤(约 2 分钟;先自己起服务):"
echo "      1) $PYTHON -m uvicorn app.main:app --port 8000"
echo "      2) 浏览器开 http://localhost:8000;左侧应列出若干会话(新在前),带预览与「已摘要」标记;"
echo "      3) 新开一个会话聊一句,它应立刻出现在侧栏顶部;"
echo "      4) 点侧栏里一个**旧会话**:历史原文应回载(不含工具载荷、不含空气泡);"
echo "      5) 在那个旧会话里接着发一句,回复应正常流式回来;"
echo "      6) 刷新页面:会话应还在(localStorage 里的 sessionId),没有跳回新会话。"
echo "    (本次脚本用过的两个会话 id:$SID1 / $SID3 —— 它们都在库里,可以直接点开看)"
echo "════════════════════════════════════════════════════════════════════════"
echo
echo "结果:$PASS 通过,$FAIL 失败,$WARN 项未复现(见上面的 WARN 行 —— 它们**没有**被验过,别当通过)"
[ "$FAIL" -eq 0 ]
