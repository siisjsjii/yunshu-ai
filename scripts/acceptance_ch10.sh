#!/usr/bin/env bash
# ch10 验收(spec §12 验收 ④;§11.4 那个「可测的数」)。
#
# ⚠️ **本文件是 ch10-A 建的,目前只有「转人工」那一节。** ch10-B 的任务会在它后面
#    **补上验收 ①–③ 三节**(主题分类器的 F1/分布页/多标签)。为此:
#      * helper 一律收在文件上半段、**不藏在某一节里**,后来的节可以直接用;
#      * 收尾的判词/局限清单是**分节**打印的,补节时各自加一段,不用动别节;
#      * `$WORK` 是 `.ch10_acceptance`(与 ch09 的 `.ch09_acceptance` 分开,互不覆盖)。
#
# 前置:**MySQL + 真实 key**。
#   * **不需要 Milvus**:转人工的路由值是 `HANDOFF → agent`(app/agent/graph.py),
#     这条路上**没有检索**(只有 `商品咨询` 走 `retrieve_knowledge` + 置信度闸)。
#     Milvus 没起只是 WARN,不是预检失败 —— 与 ch06/ch09 那两份**刻意不同**。
#   * **不需要 Langfuse**:本节的断言全在 SSE 帧与回复文本上,不读观测。
#
# 本脚本自己起三样东西:客服服务(8000)+ 两个 MCP Server(8101 物流 / 8102 售后,
# **尽力而为**:起不来只 WARN)。
#   * 客服服务必须自己起:「端口被旧进程占着」时 `wait_ready` 会连到**那个旧进程**上
#     并「通过」,于是整套验收测的是别的代码(本仓记过的「起服务前先查端口」)。
#   * 两个 MCP Server 起起来是为了让这一轮的工具集**与生产逐字相同**(内置 + MCP)。
#     起不来不影响结论:`transfer_to_human` 是**内置**工具(任务 2),
#     与 MCP 无关;少的只是 `query_logistics` 那类。
#
# ────────────────────────────────────────────────────────────────────────
# 本节断什么:**三层,缺一不可 —— 每一层失败的原因不同,所以分开报**
# ────────────────────────────────────────────────────────────────────────
#   ① 分类对     :done 帧的 intent == 转人工                (分类器的问题)
#   ② 工具被调了 :SSE 里有 transfer_to_human 的 tool_call 帧 (模型没调工具)
#   ②b 那次调用成了:同一个 `tool_call_id` 的 tool_result 是 `ok: true` (调了但失败了)
#   ③ 用户拿到了 :回复里有**工号 A###** 或**等待/排队**   (结果没被转述给用户)
#
# **只断 ① 的话,「分类对了但模型没调工具、用户什么也没得到」会全绿通过。**
#
# ⚠️ **③ 的口径是 spec 的字面口径,不是「只认工号」**(fix round 1,I1):
#   spec §12 验收 ④ 与 §11.4 写的是「工号**/等待时长**」—— 那个斜杠是**或**。
#   实测 9 句里 4 句只给了「当前排队第 2/3 位,预计等待约 7/8 分钟」而没报工号,
#   而那种回复**用户是拿到了结果的**。⇒ ③ 判红 = **两个 artefact 一个都没有**;
#   同时**两个子数分开打**(报出工号 / 只给了等待),不许合成一个 ——
#   合成会把「模型时时报工号、时时不报」这个不一致**平均掉**,而那正是要看的。
#
# ⚠️ **②b 不是装饰**(fix round 1,I2):没有它的话,「工具**失败了**(`ok:false`)、
#   而模型自己编了一个 `A###`」会判绿 —— ③ 只看回复文本时它看不出来。
#   真实结果必须由**帧**证明,且要**按 `tool_call_id` 配对**(`tool_result` 帧
#   不带工具名,一轮里可能有多个工具 —— 见 `tool_result_check` 的说明)。
#
# **第 ② 层失败是一个读数,不是一场事故**(spec §11.4):转人工走主力 Agent 意味着
# 「它发生不发生取决于模型记不记得调工具」—— 这是 ch06「模型只决定意图标签、不决定
# 走向」那条立身之本的**唯一一处例外**(已当面告知用户、用户知情选择)。结构上拦不住,
# 所以这里把它**测出来**:第 ② 层的通过率就是 spec §11.4 要的那个比例。
#
# ② 的判据是 **SSE 的 `tool_call` 帧**,不是回复里的措辞。
#   ⚠️ 「预计/等待」那类词是在断**模型的措辞**:模型完全可能把工具返回改写成
#   「客服马上接入,您的号码是 A205」,于是真实故障与措辞差异分不开 —— 而这条断言
#   的全部意义是回答「模型到底调工具了没有」,那是个**结构**问题,SSE 里有直接证据
#   (`app/agent/nodes.py` 的 `emit({"frame": "tool_call", "name": ...})`)。
#
# ────────────────────────────────────────────────────────────────────────
# 四条**装置**规矩(不遵守就会得到一条指向错方向的判词)
# ────────────────────────────────────────────────────────────────────────
#   A. **没有 done 帧 / 有 error 帧而没有 done 帧 ⇒ 判装置故障,不判「模型没调工具」。**
#      上游 502、Milvus/tool 基础设施故障都会让流里**没有 tool_call 帧** ——
#      照 ② 报出去就是「模型没调工具」,而真因是服务坏了。这是本仓点名的
#      「报错指向别处」,所以那两种先分出去(`check_handoff` 开头两支)。
#   B. **`has_needle` 是三个退出码,别把 1 与 2 混成一个非零**:
#      `0` = 针在;`1` = 文件读得动、针确实不在;`2` = 文件读不了/解不开(装置故障)。
#      本节的两条断言都是**肯定**断言(针必须在),所以用 `grep` 就够;
#      若要加否定断言,**必须**走 `assert_needle_absent`(它把 2 报成 boom)。
#      ⚠️ 同一条规矩适用于 `tool_result_check`(它也把「读不了」单列成 2)。
#   C. **不许直接 grep 原始 SSE 流**:回复逐 token 推送,`A205` 会被切成独立帧
#      (「A」「2」「05」),必须先用 `join_tokens` 拼回来。同理,`done` 帧的
#      `intent` 要从**帧载荷**里取(`intent_label`),不是整份文件 grep 一遍 ——
#      后者会被 `tool_result` 里回灌的 JSON 撞上。
#   D. **`intent_label` 交不出 intent(字面量 `?` 或**空串**)⇒ 判装置故障,
#      不判「分类器判错」**(fix round 1,I3)。done 帧在、但载荷里没有 `intent`
#      (上游是 `final.get("intent")`)或载荷不是 JSON(helper 崩了、stdout 空)时,
#      ① 会把装置故障报成「意图没落「转人工」」。`?` 与空串都**不在**九类闭集里
#      ⇒ 这一支不会误伤一次真的判错。
#   E. **`tool_call` 配不到 `tool_result`(或它根本不是 `ok:true`)⇒ 分开报**
#      (fix round 1,I2):是**装置问题**(配不到)就 boom,是**工具真失败**(ok:false)
#      就判 ③ 红 —— 后者**不许**被一个模型自己编出来的 `A###` 蒙过去。
#
# 平台陷阱(Windows + Git Bash,本仓在 ch02/ch05/ch08 各栽过):
#   1) **含中文的请求体不能走 curl 的 argv**(MSYS2 按 CP936 重编码,服务端只回
#      `error parsing the body`)⇒ 一律走 stdin heredoc(`ask`)。
#   2) **中文 needle 一旦要当命令行参数传出去就要用十六进制码点**。
#      * ASCII 的那两个(`transfer_to_human` 的子串、`A[0-9]{3}` 正则)走 argv 没问题;
#      * ③ 的「等待」「排队」**不走 argv** —— 它们经 `has_needle` 的 hex needle 比
#        (needle 在 python 内部由码点拼出),理由与码点见文件头的 `H_WAIT/H_QUEUE`;
#      * 唯一那处**中文比较**(intent 是不是「转人工」)用的是**脚本内的字面量直接比**,
#        不经 argv —— `scripts/acceptance_ch06.sh` 的 `intent_check` 就是这么写的
#        (它跑得通),而本脚本另有一道**编码自检**兜它(见 `check_handoff` 的①)。
#      ⚠️ 但那种比较**失败时长什么样要想清楚**:
#      它失败时会把实得值一起打出来(见 `check_handoff` 的 ① 分支),所以一次编码性
#      失败会表现为「期望值与实得值**看起来一样**却判红」—— 见到那种行,先怀疑装置。
#   3) **退出码要能区分「红」与「没跑」**:`boom`(装置故障)与 `bad` 分开计,
#      与 ch09 同款。空的/残缺的转录上,「没有这一帧」这类断言**恒真** ⇒
#      每个断言块先跑 `frames_sane`。
set -uo pipefail

cd "$(dirname "$0")/.." || exit 2
[ -f app/main.py ] || { echo "请在项目根目录运行本脚本(找不到 app/main.py)" >&2; exit 2; }

# ── 自我转录(收尾那条「输出干净性自检」的输入)────────────────────────────
# 形状与 ch09 同款:**父进程** `tee` 一份转录并等子进程(自己的另一份)跑完,用
# `PIPESTATUS[0]` 原样传回它的退出码。`tee` 是流式的 ⇒ 人照样实时看到输出。
# ⚠️ 转录放 `log/`(已 gitignore),**不放 `$WORK`** —— `$WORK` 开头会被 `rm -rf` 重建,
# 放进去等于让 tee 写一个已经被删掉的文件。
SELF_LOG="${CH10_SELF_LOG:-$PWD/log/acceptance_ch10_self.log}"
if [ -z "${CH10_SELF_LOG:-}" ]; then
  mkdir -p log
  export CH10_SELF_LOG="$SELF_LOG"
  bash "$0" "$@" 2>&1 | tee "$SELF_LOG"
  exit "${PIPESTATUS[0]}"
fi

# ── needle 一律十六进制码点(平台陷阱 2)。
#   * 「转人工」不当命令行参数传,只在①的装置自检里当**码点**比;
#   * 「等待」「排队」是③的**第二组** artefact(spec §12 验收 ④ 的「工号/等待时长」),
#     走 `has_needle`(needle 由码点在 **python 内部**拼出来,argv 上只有 ASCII 的
#     hex)⇒ 完全不碰「中文进 argv 会不会被 MSYS2 重编码」这个问题。
#     ⚠️ **不用 `grep -qE '等待|排队'`**:那种写法把中文放进了 argv ——
#     本仓在 curl 上栽过(MSYS2 按 CP936 重编码),而这里一旦重编码就是
#     「永远匹配不上」⇒ **9 句全判红**,一条查不出来的假红。
#     码点已用脚本自己的解码器核过:`chr(int(h,16))`。
H_HANDOFF="8f6c 4eba 5de5"                # 转人工
H_WAIT="7b49 5f85"                        # 等待
H_QUEUE="6392 961f"                       # 排队

PYTHON="${PYTHON:-.venv/Scripts/python.exe}"
PORT="${PORT:-8000}"
BASE="http://localhost:$PORT"
MCP_LOGISTICS_PORT="${MCP_LOGISTICS_PORT:-8101}"
MCP_AFTERSALES_PORT="${MCP_AFTERSALES_PORT:-8102}"
LOG="log/app.log"
WORK=".ch10_acceptance"

PASS=0; FAIL=0; WARN=0
# 读数(不进判词)里没通过**层**数。见下面 `check_handoff` 的 `mode` 说明。
PROBE_MISS=0
# 3 句里的几个读数:模型调了工具的句数 / 三层全过的句数 / 其中报出工号的句数 /
# 其中只给了等待·排队的句数。**后两个必须分开打**(见 §读数区那段注释)。
CALL_OK=0; ALL3_OK=0; H_ROUNDS=0; AGENT_NO_OK=0; WAIT_ONLY=0

ok()   { echo "  PASS  $1"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL  $1"; FAIL=$((FAIL+1)); }
# 未复现(不是失败,也不是通过)。**必须显式计数并打印** —— 否则它退化成一条
# 悄悄跳过的检查。
warn() { echo "  WARN  $1"; WARN=$((WARN+1)); }
# 装置坏了(该出现的东西没出现、探针跑不动)。比 FAIL 更严重:此时**所有**
# 「没有命中」的断言都恒真,必须响亮地说出来。
boom() { echo "  !!!!  $1"; FAIL=$((FAIL+1)); }

if [ ! -x "$PYTHON" ]; then
  echo "找不到 $PYTHON —— 请在项目根目录运行本脚本。" >&2
  exit 2
fi

rm -rf "$WORK"; mkdir -p "$WORK" log
CS_PID=""; MCP_LOG_PID=""; MCP_AFTER_PID=""
KEEP=0

cleanup() {
  # 先杀客服服务:它是唯一持有会话锁的,留着会挡住下一次运行。
  for pid in "$CS_PID" "$MCP_LOG_PID" "$MCP_AFTER_PID"; do
    [ -n "$pid" ] || continue
    kill "$pid" 2>/dev/null
  done
  sleep 1
  for pid in "$CS_PID" "$MCP_LOG_PID" "$MCP_AFTER_PID"; do
    [ -n "$pid" ] || continue
    kill -9 "$pid" 2>/dev/null
  done
  # 失败时**保留**工作目录(SSE 原文、三个服务的控制台)。
  # 判据是 `FAIL -gt 0` **或** `KEEP -eq 1` —— 不能只看 FAIL:几条 preflight
  # 分支在**一条断言都还没打**的时候就 `exit 1`(那时 FAIL 仍是 0),
  # 只认 FAIL 的话 cleanup 会把**唯一的**那份证据删掉。
  if [ "$FAIL" -eq 0 ] && [ "$KEEP" -eq 0 ]; then
    rm -rf "$WORK"
  else
    [ -n "${CH10_SELF_LOG:-}" ] && [ -f "$CH10_SELF_LOG" ] && \
      cp "$CH10_SELF_LOG" "$WORK/transcript.txt" 2>/dev/null
    echo "  (证据留在 $(pwd)/$WORK/)"
  fi
}
trap cleanup EXIT
# ⚠️ **外部打断必须自己 `exit`**:装了 trap 之后 bash 不会因为收到信号就退出 ——
# 信号处理函数返回后脚本**继续往下跑**(ch07 实测:TERM 到了 → cleanup 以
# `FAIL=0 && KEEP=0` 判定「这是一次成功」⇒ 把工作目录删掉,而脚本接着走到失败
# 分支,那里再读 SSE 就只剩 No such file or directory)。
on_signal() { KEEP=1; exit 130; }
trap on_signal INT TERM HUP

# preflight 失败的统一出口:**必须用它,不要直接 `exit 1`**(理由见 cleanup)。
fail_exit() { KEEP=1; exit 1; }

# ── 预检:三个端口必须是空的 ────────────────────────────────────────────
for p in "$PORT" "$MCP_LOGISTICS_PORT" "$MCP_AFTERSALES_PORT"; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://localhost:$p/")
  if [ "$code" != "000" ]; then
    echo "预检失败:端口 $p 上已经有服务在监听(返回 $code)。本脚本要自己起服务,请先清掉:"
    echo "  netstat -ano | grep ':$p'                                    # 记下 PID"
    echo "  powershell -NoProfile -Command \"Stop-Process -Id <PID> -Force\""
    exit 1
  fi
done

# ── 日志:先截断,再起服务(顺序反了就是「它在写、我在砍」)──────────────────
: > "$LOG"

# ══════════════════════════════════════════════════════════════════════════
# helper(与 ch08/ch09 逐字同款;ch10-B 补 ①–③ 节时直接用,不要再抄一份)
# ══════════════════════════════════════════════════════════════════════════

# 发一条消息并录 SSE。$1=sid $2=消息 $3=SSE 输出文件
# 中文请求体走 **stdin heredoc**,不走 argv(平台陷阱 1)。
# `--max-time 300`:上游卡住时不加这个,脚本会**无限期阻塞且什么都不打印**,
# 收尾时表现为「挂住」而不是「红」—— 那是查不出来的失败。
ask() {
  curl -s -N --max-time 300 -X POST "$BASE/api/chat/stream" \
    -H "Content-Type: application/json" \
    --data-binary @- > "$3" <<JSON
{"session_id":"$1","message":"$2"}
JSON
}
new_sid() { "$PYTHON" -c "import uuid;print(uuid.uuid4().hex)"; }

has_event() { grep -q "^event: $2\$" "$1"; }
has_done()  { grep -q "^event: done" "$1"; }
has_error() { grep -q "^event: error" "$1"; }

# 帧的自检。**每个断言块都先跑它** —— 「没有这一帧」这类断言在一份空文件上恒真。
frames_sane() {   # $1=文件 → 0/1
  [ -s "$1" ] && has_event "$1" meta && return 0
  return 1
}

# 帧序列(逗号分隔的**事件名**)。诊断用:一张卡片该出而没出时,这一行是最直接的证据。
frame_names() {
  "$PYTHON" -c '
import sys
names = []
for l in sys.stdin.buffer.read().decode("utf-8", "replace").splitlines():
    if l.startswith("event: "):
        names.append(l[7:].strip())
sys.stdout.buffer.write(",".join(names).encode("ascii"))'
}

# 出错的 error 帧文本(诊断用;已经过 redact_api_key)。
error_text() {
  "$PYTHON" -c '
import json, sys
lines = sys.stdin.buffer.read().decode("utf-8", "replace").splitlines()
for i, l in enumerate(lines):
    if l.strip() == "event: error" and i + 1 < len(lines) and lines[i+1].startswith("data: "):
        sys.stdout.buffer.write(json.loads(lines[i+1][6:]).get("message", "").encode("utf-8"))
        break'
}

# 把 token 帧拼回整段回复(逐 token 推送,不能直接 grep)。读 stdin。
# ⚠️ 一律按**字节**读写 stdin/stdout:Windows 上 Python 对管道默认用 ANSI 代码页
# (cp936)+ surrogateescape,`sys.stdin.read()` 会把 UTF-8 的 SSE 字节解成乱码与
# 孤立代理对,随后 `.encode("utf-8")` 抛 `UnicodeEncodeError` —— ch05 实测因此
# **在内容其实满足条件的情况下判失败**(假红)。
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

# done 帧里的 intent(**吃文件参数,不是 stdin**)。取不到时输出字面量 `?` ——
# 调用方必须先把「没有 done 帧」分出去,否则挂起/故障会被报成「意图判错」。
intent_label() {
  "$PYTHON" -c '
import json, sys
lines = open(sys.argv[1], "rb").read().decode("utf-8", "replace").splitlines()
intent = None
for i, l in enumerate(lines):
    if l.strip() == "event: done" and i + 1 < len(lines) and lines[i+1].startswith("data: "):
        intent = json.loads(lines[i+1][6:]).get("intent")
sys.stdout.buffer.write((intent or "?").encode("utf-8"))' "$1"
}

# SSE 里所有 `tool_call` 帧的工具名(空格分隔;没有则空串)。
#
# ⚠️ **为什么不看回复里的「预计/等待」**:那是在断**模型的措辞**。模型完全可能把
# 工具返回改写成「客服马上接入,您的号码是 A205」,于是真实故障与措辞差异分不开 ——
# 而这条断言的**全部意义**是回答「模型到底调工具了没有」,那是个**结构**问题,
# SSE 里有直接证据(`app/agent/nodes.py` 的 `emit({"frame": "tool_call", "name": ...})`)。
called_tools() {   # $1=文件
  "$PYTHON" -c '
import json, sys
lines = open(sys.argv[1], "rb").read().decode("utf-8", "replace").splitlines()
names = []
for i, l in enumerate(lines):
    if l.strip() == "event: tool_call" and i + 1 < len(lines) and lines[i+1].startswith("data: "):
        names.append(json.loads(lines[i+1][6:]).get("name", "?"))
sys.stdout.buffer.write(" ".join(names).encode("utf-8"))' "$1"
}

# 某个工具的**那一次调用到底成没成**。$1=文件 $2=工具名
#   stdout = 一行诊断串;退出码 **0**=配到且 `ok: true`;**1**=配到但**不是**全 ok;
#   **2**=没配到(没有该工具的 `tool_call` 帧,或没有它的 `tool_result` 帧)⇒ 装置/协议问题。
#
# ⚠️ **必须按 `tool_call_id` 配对,不能只看「流里出现过 ok:true」**:
#   `tool_result` 帧**不带工具名**(`app/agent/nodes.py` 的
#   `emit({"frame": "tool_result", "tool_call_id": …, "ok": …, "summary": …})`),
#   而一轮里可能有多个工具 ⇒ 不配对就会把**别的工具**的成功当成这次转接的成功。
# ⚠️ 这条存在的理由(I2):③ 只看回复文本时,**一次失败的调用 + 模型自己编的工号**
#   会判绿。真实结果必须由**帧**证明,不能由回复证明。
tool_result_check() {   # $1=文件 $2=工具名
  "$PYTHON" -c '
import json, sys
name = sys.argv[2]
# ⚠️ **读不了 ≠ ok:false**(与 `has_needle` 同一条规矩):不 try 的话,一个路径写错 /
# 文件被删会抛 FileNotFoundError ⇒ 进程退 1 ⇒ 被读成「这次调用失败了」——
# 装置故障冒充产品结论。文件读不了必须是 **2**。
try:
    lines = open(sys.argv[1], "rb").read().decode("utf-8").splitlines()
except (OSError, UnicodeDecodeError) as exc:
    print("读不了 %s: %s" % (sys.argv[1], exc)); raise SystemExit(2)
def frames(ev):
    out = []
    for i, l in enumerate(lines):
        if l.strip() == ev and i + 1 < len(lines) and lines[i+1].startswith("data: "):
            try:
                out.append(json.loads(lines[i+1][6:]))
            except ValueError:
                pass
    return out
ids = {c.get("tool_call_id") for c in frames("event: tool_call") if c.get("name") == name}
if not ids:
    print("没有 %s 的 tool_call 帧" % name); raise SystemExit(2)
res = [r for r in frames("event: tool_result") if r.get("tool_call_id") in ids]
if not res:
    print("有 %s 的 tool_call 帧,却配不到它的 tool_result 帧(共 %d 个 id)" % (name, len(ids)))
    raise SystemExit(2)
bad = [r for r in res if r.get("ok") is not True]
if bad:
    print("ok=false(%d/%d 条): %s" % (len(bad), len(res),
          json.dumps(bad[0].get("summary", ""), ensure_ascii=False)[:200]))
    raise SystemExit(1)
print("ok=true(%d 条)" % len(res)); raise SystemExit(0)' "$1" "$2"
}

# 读文件前 N 个**码点**(诊断用)。⚠️ 不用 `${var:0:N}`:bash 的下标切片是否按
# **字符**走取决于 `LC_CTYPE`,而本机 locale 是 cp936/ch05 记过的那一类 ——
# 切在半个 UTF-8 字符上会打出一串乱码,而那是**诊断行**,读到乱码的人会先怀疑产品。
head_chars() {   # $1=文件 $2=N
  "$PYTHON" -c '
import sys
text = open(sys.argv[1], "rb").read().decode("utf-8", "replace")
sys.stdout.buffer.write(text[:int(sys.argv[2])].encode("utf-8"))' "$1" "$2"
}

# 回复(拼回后)里有没有这个 needle。**needle 用十六进制码点传**(平台陷阱 2)。
# ⚠️ **退出码有三个,别把 1 与 2 混成一个非零**:0=针在;1=文件读得动、针确实不在;
# 2=**文件读不了/解不开**(装置故障)。
has_needle() {   # $1=文件 $2="空格分隔的码点"
  "$PYTHON" -c '
import sys
try:
    text = open(sys.argv[1], "rb").read().decode("utf-8")
except (OSError, UnicodeDecodeError):
    raise SystemExit(2)
needle = "".join(chr(int(h, 16)) for h in sys.argv[2].split())
raise SystemExit(0 if needle in text else 1)' "$1" "$2"
}
# 「针**不在**」这种**否定**断言。$1=文件 $2=hex $3=绿话 $4=红话(针在时)
#
# ⚠️ **本节的断言全是肯定断言,所以这条今天没人调用** —— 留着它是刻意的:
#    它是本脚本里**唯一**允许写否定断言的出口(理由见 `has_needle` 上面那段),
#    而 helper 块按文件头的约定是**共用**的(ch10-B 补 ①–③ 节时会用到)。
#    **别**为了「让它被用到」去加一条否定断言 —— 一条多余的断言比没有更坏。
assert_needle_absent() {
  has_needle "$1" "$2"; local rc=$?
  case "$rc" in
    0) bad "$4" ;;                    # 针在
    1) ok "$3" ;;                     # 文件读得动、针确实不在
    # ⚠️ **除 1 以外的非零一律当装置故障**,不做「反正不是 0,就当针不在」的兜底 ——
    # 那个兜底与「把 OSError 判成绿」同源:127(has_needle 根本没定义)、2(读不了)
    # 都会以「否定断言成立」的样子绿过去。
    2) boom "装置故障:读不了 $1 —— 这条否定断言判不了(不是「针不在」)" ;;
    *) boom "装置故障:has_needle 返回 $rc(只该是 0/1/2)⇒ 这条否定断言判不了" ;;
  esac
}

# ── 起服务 ──────────────────────────────────────────────────────────────
_kill_pid() { [ -n "$1" ] || return 0; kill "$1" 2>/dev/null; sleep 1; kill -9 "$1" 2>/dev/null; }
# **等端口真的空出来**。Windows 上监听套接字不是进程一死就归还的(实测:新进程会以
# `[Errno 10048] error while attempting to bind` 退出,而那条报错出现在**新进程**的
# 控制台里,读起来像「新起的服务坏了」)⇒ 轮询,不靠固定 sleep 猜。
wait_port_free() {   # $1=端口 $2=秒数
  local deadline=$((SECONDS + $2))
  while [ "$SECONDS" -lt "$deadline" ]; do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://localhost:$1/")" = "000" ] && return 0
    sleep 1
  done
  return 1
}
wait_ready() {   # $1=秒数
  local deadline=$((SECONDS + $1))
  while [ "$SECONDS" -lt "$deadline" ]; do
    [ "$(curl -s --max-time 5 -o /dev/null -w '%{http_code}' "$BASE/api/conversations")" = "200" ] && return 0
    sleep 1
  done
  return 1
}
wait_mcp() {     # $1=端口 $2=秒数(MCP 的 `/mcp` 对 GET 回 406 ⇒「非 000」即已监听)
  local deadline=$((SECONDS + $2))
  while [ "$SECONDS" -lt "$deadline" ]; do
    [ "$(curl -s --max-time 3 -o /dev/null -w '%{http_code}' "http://localhost:$1/mcp")" != "000" ] && return 0
    sleep 1
  done
  return 1
}
start_cs() {   # $1=控制台输出文件;其余是给这一次启动的 env 覆盖
  local out="$1"; shift
  # **`>>` 不是 `>`**:wait 会重试,重试时若把上一轮截掉,那条真正的错因
  # (`[Errno 10048] ... bind`)就没了 —— 而那正是要看的东西。
  env "$@" "$PYTHON" -m uvicorn app.main:app --port "$PORT" >> "$out" 2>&1 &
  CS_PID=$!
}
start_cs_ready() {   # $1=控制台输出文件;其余=env 覆盖
  local out="$1"; shift
  : > "$out"
  local attempt
  for attempt in 1 2 3; do
    start_cs "$out" "$@"
    if wait_ready 60; then
      [ "$attempt" -gt 1 ] && echo "  (第 $attempt 次启动才起来 —— 上一次的端口还没归还)"
      return 0
    fi
    _kill_pid "$CS_PID"; CS_PID=""
    wait_port_free "$PORT" 20
  done
  return 1
}
start_mcp_ready() {  # $1=模块 $2=日志 $3=端口 → 成功时 pid 在 MCP_NEW_PID
  : > "$2"
  local attempt
  for attempt in 1 2 3; do
    nohup "$PYTHON" -m "$1" >> "$2" 2>&1 &
    MCP_NEW_PID=$!
    if wait_mcp "$3" 30; then return 0; fi
    _kill_pid "$MCP_NEW_PID"
    wait_port_free "$3" 20
  done
  return 1
}
# 等预热。**看日志而不是打检索**(`?q=` 里的中文走 argv 会被重编码)。
# 等不到不致命 —— 本节的链路**不检索**,预热只是让服务稳定下来。
wait_warmup() {   # $1=秒数 → 0=预热完成 1=没等到 2=预热**失败**
  local deadline=$((SECONDS + $1))
  while [ "$SECONDS" -lt "$deadline" ]; do
    grep -q "预热完成" "$LOG" && return 0
    grep -q "预热失败" "$LOG" && return 2
    sleep 2
  done
  return 1
}
show_console_head_tail() {   # $1=文件(失败时打印头尾,不整份 cat)
  "$PYTHON" -c '
import sys
path = sys.argv[1]
lines = open(path, "rb").read().decode("utf-8", "replace").splitlines()
head, tail = 40, 15
out = list(lines[:head])
if len(lines) > head + tail:
    out.append("... [中间省略 %d 行,全文见文件] ..." % (len(lines) - head - tail))
out += lines[-tail:] if len(lines) > tail else []
out.append("(共 %d 行)" % len(lines))
sys.stdout.buffer.write(("\n".join(out)).encode("utf-8") + b"\n")' "$1"
}

# ══════════════════════════════════════════════════════════════════════════
# 转人工的三层判定(本节核心)
# ══════════════════════════════════════════════════════════════════════════

# 读数(不进判词)。$1 = 话。
probe_bad() { echo "  ⚠ 读数(不进判词)$1"; PROBE_MISS=$((PROBE_MISS+1)); }
probe_ok()  { echo "  读数  $1"; }

# 判一条 SSE **三层**。$1=标签(诊断用) $2=SSE 文件 $3=`judge`(承重,进判词)/ `probe`(只当读数)
# 返回:0=三层全过 1=① 失败 2=② 失败 3=③ 失败 4=装置故障(已 boom)
#
# ⚠️ **承重与读数走的是同一条判定路径**(只换报话人)。两条路各写一份的话,
#    读数就可能在判词之外**悄悄放宽**,而「读数」正是 spec §11.4 要的那个数。
check_handoff() {
  local label="$1" sse="$2" mode="$3" intent tools reply agent_no reply_file slug guard_rc
  local report_bad=bad report_ok=ok has_wait artefact tr_rc TR_DIAG rc_wait rc_queue
  slug=$(basename "$sse" .sse)                    # 文件名是 ASCII(调用方控制)
  reply_file="$WORK/reply_${slug}.txt"
  if [ "$mode" = "probe" ]; then report_bad=probe_bad; report_ok=probe_ok; fi

  # ── 装置:先分两种「不是模型没调工具」的病因 ──────────────────────────
  if ! frames_sane "$sse"; then
    boom "$label:SSE 转录不完整(没有 meta 帧,文件 $(wc -c < "$sse" | tr -d ' ') 字节)—— 这一轮判不了,不是「模型没调工具」"
    return 4
  fi
  if has_error "$sse" && ! has_done "$sse"; then
    boom "$label:端点发了 error 帧、没有 done 帧(上游/基础设施故障,不是「模型没调工具」):$(error_text < "$sse")"
    return 4
  fi
  if ! has_done "$sse"; then
    # 转人工这一轮**不该**挂起(`transfer_to_human` 是只读工具、不过确认流;
    # 这条路上也没有 `interrupt`)。没有 done 帧 = 流的形状与预期不符 ⇒ 装置问题。
    boom "$label:没有 done 帧(帧序列 $(frame_names < "$sse"))—— 这一轮判不了"
    return 4
  fi
  # 「error 与 done 同时在」是一种上层恢复过的流。**不当红** —— 判词说的三层
  # 它可能全过;但它也不该被静默吞掉,故 WARN 并打出 error 文本。
  if has_error "$sse"; then
    warn "$label:这一轮的流里**同时**有 error 帧与 done 帧(上层恢复过):$(error_text < "$sse")"
  fi

  intent=$(intent_label "$sse")
  tools=$(called_tools "$sse")
  join_tokens < "$sse" > "$reply_file"
  reply=$(cat "$reply_file")

  # ── 装置(I3):harvester **交不出** intent ───────────────────────────────
  # `intent_label` 的 docstring 自己写着「取不到时输出字面量 `?`,调用方必须先把
  # 这一支分出去」—— 而只查 `has_done` 是不够的:done 帧在、但它的载荷里没有
  # `intent`(上游的 `final.get("intent")` 可能是 None),或载荷根本不是 JSON
  # (helper 崩了、stdout 一个字节都没写 ⇒ 变量是**空串**)时,
  # `[ "$intent" != "转人工" ]` 会把**装置故障**报成「① 意图没落「转人工」」。
  # ⚠️ 判据不会误伤:`?` 与空串都**不在**九类闭集里 ⇒ 一次真的「判成别的标签」
  #    到不了这一支。
  if [ "$intent" = "?" ] || [ -z "$intent" ]; then
    boom "$label:装置故障:intent_label 从 done 帧取不出 intent(实得 [$intent])—— 这一轮判不了,不是「分类器判错」"
    return 4
  fi
  H_ROUNDS=$((H_ROUNDS+1))

  # ── ① 分类对(分类器的问题)────────────────────────────────────────────
  # 判据是**标签本身**(闭集里的原值),不是「路由出口」—— 判成「其他」也会
  # 同样拿到一个非空的 intent,所以「非空」证明不了任何事。
  if [ "$intent" != "转人工" ]; then
    # 装置自检(把一次**编码性**失败与一次**真的判错**分开):
    # 若实得标签的**码点**就是那三个字,却与脚本里的字面量不相等,那只能是本脚本/
    # 控制台的编码问题 —— 报红会指向「分类器坏了」,而那是指错方向。
    # ⚠️ 这条自检**不会误报**:`classify_intent` 只可能吐出九类原值或「其他」之一
    #    (`app/agent/nodes.py`:`if intent not in INTENT_TO_ROUTE: intent = OTHER`,
    #    而那是**字典键的精确比较**)⇒「带空格/带前缀」这类近似值到不了这里,
    #    码点里含「转人工」就只能是它本身。
    printf '%s' "$intent" > "$WORK/intent_${slug}.txt"
    # ⚠️ **1 与 2 必须分开**(与 `assert_needle_absent` 同规矩):把 2 当 1 处理,
    # 就是把一次「读不了」静默判成「码点不匹配 ⇒ 真的判错」—— 又是那条
    # 「装置故障不许冒充结论」。
    has_needle "$WORK/intent_${slug}.txt" "$H_HANDOFF"; guard_rc=$?
    case "$guard_rc" in
      0) boom "$label:装置故障:实得标签的**码点**就是「转人工」,却与本脚本的字面量不相等 —— 这是编码问题,不是分类器判错"; return 4 ;;
      1) : ;;                                   # 码点就不是那三个字 ⇒ 是真的判错
      2) boom "$label:装置故障:读不了刚写出的意图文件($WORK/intent_${slug}.txt)—— 这一轮判不了"; return 4 ;;
      *) boom "$label:装置故障:has_needle 返回 $guard_rc(只该是 0/1/2)—— 这一轮判不了"; return 4 ;;
    esac
    "$report_bad" "$label ① 意图没落「转人工」,实得:$intent(期望:转人工)"
    return 1
  fi
  # ── ② 工具被调了(模型没调工具 —— spec §11.4 的那个风险)───────────────
  if ! printf '%s' "$tools" | grep -qw transfer_to_human; then
    "$report_bad" "$label ② 意图是「转人工」但**模型没调 transfer_to_human** —— 用户什么也没得到。实际调用:[$tools]"
    return 2
  fi
  CALL_OK=$((CALL_OK+1))

  # ── ②b 那次调用**真的成了吗**(I2)────────────────────────────────────
  # ③ 只看回复文本时,**一次失败的调用 + 模型自己编出来的工号**会判绿 ——
  # 而「工具失败了、模型还编了号码」正是本仓最不该放过的那一类。真实结果必须由
  # **帧**证明:`tool_result` 的 `ok` 字段(且要**按 `tool_call_id` 配到**这次调用,
  # 一轮里可能有好几个工具)。
  TR_DIAG=$(tool_result_check "$sse" transfer_to_human); tr_rc=$?
  case "$tr_rc" in
    0) : ;;                                   # ok:true,继续看回复
    1) "$report_bad" "$label ③ 调了 transfer_to_human,但**那次调用的 tool_result 是 ok=false** —— 没有真结果可报(就算回复里出现了 A###,那也是模型自己编的)。$(printf '%s' "$TR_DIAG")"
       return 3 ;;
    2) boom "$label:装置故障:流里配不到 transfer_to_human 的 tool_result 帧($(printf '%s' "$TR_DIAG"))—— 这一轮判不了"; return 4 ;;
    *) boom "$label:装置故障:tool_result_check 返回 $tr_rc(只该是 0/1/2)—— 这一轮判不了"; return 4 ;;
  esac

  # ── ③ 用户拿到了(调了但没转述)──────────────────────────────────────
  # **判据是 spec 的字面口径**:回复里出现 **工号 `A###`** 或 **等待时长 / 排队位次**
  # —— 那个斜杠是「**或**」(spec §12 验收 ④ 与 §11.4 都写「工号/等待时长」)。
  # 实测(2026-09-25/26,本机 9 句):4 句只给了「当前排队第 2/3 位,预计等待约
  # 7/8 分钟」而没报工号 —— 那种回复**用户是拿到了结果的**,不该被判成产品坏了。
  # ⚠️ 所以这一层断的是「**结果有没有被转述**」,不是「模型有没有照抄某一个字段」;
  #    判红 = **两个 artefact 一个都没有**(既没工号、也没等待/排队)。
  agent_no=$(printf '%s' "$reply" | grep -oE 'A[0-9]{3}' | head -1)
  # 「等待」或「排队」:两串都走 `has_needle`(hex needle,见文件头)。**三值必须分开**:
  # 0=在 / 1=不在 / 2=文件读不了(装置故障)—— 在 `||` 链里把 2 当 1 就是
  # 「装置故障冒充结论」。
  has_wait=no
  has_needle "$reply_file" "$H_WAIT";  rc_wait=$?
  has_needle "$reply_file" "$H_QUEUE"; rc_queue=$?
  case "$rc_wait:$rc_queue" in
    0:*|*:0) has_wait=yes ;;
    1:1) : ;;                                   # 两串都确实不在 ⇒ 才是「没转述」
    *) boom "$label:装置故障:读不了回复文件(has_needle 返回 $rc_wait/$rc_queue)—— 这一轮判不了"; return 4 ;;
  esac
  if [ -z "$agent_no" ] && [ "$has_wait" = "no" ]; then
    "$report_bad" "$label ③ 调用成功了(ok=true),但回复里**既没有工号 A###、也没有等待/排队信息** —— 模型没把这次转接的结果转述给用户(这是**模型没转述**,不是产品没转出去)。回复前 200 字:$(head_chars "$reply_file" 200)"
    return 3
  fi
  ALL3_OK=$((ALL3_OK+1))
  # **两个子数分开记**(见下面读数区的说明):报出工号 / 只给了等待·排队。
  if [ -n "$agent_no" ]; then
    AGENT_NO_OK=$((AGENT_NO_OK+1)); artefact="工号=$agent_no"
  else
    WAIT_ONLY=$((WAIT_ONLY+1)); artefact="只给了等待/排队(**没报工号**)"
  fi
  "$report_ok" "$label 三层全过:intent=转人工,调了 transfer_to_human(ok=true),${artefact}"
  return 0
}

# ══════════════════════════════════════════════════════════════════════════
# 起环境
# ══════════════════════════════════════════════════════════════════════════
echo "== 起环境:客服服务($PORT)+ 两个 MCP Server(尽力而为)=="
if ! start_cs_ready "$WORK/cs.log"; then
  echo "预检失败:客服服务起不来。控制台输出:"
  show_console_head_tail "$WORK/cs.log"
  fail_exit
fi
echo "  客服服务已就绪(pid=$CS_PID)"
if start_mcp_ready mcp_servers.logistics "$WORK/mcp_logistics.log" "$MCP_LOGISTICS_PORT"; then
  MCP_LOG_PID="$MCP_NEW_PID"
  echo "  物流 MCP Server 已就绪($MCP_LOGISTICS_PORT,pid=$MCP_LOG_PID)"
else
  warn "物流 MCP Server($MCP_LOGISTICS_PORT)起不来 —— 继续(工具集少一个,但不含 transfer_to_human)"
fi
if start_mcp_ready mcp_servers.aftersales "$WORK/mcp_aftersales.log" "$MCP_AFTERSALES_PORT"; then
  MCP_AFTER_PID="$MCP_NEW_PID"
  echo "  售后 MCP Server 已就绪($MCP_AFTERSALES_PORT,pid=$MCP_AFTER_PID)"
else
  warn "售后 MCP Server($MCP_AFTERSALES_PORT)起不来 —— 继续"
fi

# 预热(BGE-M3)。等不到不致命(本节链路**不检索**),但要说清楚。
wait_warmup 180
case $? in
  0) echo "  BGE-M3 预热完成" ;;
  2) warn "BGE-M3 预热**失败**(日志里有那条)—— 本节不检索,不影响本节的结论" ;;
  *) warn "180s 内没等到预热完成 —— 本节不检索,不影响本节的结论" ;;
esac

# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "== 验收 ④(★ 承重句):「我要转人工」端到端拿到工号 =="
# 承重 = **这一句**的判词进 PASS/FAIL。spec §12 验收 ④ 说的就是这句话。
# 三层缺一不可,理由见文件头。
SID_H=$(new_sid)
ask "$SID_H" "我要转人工" "$WORK/handoff_main.sse"
# 判词由 `check_handoff` 自己打(ok/bad/boom),返回值这里不用 —— `set -uo pipefail`
# 下非零返回不会中断脚本,而它已经在该报的级别上报过了。
check_handoff "承重句" "$WORK/handoff_main.sse" judge
echo "  ── 诊断:intent=$(intent_label "$WORK/handoff_main.sse") 调用=[$(called_tools "$WORK/handoff_main.sse")] 帧=[$(frame_names < "$WORK/handoff_main.sse")]"
echo

echo "== 附加读数:同性质的 2 句换说法(**不进判词**,spec §11.4 要的比例) =="
# ⚠️ **为什么加这两句**:第 ② 层的失败率是**一个比例**,而单句样本给不出比例
# (spec §11.4 的原文是「模型偶尔不调工具的比例,是写进报告的一个数」)。
# ⚠️ **为什么不把它们也做成承重**:spec §12 验收 ④ 的题面是「一句话『我要转人工』」
# —— 把验收的通过与否挂到 3 次模型抽样上,红/绿就由模型运气决定,而不是由产品决定。
# 这两句的每一层都在**同一张表**上如实打出来(`probe_bad` 计 PROBE_MISS),
# 并在收尾的判词区**再提一次** —— 「不进判词」不等于「可以不读」。
SID_H2=$(new_sid)
ask "$SID_H2" "帮我转接一下人工客服" "$WORK/handoff_b.sse"
check_handoff "第2句" "$WORK/handoff_b.sse" probe
SID_H3=$(new_sid)
ask "$SID_H3" "有真人在吗,这问题我想找人处理" "$WORK/handoff_c.sse"
check_handoff "第3句" "$WORK/handoff_c.sse" probe

echo ""
echo "  ── 转人工的几个数(可判 $H_ROUNDS/3 句)──"
echo "     **模型调了 transfer_to_human** 的句数:$CALL_OK/$H_ROUNDS   ← spec §11.4 的那个比例"
echo "     结果真的转出去了(意图 + 调用且 ok=true + 回复里有工号或等待)的句数:$ALL3_OK/$H_ROUNDS"
# ⚠️ **两个子数必须分开打,不许合成一个**:③ 的口径是「工号**或**等待」(spec 的字面),
# 但「报了工号」与「只给了等待」是**两种不同的回复**,合成一个数会把模型的不一致
# **平均掉** —— 而那个不一致正是本节要看的东西(实测 9 句里 4 句只给了等待)。
echo "       其中 **回复里报出工号** 的:$AGENT_NO_OK/$H_ROUNDS"
echo "       其中 **只给了等待/排队**(没报工号)的:$WAIT_ONLY/$H_ROUNDS   ← 模型没报工号,但用户拿到了结果"
if [ "$H_ROUNDS" -lt 3 ]; then
  boom "只有 $H_ROUNDS/3 句可判 —— 其余几轮的转录是坏的,比例读数的分母不完整"
fi
echo

# ══════════════════════════════════════════════════════════════════════════
# 输出干净性自检 —— **判词之前**跑,让「装置自己喷错误行」不再可能与「通过」共存。
#
# 判据只有三条 ASCII 串(都用 grep -E,不会与中文判词撞车)。它自己**可能红**:
# 把任意一段这样的行进转录(或让脚本自己在输出里喷一行),这条就 `boom`。
# ⚠️ 与 ch09 那版的一处差别:那条**绿判词的文案**原先把这三个串写进了输出 ——
# 于是同一份转录上再跑第二次会**自己咬自己**(ch09 已记账)。这里不写出那三个串。
# ══════════════════════════════════════════════════════════════════════════
output_cleanliness_check() {
  echo ""
  echo "== 输出干净性自检(装置自己喷错误行,就不许与「通过」共存)=="
  if [ -z "${CH10_SELF_LOG:-}" ] || [ ! -f "$CH10_SELF_LOG" ]; then
    # 只有绕过文件头那个 tee 包装才会走到这里(手工设了 CH10_SELF_LOG 之类)。
    # **不静默**:说清「这一条本轮跑不了」,而不是让它悄悄绿着。
    warn "没有转录文件(${CH10_SELF_LOG:-未设置})⇒ 这条自检本轮跑不了"
    return
  fi
  # **屏障**:先打一个哨兵行,再**轮询转录直到哨兵出现** —— 那说明 tee 已经追平,
  # 之后扫到的就是**本轮全部**输出。不靠 sleep 猜(猜短了会漏、猜长了白等)。
  local sentinel="ch10-selfcheck-sentinel-$$-$RANDOM" i hits
  echo "$sentinel"
  for i in $(seq 1 50); do
    grep -qF "$sentinel" "$CH10_SELF_LOG" && break
    sleep 0.1
  done
  if ! grep -qF "$sentinel" "$CH10_SELF_LOG"; then
    boom "转录里没有刚才那行哨兵 ⇒ tee 没追上(文件被别的东西截断?),这条自检可信度不足"
  fi
  hits=$(grep -nE 'command not found|syntax error|unexpected EOF' "$CH10_SELF_LOG" | head -5)
  if [ -n "$hits" ]; then
    boom "输出里出现了 shell 级错误行(装置自己喷的)—— 不许与「通过」共存:"
    printf '%s\n' "$hits" | sed 's/^/      /'
  else
    ok "输出干净(转录里没有 shell 级错误行)"
  fi
}
output_cleanliness_check

# ══════════════════════════════════════════════════════════════════════════
# ⚠️ **判词必须打在上面那条自检的后面**(审查 I2):`output_cleanliness_check`
# 里的 `ok` / `boom` 会**改 `PASS`/`FAIL`**,而它下面这两段要读那两个数。
# 原先它打在前面 ⇒ 印出来的总数**永远少一条**;更难看的是一种自相矛盾的转录:
# 自检 `boom` 之后 `失败` 仍是 `0`,而脚本以 1 退出 ——
# **转录就是本节的证据本体**,一份自相矛盾的证据比没有证据更糟。
# 顺带:上面那段注释原本写着「判词之前跑」,而代码里它在判词**之后** ——
# 挪过来之后,那句话与代码终于一致了(注释没错,错的是顺序)。
# ══════════════════════════════════════════════════════════════════════════
echo "════════════════════════════════════════════════════════════"
echo "通过 $PASS / 失败 $FAIL / 未复现 $WARN"
echo "  ④ 转人工端到端   见 $WORK/handoff_main.sse(承重句)/ handoff_b.sse / handoff_c.sse(读数)"
echo ""
echo "== 局限(「通过」不等于这些也被验过)—— 如实列在这里,不许读成全绿 =="
# ⚠️ **这几行一律用单引号**:里面有反引号(在双引号里会被 bash 当**命令替换**执行 ——
# ch09 实测踩过,喷了一屏 `import: command not found` 而脚本照样打「通过」)。
echo '  * **承重的是 1 句**。另外 2 句是读数(不进判词)⇒ 「承重句过了」**不等于**'
echo '    「转人工稳定」,后者要看上面那几个数(3 句里调了几次、全过几次、报了几次工号)。'
echo '  * **第 ② 层失败的成因是结构性的,不是本脚本能修的东西**:转人工走主力 Agent'
echo '    意味着「发生不发生」取决于模型记不记得调工具,而模型只被要求决定**意图标签**。'
echo '    这是 ch06「模型只决定标签、不决定走向」的唯一一处例外(spec §11.3/§13.5),'
echo '    **没有结构保证,只有这个比例**。'
echo '  * **③ 的口径是「工号**或**等待/排队」(spec 的字面),它证明的是「结果被转述了」**;'
echo '    「模型报不报工号」是**另一个**读数,已单列(报出工号 / 只给等待),**没有合成一个数**。'
echo '  * **「投诉 vs 转人工」的边界不在本脚本覆盖内**:那两条边界负例在'
echo '    `evals/intent_cases.jsonl` 上由 `scripts/run_intent_eval.py` 测(打网络、不进验收)。'
# ⚠️ **这一行原先自相矛盾**(审查 M3):标题说「确实不留痕」,正文说「留痕由
# `conversations.status` 与建单路径负责」—— 而 `conversations.status` 至今只有
# **一个**写方(`app/tools/builtin/tickets.py` 的 `create_ticket`),它要用户
# **另外**要建工单才触发;只说「转人工」时那一列一动不动。改成实际的样子:
echo '  * **「转人工之后会话有没有留痕」不在覆盖内**:`transfer_to_human` **刻意不落库**'
echo '    (不加 session 依赖)⇒ 转人工这一轮**没有自己的业务行**(`conversations.status`'
echo '    只有建工单那条路会写,而用户没有要工单)。这轮唯一的痕是**执行器为这次调用'
echo '    落的 `tool_audit_logs` 行**(`status=success`)—— 它是痕,但不是业务留痕,'
echo '    而本脚本**不断那张表**。'
echo '  * **「工号是不是真的」不在覆盖内**:②b 断的是那次调用的 `tool_result` 是 `ok:true`'
echo '    (⇒ 编号来自工具、不是模型编的),但**没有**核对回复里那个 `A###` 与'
echo '    `tool_result` 里的 `agent_no` **是同一个**。模型理论上可以报出另一个合法的工号'
echo '    —— 概率极低、且今天没有任何证据,但**这一条确实没被断**。'
echo '  * **前端那个按钮不在覆盖内**:投诉出口的「转人工」按钮改走真实路径是 ch10-A'
echo '    的另一件任务(Vibe Coding),本节走的是**接口**,不点按钮。'
if [ "$PROBE_MISS" -gt 0 ]; then
  echo "  ⚠ **本节有 $PROBE_MISS 层读数没通过**(不进判词,但必须被读):见上面那几行 ⚠"
fi

if [ "$FAIL" -eq 0 ]; then
  echo ""
  echo "转人工那一节通过(④);其余几节由 ch10-B 的验收补"
else
  echo ""
  echo "有失败项 —— 证据留在 $WORK/,**不许把它读成全绿**"
fi
[ "$FAIL" -eq 0 ] || exit 1
