#!/usr/bin/env bash
# ch08 验收 1–6(端到端)。前置:MySQL 在跑、`.env` 里有真实 key。
#
# **本脚本自己起三样东西**:两个 MCP Server(8101 物流 / 8102 售后)与客服服务(8000)。
# 三个都必须自己起,各有各的理由:
#   * 验收 3 要**只重启其中一个 Server**再看客服服务的进程号有没有变 ——
#     Server 的生死必须捏在脚本手里;
#   * 验收 1 要**重启客服服务**,进程号同样得自己记;
#   * 端口被别人占着时,`wait_ready` 会连到**那个旧进程**上并「通过」,
#     于是整套验收测的是别的代码(本仓记过的「起服务前先查端口」)。
#
# 断言依据一律是**确定性的东西**:SSE 帧的名字与载荷、`tool_audit_logs` 里
# 本次会话的行、以及 MCP Server 的 `tools/list` 原始返回。**不靠模型自由文本**
# (deepseek 在 temperature=0 下依然非确定)—— 唯一读模型产出的是「回复里有没有
# 那句话 / 工单号」,而那两条正是本章要验的用户可见结果。
#
# ── 沿用的五条(ch07 那份 892 行的脚本是模板)────────────────────────────
#   1. **`KEEP` 与 `FAIL` 分离**,`fail_exit()` 先置 `KEEP=1` **再**退出 ——
#      失败路径不许删自己的证据;
#   2. **`EXIT` 与 `INT TERM` 分开 trap**;
#   3. **中文 needle 用 `chr()` 从十六进制码点构造**,不写进脚本字面量
#      (凡把中文当**命令行参数**传出去都会走 argv,MSYS2 按 CP936 重编码 ⇒
#       **沉默地匹配不上**,永远判「没有」);
#   4. **`wait_ready` 用墙钟 + `curl --max-time`**,不用固定 `sleep`;
#   5. **断言前先 `join_tokens` 把 SSE 帧拼回**(逐 token 推送会把 `7391` 切碎)。
#
# ── 本章特有的一条硬要求 ────────────────────────────────────────────────
# **查 `tool_audit_logs` 必须带本次会话的 `conversation_id` 过滤。**
# 整套测试每轮往那张表写 ~50 行只追加的记录(实测跑本脚本之前库里已有 1243 行),
# 按「查最近这几条」去查会撞上谁全看运气 —— 表现是**偶尔红、偶尔绿**。
# 脚本自己知道本次的 `session_id`,过滤它是顺手的。
#
# ── 两处临时改文件 ─────────────────────────────────────────────────────
# 验收 1(新增内置工具)与验收 3(给 MCP Server 加工具)**都要临时改文件**,
# 用 `EXIT` trap 保证**无论成败都还原**。还原走 git:
#   * `mcp_servers/logistics.py` 是**已跟踪**文件 ⇒ `git checkout --` 逐字还原;
#   * 新加的那个模块**不在 HEAD 里**,`git checkout` 对它无解,用 `git clean -f`。
#     ⚠️ `git clean` 在路径不存在时**静默成功**,所以后面补一条 `[ ! -e ]` 断言
#     —— 还原失败必须**响亮**,那正是这条规矩的要点。
#
# 平台陷阱(本仓在 ch02/ch05/ch06/ch07 各栽过,这里逐条防):
#   1) **含中文的请求体不走 curl 的 argv**:一律走 stdin heredoc;
#   2) **`python -c` 的片段源码里不许出现中文**(那段源码走 argv,同样会被重编码),
#      需要中文一律写 `\uXXXX` 转义 —— 或干脆**写进文件**(本脚本的 `dbq.py`
#      与临时模块都走 heredoc 落盘,读的是文件不是 argv);
#   3) **不起管道截断**:要落盘就整段落盘。
set -uo pipefail

cd "$(dirname "$0")/.." || exit 2
[ -f app/main.py ] || { echo "请在项目根目录运行本脚本(找不到 app/main.py)" >&2; exit 2; }

PYTHON="${PYTHON:-.venv/Scripts/python.exe}"
PORT="${PORT:-8000}"
BASE="http://localhost:$PORT"
MCP_LOGISTICS_PORT="${MCP_LOGISTICS_PORT:-8101}"
MCP_AFTERSALES_PORT="${MCP_AFTERSALES_PORT:-8102}"
LOG="log/app.log"
WORK=".ch08_acceptance"

# 两处临时改动(# 见头部说明)。
TRACKED_TMP="mcp_servers/logistics.py"
UNTRACKED_TMP="app/tools/builtin/zz_echo_note.py"

# 验收 1 的回显暗号。**中文部分一律用十六进制码点**(规矩 3)——
# 它同时被送进 heredoc 的请求体(安全通道)与用作 `has_needle` 的判据(argv 通道)。
PHRASE_HEX="8584 8377 5c71 4e18 2d 37 33 39 31"     # 薄荷山丘-7391
PHRASE_HOST='薄荷山丘-7391'

PASS=0; FAIL=0; WARN=0
ok()   { echo "  PASS  $1"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL  $1"; FAIL=$((FAIL+1)); }
# 未复现(不是失败,也不是通过):只在「判定模型这一轮给出的结论」决定某条断言判什么时用。
# **必须显式计数并打印**,否则它就退化成一条悄悄跳过的检查。
warn() { echo "  WARN  $1"; WARN=$((WARN+1)); }
# 装置坏了(该出现的东西没出现、日志读不了、SQL 探针跑不动)。它比 FAIL 更严重:
# 此时**所有**「没有命中」的断言都恒真,必须响亮地说出来,而不是让它们静静地绿着。
boom() { echo "  !!!!  $1"; FAIL=$((FAIL+1)); }

if [ ! -x "$PYTHON" ]; then
  echo "找不到 $PYTHON —— 请在项目根目录运行本脚本。" >&2
  exit 2
fi

rm -rf "$WORK"; mkdir -p "$WORK" log
CS_PID=""
MCP_LOG_PID=""
MCP_AFTER_PID=""
RESTORE_BAD=0
KEEP=0

# ── 还原:无论成败都要跑 ────────────────────────────────────────────────
restore_tmp() {
  # ① 已跟踪文件:git checkout 逐字还原。它**失败会响亮报错**(路径不在 HEAD 里
  #    / 工作区锁住 / 权限),这正是选它而不选「手写删文件」的理由。
  if ! git checkout -- "$TRACKED_TMP" > "$WORK/restore_checkout.log" 2>&1; then
    echo "  !!!! 还原 $TRACKED_TMP 失败(git checkout 报错):"
    cat "$WORK/restore_checkout.log"
    RESTORE_BAD=1
  fi
  # ② 新加的文件:HEAD 里没有这个路径,git checkout 对它无解。
  #    ⚠️ 但**证据要先留下来** —— 否则「验收 3 失败」时没人看得到当时的补丁长什么样。
  if [ -e "$UNTRACKED_TMP" ]; then
    cp "$UNTRACKED_TMP" "$WORK/zz_echo_note.py.snapshot" 2>/dev/null
    git clean -fq -- "$UNTRACKED_TMP" > "$WORK/restore_clean.log" 2>&1 \
      || { cat "$WORK/restore_clean.log"; RESTORE_BAD=1; }
  fi
  if [ -e "$UNTRACKED_TMP" ]; then
    echo "  !!!! 临时内置工具没删掉:$UNTRACKED_TMP"
    RESTORE_BAD=1
  fi
}

cleanup() {
  # 停三个进程。**先杀客服服务**:它是唯一持有会话锁的,留着它会挡住下一次运行。
  for pid in "$CS_PID" "$MCP_LOG_PID" "$MCP_AFTER_PID"; do
    [ -n "$pid" ] || continue
    kill "$pid" 2>/dev/null
  done
  sleep 1
  for pid in "$CS_PID" "$MCP_LOG_PID" "$MCP_AFTER_PID"; do
    [ -n "$pid" ] || continue
    kill -9 "$pid" 2>/dev/null
  done
  restore_tmp
  # 失败时**保留**证据目录(SSE 原文、两个服务的控制台输出、临时补丁的快照);
  # 通过时才清掉。判据是 `FAIL -gt 0` **或** `KEEP -eq 1`,**不能只看 FAIL**:
  # 几条 preflight 分支在**一条断言都还没打**的时候就 `exit 1`(那时 FAIL 仍是 0),
  # 只认 FAIL 的话 `cleanup()` 会把工作目录连同**唯一的**那份控制台日志一起删掉
  # —— 而那几条恰恰是最可能真触发的路径,也就是**最需要证据的时候**。
  if [ "$FAIL" -eq 0 ] && [ "$KEEP" -eq 0 ] && [ "$RESTORE_BAD" -eq 0 ]; then
    rm -rf "$WORK"
  else
    echo "  (证据留在 $(pwd)/$WORK/ —— SSE 原文、服务控制台输出、临时补丁快照)"
  fi
  if [ "$RESTORE_BAD" -ne 0 ]; then
    echo "  !!!! 工作区**没有**还原干净 —— 下一次运行会从被污染的状态起步。"
    echo "       人工收尾:git checkout -- $TRACKED_TMP ; rm -f $UNTRACKED_TMP"
    exit 1
  fi
}
trap cleanup EXIT
# ⚠️ **外部打断(Ctrl-C / `timeout` 发的 TERM)必须自己 `exit`**:装了 trap 之后
# bash **不会**因为收到信号就退出 —— 信号处理函数返回后,脚本**继续往下跑**。
# (ch07 实测:TERM 到了 → `cleanup` 以 `FAIL=0 && KEEP=0` 判定「这是一次成功」
#  ⇒ **把工作目录删掉**;然后脚本接着走到失败分支,那里再读 SSE 就只剩
#  「No such file or directory」—— 证据在它被需要的前一刻被自己删了。)
on_signal() { KEEP=1; exit 130; }
trap on_signal INT TERM

# preflight 失败的统一出口:**必须用它,不要直接 `exit 1`**(理由见 `cleanup`)。
fail_exit() { KEEP=1; exit 1; }

# ── 预检:三个端口必须是空的 ────────────────────────────────────────────
for p in "$PORT" "$MCP_LOGISTICS_PORT" "$MCP_AFTERSALES_PORT"; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://localhost:$p/")
  if [ "$code" != "000" ]; then
    echo "预检失败:端口 $p 上已经有服务在监听(返回 $code)。本脚本要自己起服务,请先清掉:"
    echo "  netstat -ano | grep ':$p'                                    # 记下 PID"
    echo "  powershell -NoProfile -Command \"Stop-Process -Id <PID> -Force\""
    # 这一支用普通 exit:此刻工作目录是空的(一条断言都没打),没有证据可留。
    exit 1
  fi
done

# ── 日志:先截断,再起服务(顺序反了就是「它在写、我在砍」)──────────────────
: > "$LOG"

# ── DB 探针(只读)──────────────────────────────────────────────────────
# 写成文件而不是 `python -c`:里面有中文注释,而 `-c` 的源码走 argv 会被重编码。
cat > "$WORK/dbq.py" <<'PYEOF'
"""验收脚本的只读 DB 探针。**每个 mode 都按 conversation_id 过滤。**

理由(本章硬要求):`tool_audit_logs` 是只追加的,整套测试每轮往里写 ~50 行。
按「最近这几条」查会变成偶尔红、偶尔绿 —— 那正是本仓记过的「被上次运行的数据
污染」这一类假绿。
"""

import asyncio
import sys

from sqlalchemy import text

from app.db.base import get_engine

ROWS = (
    "SELECT tool_name, source, status, retry_count, duration_ms "
    "FROM tool_audit_logs WHERE conversation_id = :s {extra} ORDER BY id"
)
#: 「只读工具」的判据:**不是 create_ticket 的每一条**。
#: 本章的写工具全集就只有它一条(`app/tools/policy.py` 的 WRITE_TOOLS),
#: 所以这个补集与「kind=read」等价 —— 且不依赖脚本侧另记一份工具清单。
NOT_WRITE = "AND tool_name <> 'create_ticket'"


async def main() -> None:
    mode, sid = sys.argv[1], sys.argv[2]
    out = sys.stdout.buffer
    engine = get_engine()
    try:
        async with engine.connect() as conn:
            if mode == "tickets":
                r = await conn.execute(
                    text("SELECT COUNT(*) FROM tickets WHERE conversation_id = :s"),
                    {"s": sid},
                )
                out.write(("%d\n" % r.scalar()).encode("ascii"))
            elif mode == "last-ticket-no":
                # ⚠️ **`tickets` 表没有自增 `id`** —— 它的主键是业务工单号
                # `ticket_no`(`app/db/models.py::Ticket` 的 docstring 写着这条)。
                # 照别的表的样子写 `ORDER BY id DESC` 会得到
                # `DBQ-ERROR OperationalError: Unknown column 'id'`,
                # 而那个串**非空**,于是「回复里有没有工单号」那一条会以
                # 「回复里没有」的样子红掉 —— 现象与错因完全不在一个地方(实测踩过)。
                r = await conn.execute(
                    text(
                        "SELECT ticket_no FROM tickets WHERE conversation_id = :s "
                        "ORDER BY created_at DESC, ticket_no DESC LIMIT 1"
                    ),
                    {"s": sid},
                )
                row = r.first()
                out.write(((row[0] if row else "") + "\n").encode("utf-8"))
            else:
                extra = NOT_WRITE if mode == "read-rows" else ""
                r = await conn.execute(text(ROWS.format(extra=extra)), {"s": sid})
                for name, source, status, retries, ms in r:
                    out.write(
                        ("%s|%s|%s|%d|%d\n" % (name, source, status, retries, ms))
                        .encode("utf-8")
                    )
    except Exception as exc:                                # noqa: BLE001
        # **不许静默** —— 探针跑不动时下面所有 grep 都会落空,
        # 而「落空」在 grep 的世界里与「确实没有那一行」长得一模一样。
        out.write(("DBQ-ERROR %s\n" % type(exc).__name__).encode("ascii"))
        raise SystemExit(3)


asyncio.run(main())
PYEOF
# `PYTHONPATH="$PWD"` 是**必须的**:按**路径**跑一个脚本时,Python 把**脚本所在目录**
# (`$WORK`)塞进 `sys.path`,而不是当前目录 —— `from app.db.base import ...` 于是
# `ModuleNotFoundError: No module named 'app'`。`python -c` 不会这样(CWD 在 sys.path 上),
# 所以本仓既有的那些内联片段从来没有暴露过这条。
dbq() { PYTHONPATH="$PWD" "$PYTHON" "$WORK/dbq.py" "$@"; }

# 探针自检:跑得动吗?(跑不动 ⇒ 下面的断言全部恒假,必须现在就说)
if ! dbq tickets "probe-selfcheck" > /dev/null 2>"$WORK/dbq.err"; then
  echo "预检失败:DB 探针跑不动($WORK/dbq.err):"
  cat "$WORK/dbq.err"
  fail_exit
fi

audit_lines() { dbq rows "$1"; }
read_lines()  { dbq read-rows "$1"; }
tickets()     { dbq tickets "$1"; }
last_ticket_no() { dbq last-ticket-no "$1"; }

# ── 起服务 ──────────────────────────────────────────────────────────────
start_cs() {   # $1=控制台输出文件;其余是给这一次启动的 env 覆盖
  local out="$1"; shift
  # **`>>` 不是 `>`**:`start_cs_ready` 会重试,重试时若把上一轮截掉,
  # 那条真正的错因(`[Errno 10048] ... bind`)就没了 —— 而那正是要看的东西。
  # 截断由调用方在第一次启动前做。
  env "$@" "$PYTHON" -m uvicorn app.main:app --port "$PORT" >> "$out" 2>&1 &
  CS_PID=$!
  echo "$CS_PID" > "$WORK/cs.pid"
}
# 起服务并**等到它真的能服务为止**,最多 3 次。
#
# 为什么要重试而不是把 sleep 调大:`kill -9` 之后 Windows **并不立刻归还监听端口**
# —— 实测(2026-09-23,本脚本第二次跑)新进程会以
#   `ERROR: [Errno 10048] error while attempting to bind on address ('127.0.0.1', 8000)`
# 退出,而这条报错出现在**新进程**的控制台里,读起来像「新起的服务坏了」。
# `wait_port_free` 能把窗口缩小但关不掉它(curl 拿到 000 只说明 accept 路径没了,
# 不等于 bind 立刻能成)—— 所以这里重试的是**整个启动**,不是把 sleep 调大。
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
    stop_cs
  done
  return 1
}
_kill_pid() {
  [ -n "$1" ] || return 0
  kill "$1" 2>/dev/null; sleep 1; kill -9 "$1" 2>/dev/null
}
# **等端口真的空出来**。实测(2026-09-23,本脚本第一次跑):`kill -9` 之后立刻
# `uvicorn` 会以
#   `ERROR: [Errno 10048] error while attempting to bind on address ('127.0.0.1', 8000)`
# 退出 —— 而这条报错出现在**新进程**的控制台里,读起来像「新起的服务坏了」。
# Windows 上监听套接字不是进程一死就归还的,这里必须轮询,不能靠固定 sleep 猜。
wait_port_free() {   # $1=端口 $2=秒数
  local deadline=$((SECONDS + $2))
  while [ "$SECONDS" -lt "$deadline" ]; do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://localhost:$1/")" = "000" ] && return 0
    sleep 1
  done
  return 1
}
stop_cs() {
  [ -n "$CS_PID" ] || return 0
  _kill_pid "$CS_PID"
  CS_PID=""
  wait_port_free "$PORT" 20 || echo "  (警告:端口 $PORT 20 秒内没释放,下一次启动大概会 bind 失败)"
}
start_mcp() {  # $1=模块 $2=控制台输出文件 → 把 pid 打印出来
  nohup "$PYTHON" -m "$1" >> "$2" 2>&1 &
  echo $!
}
# 与 `start_cs_ready` 同款(同一种端口归还竞态):起 + 等就绪,最多 3 次。
# 成功时 pid 在 `MCP_NEW_PID` 里。
start_mcp_ready() {   # $1=模块 $2=日志文件 $3=端口
  : > "$2"
  local attempt
  for attempt in 1 2 3; do
    MCP_NEW_PID=$(start_mcp "$1" "$2")
    if wait_mcp "$3" 30; then
      [ "$attempt" -gt 1 ] && echo "  (第 $attempt 次启动才起来 —— 上一次的端口还没归还)"
      return 0
    fi
    _kill_pid "$MCP_NEW_PID"
    wait_port_free "$3" 20
  done
  return 1
}

# 就绪轮询。**按挂钟计时(`SECONDS`),不是按次数**,而且 curl 自带 `--max-time`:
# 按次数循环时,每次 curl 若因为服务半死而卡上几秒,「60 秒」会变成好几分钟。
wait_ready() {   # $1=秒数
  local deadline=$((SECONDS + $1))
  while [ "$SECONDS" -lt "$deadline" ]; do
    [ "$(curl -s --max-time 5 -o /dev/null -w '%{http_code}' "$BASE/api/conversations")" = "200" ] && return 0
    sleep 1
  done
  return 1
}
# MCP 的 `/mcp` 对 GET 回 **406**(它要 POST + Accept 头)⇒「非 000」即已监听。
wait_mcp() {     # $1=端口 $2=秒数
  local deadline=$((SECONDS + $2))
  while [ "$SECONDS" -lt "$deadline" ]; do
    [ "$(curl -s --max-time 3 -o /dev/null -w '%{http_code}' "http://localhost:$1/mcp")" != "000" ] && return 0
    sleep 1
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

# ── SSE 收发 ────────────────────────────────────────────────────────────
ask() {    # $1=sid $2=消息 $3=SSE 输出文件(中文请求体走 stdin heredoc,不走 argv)
  curl -s -N --max-time 300 -X POST "$BASE/api/chat/stream" \
    -H "Content-Type: application/json" \
    --data-binary @- > "$3" <<JSON
{"session_id":"$1","message":"$2"}
JSON
}
resume() { # $1=sid $2=true|false $3=SSE 输出文件
  curl -s -N --max-time 300 -X POST "$BASE/api/chat/stream" \
    -H "Content-Type: application/json" \
    --data-binary @- > "$3" <<JSON
{"session_id":"$1","resume":{"approved":$2}}
JSON
}

# 提问,**直到模型真的做出了我们要它做的那件事**为止;最多 `$3` 次。
#
# 重试的**只是提问这一步,断言一个字没变**:每一次尝试都要满足同一条断言,
# 变的只是模型选哪条路。为什么必须这样 —— 实测(2026-09-23):
#   * 同一个题面「订单 1011 的物流有问题,帮我建个工单」,一次被判成 **物流**
#     (进 Agent → 命中确认流 → 卡片出现),另一次被判成 **投诉**
#     (进固定话术出口 → 一个工具都不调)。分类与「选哪个工具」**都非确定**
#     (deepseek 在 temperature=0 下亦然)。不重试的话验收 4/5 会**偶尔红**,
#     而那个红与代码无关 —— 一条会说谎的断言比没有断言更糟。
#   * 成功的那些尝试**必须打印第几次才成**:第 1 次就成与第 3 次才成,
#     读起来不是一回事。
# 判据两种:`tool=名字`(SSE 里有这个名字的 tool_call 帧)/ `frame=名字`(有这一帧)。
# 成功时 `$ASK_SID` / `$ASK_SSE` 有值。
ask_until() {   # $1=题面 $2=判据 $3=最多几次
  local text="$1" want="$2" tries="$3" attempt sid sse got
  ASK_SID=""; ASK_SSE=""
  for attempt in $(seq 1 "$tries"); do
    sid=$("$PYTHON" -c "import uuid;print(uuid.uuid4().hex)")
    sse="$WORK/ask_$attempt.sse"
    ask "$sid" "$text" "$sse"
    tool_call_names < "$sse" > "$WORK/ask_$attempt.calls" 2>/dev/null
    case "$want" in
      tool=*)  got=$(grep -qx "${want#tool=}" "$WORK/ask_$attempt.calls" && echo 1) ;;
      frame=*) got=$(has_event "$sse" "${want#frame=}" && echo 1) ;;
    esac
    if [ -n "$got" ]; then
      ASK_SID="$sid"; ASK_SSE="$sse"
      [ "$attempt" -gt 1 ] && echo "  (第 $attempt 次提问才满足 $want —— 模型选择/分类非确定)"
      return 0
    fi
    echo "  第 $attempt 次提问没满足 $want:意图=$(intent_label "$sse") / 调用了 [$(tr '\n' ' ' < "$WORK/ask_$attempt.calls")]"
  done
  return 1
}
# done 帧里的 intent(诊断用;挂起的那一轮没有 done 帧 ⇒ 回 `?`)。
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

has_done()  { grep -q '^event: done' "$1"; }
done_n()    { grep -c '^event: done' "$1"; }
error_n()   { grep -c '^event: error' "$1"; }
has_event() { grep -q "^event: $2\$" "$1"; }   # $1=文件 $2=帧名(ASCII)

# 帧的自检。**每个断言块都先跑它** —— 「没有这一帧」这类断言在一份空文件上恒真。
frames_sane() {   # $1=文件 → 0/1
  [ -s "$1" ] && has_event "$1" meta && return 0
  return 1
}

error_text() {
  "$PYTHON" -c '
import json, sys
lines = sys.stdin.buffer.read().decode("utf-8", "replace").splitlines()
for i, l in enumerate(lines):
    if l.strip() == "event: error" and i + 1 < len(lines) and lines[i+1].startswith("data: "):
        sys.stdout.buffer.write(json.loads(lines[i+1][6:]).get("message", "").encode("utf-8"))
        break'
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
# 这一轮调了哪些工具(按帧序;ASCII 名字,直接给 bash 用)。
tool_call_names() {
  "$PYTHON" -c '
import json, sys
lines = sys.stdin.buffer.read().decode("utf-8", "replace").splitlines()
for i, l in enumerate(lines):
    if l.strip() == "event: tool_call" and i + 1 < len(lines) and lines[i+1].startswith("data: "):
        sys.stdout.buffer.write((json.loads(lines[i+1][6:]).get("name", "") + "\n").encode("utf-8"))'
}
# 回复(拼回后)里有没有这个 needle。**needle 用十六进制码点传**(规矩 3)。
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
# done 帧的 intent 是不是「业务类」(物流 / 订单)。**这是本脚本的关键装置判据**:
# 非业务类不进 Agent(`app/agent/routing.py`),那时「模型没调工具」根本不是缺陷。
# 判据也走 chr() 构造 —— 两个标签都是中文。
done_intent_is_business() {
  "$PYTHON" -c '
import json, sys
lines = open(sys.argv[1], "rb").read().decode("utf-8", "replace").splitlines()
BUSINESS = {chr(0x7269)+chr(0x6d41), chr(0x8ba2)+chr(0x5355)}   # 物流 / 订单
intent = None
for i, l in enumerate(lines):
    if l.strip() == "event: done" and i + 1 < len(lines) and lines[i+1].startswith("data: "):
        intent = json.loads(lines[i+1][6:]).get("intent")
sys.stdout.buffer.write(("intent=%s\n" % intent).encode("utf-8"))
raise SystemExit(0 if intent in BUSINESS else 1)' "$1"
}
# MCP Server 的 tools/list(原始 JSON-RPC;`stateless_http=True` 下不需要握手)。
mcp_tools() {    # $1=端口 → 每行一个工具名
  curl -s --max-time 8 -X POST "http://localhost:$1/mcp" \
    -H 'Content-Type: application/json' \
    -H 'Accept: application/json, text/event-stream' \
    --data-binary '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' \
  | "$PYTHON" -c '
import json, sys
try:
    obj = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    names = [t["name"] for t in obj["result"]["tools"]]
except Exception as exc:
    sys.stdout.buffer.write(("MCP-LIST-ERROR-%s\n" % type(exc).__name__).encode("ascii"))
    raise SystemExit(3)
sys.stdout.buffer.write(("\n".join(names) + "\n").encode("utf-8"))'
}

# ── 临时内置工具(验收 1 用)───────────────────────────────────────────
write_temp_builtin() {
  cat > "$UNTRACKED_TMP" <<'PYEOF'
"""临时验收工具(ch08 验收 1)—— 由验收脚本创建,结束即删。

它存在的唯一目的:证明「往 builtin/ 里新加一个模块 = 新加一个工具,核心代码零改动」。
"""

import json

from langchain.tools import tool


@tool
async def echo_note(text: str) -> str:
    """把用户指定的那句话原样回显,一个字都不改。用户明确要求「回显」「复述」某句话时使用。"""
    return json.dumps({"echo": text}, ensure_ascii=False)


def build(*, session, conversation_id, retriever):
    return [echo_note]
PYEOF
}
# 给 `mcp_servers/logistics.py` **插入**一个临时工具(验收 3)。
# ⚠️ 必须**插在 `def main()` 之前** —— 文件末尾那句
# `if __name__ == "__main__": main()` 会**阻塞**在 `mcp.run()` 上,追加到它后面的
# `@mcp.tool()` 装饰器**永远不会执行** ⇒ 工具静默地不存在(实测:追加版跑完
# `tools/list` 里只有 query_logistics,而服务照常起来、一行报错都没有)。
# ⚠️ 写成**文件**再跑,不用 `python - <<EOF`:`python -` 读的是 `sys.stdin`,
# 而 Windows 上 pipe 的 stdin 编码取自 locale(cp936)—— heredoc 里的 UTF-8 中文
# 会被解成乱码,而那个乱码会**写进工具的用途描述**里(模型的工具清单变成乱码,
# 它就不再选这个工具),报错却一个都没有。
patch_mcp_server() {
  PYTHONPATH="$PWD" "$PYTHON" "$WORK/patch_mcp.py"
}
_write_patch_mcp() {
  cat > "$WORK/patch_mcp.py" <<'PYEOF'
import sys

PATH = "mcp_servers/logistics.py"
MARKER = "\n\ndef main() -> None:"
BLOCK = '''

@mcp.tool()
async def query_delivery_estimate(order_id: str) -> str:
    """查询订单的预计送达日期与剩余天数。用户问「什么时候能送到」「还要几天」时使用。"""
    from app.tools.mock_data import rng

    order_no = require_order_no(order_id)
    r = rng("eta", order_no)
    return json.dumps(
        {
            "order_id": order_no,
            "eta_days": r.randint(1, 5),
            "eta_date": f"2026-{r.randint(10, 12):02d}-{r.randint(1, 28):02d}",
        },
        ensure_ascii=False,
    )
'''

src = open(PATH, encoding="utf-8").read()
if "query_delivery_estimate" in src:
    sys.stdout.buffer.write(b"already-patched\n")
    raise SystemExit(0)
if src.count(MARKER) != 1:
    sys.stdout.buffer.write(
        ("anchor-not-unique:%d\n" % src.count(MARKER)).encode("ascii"))
    raise SystemExit(3)
open(PATH, "w", encoding="utf-8", newline="").write(
    src.replace(MARKER, BLOCK + MARKER))
sys.stdout.buffer.write(b"patched\n")
PYEOF
}
_write_patch_mcp

# ══════════════════════════════════════════════════════════════════════════
# 起环境
# ══════════════════════════════════════════════════════════════════════════
echo "== 起环境:两个 MCP Server(8101/8102)+ 客服服务(8000)=="
if ! start_mcp_ready mcp_servers.logistics "$WORK/mcp_logistics.log" "$MCP_LOGISTICS_PORT"; then
  echo "预检失败:MCP Server $MCP_LOGISTICS_PORT 起不来。控制台输出:"
  show_console_head_tail "$WORK/mcp_logistics.log"
  fail_exit
fi
MCP_LOG_PID="$MCP_NEW_PID"
if ! start_mcp_ready mcp_servers.aftersales "$WORK/mcp_aftersales.log" "$MCP_AFTERSALES_PORT"; then
  echo "预检失败:MCP Server $MCP_AFTERSALES_PORT 起不来。控制台输出:"
  show_console_head_tail "$WORK/mcp_aftersales.log"
  fail_exit
fi
MCP_AFTER_PID="$MCP_NEW_PID"
echo "  两个 MCP Server 已就绪(logistics=$MCP_LOG_PID aftersales=$MCP_AFTER_PID)"
# 两个 Server 的 tools/list 都要能读到 —— 读不到的话,验收 2/3 的
# 「审计里 source=mcp:*」在**装置层面**就不可能成立,得先说清楚。
L_TOOLS=$(mcp_tools "$MCP_LOGISTICS_PORT")
if ! printf '%s\n' "$L_TOOLS" | grep -q '^query_logistics$'; then
  echo "预检失败:8101 的 tools/list 读不到 query_logistics(读到的是:$L_TOOLS)"
  fail_exit
fi
echo "  8101 暴露的工具:$(printf '%s' "$L_TOOLS" | tr '\n' ' ')"

if ! start_cs_ready "$WORK/cs_1.log"; then
  echo "预检失败:$BASE 60 秒内没起来。控制台输出(头 40 行 + 末 15 行):"
  show_console_head_tail "$WORK/cs_1.log"
  fail_exit
fi
echo "  客服服务已就绪(pid $CS_PID)"
echo

# ══════════════════════════════════════════════════════════════════════════
# ⚠️ **项号顺序与执行顺序不同**(ch07 也这么干),这里跑的是 **2 → 1 → 3 → 4 → 5 → 6**:
#   * 验收 2 要先跑 —— 它是一条**基线**,跑在一个没被任何临时改动碰过的服务上;
#   * 验收 1 要把客服服务重启一次(它加的是内置工具),此后所有项都跑在重启过的进程上;
#   * 验收 3 必须排在验收 1 的**还原之后** —— 它断言「客服服务进程号没变」,
#     而验收 1 恰好会换进程号。脚本把这条断言建在**它自己读到的前后两个值**上,
#     所以换成别的顺序也成立,但读起来会很难受。
echo "== 验收 2:业务 MCP Server 的工具能查到真数据(基线,未改任何文件)=="
# 单号 **1011 是挑过的**:它在 mock 数据源里是「已发货 / 派送中」——
# 物流工具对它**真的**有记录可返回。挑「真的发了货」的单号是题面明写的,
# 拿一个未发货的单号去问,模型只会拿到一条 ToolNotFound,那时
# 「done 帧非空」照样成立,而这条验收什么都没验到。
# (对照:`1002` 在本章的 mock 里是「已取消」,压根没有物流记录。)
ASK2="订单 1011 到哪了"
if ! ask_until "$ASK2" "tool=query_logistics" 3; then
  boom "验收 2:三次提问模型都没调 query_logistics —— 本项没测到它要测的东西"
  SID2=""; cp "$WORK/ask_3.sse" "$WORK/acc2.sse" 2>/dev/null
else
  SID2="$ASK_SID"; cp "$ASK_SSE" "$WORK/acc2.sse"
fi
if ! frames_sane "$WORK/acc2.sse"; then
  boom "验收 2 的 SSE 不成形(没有 meta 帧)—— 下面的断言全部不可信"
else
  REP2=$(join_tokens < "$WORK/acc2.sse")
  CALLS2=$(tool_call_names < "$WORK/acc2.sse" | tr '\n' ' ')
  INTENT2=$(done_intent_is_business "$WORK/acc2.sse")
  echo "  题面:$ASK2"
  echo "  $INTENT2 / 调用的工具:$CALLS2 / error 帧 $(error_n "$WORK/acc2.sse") 条"
  # **只作证据打印,不作断言**:非业务类不进 Agent(`routing.py`),那时模型
  # 根本调不到工具 —— 而上面的门(模型必须调了 query_logistics)在结构上已经
  # 排除了那种情况。再断一次就是恒真的。
  done_intent_is_business "$WORK/acc2.sse" > /dev/null \
    || echo "  ── 注:这一轮的意图不是业务类,但工具仍被调到了(见上面的工具名)"
  if [ -n "$REP2" ]; then
    ok "done 帧拼回后非空(${#REP2} 字节):$(printf '%s' "$REP2" | head -c 120)"
  else
    bad "done 帧拼回后是空的 —— 模型没有产出任何用户可见的答复"
  fi
  RO2=$(audit_lines "$SID2")
  if [ -z "$RO2" ]; then
    boom "本次会话在 tool_audit_logs 里一行都没有 —— 审计写口没接上,下面那条断言恒假"
  fi
  if printf '%s\n' "$RO2" | grep -q '|mcp:logistics|'; then
    ok "审计里有 source='mcp:logistics' 的调用:$(printf '%s\n' "$RO2" | grep '|mcp:logistics|' | head -1)"
  else
    bad "审计里没有 source='mcp:logistics' 的调用(本次会话的审计行:$RO2)"
  fi
  # ---- 落点(模型判定,**WARN 档**)------------------------------------
  # 「回复非空」只说明模型说了话,不说明它**把工具取回来的数据用上了**。
  # 后者是模型行为,按 ch07 §10.5 的规矩走 WARN:**显式计数并打印**,
  # 既不算通过也不算失败 —— 悄悄跳过是最坏的处理。
  # 判据取 1011 在本章 mock 里那个**确定性**的物流状态「派送」(6D3E 9001);
  # 换成任何中文都比较不了(argv 通道),所以同样走十六进制码点。
  join_tokens < "$WORK/acc2.sse" > "$WORK/acc2_reply.txt"
  if has_needle "$WORK/acc2_reply.txt" "6d3e 9001"; then
    warn "落点成立:回复里出现了订单 1011 的物流状态「派送」⇒ 工具数据确实被用进了答复。**仍然不计为通过**(模型判定,不是代码判定)"
  else
    warn "落点未复现:回复里没有「派送」二字 ⇒ 模型可能没把工具结果用进答复。**模型行为**,不是代码缺陷;显式记下来而不是悄悄跳过"
  fi
fi
echo

# ══════════════════════════════════════════════════════════════════════════
echo "== 验收 1:往 app/tools/builtin/ 丢一个新模块 → 重启客服服务 → 它就能用 =="
write_temp_builtin
echo "  已写入 $UNTRACKED_TMP(未跟踪文件,退出时由 git clean 还原)"
# 重启客服服务。**这一步是题面要求的**,脚本照做。
# ⚠️ 但别把「必须重启」当成已验证的事实:实测(2026-09-23)不重启也能用 ——
# `builtin.discover()` 每请求都 `pkgutil.iter_modules` + `import_module` 重新扫一遍
# 包目录,新增的模块**当场**就进注册表。`app/tools/builtin/__init__.py` 的 docstring
# 写着「新增内置工具需要重启客服服务」,与这条实测**不符**(已作为发现上报,本脚本
# 不据此下断言 —— 重启在两种实现下都成立,所以它不构成本项判据)。
stop_cs
if ! start_cs_ready "$WORK/cs_2.log"; then
  echo "预检失败:重启后起不来。控制台输出:"
  show_console_head_tail "$WORK/cs_2.log"
  fail_exit
fi
echo "  客服服务已重启(pid $CS_PID)"
ASK1="帮我查一下订单 1011 的物流,并且调用 echo_note 工具把这句话原样回显给我:$PHRASE_HOST"
# 门设成「模型调了 echo_note」,**因此下面不再把「SSE 里有 echo_note 的 tool_call 帧」
# 当成一条 PASS** —— 门已经保证了它,再报一次就是**恒真的断言**(读起来像证据,
# 其实什么也没说)。留下的是两条与这个门**无关**的:回复里的话、审计里的行。
if ! ask_until "$ASK1" "tool=echo_note" 3; then
  boom "验收 1:三次提问模型都没调 echo_note —— 本项没测到它要测的东西"
  SID1=""; cp "$WORK/ask_3.sse" "$WORK/acc1.sse" 2>/dev/null
else
  SID1="$ASK_SID"; cp "$ASK_SSE" "$WORK/acc1.sse"
fi
if ! frames_sane "$WORK/acc1.sse"; then
  boom "验收 1 的 SSE 不成形(没有 meta 帧)"
else
  # ⚠️ **必须先把 token 帧拼回再找 needle**(规矩 5):回复是逐 token 推的,
  # 「薄荷山丘-7391」完全可能被切成两三帧,直接在原始 SSE 里找是**找得到也靠运气**。
  join_tokens < "$WORK/acc1.sse" > "$WORK/acc1_reply.txt"
  REP1=$(cat "$WORK/acc1_reply.txt")
  echo "  调用的工具:$(tool_call_names < "$WORK/acc1.sse" | tr '\n' ' ')"
  # ⚠️ **单看这一条不足以证明工具被调过** —— 暗号就在问题里,模型原样复述
  # 用户的话也能命中。它要跟下面那条审计行**合起来**读。
  if has_needle "$WORK/acc1_reply.txt" "$PHRASE_HEX"; then
    ok "回复里有那句话(十六进制码点构造的 needle 命中)"
  else
    bad "回复里没有那句话。回复原文:$(printf '%s' "$REP1" | head -c 300)"
  fi
  RO1=$(audit_lines "$SID1")
  if [ -z "$RO1" ]; then
    boom "本次会话在 tool_audit_logs 里一行都没有 —— 审计写口没接上,下面那条断言恒假"
  fi
  if printf '%s\n' "$RO1" | grep -q '^echo_note|builtin|success|'; then
    ok "审计里有 source='builtin'、tool_name='echo_note' 的成功调用"
  else
    bad "审计里没有 echo_note|builtin|success(本次会话的审计行:$RO1)"
  fi
fi
echo

# ══════════════════════════════════════════════════════════════════════════
echo "== 还原验收 1 的临时模块,重启客服服务(后面几项要在干净的服务上跑)=="
# 走**与 EXIT trap 同一个** `restore_tmp()` —— 不变量要放在唯一写口上,
# 不要在两个调用点各写一份「怎么算还原干净」(本仓的元教训之一)。
# 在 trap 里它是兜底,在这里它是主动调用;两处的判据必须逐字相同。
restore_tmp
if [ "$RESTORE_BAD" -eq 0 ]; then
  echo "  已还原(临时模块 git clean 掉、$TRACKED_TMP git checkout 回原样)"
else
  echo "  !!!! 还原没干净 —— 工作区被污染,后面的读数不可信"
fi
stop_cs
if ! start_cs_ready "$WORK/cs_3.log"; then
  echo "预检失败:重启后起不来。控制台输出:"
  show_console_head_tail "$WORK/cs_3.log"
  fail_exit
fi
echo "  客服服务已重启(pid $CS_PID)"
echo

# ══════════════════════════════════════════════════════════════════════════
echo "== 验收 3:给 MCP Server 加工具 → 只重启该 Server → 客服服务不用动 =="
BEFORE_PID=$(cat "$WORK/cs.pid")
PATCH_OUT=$(patch_mcp_server)
echo "  补丁:$PATCH_OUT"
if [ "$PATCH_OUT" != "patched" ] && [ "$PATCH_OUT" != "already-patched" ]; then
  boom "给 $TRACKED_TMP 打补丁失败($PATCH_OUT)—— 本项没有测到它要测的东西"
else
  # **只重启物流 Server**。客服服务不动 —— 这就是本项要证明的两条通道之别。
  _kill_pid "$MCP_LOG_PID"
  wait_port_free "$MCP_LOGISTICS_PORT" 20 || echo "  (警告:端口 $MCP_LOGISTICS_PORT 没释放)"
  if ! start_mcp_ready mcp_servers.logistics "$WORK/mcp_logistics_2.log" "$MCP_LOGISTICS_PORT"; then
    echo "预检失败:重启后的 $MCP_LOGISTICS_PORT 没起来。控制台输出:"
    show_console_head_tail "$WORK/mcp_logistics_2.log"
    fail_exit
  fi
  MCP_LOG_PID="$MCP_NEW_PID"
  NEW_TOOLS=$(mcp_tools "$MCP_LOGISTICS_PORT")
  echo "  8101 现在的工具:$(printf '%s' "$NEW_TOOLS" | tr '\n' ' ')"
  if printf '%s\n' "$NEW_TOOLS" | grep -q '^query_delivery_estimate$'; then
    ok "Server 侧已暴露新工具 query_delivery_estimate(直接读 tools/list,不经模型)"
  else
    bad "Server 的 tools/list 里没有 query_delivery_estimate(读到:$NEW_TOOLS)"
  fi
  ASK3="订单 1011 预计什么时候能送到?"
  if ! ask_until "$ASK3" "tool=query_delivery_estimate" 3; then
    boom "验收 3:三次提问模型都没调 query_delivery_estimate —— 本项没测到它要测的东西"
    SID3=""; cp "$WORK/ask_3.sse" "$WORK/acc3.sse" 2>/dev/null
  else
    SID3="$ASK_SID"; cp "$ASK_SSE" "$WORK/acc3.sse"
  fi
  # **进程号先看**:它是本项最硬的一条,而且与模型无关。
  AFTER_PID=$(cat "$WORK/cs.pid")
  if [ "$BEFORE_PID" = "$AFTER_PID" ]; then
    ok "客服服务进程号没变(before=$BEFORE_PID after=$AFTER_PID)—— 新工具是**现问现拿**来的"
  else
    bad "客服服务被重启了(before=$BEFORE_PID after=$AFTER_PID)—— 验收 3 要求它不动"
  fi
  if ! frames_sane "$WORK/acc3.sse"; then
    boom "验收 3 的 SSE 不成形(没有 meta 帧)"
  else
    CALLS3=$(tool_call_names < "$WORK/acc3.sse" | tr '\n' ' ')
    echo "  题面:$ASK3 / 调用的工具:$CALLS3"
    RO3=$(audit_lines "$SID3")
    if [ -z "$RO3" ]; then
      boom "本次会话在 tool_audit_logs 里一行都没有 —— 下面那条断言恒假"
    fi
    if printf '%s\n' "$RO3" | grep -q '^query_delivery_estimate|mcp:logistics|success|'; then
      ok "客服系统真的调到了新工具(审计:$(printf '%s\n' "$RO3" | grep '^query_delivery_estimate|' | head -1))"
    else
      bad "审计里没有 query_delivery_estimate|mcp:logistics|success(本次会话:$RO3;本轮调用:$CALLS3)"
    fi
  fi
fi
echo

# ══════════════════════════════════════════════════════════════════════════
# 确认流(验收 4/5)。**这一段是模型选择工具** —— 题面必须落在业务类意图上,
# 否则不进 Agent、`create_ticket` 永远不会被调。实测(2026-09-23):
#   * 「帮我建个工单」→ **其他** → 兜底出口,压根不进 Agent(不能用作题面);
#   * 「订单 1011 有没有问题,帮我建个工单」→ 投诉 → 固定话术出口,同样不进 Agent;
#   * 「订单 1011 的物流有问题,帮我建个工单」→ **物流** → Agent → 命中确认流。
# 题面因此**必须带一个业务类的落点**(这里是订单 1011 的物流)。
TICKET_ASK="订单 1011 的物流有问题,帮我建个工单"
confirm_flow() {   # $1=标签 $2=approved(true/false)
  local tag="$1" approved="$2"
  local sid ask_sse res_sse
  CONFIRM_SID=""
  # 门 = **模型调了 create_ticket**(它的动作);
  # 断言 = **卡片帧出现了**(系统的反应)。两者**故意不是同一件事** ——
  # 门若也设成「卡片出现」,那条断言就恒真了(本仓明令:恒真的断言比没有更糟)。
  if ! ask_until "$TICKET_ASK" "tool=create_ticket" 3; then
    boom "$tag:三次提问模型都没调 create_ticket(意图会随措辞抖动:同一个题面实测被判成过「物流」也判成过「投诉」,后者压根不进 Agent)—— 本项没测到它要测的东西"
    CONFIRM_SID=""
    return
  fi
  sid="$ASK_SID"; ask_sse="$WORK/${tag}_ask.sse"; res_sse="$WORK/${tag}_resume.sse"
  cp "$ASK_SSE" "$ask_sse"
  if ! frames_sane "$ask_sse"; then
    boom "$tag 的提问轮 SSE 不成形(没有 meta 帧)"
    CONFIRM_SID=""
    return
  fi
  local before after
  before=$(tickets "$sid")
  echo "  题面:$TICKET_ASK"
  echo "  提问轮的帧:$(grep -o '^event: .*' "$ask_sse" | sort -u | tr '\n' ' ')"
  if has_event "$ask_sse" ticket_confirm; then
    ok "$tag:模型调了 create_ticket 之后,卡片帧真的出现了(挂起,本轮不发 done —— done=$(done_n "$ask_sse") 条)"
  else
    bad "$tag:模型调了 create_ticket 但**没有** ticket_confirm 帧(本轮帧:$(grep -o '^event: .*' "$ask_sse" | sort -u | tr '\n' ' '))—— 确认闸没接上"
  fi
  resume "$sid" "$approved" "$res_sse"
  if ! frames_sane "$res_sse"; then
    boom "$tag 的续跑轮 SSE 不成形(没有 meta 帧)"
    CONFIRM_SID=""
    return
  fi
  echo "  续跑轮的帧:$(grep -o '^event: .*' "$res_sse" | sort -u | tr '\n' ' ')(error 帧 $(error_n "$res_sse") 条)"
  [ "$(error_n "$res_sse")" -ge 1 ] && echo "    error 文案:$(error_text < "$res_sse" | head -c 200)"
  after=$(tickets "$sid")
  CONFIRM_SID="$sid"
  CONFIRM_BEFORE="$before"
  CONFIRM_AFTER="$after"
  CONFIRM_ASK="$ask_sse"
  CONFIRM_RESUME="$res_sse"
}

echo "== 验收 4:走确认流并点「确认提交」= 工单真的建出来 =="
confirm_flow acc4 true
SID4="$CONFIRM_SID"
if [ -z "$SID4" ]; then
  boom "验收 4 没跑起来(没拿到确认卡片 —— 具体错因见上面那行 !!!!)"
else
  echo "  会话 $SID4 / tickets:$CONFIRM_BEFORE → $CONFIRM_AFTER"
  if [ "$CONFIRM_AFTER" -eq $((CONFIRM_BEFORE + 1)) ]; then
    ok "tickets 表多了一行($CONFIRM_BEFORE → $CONFIRM_AFTER)"
  else
    bad "tickets 表不是恰好 +1($CONFIRM_BEFORE → $CONFIRM_AFTER)"
  fi
  # 工单号**从库里现读**(不是从帧里抄),再要求它出现在**拼回后的回复**里 ——
  # 这样断的是「用户看到的那个号与真正落库的那一行是同一个」,而不是「帧里有个号」。
  # (工单号是纯 ASCII,`grep -F` 按字节匹配即可;UTF-8 的续字节都 >= 0x80,
  #  不会与 ASCII 模式撞上 —— 这一点与中文 needle 不同,那条必须走码点。)
  join_tokens < "$CONFIRM_RESUME" > "$WORK/acc4_reply.txt"
  TNO=$(last_ticket_no "$SID4")
  # 探针自己坏了与「回复里没有」是两件事:前者是**装置**问题(下面所有断言都不可信),
  # 后者才是结果。混在一起报的话,一条 SQL 写错会伪装成产品缺陷。
  case "$TNO" in
    DBQ-ERROR*)
      boom "取工单号的探针挂了($TNO)—— 这一条没验到,别当成产品缺陷" ;;
    "")
      bad "该会话在 tickets 表里没有行 —— 上一条「多了一行」与这一条矛盾,先看那条" ;;
    *)
      if grep -qF "$TNO" "$WORK/acc4_reply.txt"; then
        ok "回复里有工单号 $TNO(与库里那一行**同一个**号)"
      else
        bad "回复里没有库里那个工单号 $TNO。回复原文:$(head -c 300 "$WORK/acc4_reply.txt")"
      fi ;;
  esac
  RO4=$(audit_lines "$SID4")
  if [ -z "$RO4" ]; then
    boom "本次会话在 tool_audit_logs 里一行都没有 —— 下面那条断言恒假"
  fi
  if printf '%s\n' "$RO4" | grep -q '^create_ticket|builtin|success|'; then
    ok "审计里 create_ticket 是 success"
  else
    bad "审计里没有 create_ticket|builtin|success(本次会话:$RO4)"
  fi
fi
echo

echo "== 验收 5:同一条路径,这回点「取消」= 工单**不**建,审计记权限拒绝 =="
confirm_flow acc5 false
SID5="$CONFIRM_SID"
if [ -z "$SID5" ]; then
  boom "验收 5 没跑起来(没拿到确认卡片 —— 具体错因见上面那行 !!!!)"
else
  echo "  会话 $SID5 / tickets:$CONFIRM_BEFORE → $CONFIRM_AFTER"
  # **必须是「相等」,不是「<=」** —— `<=` 在 tickets 表没有写入的任何实现下都成立。
  if [ "$CONFIRM_AFTER" -eq "$CONFIRM_BEFORE" ]; then
    ok "tickets 表**没有**多行($CONFIRM_BEFORE → $CONFIRM_AFTER)"
  else
    bad "取消之后 tickets 居然多了行($CONFIRM_BEFORE → $CONFIRM_AFTER)—— 写操作被放行了"
  fi
  RO5=$(audit_lines "$SID5")
  if [ -z "$RO5" ]; then
    boom "本次会话在 tool_audit_logs 里一行都没有 —— 下面那条断言恒假"
  fi
  if printf '%s\n' "$RO5" | grep -q '^create_ticket|builtin|permission_denied|'; then
    ok "审计里 create_ticket 是 permission_denied"
  else
    bad "审计里没有 create_ticket|builtin|permission_denied(本次会话:$RO5)"
  fi
fi
echo

# ══════════════════════════════════════════════════════════════════════════
echo "== 验收 6:TOOL_TIMEOUT_SECONDS=0.001 —— 只读可重试,写操作**结构性**不重试 =="
# ⚠️ 这一项**只断审计行**,不断 `tickets` 的增减:超时的写操作**未必没执行**
# (0.001 秒的窗口里,SQL 可能已经提交了才被取消)。断表里的行数会变成一条靠运气的断言。
stop_cs
if ! start_cs_ready "$WORK/cs_4.log" TOOL_TIMEOUT_SECONDS=0.001; then
  echo "预检失败:带 TOOL_TIMEOUT_SECONDS=0.001 的服务起不来。控制台输出:"
  show_console_head_tail "$WORK/cs_4.log"
  fail_exit
fi
echo "  客服服务已就绪(pid $CS_PID,TOOL_TIMEOUT_SECONDS=0.001)"
# ---- 只读:重试到用尽 ----
# 门同样是「模型调了 query_logistics」(它是 MCP 工具,0.001 秒必然超时);
# 断言是审计行里的 `timeout|2` —— 那条取决于**执行器**的行为,与门无关。
if ! ask_until "订单 1011 到哪了" "tool=query_logistics" 3; then
  boom "验收 6:三次提问模型都没调 query_logistics —— 本项没测到它要测的东西"
  SID6R=""
else
  SID6R="$ASK_SID"; cp "$ASK_SSE" "$WORK/acc6_read.sse"
fi
RR=$(read_lines "$SID6R")
echo "  只读那轮的审计行(tool|source|status|retry|ms):"
printf '%s\n' "$RR" | sed -n 's/^/    /p'
if [ -z "$RR" ]; then
  boom "只读那轮在 tool_audit_logs 里一行都没有 —— 下面两条断言恒假"
fi
# `retry_count=2` 是**默认值 2 的直接读数**(共 3 次尝试):这条断言同时钉住
# `tool_retry_attempts=2` 真的生效,以及执行器把「**真实发生的**重试次数」
# 而不是配置值写进表里(写配置值的话,一个首次就成功的查询会被记成「重试了 2 次」)。
if printf '%s\n' "$RR" | grep -q '|timeout|2|'; then
  ok "只读工具超时且重试到用尽:$(printf '%s\n' "$RR" | grep '|timeout|2|' | head -1)"
else
  bad "只读那轮没有 `status=timeout 且 retry_count=2` 的行(实际:$RR)"
fi
DUR=$(printf '%s\n' "$RR" | grep '|timeout|2|' | head -1 | cut -d'|' -f5)
if [ -n "$DUR" ] && [ "$DUR" -gt 0 ] 2>/dev/null; then
  ok "duration_ms>0($DUR ms)—— 它数的是**整轮尝试**的墙钟,不是单次"
else
  bad "duration_ms 不是正数(${DUR:-缺})"
fi
# ---- 写操作:一次都不重试 ----
confirm_flow acc6_w true
SID6W="$CONFIRM_SID"
if [ -z "$SID6W" ]; then
  boom "验收 6 的写路径没跑起来(没拿到确认卡片 —— 具体错因见上面那行 !!!!)"
else
  RW=$(audit_lines "$SID6W" | grep '^create_ticket|')
  echo "  写操作的审计行:$(printf '%s' "$RW" | tr '\n' ' ')"
  if printf '%s\n' "$RW" | grep -q '^create_ticket|builtin|timeout|0|'; then
    ok "写操作超时且 retry_count=0 —— 「永不重试」是结构保证(kind 推出),不是配置约定"
  else
    bad "写操作那条不是 timeout|0(实际:$RW)"
  fi
fi
echo

# ══════════════════════════════════════════════════════════════════════════
echo "════════════════════════════════════════════════════════════════════════"
echo "结果:$PASS 通过,$FAIL 失败,$WARN 项未复现(见上面的 WARN 行 —— 它们**没有**被验过,别当通过)"
[ "$FAIL" -eq 0 ]
