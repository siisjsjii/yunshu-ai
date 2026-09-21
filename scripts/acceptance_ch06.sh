#!/usr/bin/env bash
# ch06 验收 1–4(端到端)。前置:
#   1) 服务已在 8000 上跑**当前代码**(uvicorn 无 --reload,改了端点不会自己生效);
#   2) MySQL 在跑(docker start mysql);
#   3) Milvus 在跑(docker start milvus-standalone)—— 退款子流程要检索政策;
#   4) 真实 API key(.env)。
#
# **断言依据一律是确定性的东西**:SSE 帧的**名字与载荷**、done 帧的 trace / intent、
# 以及**直接查库**的行数。不靠模型自由文本(deepseek 在 temperature=0 下依然非确定),
# 也不 grep 原始 SSE 流(逐 token 推送会把 20240915 切成三帧)。
# 唯一一处读自由文本的地方是「回复里有没有固定兜底话术」—— 那是**常量**,不是自由文本。
#
# 与 `scripts/acceptance.sh`(ch01–ch04 的网)的关系:**本脚本不依赖它**。
# 它那条题面「退货政策是什么」在 ch06 起会走退款子流程并挂起(合法行为),
# 由下一件任务单独处置 —— 不是本脚本该管的事。
set -uo pipefail
BASE="${BASE:-http://localhost:8000}"
# 与 `scripts/acceptance.sh` 一致:一律用 venv 里的解释器。裸 `python` 在本机不保证
# 存在、也不保证是 venv 那个 —— 那样 new_sid 会产出空串,整个脚本静默走偏。
PYTHON="${PYTHON:-.venv/Scripts/python.exe}"
PASS=0; FAIL=0; WARN=0

ok()   { echo "  PASS  $1"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL  $1"; FAIL=$((FAIL+1)); }
# 未复现(不是失败,也不是通过):只在「判定模型这一轮给出的结论」决定某一帧
# 出不出来时用。**必须显式计数并打印**,否则它就退化成一条悄悄跳过的检查 ——
# 而「悄悄跳过」正是本项目最怕的假绿形态。
warn() { echo "  WARN  $1"; WARN=$((WARN+1)); }

if [ ! -x "$PYTHON" ]; then
  echo "找不到 $PYTHON —— 请在项目根目录运行本脚本。" >&2
  exit 2
fi

# ── 平台陷阱(Windows + Git Bash,本仓在 ch02/ch05 各栽过)──────────────────
#
# 1) **含中文的请求体不能走 curl 的 argv**。MSYS2 会按 CP936 重编码,服务端只回
#    `error parsing the body` —— 而本脚本的断言全是 ASCII,那种全盘失败还算好认,
#    怕的是它被掩盖成「通过」。故所有请求体一律走 stdin heredoc(管道是字节流)。
# 2) **取帧的 helper 一律按字节读写 stdin/stdout**(`sys.stdin.buffer`)。Windows 上
#    Python 对**管道**默认用 ANSI 代码页(cp936)+ surrogateescape,`sys.stdin.read()`
#    会把 UTF-8 的 SSE 字节解成乱码与孤立代理对,随后 `.encode("utf-8")` 抛
#    `UnicodeEncodeError: ... surrogates not allowed` —— ch05 实测因此**在内容
#    其实满足条件的情况下判失败**(假红)。
# 3) **服务没起时 curl 给 `000`**,与「旧进程」是两种病。预检先分这一支。

# 发一条消息。请求体走 stdin heredoc(理由见上)。
# --max-time:上游卡住时不加这个,脚本会**无限期阻塞且什么都不打印**,收尾时表现为
# 「挂住」而不是「红」—— 那是查不出来的失败。180s 足够:最长的一轮(退款子流程,
# 三次 LLM 调用 + 检索)实测 10~40s。
ask() {
  local sid="$1" msg="$2"
  curl -s -N --max-time 180 -X POST "$BASE/api/chat/stream" \
    -H "Content-Type: application/json" \
    --data-binary @- <<JSON
{"session_id":"$sid","message":"$msg"}
JSON
}

# 从挂起点续跑(点订单卡片)。载荷是 ASCII,但仍走 heredoc —— 通道只有一条,
# 「这条要不要走 heredoc」每次判断都会判错一次。
resume() {
  local sid="$1" order_no="$2"
  curl -s -N --max-time 180 -X POST "$BASE/api/chat/stream" \
    -H "Content-Type: application/json" \
    --data-binary @- <<JSON
{"session_id":"$sid","resume":{"order_no":"$order_no"}}
JSON
}

new_sid() { "$PYTHON" -c "import uuid;print(uuid.uuid4().hex)"; }

# 把 token 帧拼回整段回复(逐 token 推送,不能直接 grep)。
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

# 帧序列(逗号分隔的**事件名**)。这是本脚本最要紧的一把尺子:验收 4 的「挂起轮
# 只有 meta + order_choice、没有 done」和「续跑轮 meta/citations/token/refund_offer/done」
# 都靠它逐帧断,**不是**「流里出现过 order_choice 就算数」—— 后者对
# 「挂起之后又误发了一个 done」毫无判别力。
frame_names() {
  "$PYTHON" -c '
import sys
names = []
for l in sys.stdin.buffer.read().decode("utf-8", "replace").splitlines():
    if l.startswith("event: "):
        names.append(l[7:].strip())
sys.stdout.buffer.write(",".join(names).encode("ascii"))'
}

# 从最后一帧(done)的 data 里取一个字段;取不到输出字面量 null。
# 注意:端点故障时只发 error、**不发 done**,那时最后一帧是 error 载荷,
# `.get(字段)` 得到 null —— 调用方必须先分这一支(见 has_error),否则会把
# **基础设施故障**报成「路由不对」,正是 CLAUDE.md 点名的「报错指向别处」。
done_field() {
  "$PYTHON" -c '
import json, sys
lines = sys.stdin.buffer.read().decode("utf-8", "replace").splitlines()
data = [l[6:] for l in lines if l.startswith("data: ")]
try:
    payload = json.loads(data[-1])
except (IndexError, json.JSONDecodeError):
    payload = {}
value = payload.get(sys.argv[1])
sys.stdout.buffer.write(
    (value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)).encode("utf-8"))' "$1"
}

# done 帧存不存在(挂起轮**没有** done,验收 4 断的就是这个)。
has_done() { grep -q "^event: done" "$1"; }

# 有 error 帧。端点/上游故障时只发 error、不发 done。
has_error() { grep -q "^event: error" "$1"; }

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

# done 帧的 trace 数组里是否含某段子串 → yes/no。判据是 ASCII(trace 的取值全是
# ASCII:节点名 / 工具名 / 订单号),故可以直接走 argv。
# ⚠️ 没有 done 帧时它返回 no —— 调用方必须先确认 done 帧在(否则「trace 里没有 X」
# 会把一次基础设施故障说成路由问题)。
trace_has() {
  "$PYTHON" -c '
import json, sys
needle = sys.argv[1]
lines = sys.stdin.buffer.read().decode("utf-8", "replace").splitlines()
data = [l[6:] for l in lines if l.startswith("data: ")]
trace = []
try:
    trace = json.loads(data[-1]).get("trace") or []
except (IndexError, json.JSONDecodeError):
    trace = []
sys.stdout.buffer.write(b"yes" if any(needle in t for t in trace) else b"no")' "$1"
}

# done 帧里 `intent` 是否落在**八类闭集**内(app/agent/routing.py 是那份标签表的
# 唯一来源,这里读它而不是抄一份 —— 抄一份的话,标签表改了脚本不会跟着改)。
# 「JSON 可解析」的落点就是这个:分类器解析失败时会**降级成「其他」**,那条路径同样
# 输出一个非空的 intent,所以「非空」本身证明不了任何事,必须在**枚举**上断。
intent_in_enum() {
  "$PYTHON" -c '
import json, sys
from app.agent.routing import INTENT_LABELS
lines = sys.stdin.buffer.read().decode("utf-8", "replace").splitlines()
data = [l[6:] for l in lines if l.startswith("data: ")]
intent = None
try:
    intent = json.loads(data[-1]).get("intent")
except (IndexError, json.JSONDecodeError):
    intent = None
sys.stdout.buffer.write(
    b"yes" if isinstance(intent, str) and intent in INTENT_LABELS else b"no")'
}

# 退款检索的命中数:从 trace 的 `refund_expand_retrieve:3 路 7 命中 top=0.87` 里取 7
# (第一个数字是路数、第二个是命中数)。正则**不写那两个中文词**:写死了的话,
# 改动 trace 措辞会让它静默取不到,而「取不到」在那条断言里长得像「检索为空」。
# **取不到一律输出 -1**,不是 0 —— 0 与「这个节点根本没跑」在数值上不可区分,而
# 「没跑」正是这条断言要挡的故障(拿 0 当默认值 = 把「节点没跑」读成「检索为空」)。
refund_hits() {
  "$PYTHON" -c '
import json, re, sys
lines = sys.stdin.buffer.read().decode("utf-8", "replace").splitlines()
data = [l[6:] for l in lines if l.startswith("data: ")]
trace = []
try:
    trace = json.loads(data[-1]).get("trace") or []
except (IndexError, json.JSONDecodeError):
    trace = []
for t in trace:
    m = re.search(r"refund_expand_retrieve:[^\d]*\d+[^\d]+(\d+)", t)
    if m:
        sys.stdout.buffer.write(m.group(1).encode("ascii"))
        break
else:
    sys.stdout.buffer.write(b"-1")'
}

# citations 帧的条目数(没有该帧 → 0)。逐帧解析而不是 grep:事件名与 data 分处
# 两行、条数是 JSON 结构,拿空格敏感的子串匹配去断它属于自找假红。
citations_count() {
  "$PYTHON" -c '
import json, sys
lines = sys.stdin.buffer.read().decode("utf-8", "replace").splitlines()
for i, l in enumerate(lines):
    if l.strip() == "event: citations" and i + 1 < len(lines) and lines[i+1].startswith("data: "):
        sys.stdout.buffer.write(str(len(json.loads(lines[i+1][6:]).get("items") or [])).encode("ascii"))
        break
else:
    sys.stdout.buffer.write(b"0")'
}

# 第一条引用的标题(打印给人工看「拿到的是不是退款政策那一类块」)。
first_citation_q() {
  "$PYTHON" -c '
import json, sys
lines = sys.stdin.buffer.read().decode("utf-8", "replace").splitlines()
for i, l in enumerate(lines):
    if l.strip() == "event: citations" and i + 1 < len(lines) and lines[i+1].startswith("data: "):
        items = json.loads(lines[i+1][6:]).get("items") or []
        if items:
            sys.stdout.buffer.write(str(items[0].get("question", "")).encode("utf-8"))
        break'
}

# order_choice 帧的订单号列表(逗号分隔)。逐帧解析,不做子串匹配:
# 子串只证明「某个号码在流里出现过」,不证明它在 **order_choice 帧的 options 数组**里。
order_choice_nos() {
  "$PYTHON" -c '
import json, sys
lines = sys.stdin.buffer.read().decode("utf-8", "replace").splitlines()
for i, l in enumerate(lines):
    if l.strip() == "event: order_choice" and i + 1 < len(lines) and lines[i+1].startswith("data: "):
        opts = json.loads(lines[i+1][6:]).get("options") or []
        sys.stdout.buffer.write(",".join(str(o.get("order_no") or "") for o in opts).encode("utf-8"))
        break'
}

# 卡片选项的形状:每一项都得是 {order_no, status, product, amount},且**四值都非空**。
# 这就是「可点性」的边界 —— 缺一个字段的卡片在浏览器里表现为「点了没反应」,
# 而脚本这边看不出任何异常。输出 yes/no。
order_choice_shape() {
  "$PYTHON" -c '
import json, sys
lines = sys.stdin.buffer.read().decode("utf-8", "replace").splitlines()
keys = ("order_no", "status", "product", "amount")
for i, l in enumerate(lines):
    if l.strip() == "event: order_choice" and i + 1 < len(lines) and lines[i+1].startswith("data: "):
        opts = json.loads(lines[i+1][6:]).get("options") or []
        ok = bool(opts) and all(
            isinstance(o, dict)
            and all(isinstance(o.get(k), str) and o.get(k).strip() for k in keys)
            for o in opts
        )
        sys.stdout.buffer.write(b"yes" if ok else b"no")
        break
else:
    sys.stdout.buffer.write(b"no")'
}

# refund_offer 帧的某个字段(数组拼成逗号串);没有该帧 → 空串。
refund_offer_field() {
  "$PYTHON" -c '
import json, sys
key = sys.argv[1]
lines = sys.stdin.buffer.read().decode("utf-8", "replace").splitlines()
for i, l in enumerate(lines):
    if l.strip() == "event: refund_offer" and i + 1 < len(lines) and lines[i+1].startswith("data: "):
        v = json.loads(lines[i+1][6:]).get(key)
        if isinstance(v, list):
            v = ",".join(str(x) for x in v)
        sys.stdout.buffer.write(str(v).encode("utf-8"))
        break' "$1"
}

# 不在白名单里的帧名(逗号分隔;全在白名单里 → 空)。用来断「这一轮只出现了
# 该出现的帧」—— 帧协议多了/少了东西时,前端的表现是静默的。
unexpected_frames() {   # $1 = 允许的帧名;stdin = sse
  "$PYTHON" -c '
import sys
allowed = set(sys.argv[1].split(","))
names = []
for l in sys.stdin.buffer.read().decode("utf-8", "replace").splitlines():
    if l.startswith("event: "):
        names.append(l[7:].strip())
sys.stdout.buffer.write(",".join(n for n in names if n not in allowed).encode("ascii"))' "$1"
}

# 某个帧出现过(有该帧 → 退出码 0,故可当断言用,也可用在 && || 里)。
has_frame() { grep -q "^event: $2$" "$1"; }

# 固定退款类目,**从单一来源读**(app/refund/categories.py),脚本不抄一份 ——
# 抄一份的话,类目表改了脚本会红在一个与它无关的地方。
app_categories() {
  "$PYTHON" -c '
import sys
from app.refund.categories import REFUND_REASON_CATEGORIES
sys.stdout.buffer.write(",".join(REFUND_REASON_CATEGORIES).encode("utf-8"))'
}

# 从 stdin 的一份 JSON 对象里取一个字段(值是字符串/数字);解析不出或没有该键 → 空串。
# 用途:`POST /api/refund` 的**回显逐字段比对**。只断状态码是不够的 —— 200 只说明
# 「没被拒」,说明不了它记下的是我们提交的那一单(order_no / reason 都可能被写错)。
json_field() {
  "$PYTHON" -c '
import json, sys
raw = sys.stdin.buffer.read().decode("utf-8", "replace").strip()
try:
    payload = json.loads(raw)
except json.JSONDecodeError:
    payload = {}
value = payload.get(sys.argv[1]) if isinstance(payload, dict) else None
sys.stdout.buffer.write(("" if value is None else str(value)).encode("utf-8"))' "$1"
}

# 直接查库:某会话的 messages 行数 / 角色序列 / 第一条 user 内容 / refund_requests 行数。
# 输出若干 "key=value" 行(ASCII 键),调用方用 db_get 取值。
#
# **按会话 id 查,不查全表**:全表计数会被上一次运行的残留带偏,那是「靠运气的绿」。
# 用**新进程新 engine** 读 —— 同 session 重读是否打到库取决于身份映射与 refcount,
# 会变成「靠 refcount 走运」的断言(CLAUDE.md 记过)。
db_state() {
  SID="$1" "$PYTHON" -c '
import asyncio, os, sys

from sqlalchemy import select

from app.db.base import get_engine, get_sessionmaker

# 分隔符写成常量:片段整体套在 bash 的单引号里,**片段里出现单引号会把 bash
# 的引号提前闭合**(那种错 bash -n 抓不到 —— 它只是把两个字符串拼接起来,
# 语法完全合法,要到运行时才炸)。
SEP = ","
SEP2 = "|"
NONE = ""
from app.db.models import MessageRecord, RefundRequest


def row_key(x):
    """退款单的行摘要(片段里不许出现单引号,故不写内联 f-string)。"""
    return SEP2.join([x.order_no, x.reason_category, x.status])

async def main():
    sid = os.environ["SID"]
    async with get_sessionmaker()() as s:
        msgs = (await s.execute(
            select(MessageRecord.role, MessageRecord.content)
            .where(MessageRecord.conversation_id == sid)
            .order_by(MessageRecord.id))).all()
        rows = (await s.execute(
            select(RefundRequest).where(RefundRequest.conversation_id == sid)
        )).scalars().all()
    out = [
        f"messages={len(msgs)}",
        f"roles={SEP.join(r for r, _ in msgs)}",
        f"first_user={msgs[0][1] if msgs else NONE}",
        f"refunds={len(rows)}",
        f"refund_rows={SEP2.join(row_key(x) for x in rows)}",
        f"refund_ids={SEP.join(str(x.id) for x in rows)}",
    ]
    sys.stdout.buffer.write("\n".join(out).encode("utf-8"))
    await get_engine().dispose()


asyncio.run(main())
'
}

# 从 db_state 的输出里取一个 key 的值。
db_get() { printf '%s\n' "$1" | sed -n "s/^$2=//p" | head -1; }

# ── 预检:8000 上那个进程**是不是当前代码** ────────────────────────────────
# 不是形式主义:开发期起的 uvicorn **没有 --reload**,改了代码它照跑旧的。
# 不预检的话四条验收**全部假红**,而且看起来像「新代码坏了」—— 本项目已栽过一次。
#
# 两道,缺一不可:
#   ① `POST /api/ticket` 33 个 x → **422**(session_id 的 max_length=32)。它证明
#      「ch05 之后的路由与校验层都在」,ch05 之前的进程会回 405(静态目录的兜底)。
#      ⚠️ 它**分不出 ch05 与 ch06** —— ch05 的 /api/ticket 一模一样。故加 ②。
#   ② `POST /api/refund` 带一个**不在固定集里的类目** → **422**。这是 ch06 才有的
#      路由;旧进程回 404/405。类目校验**先于任何 IO**(端点第一行、锁之前),
#      所以这一发**不会写出任何行**,可以放心在预检里打。
PRE_CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/api/ticket" \
  -H 'Content-Type: application/json' \
  --data-binary '{"session_id":"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}')
# 服务**没起**时 curl 给的是 000,不是 405/404。先分这一支 —— 否则下面那段
# 「多半是旧进程」会把「服务没起」误诊成「旧进程」,让人去 netstat 一个不存在的 PID。
if [ "$PRE_CODE" = "000" ]; then
  echo "预检失败:$BASE 上没有服务在监听(curl 返回 000)。先起服务:"
  echo "  docker start mysql milvus-standalone"
  echo "  $PYTHON -m uvicorn app.main:app --port 8000"
  exit 1
fi
if [ "$PRE_CODE" != "422" ]; then
  echo "预检失败:$BASE 上的服务不是当前代码 —— POST /api/ticket 期望 422,实得 $PRE_CODE。"
  echo "8000 上多半是旧进程(无 --reload,不会自己更新)。按序执行:"
  echo "  netstat -ano | grep ':8000'                                   # 记下 PID"
  echo "  powershell -NoProfile -Command \"Stop-Process -Id <PID> -Force\""
  echo "  $PYTHON -m uvicorn app.main:app --port 8000"
  exit 1
fi
PRE_REFUND=$(printf '%s' '{"session_id":"preflight-ch06","order_no":"20240915","reason_category":"这个类目不存在"}' \
  | curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/api/refund" \
    -H 'Content-Type: application/json' --data-binary @-)
if [ "$PRE_REFUND" != "422" ]; then
  echo "预检失败:POST /api/refund 期望 422(类目不在固定集),实得 $PRE_REFUND。"
  echo "这道断的是「8000 上是 ch06 的代码」—— 上面那条 /api/ticket 分不出 ch05 与 ch06。"
  echo "旧进程处理:同上(kill 掉再起)。"
  exit 1
fi
echo "预检通过:/api/ticket=422 /api/refund=422(当前代码)"
echo

# 会话 id 必须是 32 位十六进制(uuid4().hex)。它同时是 conversations.id(varchar(32))
# 与 checkpointer 的 thread_id:**短了/空了会让四次验收共用同一段 state 与同一批
# messages 行**,而那种错看起来像「上一轮的东西串味了」。
sid_or_die() {
  local sid
  sid=$(new_sid)
  case "$sid" in
    *[!0-9a-f]*|"") echo "new_sid 产出非法会话 id「$sid」—— 检查 $PYTHON 能不能跑。" >&2; exit 2;;
  esac
  [ "${#sid}" -eq 32 ] || { echo "new_sid 产出的 id 不是 32 位:「$sid」" >&2; exit 2; }
  printf '%s' "$sid"
}

# ══════════════════════════════════════════════════════════════════════════
echo "== 验收 1:多轮意图(物流 → 退款 → 物流)每轮 intent 判对 =="
# 每轮都断 done 帧的 `intent`(确定性的**标签**,不是回复文本)。
# ⚠️ 判据是**标签本身**,不是「路由出口」:物流与订单同走 BUSINESS,
# 但那两条标签在本章是两回事,把 物流 判成 订单 就是判错 —— 这条不能放宽。
# 第 2 轮若是**挂起**(弹了订单卡片),就一定没有 done 帧、也就取不到 intent:
# 那种红必须与「intent 判错」在输出上分开(挂起轮会单独打一行说明)。
SID1=$(sid_or_die)
ask "$SID1" "订单 1005 的物流到哪了" > /tmp/ch06_1a.sse
ask "$SID1" "这个能退吗"            > /tmp/ch06_1b.sse
ask "$SID1" "那订单 1005 的物流到哪儿了" > /tmp/ch06_1c.sse

intent_check() {   # $1=轮次名 $2=sse 文件 $3=期望 intent
  local name="$1" f="$2" want="$3" got
  if has_error "$f" && ! has_done "$f"; then
    bad "$name:端点发了 error 帧、没有 done 帧(基础设施/上游故障,不是意图判错):$(error_text < "$f")"
    return
  fi
  if ! has_done "$f"; then
    # 挂起 = 没有 done。第 2 轮挂起说明**指代没补上**(上下文里那个号码没被取到),
    # 而不是「意图判错」—— 两者的修法完全不同,不能报成同一句话。
    bad "$name:这一轮**挂起**了(帧序列 $(frame_names < "$f")),没有 done 帧 —— 缺订单号才会挂起,即上下文没接上"
    return
  fi
  got=$(done_field intent < "$f")
  if [ "$got" = "$want" ]; then
    ok "$name intent=$got"
  else
    bad "$name 期望 intent=$want,实得 $got(trace=$(done_field trace < "$f"))"
  fi
}
intent_check "第 1 轮(物流)" /tmp/ch06_1a.sse "物流"
intent_check "第 2 轮(退款)" /tmp/ch06_1b.sse "退款退货"
intent_check "第 3 轮(物流)" /tmp/ch06_1c.sse "物流"
echo "  ── 第 2 轮 trace(上下文有没有接上,看这一行):$(done_field trace < /tmp/ch06_1b.sse)"
echo "  ── 第 1 轮 trace:$(done_field trace < /tmp/ch06_1a.sse)"
echo

# ══════════════════════════════════════════════════════════════════════════
echo "== 验收 2:意图 JSON 可解析 + 怪问题落「其他」 =="
# ①「可解析」的落点是**八类闭集**。分类器解析失败时会把 intent 降级成「其他」并
#    把 confidence 置 0 —— 那条路径同样给出一个**非空的 intent**,所以
#    「done 帧的 intent 非空」本身证明不了任何事(它是恒真的),必须在枚举上断。
# ② 怪问题落「其他」要三处同时成立:标签是「其他」、trace 走了 fallback_reply 出口、
#    回复是**固定兜底话术**。三处独立,少一处都可能被别的实现满足。
SID2=$(sid_or_die)
ask "$SID2" "你好" > /tmp/ch06_2a.sse
ask "$SID2" "请用 Python 写一个快速排序算法" > /tmp/ch06_2b.sse

if [ "$(intent_in_enum < /tmp/ch06_2a.sse)" = "yes" ]; then
  ok "普通问题的 intent 落在八类闭集内:$(done_field intent < /tmp/ch06_2a.sse)(confidence=$(done_field confidence < /tmp/ch06_2a.sse))"
else
  bad "done 帧的 intent 不在八类闭集内或取不到:$(done_field intent < /tmp/ch06_2a.sse) —— 分类器没解析出结构化出参(降级成「其他」)"
fi

W2=$(done_field intent < /tmp/ch06_2b.sse)
T2=$(done_field trace < /tmp/ch06_2b.sse)
TXT2=$(join_tokens < /tmp/ch06_2b.sse)
case "$T2" in
  *fallback_reply*) F2=yes;;
  *) F2=no;;
esac
if [ "$W2" = "其他" ] && [ "$F2" = "yes" ]; then
  ok "怪问题落「其他」并走 fallback_reply:$T2"
else
  bad "怪问题期望 intent=其他 且 trace 含 fallback_reply,实得 intent=$W2 trace=$T2"
fi
# 空回复**判红**:空文本里当然不含兜底话术,不先判空的话,端点出错、一个 token 帧
# 都没发时这里会白打一行绿(`scripts/acceptance.sh` 被审查抓过这一条)。
if [ -z "$TXT2" ]; then
  bad "怪问题的回复为空 —— 没有任何 token 帧(端点出错?)"
elif case "$TXT2" in *"我没太理解您的意思"*) true;; *) false;; esac; then
  ok "回复是固定兜底话术:$TXT2"
else
  bad "回复不是固定兜底话术:$TXT2"
fi
echo

# ══════════════════════════════════════════════════════════════════════════
echo "== 验收 3:「这个能退吗」补全指代 → 走退款子流程 → 拿到订单与政策 =="
# 会话第一轮先把订单号 1007 放进**上下文**(题面本身只提物流,不提退款);
# 第二轮的题面**只有「这个」**,订单号只能从上下文里来。
# 1007 **不在演示订单池**(池子是 20240915/20240901/20240818/20240808)——
# 这一点是本条判别力的来源:若指代/上下文没接上,子流程会**挂起弹卡片**,
# 而卡片上的号码来自演示池,与 1007 一眼可分。断言里写 1007 就同时钉住了这件事。
SID3=$(sid_or_die)
ask "$SID3" "订单 1007 的物流到哪了" > /tmp/ch06_3a.sse
ask "$SID3" "这个能退吗"            > /tmp/ch06_3b.sse

if has_error /tmp/ch06_3b.sse && ! has_done /tmp/ch06_3b.sse; then
  bad "端点发了 error 帧、没有 done 帧(基础设施/上游故障,不是路由问题):$(error_text < /tmp/ch06_3b.sse)"
elif ! has_done /tmp/ch06_3b.sse; then
  bad "第二轮**挂起**了(帧序列 $(frame_names < /tmp/ch06_3b.sse))—— 上下文里的 1007 没被取到,于是弹了订单卡片"
else
  TR3=$(done_field trace < /tmp/ch06_3b.sse)
  # ① resolve_references:⚠️ 它**近乎恒真**(那是每轮第一个节点,任何一轮的 trace
  #    里都有它)。留着是因为验收标准点名了它,但**判别力不在这里** ——
  #    真正承重的是下面 ② 那一行:子流程自己报出「从问题/历史里取到订单号 1007」。
  #    没有 ② 的话,「trace 有 resolve_references」只能证明图跑起来了。
  if [ "$(trace_has resolve_references < /tmp/ch06_3b.sse)" = "yes" ]; then
    ok "trace 有 resolve_references(每轮都会有的节点,弱判据)"
  else
    bad "trace 里没有 resolve_references:$TR3"
  fi
  # ② 指代补全的**唯一确定性证据**:子流程报出的订单号是上下文里那个 1007。
  if [ "$(trace_has "refund:问题/历史里取到订单号 1007" < /tmp/ch06_3b.sse)" = "yes" ]; then
    ok "指代补全成立:子流程取到上下文里的订单号 1007"
  else
    bad "子流程没取到上下文里的 1007(这一行本该是「refund:问题/历史里取到订单号 1007」):$TR3"
  fi
  # ③ 拿到订单:query_order 真的查到了(走的是 execute_tool,ok 才会是这个 trace)。
  if [ "$(trace_has "refund:fetch ok" < /tmp/ch06_3b.sse)" = "yes" ]; then
    ok "子流程取到了这一单(refund:fetch ok)"
  else
    bad "没有 refind:fetch ok —— 订单没查成:$TR3"
  fi
  # ④ 拿到政策:两条独立证据 —— trace 里的命中数,与 citations 帧的条目数。
  #    citations 帧**只在检索返回非空块时才发**,所以「有帧且条目 >= 1」本身就是
  #    「证据非空」的直接证据;命中数是同一件事的另一个出口(trace 字符串)。
  HITS3=$(refund_hits < /tmp/ch06_3b.sse)
  CIT3=$(citations_count < /tmp/ch06_3b.sse)
  if [ "${HITS3:-0}" -ge 1 ] 2>/dev/null && [ "${CIT3:-0}" -ge 1 ] 2>/dev/null; then
    ok "政策证据非空:检索命中 $HITS3 条、citations 帧 $CIT3 条(首条:$(first_citation_q < /tmp/ch06_3b.sse))"
  else
    bad "政策证据为空:命中=$HITS3(取不到是 -1)citations=$CIT3 —— 检索被阈值滤空或节点没跑:$TR3"
  fi
  echo "  ── 第 2 轮 trace:$TR3"
fi
echo

# ══════════════════════════════════════════════════════════════════════════
echo "== 验收 4:不带订单号问退款 → 订单卡片 → resume 走完 → 落库 =="
SID4=$(sid_or_die)
ask "$SID4" "我想退款" > /tmp/ch06_4a.sse
NAMES4A=$(frame_names < /tmp/ch06_4a.sse)
# ① 挂起轮的**帧序列**必须**恰好**是 meta,order_choice。
#    精确序列而不是「含 order_choice」:后者对「挂起之后又漏发了一个 done」无判别力,
#    而 done 帧自报的是「这一轮跑完了」(它带 trace/intent/agent_steps,挂起时全是初值)
#    —— 发了就是在撒谎,且挂起轮**不落库**,前端会以为这轮有回复。
if [ "$NAMES4A" = "meta,order_choice" ]; then
  ok "挂起轮帧序列恰好是 meta,order_choice(没有 done)"
else
  bad "挂起轮帧序列期望 meta,order_choice,实得 [$NAMES4A]"
fi

NOS4=$(order_choice_nos < /tmp/ch06_4a.sse)
SHAPE4=$(order_choice_shape < /tmp/ch06_4a.sse)
# ② 卡片载荷**可点**:选项非空、每项四字段齐全且非空。这就是「验收标准 4 的浏览器
#    那一半」在本脚本里能验到的**边界** —— 再往下(真的渲染出来、真的能点)只能人工。
if [ "$SHAPE4" = "yes" ]; then
  ok "订单卡片载荷可点:$NOS4"
else
  bad "订单卡片载荷不完整(选项为空或缺 order_no/status/product/amount):[$NOS4]"
fi

# ③ 挂起轮**不落库**(spec §5.1:log_turn 没跑)。这一条只有查库能证明 ——
#    帧上完全看不出「这轮写没写历史」。
DB4A=$(db_state "$SID4")
M4A=$(db_get "$DB4A" messages)
if [ "$M4A" = "0" ]; then
  ok "挂起轮没往 messages 写任何行"
else
  bad "挂起轮写了 $M4A 行 messages —— 挂起的一轮不该落库(帧序列 $NAMES4A)"
fi

# ④ resume 走完。用卡片上的**第一个**选项(人工点第一张卡就是这个动作)。
ORDER4=$(printf '%s' "$NOS4" | cut -d, -f1)
resume "$SID4" "$ORDER4" > /tmp/ch06_4b.sse
NAMES4B=$(frame_names < /tmp/ch06_4b.sse)
TR4B=$(done_field trace < /tmp/ch06_4b.sse)

# ④ 续跑轮的形状:首帧 meta、末帧 done、**不得再挂起**(再出现 order_choice 就是
#    回填的订单号没被接纳)、不得出现白名单之外的帧。
#    **刻意不写死整条序列**:citations 帧只在检索返回非空块时才发,把它写进序列
#    会让「政策被阈值滤空」这种环境问题伪装成「协议变了」。
UNK4B=$(unexpected_frames "meta,citations,token,refund_offer,done" < /tmp/ch06_4b.sse)
if [ -z "$UNK4B" ] && [ "${NAMES4B%%,*}" = "meta" ] && [ "${NAMES4B##*,}" = "done" ] \
   && ! has_frame /tmp/ch06_4b.sse order_choice; then
  ok "续跑轮走完且没有再次挂起:[$NAMES4B]"
else
  bad "续跑轮形状不对:[$NAMES4B](白名单外的帧:[$UNK4B];再挂起=$(has_frame /tmp/ch06_4b.sse order_choice && echo yes || echo no))"
fi

# ⑤ 回填的订单号**真的被用上了**:`refund:fetch ok` 只在 order_no 非空、且这一单
#    查得到时出现。载荷没被采纳(端点递错形状)时 order_no 是**空串** →
#    `query_order("")` 抛 ToolNotFound → trace 变 `refund:fetch not_found`、
#    话术变成「没能查到订单 ,请核对订单号」。resume 那条 trace **不带号码**,
#    所以号码本身只能靠 fetch 的成败 + 下面的 refund_offer 帧来钉(后者只在
#    判定模型说「能退」时才发)。
if [ "$(trace_has "refund:fetch ok" < /tmp/ch06_4b.sse)" = "yes" ]; then
  ok "回填的订单号被采纳(refund:fetch ok,而不是 not_found)"
else
  bad "订单没取成 —— 回填的号码可能是空的:$TR4B"
fi

# ⑥ 退款入口的**两个证据必须同向**:refund_offer 帧在 ⟺ trace 里有 refund:offer。
#    只看一边都会漏掉真故障:只有帧没 trace = 前端有表单但流程没走 offer 出口;
#    只有 trace 没帧 = 流程走完了,用户却看不到表单(前端「点了没反应」那类故障)。
HAS_OFFER_FRAME=no; has_frame /tmp/ch06_4b.sse refund_offer && HAS_OFFER_FRAME=yes
HAS_OFFER_TRACE=$(trace_has "refund:offer" < /tmp/ch06_4b.sse)
if [ "$HAS_OFFER_FRAME" = "$HAS_OFFER_TRACE" ]; then
  ok "退款入口的帧与 trace 同向(refund_offer 帧=$HAS_OFFER_FRAME)"
else
  bad "refund_offer 帧与 trace 不一致(帧=$HAS_OFFER_FRAME,trace=$HAS_OFFER_TRACE):$TR4B"
fi

CATS_APP=$(app_categories)
if [ "$HAS_OFFER_FRAME" = "yes" ]; then
  # 判定模型说「能退」—— 表单帧出现,把它的载荷钉死。
  GOT4=$(refund_offer_field order_no < /tmp/ch06_4b.sse)
  CATS4=$(refund_offer_field categories < /tmp/ch06_4b.sse)
  if [ "$GOT4" = "$ORDER4" ]; then
    ok "refund_offer 的订单号 = resume 载荷里的 $ORDER4"
  else
    bad "refund_offer 的订单号实得「$GOT4」,期望「$ORDER4」(resume 载荷没被采纳?)"
  fi
  if [ "$CATS4" = "$CATS_APP" ]; then
    ok "表单类目与单一来源一致:$CATS4"
  else
    bad "表单类目与 app/refund/categories.py 不一致:帧=$CATS4 单一来源=$CATS_APP —— 前端能选、端点回 422 的那种漂移"
  fi
else
  # ⚠️ 这一支**不是失败**:判定模型这一轮说「不能退」,子流程走 refund_explain,
  #    于是不发 refund_offer —— 那是**设计如此**(不能退的单子不该弹退款表单)。
  #    但它意味着本轮**没有验到表单帧**,必须显式说出来、并计入 WARN:
  #    「悄悄跳过」正是本项目最怕的假绿形态。人工步骤 4 会覆盖它。
  # 一条 warn、一个计数(`warn` 调三次的话汇总行会写成「3 项未复现」,而事实是 1 项)。
  warn "本轮**没有**验到 refund_offer 表单帧(判定模型说「不能退」→ 走 refund_explain,不发表单):
       trace=$TR4B
       ↳ 这是**模型行为**:同一张卡、同一个订单号,本机 2026-09-21 多次实测里「能退」与
         「不能退」两种结论**都出现过**(检索到的政策片段每次略有不同)。不是缺陷。
       ↳ 它不改变本轮的其余结论:卡片载荷 → resume 走完 → 落库 全都验过了。
         表单帧本身由人工步骤 4 覆盖(浏览器里看得见下拉框就说明它出来了)。"
fi
CAT4=$(printf '%s' "$CATS_APP" | cut -d, -f1)

# ⑦ 提交退款单(前端表单那一下,验收标准 4 里「落库」那一半的入口)。
REFUND_RESP=$(printf '{"session_id":"%s","order_no":"%s","reason_category":"%s"}' \
  "$SID4" "$ORDER4" "$CAT4" \
  | curl -s -w $'\n%{http_code}' -X POST "$BASE/api/refund" \
    -H 'Content-Type: application/json' --data-binary @-)
RF_CODE=$(printf '%s' "$REFUND_RESP" | tail -1)
RF_BODY=$(printf '%s' "$REFUND_RESP" | head -n -1)
# 逐字段比对**响应回显**:状态码只能证明「没被拒」,证明不了它记下的是我们提交的那一单。
RF_STATUS=$(printf '%s' "$RF_BODY" | json_field status)
RF_CONV=$(printf '%s' "$RF_BODY" | json_field conversation_id)
RF_ORDER=$(printf '%s' "$RF_BODY" | json_field order_no)
RF_CAT=$(printf '%s' "$RF_BODY" | json_field reason_category)
RF_ID=$(printf '%s' "$RF_BODY" | json_field id)
if [ "$RF_CODE" = "200" ] && [ "$RF_STATUS" = "pending" ] \
   && [ "$RF_CONV" = "$SID4" ] && [ "$RF_ORDER" = "$ORDER4" ] && [ "$RF_CAT" = "$CAT4" ] \
   && [ -n "$RF_ID" ]; then
  ok "退款单提交成功(HTTP 200,回显一致:id=$RF_ID status=pending order_no=$RF_ORDER reason=$RF_CAT)"
else
  bad "退款单提交失败:HTTP $RF_CODE body=$RF_BODY(期望 status=pending / conversation=$SID4 / order_no=$ORDER4 / reason=$CAT4,且回显 id)"
fi

# ⑧ **直接查库**:该会话恰好一行 refund_requests,且三个字段与提交的一致。
#    字段分隔符是 `|`(db_state 里定的),**不是** `/`:写错分隔符会让这条永远红 ——
#    而它红的样子像「没落库」(实测撞过一次)。
DB4C=$(db_state "$SID4")
R4C=$(db_get "$DB4C" refunds)
ROWS4=$(db_get "$DB4C" refund_rows)
if [ "$R4C" = "1" ] && case "$ROWS4" in "$ORDER4|$CAT4|pending") true;; *) false;; esac; then
  ok "refund_requests 里有对应行:$ROWS4"
else
  bad "refund_requests 期望 1 行「$ORDER4|$CAT4|pending」,实得 $R4C 行:[$ROWS4]"
fi

# ⑨ 类目不在固定集 → 422,且**一行都不写**。后半句才是这条的重点:把校验挪到
#    `session.add` 之后,状态码照样是 422,库里却留下了一次**成功的退款申请** ——
#    「先落库再校验」这个错,只看状态码是看不出来的。
printf '%s' '{"session_id":"'"$SID4"'","order_no":"'"$ORDER4"'","reason_category":"随便编一个不存在的类目"}' \
  | curl -s -o /tmp/ch06_4c.json -w '%{http_code}' -X POST "$BASE/api/refund" \
    -H 'Content-Type: application/json' --data-binary @- > /tmp/ch06_4c.code
BAD_CODE=$(cat /tmp/ch06_4c.code)
DB4D=$(db_state "$SID4")
R4D=$(db_get "$DB4D" refunds)
if [ "$BAD_CODE" = "422" ] && [ "$R4D" = "1" ]; then
  ok "非法类目 422 且没有写出任何行(库里仍是 1 行)"
else
  bad "非法类目期望 422 且库里仍是 1 行,实得 HTTP $BAD_CODE、$R4D 行(先落库再校验?)"
fi

# ⑩ 没有待续流程时 resume → **409**(不是 200 的 error 帧,更不是把 'user_input'
#    这个 Python 标识符糊到用户脸上)。用刚刚**跑完**的那个会话(它存在、跑过、已无待续)
#    —— 这正是「服务重启后用户点了页面上残留下的旧卡片」那个可达场景。
#    **响应不能是 SSE 流**:一旦是流,状态码就没意义了(流里的 error 帧是 200)。
RESUME_CODE=$(printf '%s' '{"session_id":"'"$SID4"'","resume":{"order_no":"20240915"}}' \
  | curl -s -D /tmp/ch06_4d.hdr -o /tmp/ch06_4d.json -w '%{http_code}' -X POST "$BASE/api/chat/stream" \
    -H 'Content-Type: application/json' --data-binary @-)
BODY4D=$(cat /tmp/ch06_4d.json)
CT4D=$(sed -n 's/^[Cc]ontent-[Tt]ype: *//p' /tmp/ch06_4d.hdr | tr -d '\r')
if [ "$RESUME_CODE" = "409" ]; then
  ok "无待续流程时 resume → 409"
else
  bad "无待续流程时 resume 期望 409,实得 $RESUME_CODE —— 走了流式就再也没有状态码了"
fi
case "$CT4D" in
  application/json*) ok "409 的响应体是普通 JSON,不是 SSE 流(content-type: $CT4D)";;
  *) bad "409 的响应居然是流(content-type: $CT4D)—— 那状态码本身就没有意义了";;
esac
case "$BODY4D" in
  *"没有待处理的流程"*) ok "409 的文案是固定话术:$BODY4D";;
  *) bad "409 的文案不是固定话术(不得出现 Python 标识符):$BODY4D";;
esac
echo

# ══════════════════════════════════════════════════════════════════════════
# 人工那一半。**不许假装跑过,也不许悄悄丢掉**:脚本把边界之内全断完(卡片载荷的形状、
# 前端点击要发的两个请求本身),边界之外(浏览器里真的画出来了、真的发对了)只能人看。
echo "════════════════════════════════════════════════════════════════"
echo "⚠️  以下部分**只能人工验**(浏览器,脚本覆盖不到):验收标准 4 的前端一半"
echo "    脚本已经验到边界:卡片载荷的形状(选项非空、order_no/status/product/amount"
echo "    四值齐全)、以及点击/提交时前端要发的两个请求(resume 与 POST /api/refund)"
echo "    真的走通了。**没有验**的是 index.html 把这帧画成了可点的卡片、点击时发对了载荷。"
echo "    人工步骤(约 2 分钟):"
echo "      1) 浏览器开 $BASE(新开一个会话);"
echo "      2) 发「我想退款」;"
echo "      3) 应出现**可点的订单卡片**(订单号 / 状态 / 商品 / 金额);点第一张;"
echo "      4) 回复应接着流进同一个气泡,并出现**退款原因下拉 + 提交按钮**(选项来自帧);"
echo "      5) 选一个原因提交,应看到回执(退款单已创建);"
echo "      6) 复核:SELECT * FROM refund_requests WHERE conversation_id='$(printf '%s' "$SID4" | head -c 32)';"
echo "         (本次脚本用过的会话 id 与订单号就在上面几条 PASS 行里)"
echo "════════════════════════════════════════════════════════════════"
echo

echo "结果:$PASS 通过,$FAIL 失败,$WARN 项未复现(见上面的 WARN 行 —— 它们**没有**被验过,别当通过)"
[ "$FAIL" -eq 0 ]
