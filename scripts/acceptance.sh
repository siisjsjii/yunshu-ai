#!/usr/bin/env bash
# ch01–ch03 端到端验收。前置:
#   1) .venv/Scripts/python.exe -m uvicorn app.main:app --port 8000
#   2) MySQL 在跑(见 .env 的 DATABASE_URL)
#   3) ch03 的验收 7 还需要 Milvus 容器在跑:
#      docker start milvus-standalone
# 需要真实 API key(.env)。
#
# **起服务之前先查 8000 端口**:残留的僵尸进程会让你 curl 到**旧代码**,
# 于是得到"新代码坏了"的假红(ch02 的最终验证差点栽在这上面)。本脚本开头
# 会探一次服务可达性,但探不出"在跑的是哪一版" —— 若下面验收 5/6 的知识路由
# 断言报红,第一件该查的就是 8000 上跑的是不是旧进程;改了 `app/config.py`
# 之后尤其要重启(uvicorn 没有 --reload,`get_settings` 是 lru_cache,
# 阈值改了对已在跑的进程**无效**)。
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

# 取「某个工具的每次调用」的结果(逗号分隔;从未调用 → 空)。
# **按下标配对**,不要分别在两个列表里做包含匹配:`called_tools` 与
# `tool_result_states` 是同序的,而 `*query_faq*` + `*ok*` 这种分开的包含匹配会
# 把「Agent 调了 query_faq 但**它失败了**,另一次别的调用成功」判成通过 —— 那正是
# 假绿(知识没召回到,断言却绿了)。
states_of_tool() {
  NAMES="$1" STATES="$2" NEEDLE="$3" "$PYTHON" -c '
import os, sys

names = os.environ["NAMES"].split(",")
states = os.environ["STATES"].split(",")
needle = os.environ["NEEDLE"]
picked = [states[i] for i, n in enumerate(names) if n == needle and i < len(states)]
sys.stdout.buffer.write(",".join(picked).encode("ascii"))'
}

# 从**最后一帧(done)**的 data 里取一个字段;取不到输出空。
# 同 join_tokens:按字节读写 stdin/stdout。done 帧里除了 ASCII 的 trace/intent
# 还有模型输出,文本模式在 cp936 管道上会解成乱码甚至抛 UnicodeEncodeError
# (`scripts/acceptance_ch05.sh` 记过这个**复发型**陷阱)。
done_field() {
  "$PYTHON" -c '
import json, sys

lines = sys.stdin.buffer.read().decode("utf-8", "replace").splitlines()
data = [l[6:] for l in lines if l.startswith("data: ")]
try:
    payload = json.loads(data[-1])
except (IndexError, json.JSONDecodeError):
    sys.exit(0)
value = payload.get(sys.argv[1])
sys.stdout.buffer.write(
    (value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)).encode("utf-8"))' "$1"
}

# citations 帧里的证据条数(没有该帧 → 0)。逐帧解析而不是 grep:事件名与 data
# 分处两行、条数是 JSON 结构,按空格敏感的子串匹配去断它属于自找假红。
citations_count() {
  "$PYTHON" -c '
import json, sys

lines = sys.stdin.buffer.read().decode("utf-8", "replace").splitlines()
for i, line in enumerate(lines):
    if line.strip() == "event: citations" and i + 1 < len(lines) and lines[i + 1].startswith("data: "):
        sys.stdout.buffer.write(str(len(json.loads(lines[i + 1][6:]).get("items") or [])).encode("ascii"))
        break
else:
    sys.stdout.buffer.write(b"0")'
}

# citations 帧里**至少一条**证据的 question/answer 含运费标志词(输出 yes/no)。
#
# 为什么必须查到内容级:`citations` 帧只要 `chunks` 非空就发
# (`app/agent/nodes.py`),**不区分是哪些块** —— 只数条数的话,「召回了 2 条
# 与运费无关的块」也会被打成「运费条款被召回」,而判据并**没有**证明这件事
# (2026-09-20 审查者用这样一条流复现过:条数 >= 1 为真,内容断言为假)。
# 标志词与下面的内容断言**同一组**,两处口径必须一致。
citations_has_marker() {
  "$PYTHON" -c '
import json, sys

MARKERS = ("99", "包邮", "8 元")
lines = sys.stdin.buffer.read().decode("utf-8", "replace").splitlines()
for i, line in enumerate(lines):
    if line.strip() == "event: citations" and i + 1 < len(lines) and lines[i + 1].startswith("data: "):
        items = json.loads(lines[i + 1][6:]).get("items") or []
        found = False
        for item in items:
            text = str(item.get("question", "")) + "\n" + str(item.get("answer", ""))
            if any(m in text for m in MARKERS):
                found = True
                break
        sys.stdout.buffer.write(b"yes" if found else b"no")
        break
else:
    sys.stdout.buffer.write(b"no")'
}

# 文本里是否含**固定兜底话术**。判断放在 Python 里而不是 `grep -F "<中文>"`:
# 模式若被平台按 CP936 重编码,匹配会恒不命中 —— 于是「回复里没有兜底话术」
# 这条断言变成**恒真的假绿**,恰好把要防的故障放过去。
#
# ⚠️ **空输入返回 `no`**(空文本里当然不含这句话)—— 所以调用方**必须**先判空,
# 否则端点出错、一个 token 帧都没发时,这条会对着空气打出一行绿
# (2026-09-20 审查者发现的 Important 2)。现在是调用方(验收 5/6 的 ③)负责。
has_fallback() {
  "$PYTHON" -c '
import sys

text = sys.stdin.buffer.read().decode("utf-8", "replace")
sys.stdout.buffer.write(b"yes" if "我没太理解您的意思" in text else b"no")'
}

# 取一个**已发货**的订单号。
#
# 订单状态由种子派生,多数号码是「待付款 / 已付款 / 已取消」—— 那些单子**没有**
# 物流记录,对它们查物流会(正确地)返回 ToolNotFound,拿来做验收只会得到 ok=false。
# 号码不写死:写死会在种子函数变动时静默失效,表现为"验收标准没达成",
# 而不是"号码选错了"。
shipped_order() {
  "$PYTHON" -c '
import asyncio, json, sys

from app.tools.business import _LOGISTICS_BY_STATUS, query_order

def status(oid):
    call = {"name": "query_order", "args": {"order_id": oid}, "id": "p", "type": "tool_call"}
    return json.loads(asyncio.run(query_order.ainvoke(call)).content)["status"]

for i in range(1000, 1040):
    if status(str(i)) in _LOGISTICS_BY_STATUS:
        sys.stdout.buffer.write(str(i).encode("ascii"))
        break
'
}

# 从工具实现里取该订单的确定性物流状态。
# 动态取值而非写死 —— 工具改了种子函数也不必改脚本;
# 而"工具到底返回什么"由 Tier 1 的跨进程确定性测试守护。
expected_logistics_status() {
  ORDER="$1" "$PYTHON" -c '
import asyncio, json, os, sys

from app.tools.business import query_logistics

tool_call = {
    "name": "query_logistics",
    "args": {"order_id": os.environ["ORDER"]},
    "id": "probe",
    "type": "tool_call",
}
payload = json.loads(asyncio.run(query_logistics.ainvoke(tool_call)).content)
sys.stdout.buffer.write(payload["status"].encode("utf-8"))
'
}

# 知识库一致性:MySQL 的 pending/done 与 Milvus 条数一次性取出。
# 用 ORM 而非裸 SQL —— 裸 SQL 得在里面写单引号,而这段是套在 bash 的
# 单引号串里的,'done' 会把字符串提前截断。
kb_consistency() {
  "$PYTHON" -c '
import asyncio, sys

from sqlalchemy import func, select

from app.config import get_settings
from app.db.base import get_engine, get_sessionmaker
from app.db.models import KnowledgeChunk
from app.retrieval.milvus import get_vector_store


async def main():
    async with get_sessionmaker()() as session:
        total = (await session.execute(select(func.count(KnowledgeChunk.id)))).scalar_one()
        pending = (await session.execute(
            select(func.count(KnowledgeChunk.id)).where(
                KnowledgeChunk.vectorize_status == "pending"))).scalar_one()
    settings = get_settings()
    store = get_vector_store(settings.milvus_uri, settings.milvus_collection)
    milestones = store.count()
    sys.stdout.buffer.write(
        f"总行数={total} 已向量化={total - pending} 待向量化={pending} Milvus条数={milestones}".encode("utf-8"))
    await get_engine().dispose()


asyncio.run(main())
'
}

# 把 id 最小的 N 行打回待向量化,返回实际改动的行数。
# 用途见验收 7:先造出"建库没跑完"的确定性状态,再让重跑去证明它能补齐。
kb_reset_pending() {
  COUNT="$1" "$PYTHON" -c '
import asyncio, os, sys

from sqlalchemy import select, update

from app.db.base import get_engine, get_sessionmaker
from app.db.models import KnowledgeChunk


async def main():
    want = int(os.environ["COUNT"])
    async with get_sessionmaker()() as session:
        ids = list((await session.execute(
            select(KnowledgeChunk.id).order_by(KnowledgeChunk.id).limit(want))).scalars().all())
        if ids:
            await session.execute(
                update(KnowledgeChunk)
                .where(KnowledgeChunk.id.in_(ids))
                .values(vectorize_status="pending", vector_id=None))
            await session.commit()
    sys.stdout.buffer.write(str(len(ids)).encode("ascii"))
    await get_engine().dispose()


asyncio.run(main())
'
}

if [ ! -x "$PYTHON" ]; then
  echo "找不到 $PYTHON —— 请在项目根目录运行本脚本。" >&2
  exit 2
fi

# 轮询后台任务直到终态,输出最终任务 JSON;超时(600s)输出 running 并返回 1。
poll_job() {
  BASE="$BASE" JOBID="$1" "$PYTHON" -c '
import json, os, sys, time, urllib.request

base = os.environ["BASE"]
jid = os.environ["JOBID"]
for _ in range(600):
    with urllib.request.urlopen(f"{base}/api/kb/jobs/{jid}") as r:
        j = json.loads(r.read().decode("utf-8"))
    if j["status"] != "running":
        sys.stdout.buffer.write(json.dumps(j, ensure_ascii=False).encode("utf-8"))
        sys.exit(0)
    time.sleep(1)
sys.stdout.buffer.write(b"{\"status\":\"running\"}")
sys.exit(1)
'
}

# knowledge_chunks 里重复的 (category, questions, answer) 三元组数。
# 挖知识幂等 = 重复触发不重复入库,即这张表永远不出现重复三元组 ——
# 这是确定性的断言,不像「第二次 inserted==0」那样依赖 LLM 每次都抽同一批
# (deepseek 在 temperature=0 下依然非确定,CLAUDE.md 记过)。
kb_duplicate_triples() {
  "$PYTHON" -c '
import asyncio, sys

from sqlalchemy import text

from app.db.base import get_engine, get_sessionmaker


async def main():
    async with get_sessionmaker()() as s:
        n = (await s.execute(text(
            "SELECT COUNT(*) FROM (SELECT category, questions, answer FROM "
            "knowledge_chunks GROUP BY category, questions, answer HAVING COUNT(*) > 1) t"
        ))).scalar()
    sys.stdout.buffer.write(str(n).encode("ascii"))
    await get_engine().dispose()


asyncio.run(main())
'
}

# 直接查检索器,判断某短语是否被语义召回。输出 hit / miss。
# 不经过 LLM 聊天 —— 聊天里的工具选择/关键词抽取是非确定的(deepseek 在
# temperature=0 下依然非确定),会让「上传→向量化→召回」这条断言偶发假红。
# 这里用查询原文直接喂 retriever,确定性验证「新内容进了索引且能召回」。
kb_recall_hit() {
  QUERY="$1" PHRASE="$2" "$PYTHON" -c '
import asyncio, os, sys

from app.db.base import get_engine, get_sessionmaker
from app.tools.registry import build_retriever


async def main():
    async with get_sessionmaker()() as session:
        chunks = await build_retriever(session).search(os.environ["QUERY"])
    text = " ".join(c.question + " " + c.answer for c in chunks)
    sys.stdout.buffer.write(b"hit" if os.environ["PHRASE"] in text else b"miss")
    await get_engine().dispose()


asyncio.run(main())
'
}

echo "=== 前置:服务可达性 ==="
if ! curl -s -m 5 -o /dev/null "$BASE/"; then
  echo "  ❌ $BASE 没有响应。先起服务:" >&2
  echo "     $PYTHON -m uvicorn app.main:app --port 8000" >&2
  echo "     (并确认 8000 上没有**别的**旧进程占着 —— 那会让你验到旧代码)" >&2
  exit 2
fi
pass "$BASE 可达"
echo

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
SHIPPED_ORDER=$(shipped_order)
if [ -z "$SHIPPED_ORDER" ]; then
  fail "1000-1039 里找不到已发货的订单 —— 种子函数改坏了,验收 4 无法进行"
fi
EXPECTED_STATUS=$(expected_logistics_status "$SHIPPED_ORDER")
echo "  订单 $SHIPPED_ORDER 的确定性物流状态:$EXPECTED_STATUS"

# 工具取不到值时上面那条命令会**静默返回空**。空串会让 `grep -qF ""` 匹配
# 任何文本 —— 于是"回复复述了工具结果"这条断言永远通过,而它恰恰是本章
# 唯一证明"模型真的读到了工具返回"的证据。宁可在此处硬失败。
if [ -z "$EXPECTED_STATUS" ]; then
  fail "取不到订单 $SHIPPED_ORDER 的确定性物流状态 —— 下面的复述断言会退化成恒真,不再可信"
fi

SID_TOOL="acceptance-tool-$$"
OUT4=$(curl -sN -X POST "$BASE/api/chat/stream" \
  -H 'Content-Type: application/json' \
  --data-binary @- <<JSON
{"session_id":"$SID_TOOL","message":"订单 $SHIPPED_ORDER 的物流到哪了"}
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
echo "=== 验收 5:知识类问题走强制检索(验收标准 2) ==="
# ⚠️ 判据在 ch05 换了。旧断言是 `TOOLS5 = query_faq && STATES5 = ok`。ch05 用
# 「确定性骨架的强制预检索」替换了「模型自选工具」,`app/agent/nodes.py` 的
# retrieve_knowledge 节点**直接调 retriever、不走 query_faq 工具**(节点
# docstring 明写)。本题面 2026-09-20 实测 6/6 判为「退款退货」→ 知识路由,
# 链路上**没有任何 tool_call 帧**,旧断言再也拿不到 query_faq。
# 新判据断 ch05 之后真正该保证的三件事:
#   ① 走了知识路由 —— done 帧 trace 里含 retrieve_knowledge;
#   ② 真拿到了证据 —— 有 citations 帧且 items >= 1(citations 只在检索返回
#      **非空块**时才发,是「阈值过滤后还有块」的直接证据);
#   ③ 没跌进兜底   —— 回复不含固定兜底话术「我没太理解您的意思」。
#      这条是**最强的防回归**:兜底话术是**常量**(不是自由文本),断它的**缺失**
#      合法且确定性,而它正是阈值定错时用户**看得见**的那个故障
#      (intent=退款退货 → retrieve_knowledge:0 hits → confidence_gate:fail →
#       fallback_reply)。
OUT5=$(curl -sN -X POST "$BASE/api/chat/stream" \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{"message":"退货政策是什么"}
JSON
)

TOOLS5=$(echo "$OUT5" | called_tools)
STATES5=$(echo "$OUT5" | tool_result_states)
TRACE5=$(echo "$OUT5" | done_field trace)
CITES5=$(echo "$OUT5" | citations_count)
REPLY5=$(echo "$OUT5" | join_tokens)
echo "  调用的工具:[$TOOLS5]  结果:[$STATES5](ch05 起知识路由不发工具帧,空是正常的)"
echo "  citations 帧条目数:$CITES5"
echo "  回复:$REPLY5"

# 有 error 帧 = 端点故障,此时只发 error、**不发 done**,done_field 取到字面量
# null。不特判就会把基础设施故障报成「没走知识路由」(路由问题)—— 正是本项目
# 点名的「报错指向别处」。
if echo "$OUT5" | grep -q "event: error" && [ "$TRACE5" = "null" ]; then
  fail "端点发了 error 帧、没有 done 帧(基础设施故障,不是路由问题):$(echo "$OUT5" | grep -A1 'event: error' | tail -1)"
else
  case "$TRACE5" in
    *retrieve_knowledge*) pass "trace 里有强制检索节点(走了知识路由):$TRACE5";;
    *) fail "trace 里没有 retrieve_knowledge —— 没走知识路由:$TRACE5";;
  esac
fi

if [ "${CITES5:-0}" -ge 1 ] 2>/dev/null; then
  pass "citations 帧有 $CITES5 条证据(检索返回了非空块)"
else
  fail "没有 citations 帧或条目为 0(检索被阈值滤空,或根本没检索)"
fi

# ③ 空回复**判红**(同验收 6 那段):`has_fallback` 对空输入返回 `no`,
# 不先判空的话,端点出错、一个 token 帧都没发时这里会白打一行绿。
if [ -z "$REPLY5" ]; then
  fail "回复为空 —— 没有任何 token 帧(端点出错?)。「没跌进兜底」在空回复上不成立,不给绿"
elif [ "$(echo "$REPLY5" | has_fallback)" = "no" ]; then
  pass "回复没有跌进兜底话术(没走 fallback_reply)"
else
  fail "回复是兜底话术 —— 走了 fallback_reply:$REPLY5"
fi

if REASON=$(echo "$REPLY5" | has_cjk); then
  pass "回复非空且含 $REASON"
else
  fail "回复$REASON"
fi

echo
echo "=== 验收 6:邮费换说法召回(ch03 验收标准 1) ==="
# 这一条在 ch02 里是**反向**断言:「邮费」预期查不到(ch02 验收标准 3 的漏召回
# 是刻意设计的,faq 种子不含邮费条目)。ch03 把这个洞补上了 —— 运费说明进了
# 语料、query_faq 换成向量语义检索,**同一个问题现在必须答得上来**。
# 断言方向随本章反转:查不到才是失败。
OUT6=$(curl -sN -X POST "$BASE/api/chat/stream" \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{"message":"邮费是多少"}
JSON
)

TOOLS6=$(echo "$OUT6" | called_tools)
STATES6=$(echo "$OUT6" | tool_result_states)
QF6=$(states_of_tool "$TOOLS6" "$STATES6" query_faq)
TRACE6=$(echo "$OUT6" | done_field trace)
INTENT6=$(echo "$OUT6" | done_field intent)
CITES6=$(echo "$OUT6" | citations_count)
MARK6=$(echo "$OUT6" | citations_has_marker)
REPLY6=$(echo "$OUT6" | join_tokens)
echo "  调用的工具:[$TOOLS6]  结果:[$STATES6]"
echo "  intent=$INTENT6  citations 帧条目数:$CITES6  含运费标志词:$MARK6  query_faq 调用结果:[$QF6]"
echo "  trace=$TRACE6"
echo "  回复:$REPLY6"

# ⚠️ 本条**断需求,不断机制**:需求是「换种说法也要能召回运费条款」,
# 而本题面的**意图标签会摇**(2026-09-20 实测 12 次:7 次判「物流」→ BUSINESS
# → Agent 自己调 query_faq;5 次判「商品咨询」→ KNOWLEDGE → 强制预检索)。
# 两条路**都真的召回了运费条款、都答对了**(用户可见结果一致),所以断言认两条:
#   A 知识路由:done 帧 trace 含 retrieve_knowledge
#              **且** citations 帧 items >= 1 **且** 其中至少一条的
#              question/answer 含运费标志词(`citations_has_marker`)
#   B Agent 路由:Agent 自己调了 query_faq **且该次调用有结果(ok)**
# 走通任一条即为「走通了」。
#
# A 的第三项是 2026-09-20 审查后**收紧**的:citations 帧只要 chunks 非空就发、
# 不区分是哪些块,所以「条数 >= 1」只证明「召回了**若干**块」——
# 一条**召回 2 条无关块**的流能让旧判据打绿,而通过语却声称「运费条款被召回」。
#
# **主次关系(别弄反)**:真正承重的是 A 的 citations(条数 + 内容)与
# B 的 `ok`(「证据真的到了」),以及 ③ 不含兜底话术。A 里的
# `trace 含 retrieve_knowledge` **不是主判据、也当不了主判据** ——
# 链路坏掉时(检索被阈值滤空)trace 里**照样**有 `retrieve_knowledge:0 hits`,
# 节点跑了、只是没命中,所以它对「召回失败」这一类故障**没有判别力**
# (2026-09-20 用合成流与 8001 真链路各验过一次)。它只起「标记走的是哪条路」的作用。
#
# **两条路能证明的东西不对称,通过语因此不对称(不许说过头)**:
#   A 有内容级证据(citations 载荷里带 question/answer)→ 可以说「带回了含运费
#     标志词的块」;
#   B **没有**内容级证据 —— `tool_result` 帧只带 tool_call_id / ok / summary
#     (`app/agent/nodes.py`),没有召回文本,所以 B 只能证明「工具返回了非空结果」
#     (`query_faq` 仅在 0 条时抛 ToolNotFound)。**不为对称去改产品代码多传字段。**
#   ⇒ 「运费条款是否真被用上」由下面的**运费要点内容断言**负责。
#
# 记账:旧断言(`TOOLS6 = query_faq` 且 `STATES6 = ok`)是**同一枚硬币的另一面**
# —— 它只在「物流」路上通过、在「商品咨询」路上失败。也就是说本条验收
# **在 ch05 改动前后都是抛硬币**,这次的红不是 ch05 引入的。
A6=no
case "$TRACE6" in
  *retrieve_knowledge*)
    if [ "${CITES6:-0}" -ge 1 ] 2>/dev/null && [ "$MARK6" = "yes" ]; then A6=yes; fi;;
esac
B6=no
case ",$QF6," in *,ok,*) B6=yes;; esac

if echo "$OUT6" | grep -q "event: error" && [ "$TRACE6" = "null" ]; then
  fail "端点发了 error 帧、没有 done 帧(基础设施故障,不是路由问题):$(echo "$OUT6" | grep -A1 'event: error' | tail -1)"
elif [ "$A6" = "yes" ]; then
  pass "知识路由带回了**含运费标志词**的块(A:citations=$CITES6 条,含标志词=$MARK6)"
elif [ "$B6" = "yes" ]; then
  pass "Agent 自调 query_faq 拿到了非空结果(B:qf=[$QF6])—— 这条**只证明工具返回了结果**;帧里没有召回文本,运费条款是否真被用上由下面的要点断言负责"
else
  fail "两条路都没走通:intent=$INTENT6 cite=$CITES6 含标志词=$MARK6 qf=[$QF6] trace=$TRACE6"
  echo "     ↳ A 需要「知识路由 + citations >= 1 且其中至少一条含运费标志词」;"
  echo "       B 需要「Agent 真的调了 query_faq 且那次调用 ok」。"
  echo "       两者都空:先确认 8000 上跑的不是旧进程,再确认 build_kb 跑过"
  echo "       ($PYTHON scripts/build_kb.py;库里没有运费说明时会如实落空)"
fi

# ③ 空回复**判红**:空回复既证明不了「没跌进兜底」,也不该白拿一行绿
# (端点出错时一个 token 帧都没有,而 has_fallback 对空输入返回 no —— 2026-09-20 审查发现)。
if [ -z "$REPLY6" ]; then
  fail "回复为空 —— 没有任何 token 帧(端点出错?)。「没跌进兜底」在空回复上不成立,不给绿"
elif [ "$(echo "$REPLY6" | has_fallback)" = "no" ]; then
  pass "回复没有跌进兜底话术(没走 fallback_reply)"
else
  fail "回复是兜底话术 —— 走了 fallback_reply:$REPLY6"
fi

# 回复里必须出现运费条款的**要点**。比的是语料原文里的数字与措辞:
# 没召回就不可能说对,所以这条同时证明了"召回的内容真的被模型用上了"。
# 逐 token 推送会把 "99" 切成独立帧,必须先 join_tokens(脚本头注 2)。
if echo "$REPLY6" | grep -qE "99|包邮|8 元"; then
  pass "回复里含运费要点(包邮门槛 / 基础运费)"
else
  fail "回复里没有任何运费要点 —— 召回了但没用上,或者答的是别的"
fi

if REASON=$(echo "$REPLY6" | has_cjk); then
  pass "回复非空且含 $REASON"
else
  fail "回复$REASON"
fi

echo
echo "=== 验收 7:中断建库后重跑补齐(ch03 验收标准 2) ==="
# 两步造出"半途而废"的现场,缺一不可:
#   ① 把一部分行打回待向量化 —— 否则库里全是 done,**没有任何活可干**,
#      后面的 timeout 杀与不杀是同一件事,这条验收会变成恒真;
#   ② timeout 杀 build_kb —— 真的打断一次,而不是假装。
# 断言只认**终态**:重跑后没有待向量化行,且 Milvus 条数 == 已向量化行数。
# 不假设"被杀的那次一定提交了前几批"(那取决于机器速度,会引入 flaky)。
BEFORE=$(kb_consistency)
echo "  起始:$BEFORE"
RESET=$(kb_reset_pending 10)
echo "  打回待向量化:$RESET 行"

if timeout 25 "$PYTHON" scripts/build_kb.py > /tmp/kb_interrupted.log 2>&1; then
  echo "  (build_kb 在 25 秒内自己跑完了,没被杀死 —— 幂等补齐仍会被下面的重跑验证)"
else
  echo "  build_kb 已在 25 秒处被打断(退出码 $?)"
fi
echo "  中断后:$(kb_consistency)"

if ! "$PYTHON" scripts/build_kb.py > /tmp/kb_rerun.log 2>&1; then
  fail "重跑 build_kb 失败,详见 /tmp/kb_rerun.log"
  tail -3 /tmp/kb_rerun.log
fi

AFTER=$(kb_consistency)
echo "  重跑后:$AFTER"

if echo "$AFTER" | grep -q "待向量化=0"; then
  pass "重跑后没有待向量化的行(漏掉的块被捡起补齐)"
else
  fail "重跑后仍有待向量化行 —— 补齐路径没兜住:$AFTER"
fi

# Milvus 条数 == 已向量化行数。两者不等说明双写没对齐:
# Milvus 少了是漏写,多了是残留了已删 MySQL 行的陈旧向量。
# 注意行数一律由 MilvusVectorStore.count() 走 query(count(*)) 取 ——
# get_collection_stats 的 row_count 未扣 delete,不可信(spec §12)。
MILVUS_N=$(echo "$AFTER" | sed -n 's/.*Milvus条数=\([0-9]*\).*/\1/p')
DONE_N=$(echo "$AFTER" | sed -n 's/.*已向量化=\([0-9]*\).*/\1/p')
if [ -n "$MILVUS_N" ] && [ "$MILVUS_N" = "$DONE_N" ]; then
  pass "Milvus 条数($MILVUS_N)== 已向量化行数($DONE_N),双写对齐"
else
  fail "Milvus 条数($MILVUS_N)!= 已向量化行数($DONE_N) —— 双写没对齐"
  echo "     ↳ 若 Milvus 更多:集合里残留了已删行的陈旧向量,跑 build_kb --reindex 重建"
fi

echo
echo "=== 验收 8:上传 → 向量化 → 改说法召回(ch04 验收标准 1) ==="
# 上传一份带运费说明的新文档(文件名带 $$,避免重复跑撞 409),随后向量化,
# 再问一个改说法的问题,断言 query_faq 的工具结果里出现了新文档的独有短语
# (工具结果是单帧完整 JSON,不会像回复那样被逐 token 切开,可安全 grep)。
UP_NAME="超大件运费-$$.md"
UP_PHRASE="超大件商品运费单独计费"
UP_UPLOAD=$(curl -s -X POST "$BASE/api/kb/documents" \
  -H 'Content-Type: application/json' \
  --data-binary @- <<JSON
{"filename":"$UP_NAME","type":"policy","content":"# 超大件运费\n\n$UP_PHRASE,每公斤加收 3 元。\n"}
JSON
)
echo "  上传返回:$UP_UPLOAD"

if echo "$UP_UPLOAD" | grep -q '"chunks_added"'; then
  pass "上传成功,已切分入库"
else
  fail "上传失败:$UP_UPLOAD"
fi

VJOB=$(curl -s -X POST "$BASE/api/kb/jobs/vectorize")
VJOBID=$(echo "$VJOB" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin).get("job_id",""))')
if [ -n "$VJOBID" ]; then
  VJOBJSON=$(poll_job "$VJOBID") || fail "向量化任务超时"
  VSTATUS=$(echo "$VJOBJSON" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["status"])')
  if [ "$VSTATUS" = "done" ]; then
    pass "向量化任务完成"
  else
    fail "向量化任务未完成:$VJOBJSON"
  fi
else
  fail "向量化任务未启动(可能忙):$VJOB"
fi

if [ "$(kb_recall_hit '超大件商品运费单独计费吗' "$UP_PHRASE")" = "hit" ]; then
  pass "新上传文档的运费内容被语义召回"
else
  fail "未召回「$UP_PHRASE」—— 上传或向量化链路没生效"
fi

echo
echo "=== 验收 9:挖知识幂等(ch04 验收标准 2) ==="
MINE1=$(curl -s -X POST "$BASE/api/kb/jobs/mine")
MINE1ID=$(echo "$MINE1" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin).get("job_id",""))')
if [ -n "$MINE1ID" ]; then
  MINE1JSON=$(poll_job "$MINE1ID") || fail "第一次挖知识超时"
  MINE1_STATUS=$(echo "$MINE1JSON" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["status"])')
  if [ "$MINE1_STATUS" = "done" ]; then
    MINE1_INSERTED=$(echo "$MINE1JSON" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["result"]["inserted"])')
    echo "  第一次挖知识入库:$MINE1_INSERTED 条"
    pass "第一次挖知识完成"

    MINE2=$(curl -s -X POST "$BASE/api/kb/jobs/mine")
    MINE2ID=$(echo "$MINE2" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin).get("job_id",""))')
    if [ -n "$MINE2ID" ]; then
      MINE2JSON=$(poll_job "$MINE2ID") || fail "第二次挖知识超时"
      MINE2_STATUS=$(echo "$MINE2JSON" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["status"])')
      if [ "$MINE2_STATUS" = "done" ]; then
        MINE2_INSERTED=$(echo "$MINE2JSON" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["result"]["inserted"])')
        echo "  第二次挖知识入库:$MINE2_INSERTED 条"
        DUP=$(kb_duplicate_triples)
        if [ "$DUP" = "0" ]; then
          pass "重复触发无重复入库(知识库无重复三元组)"
        else
          fail "知识库存在 $DUP 组重复三元组 —— 幂等失效"
        fi
      else
        fail "第二次挖知识失败:$(echo "$MINE2JSON" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin).get("message",""))')"
      fi
    else
      fail "第二次挖知识未启动(可能忙):$MINE2"
    fi
  else
    fail "第一次挖知识失败:$(echo "$MINE1JSON" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin).get("message",""))')"
  fi
else
  fail "挖知识任务未启动(可能忙):$MINE1"
fi

echo
echo "================================"
echo "通过 $PASS 项，失败 $FAIL 项"
[ "$FAIL" -eq 0 ] || exit 1
