#!/usr/bin/env bash
# ch05 验收 1–5。前置:服务已在 8000 启动(docker start mysql milvus-standalone)。
#
# 断言依据:done 帧里的 trace / intent / agent_steps —— 本轮**确定性的**证据链。
# 不靠模型自由文本(deepseek 在 temperature=0 下依然非确定),也不 grep 原始
# SSE 流(逐 token 推送会把 "1001" 切成三帧)。
set -uo pipefail
BASE="${BASE:-http://localhost:8000}"
# 与 scripts/acceptance.sh 一致:一律用 venv 里的解释器。裸 `python` 在本机
# 不保证存在、也不保证是 venv 那个 —— 那样 new_sid 会产出空串,整个脚本静默走偏。
PYTHON="${PYTHON:-.venv/Scripts/python.exe}"
PASS=0; FAIL=0

ok()  { echo "  PASS  $1"; PASS=$((PASS+1)); }
bad() { echo "  FAIL  $1"; FAIL=$((FAIL+1)); }

# 发一条消息。请求体走 stdin heredoc —— 含中文的请求体不能走 curl 的 argv,
# MSYS2 会按 CP936 重编码,服务端只回 error parsing the body。
ask() {
  local sid="$1" msg="$2"
  # --max-time:上游卡住时不加这个,脚本会**无限期阻塞且什么都不打印**,
  # 收尾时表现为「挂住」而不是「红」—— 那是查不出来的失败。180s 足够:
  # 最长的一条(验收 5)实测单次 10~30s。
  curl -s -N --max-time 180 -X POST "$BASE/api/chat/stream" \
    -H "Content-Type: application/json" \
    --data-binary @- <<JSON
{"session_id":"$sid","message":"$msg"}
JSON
}

new_sid() { "$PYTHON" -c "import uuid;print(uuid.uuid4().hex)"; }

# ⚠️ 下面三个取帧工具一律**按字节读写 stdin/stdout**(`sys.stdin.buffer`),
# 不用 `sys.stdin.read()`。这不是风格问题,是修一个实测出来的**假红**:
# Windows 上 Python 对**管道**默认用 ANSI 代码页(cp936)+ surrogateescape,
# `sys.stdin.read()` 会把 UTF-8 的 SSE 字节解成乱码与孤立代理对,随后
# `.encode("utf-8")` 抛 `UnicodeEncodeError: ... '\udc80' ... surrogates not allowed`。
# 实测(2026-09-20):验收 1 与 4 因此**在内容其实满足条件的情况下判失败** ——
# 用同一份 /tmp/ch05_1.sse、/tmp/ch05_4.sse 做 A/B,文本模式崩、字节模式给出
# `retrieve_knowledge` 与 `客服小猫` 的正确结果。验收 2 之所以没崩,只是运气:
# 那一条的 trace 恰好全是 ASCII。
# 与 `scripts/acceptance.sh:36-37` 同一处平台陷阱(CLAUDE.md「子进程输出要显式钉编码」),
# 老脚本早就按字节读了,这里是**复发**,不是新问题。
#
# 把 token 帧拼回整段回复(逐 token 推送,不能直接 grep)
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

# 从最后一帧(done)的 data 里取一个字段
done_field() {
  "$PYTHON" -c '
import json, sys
lines = sys.stdin.buffer.read().decode("utf-8", "replace").splitlines()
data = [l[6:] for l in lines if l.startswith("data: ")]
v = json.loads(data[-1]).get(sys.argv[1])
sys.stdout.buffer.write(
    (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)).encode("utf-8"))' "$1"
}

# 数一数有多少个 tool_result 帧是**失败**的(ok 为假)。
# 存在的理由见验收 5 那段注释:它是 `"type": "tool_call"` 键在真实链路上的**唯一**探针。
# 逐帧解析而不是 grep —— `_frame` 的 JSON 分隔符格式不属于本脚本的契约,
# 用空格敏感的字符串匹配去断帧内容是自找假红。
bad_tool_results() {
  "$PYTHON" -c '
import json, sys
bad = 0
lines = sys.stdin.buffer.read().decode("utf-8", "replace").splitlines()
for i, l in enumerate(lines):
    if l.strip() == "event: tool_result" and i + 1 < len(lines) and lines[i+1].startswith("data: "):
        if not json.loads(lines[i+1][6:]).get("ok", True):
            bad += 1
print(bad)'
}

# 取 choices 帧的 options(逐帧解析,不做子串匹配)。
# 子串匹配只证明 '"handoff"' 与 '"ticket"' **在流里出现过** —— 不证明它们在
# **同一个 choices 帧**里、更不证明在它的 options 数组里(比如某个工具返回的
# JSON 里恰好也有这两个词就会假绿)。照 bad_tool_results 的范式逐帧解析。
choices_opts() {
  "$PYTHON" -c '
import json, sys
lines = sys.stdin.buffer.read().decode("utf-8", "replace").splitlines()
for i, l in enumerate(lines):
    if l.strip() == "event: choices" and i + 1 < len(lines) and lines[i+1].startswith("data: "):
        print(",".join(o.get("key", "") for o in json.loads(lines[i+1][6:]).get("options", [])))
        break'
}

# 有 error 帧。端点故障时(app/api/chat.py)只发 error、**不发 done**,
# 此时 done_field 取到的是字面量 null —— 不特判就会把**基础设施故障**
# 报成「trace 里没有强制检索节点」(路由问题),正是 CLAUDE.md 点名的
# 「报错指向别处」。
has_error() { grep -q "event: error" "$1"; }

# ---- 预检:8000 上那个进程**是不是当前代码** ------------------------------------
# 这条不是形式主义,是实测撞出来的:开发期起的 uvicorn **没有 --reload**,
# ch05 的 T8/T9 改了端点之后它仍跑旧代码。2026-09-20 实测:旧进程上
# `POST /api/ticket` 回 **405**(静态目录对非 GET/HEAD 的兜底),
# 而当前代码回 **422**(session_id 超过 32)。不预检的话五条验收**全部假红**,
# 而且看起来像「新代码坏了」—— 本项目已栽过一次的类型。
# 33 个 x 触发 TicketRequest 的 max_length=32 校验;422 只证明路由+校验层在,
# **不会**建出工单 —— 校验先于函数体(T9 有用例钉着)。
# 注意这依赖「部署的代码确实在 32 处校验」;若某个中间版本放宽了上限,
# 这一发会在预检判红的同时真的建出一张工单。
PRE_CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/api/ticket"   -H 'Content-Type: application/json'   --data-binary '{"session_id":"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}')
# 服务**没起**时 curl 给的是 000,不是 405/404。先分这一支 —— 否则下面那段
# 「多半是旧进程」会把「服务没起」误诊成「旧进程」,让人去 netstat 一个
# 根本不存在的 PID。
if [ "$PRE_CODE" = "000" ]; then
  echo "预检失败:$BASE 上没有服务在监听(curl 返回 000)。先起服务:"
  echo "  docker start mysql milvus-standalone"
  echo "  .venv/Scripts/python.exe -m uvicorn app.main:app --port 8000"
  exit 1
fi
if [ "$PRE_CODE" != "422" ]; then
  echo "预检失败:$BASE 上的服务不是当前代码 —— POST /api/ticket 期望 422,实得 $PRE_CODE。"
  echo "8000 上多半是旧进程(无 --reload,不会自己更新)。按序执行:"
  echo "  netstat -ano | grep ':8000'                                   # 记下 PID"
  echo "  powershell -NoProfile -Command \"Stop-Process -Id <PID> -Force\""
  echo "  .venv/Scripts/python.exe -m uvicorn app.main:app --port 8000"
  exit 1
fi

echo "== 验收 1:政策类问题走到强制检索节点 =="
SID=$(new_sid)
ask "$SID" "退货政策是怎么规定的" > /tmp/ch05_1.sse
T=$(done_field trace < /tmp/ch05_1.sse)
G=$(done_field gate_passed < /tmp/ch05_1.sse)
if has_error /tmp/ch05_1.sse && [ "$T" = "null" ]; then
  bad "端点发了 error 帧、没有 done 帧(基础设施故障,不是路由问题):$(grep -A1 'event: error' /tmp/ch05_1.sse | tail -1)"
else
  case "$T" in
    *"retrieve_knowledge"*) ok "trace 有强制检索节点:$T(gate_passed=$G)";;
    *) bad "trace 里没有强制检索节点:$T";;
  esac
fi

echo "== 验收 2:Agent 自己调工具作答 =="
SID=$(new_sid)
ask "$SID" "订单 1001 的物流到哪了" > /tmp/ch05_2.sse
# 断「调用与结果**一一配对**」,而不是只 grep 到 tool_call 就算数。
# 只看 tool_call 分不出「Agent 拿到了数据作答」与「每次调用都炸了」。
# ⚠️ 这里**不能**断 ok=true:订单 1001 状态是**已取消**,`query_logistics("1001")`
# 抛 ToolNotFound → 执行器转成 ok=false —— 那是**正确**行为,断 ok 会把对的判成红。
# ok 的探针留在验收 5(BAD)。
TC=$(grep -c "event: tool_call" /tmp/ch05_2.sse)
TRC=$(grep -c "event: tool_result" /tmp/ch05_2.sse)
if [ "$TC" -ge 1 ] && [ "$TC" -eq "$TRC" ]; then
  ok "Agent 自己调了工具,且每个调用都有结果($TC 次)(trace=$(done_field trace < /tmp/ch05_2.sse))"
else
  bad "调了 $TC 次、只回了 $TRC 个结果(调用未被执行?)"
fi

echo "== 验收 3:投诉 → 两个独立选项 =="
SID=$(new_sid)
ask "$SID" "我要投诉" > /tmp/ch05_3.sse
OPTS=$(choices_opts < /tmp/ch05_3.sse)
case "$OPTS" in
  *handoff*ticket*|*ticket*handoff*) ok "choices 帧的 options 同时含 handoff 与 ticket:$OPTS";;
  *) bad "choices 帧 options 不完整:$OPTS";;
esac
# 投诉**不许**进 Agent(用户需求 3:投诉 → 安抚话术 + 两个选项,不走 Agent)。
# 证据就在同一个 done 帧里:走 Agent 的路径 trace 里必有 `agent:step`。
TR3=$(done_field trace < /tmp/ch05_3.sse)
case "$TR3" in
  *"agent:"*) bad "投诉路径进了 Agent(应只走 complaint_reply):$TR3";;
  *) ok "投诉未进 Agent(agent_steps=$(done_field agent_steps < /tmp/ch05_3.sse))";;
esac

echo "== 验收 4:闲聊拿到固定话术 =="
SID=$(new_sid)
ask "$SID" "你好" > /tmp/ch05_4.sse
TXT=$(join_tokens < /tmp/ch05_4.sse)
case "$TXT" in *"客服小猫"*) ok "闲聊固定话术:$TXT";; *) bad "闲聊话术不符(可能空了):$TXT";; esac

echo "== 验收 5:复杂问题 ReAct 不止一步 =="
SID=$(new_sid)
# 问题必须是**强制串行**的:第二次工具调用的入参**只能**来自第一次的返回。
# `query_order` 与 `query_logistics` **都直接收 order_id**,所以
# 「订单 1001 的物流到哪了」这类问法,模型完全可以在**同一轮里并发**发两个
# tool_call(T6 审查实测过:那次 `agent_steps` 是 2 —— `agent_steps` 数的是
# **绑工具的轮数,含收敛的那一轮**,并发一轮 + 收敛一轮 = 2,所以它**也能过**)。
# 也就是说那个问法不会假红,但它**过的没有道理**:两步之间没有数据依赖,
# 「不止一步」成立只是因为**收敛那一轮也被算了一步**。
# 「订单 1001 买的是什么商品?那件商品现在还有货吗」在结构上不可能并发:
# 订单里有 `product` 字段(`app/tools/business.py:79`),商品名**只能**先查订单
# 才知道,所以「不止一步」是**问题本身的形状**保证的,与轮数怎么数无关。
ask "$SID" "订单 1001 买的是什么商品?那件商品现在还有货吗" > /tmp/ch05_5.sse
# ⚠️ 这一条是**概率性**的,绿色不可重放 —— 2026-09-20 实测 16 次只过约 10 次。
# 不是脚本缺陷:意图分类器对这个**混合问题**在「订单」与「商品咨询」之间摇摆,
# 而「商品咨询 → KNOWLEDGE」是设计如此(`app/agent/routing.py`),标签一变就走
# 强制检索、Agent 根本不跑(`agent_steps=0`)。**看到红先读 `intent`**:
#   intent=订单    → Agent 跑,query_order + query_product 两次串行调用,steps=3 过
#   intent=商品咨询 → retrieve_knowledge:0 hits → confidence_gate:fail → fallback
# 后半段那个 0 hits 另有根因(与本脚本无关、也是验收 1 拿不到真答案的原因):
# `retrieval_score_threshold` 0.58 是 ch03 在 **dense 余弦**分数上定的
# (ch03 spec §12:正例最低 0.609/干扰最高 0.560),ch04 加重排时**原值沿用**
# (ch04-hybrid-rerank spec §9),于是拿 dense 的阈值去卡 **reranker sigmoid** 分数。
# 实测(2026-09-20,直连 retriever):自然长问句 rerank 分 0.27~0.34(正确块在内),
# 光问「退货」0.73。**判据不变、阈值不改**(ch05 spec §11 只写「复用」),记录在案。
STEPS=$(done_field agent_steps < /tmp/ch05_5.sse)
N=$(grep -c "event: tool_call" /tmp/ch05_5.sse)
BAD=$(bad_tool_results < /tmp/ch05_5.sse)
TR=$(done_field trace < /tmp/ch05_5.sse)
# 「不止一步」必须断在 **agent:step2** 上,不能断 agent_steps>=2:
# agent_steps 是**绑工具轮次的序号**(nodes.py:243-244 `steps = step`),
# **把收敛那一轮也算了一步**,所以任何一次工具调用都会得到 2
# (实测:验收 2 只调一次 query_logistics,done 帧照样 agent_steps=2)。
# 于是 `agent_steps >= 2` 在 `N >= 2` 之外**一点判别力都没有** ——
# 把循环改成单轮、让模型在一轮里并发发两个调用(商品名可以瞎猜),
# N=2、BAD=0、STEPS=2,照样绿 —— 而那正是「ReAct 根本没有第二轮」。
# `agent:step2` 由 nodes.py:265 在**第 2 轮真的发了工具调用**时才追加,
# 所以它才真能区分「走了第二轮」与「一轮里并发发了两个」。
case "$TR" in
  *"agent:step2"*) MULTI=1;;
  *) MULTI=0;;
esac
if [ "$MULTI" -eq 1 ] && [ "$N" -ge 2 ] && [ "$BAD" -eq 0 ]; then
  ok "ReAct 走了 $STEPS 步、$N 次工具调用、0 次工具失败(trace=$TR)"
else
  # 红的时候必须打印 intent:这一条会因**意图摇摆**而红(问句被判成「商品咨询」
  # 就绕开 Agent 走知识路),那与「ReAct 真坏了」在输出上必须能分开 ——
  # 而意图摇摆是**已记账的概率性事实**(约 10/16),不是缺陷。
  bad "非真正多轮或多/少调工具:agent_steps=$STEPS tool_calls=$N 失败工具数=$BAD intent=$(done_field intent < /tmp/ch05_5.sse) trace=$TR"
fi

# ⚠️ `BAD -eq 0` 这一条**不是补充,是唯一的探针**,别删。
# `"type": "tool_call"` 键的丢失在**单测层捕获不到**:替身 `FakeTool` 不查这个键、
# `execute_tool` 原样透传 —— T6 审查实测,把该键从 `FakeChunk.__init__` 与
# `__add__` 同时删掉,`tests/test_agent_node.py` **仍然 7 passed**。而真实链路上,
# `BaseTool.ainvoke` 判「这是不是工具调用」**只看这一个键**,缺了它就把整个 dict 当
# **参数**去校验 schema → `ValidationError` → `app/tools/executor.py:77` 转成
# `ok=False` +「工具参数不合法」。**不抛异常、不报错、不写日志**,
# 只是模型永远拿不到数据、于是开始编 —— 本项目最怕的那类静默故障。
# `BAD` 就是它的探针:真模型 + 真 `@tool` + 真执行器,缺键则必然非 0。

echo
echo "结果:$PASS 通过,$FAIL 失败"
[ "$FAIL" -eq 0 ]
