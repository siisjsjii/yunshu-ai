#!/usr/bin/env bash
# ch10 验收(spec §12 验收 ④;§11.4 那个「可测的数」)。
#
# ⚠️ **本文件是 ch10-A 建的(ch10-B 的 T16 在它后面补上了验收 ①–③)**,四节分别是:
#
#   | 节 | 断什么 | 依赖 |
#   |---|---|---|
#   | ① | 冻结测试集上的评测**报告跑得出来** + 两次运行**逐字节相同** + 报告里的数与 `report.json` 对得上 + 测试集构成/切分算术 | 无(不碰网络、不碰库) |
#   | ② | 跑批**写库** + 分布接口的每个数**与另一条独立算法逐个吻合** | 旁路服务 8103 + 客服服务 8000 + MySQL |
#   | ③ | 多诉求句**同时命中多个类目** + 标签顺序与阈值一律来自产物 | 旁路服务 8103 |
#   | ④ | 转人工第九类:一句话端到端拿到工号/等待 | 客服服务 8000 + 真实 key |
#
#   为此三条装置约定:
#      * helper 一律收在文件上半段、**不藏在某一节里**,各节直接用;
#      * 收尾的判词/局限清单是**分节**打印的,补节时各自加一段,不用动别节;
#      * `$WORK` 是 `.ch10_acceptance`(与 ch09 的 `.ch09_acceptance` 分开,互不覆盖)。
#      * **①–③ 的判定逻辑过一个 heredoc 落在 `$WORK/topic_helpers.py` 里**(见
#        `write_topic_helpers`),不写成 `python -c`:含中文的源码走 argv 会被 MSYS2
#        按 CP936 重编码(ch05/ch08 各栽过一次)。argv 上只走 ASCII 的路径与数字。
#
# ⚠️ **跨工具的路径一律用 `$TEMP`(计划订正 19-A,实测)**:bash 的 `/tmp` 是 MSYS 的
#    `/tmp`,而 **Python 的 `/tmp` 是 `D:\tmp`** —— 不是同一个地方。同一段脚本里一个
#    工具写、另一个工具查,就会得到「报告明明生成了、检查却判红」这种查不出来的假红。
#    `$TEMP` 与 `tempfile.gettempdir()` 实测一致。① 的两次运行产物就写在它下面。
#
# 前置:**MySQL + 真实 key**。
#   * **不需要 Milvus**:转人工的路由值是 `HANDOFF → agent`(app/agent/graph.py),
#     这条路上**没有检索**(只有 `商品咨询` 走 `retrieve_knowledge` + 置信度闸)。
#     Milvus 没起只是 WARN,不是预检失败 —— 与 ch06/ch09 那两份**刻意不同**。
#   * **不需要 Langfuse**:本节的断言全在 SSE 帧与回复文本上,不读观测。
#
# 本脚本自己起四样东西:客服服务(8000)+ 两个 MCP Server(8101 物流 / 8102 售后,
# **尽力而为**:起不来只 WARN)+ **主题分类旁路服务(8103,T16 起)**。
#   * 客服服务必须自己起:「端口被旧进程占着」时 `wait_ready` 会连到**那个旧进程**上
#     并「通过」,于是整套验收测的是别的代码(本仓记过的「起服务前先查端口」)。
#   * 两个 MCP Server 起起来是为了让这一轮的工具集**与生产逐字相同**(内置 + MCP)。
#     起不来不影响结论:`transfer_to_human` 是**内置**工具(任务 2),
#     与 MCP 无关;少的只是 `query_logistics` 那类。
#   * **旁路服务(8103)不是「尽力而为」**:②③ 两节全靠它。起不来时那两节报**装置故障**
#     (boom),而 ④ 照常跑完 —— 一次跑完把话说全,比中途 `exit` 好。
#   * 旁路服务**只在这一个端口上**、只读产物目录,它**不在对话链路上**(spec §9.4)。
#
# ────────────────────────────────────────────────────────────────────────
# 本节断什么:**三层,缺一不可 —— 每一层失败的原因不同,所以分开报**
# ────────────────────────────────────────────────────────────────────────
#   ① 分类对     :done 帧的 intent == 转人工                (分类器的问题)
#   ② 工具被调了 :SSE 里有 transfer_to_human 的 tool_call 帧 (模型没调工具)
#   ②b 那次调用成了:同一个 `tool_call_id` 的 tool_result 是 `ok: true` (调了但失败了)
#   ③ 用户拿到了 :回复里有**工号 A###** 或**等待位次·时长**或**只说「接入」**(结果没被转述给用户)
#
# **只断 ① 的话,「分类对了但模型没调工具、用户什么也没得到」会全绿通过。**
#
# ⚠️ **③ 的口径是 spec 的字面口径,不是「只认工号」**(fix round 1,I1):
#   spec §12 验收 ④ 与 §11.4 写的是「工号**/等待时长**」—— 那个斜杠是**或**。
#   实测 9 句里 4 句只给了「当前排队第 2/3 位,预计等待约 7/8 分钟」而没报工号,
#   而那种回复**用户是拿到了结果的**。⇒ ③ 判红 = **三类 artefact 一个都没有**;
#   同时**三个子数分开打**(报出工号 / 只给位次·时长 / 只说「接入」),不许合成一个 ——
#   合成会把「模型时时报工号、时时不报」这个不一致**平均掉**,而那正是要看的。
#   ⚠️ **订正轮 1 把第三类补上了**(复审 F6):实测「已为您转接人工客服,**当前排在您前面
#   还有 1 位,预计 7 分钟左右接入**」曾因**不含「排队」「等待」**被判红 —— 位次与时长
#   都给了。⇒ needle 加「排在」「预计」;再加一类最弱的「接入」。**加词就是放松判据**,
#   这一层的性质(关键词匹配,不是语义判定)如实写在下面的局限清单里。
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

# ── 装置自检:跑它的必须是 **Git Bash**,不能是 **WSL 的 bash**(2026-09-27 实测)──
# ⚠️ 从 Windows 的 cmd / PowerShell / **Python 的 `subprocess.run(["bash", …])`**
#    里调 `bash` 时,`CreateProcess` 的搜索顺序把 **System32 排在 PATH 之前**
#    ⇒ 解析到的是 `C:\Windows\System32\bash.exe`,**那是 WSL 的 bash**,不是本仓
#    要的那个(Git Bash 在 `D:\kit\Git\usr\bin\bash.exe`,排在 PATH 第一也没用)。
#    两者的 `localhost` 与 `/tmp` **都不是同一个地方**:
#      * WSL 里 `curl http://localhost:8000` 到不了 Windows 上的客服服务(WSL2 有
#        自己的网络命名空间)⇒ 表现为「客服服务起不来」,而真因是跑错了 shell;
#      * WSL 的 `/tmp` 是 Linux 的 /tmp,而 Python 的 `/tmp` 是 `D:\tmp` ——
#        正是 19-A 那条跨工具路径陷阱的另一副面孔。
#    ⇒ 判据用 `OSTYPE`(Git Bash 实测 = `cygwin`,WSL = `linux-gnu`),**响亮地停**,
#    而不是继续跑出一堆指向别处的红。
case "${OSTYPE:-}" in
  cygwin*|msys*|mingw*|win32*) : ;;
  *)
    echo "装置自检失败:本脚本必须在 **Git Bash** 里运行,当前 shell 的 OSTYPE=[${OSTYPE:-未设置}]。" >&2
    echo "  ⚠️ 从 cmd / PowerShell / Python 的 subprocess 直接调 'bash' 会解析到 **WSL 的 bash**" >&2
    echo "     (System32 排在 PATH 之前),而 WSL 的 localhost 与 /tmp 都不是 Windows 这边的。" >&2
    echo "  请用 Git Bash 打开仓库再跑:bash scripts/acceptance_ch10.sh" >&2
    exit 2 ;;
esac

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
# ⚠️ **③ 的 artefact 集在订正轮 1 被放宽过**,理由是一条**实测的假红**:
#   模型回复「已为您转接人工客服,**当前排在您前面还有 1 位,预计 7 分钟左右接入**」
#   —— 位次与时长都给了,却因为**没有「排队」「等待」这两个词**被判红(ch10-A 那层
#   原本是这两个词的字面匹配)。5 次运行里红了 1 次,而红落在**承重句**上。
#   ⇒ 补进「排在」「预计」:它们是同一件事(等待位次 / 预计时长)的另一种说法。
H_LINEUP="6392 5728"                      # 排在(「排在您前面还有 N 位」)
H_ETA="9884 8ba1"                         # 预计(「预计 N 分钟左右接入」)
# 第三类(最弱):只说「接入」这类话 —— 用户知道在转接了,但**没拿到位次也没拿到时长**。
# **单独计数、不合成**(合成会把「模型时时报位次、时时只报一句套话」这个不一致平均掉)。
H_JOIN="63a5 5165"                        # 接入

# ⚠️ **一律用显式的 `.venv/Scripts/python.exe`**(计划订正 19-D):裸 `python` 今天
# 恰好解析到 venv,但那是**环境碰巧**,不是保证 —— 跑批/评测那几节依赖 venv 里的
# torch / transformers,解析错了会在很靠后的地方炸,而报错指向别处。
PYTHON="${PYTHON:-.venv/Scripts/python.exe}"
PORT="${PORT:-8000}"
BASE="http://localhost:$PORT"
MCP_LOGISTICS_PORT="${MCP_LOGISTICS_PORT:-8101}"
MCP_AFTERSALES_PORT="${MCP_AFTERSALES_PORT:-8102}"
#: 主题分类旁路服务(ch10-B,spec §9.1)。**8103,不是 8101/8102** —— 那两个是 MCP 的。
TOPIC_PORT="${TOPIC_PORT:-8103}"
TOPIC_MODEL_DIR="${TOPIC_MODEL_DIR:-models/topic-clf}"
LOG="log/app.log"
WORK=".ch10_acceptance"
#: ① 的两个目录。**都在 `$TEMP` 下**(见文件头那条跨工具路径陷阱)。
#: ⚠️ `EV_COPY` 必须在 `EV` **外面**:两次运行都写 `$EV`(同一个路径),第一次的
#:    五份产物**复制**到 `EV_COPY` 再比 —— 把第二次写进另一个目录的话,`report.json`
#:    里那个产物路径字段会跟着变,于是「跑两次不同」是装置自己造出来的。
EV_BASE="${TMPDIR:-${TEMP:-${TMP:-/tmp}}}"
EV="$EV_BASE/ch10_acceptance_eval"
EV_COPY="$EV_BASE/ch10_acceptance_eval_run1"

PASS=0; FAIL=0; WARN=0
# 读数(不进判词)里没通过**层**数。见下面 `check_handoff` 的 `mode` 说明。
PROBE_MISS=0
# 3 句里的几个读数:模型调了工具的句数 / 三层全过的句数 / 其中报出工号的句数 /
# 其中只给了等待·排队的句数。**后两个必须分开打**(见 §读数区那段注释)。
CALL_OK=0; ALL3_OK=0; H_ROUNDS=0; AGENT_NO_OK=0; WAIT_ONLY=0; HANDOFF_ONLY=0

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
CS_PID=""; MCP_LOG_PID=""; MCP_AFTER_PID=""; TOPIC_PID=""
KEEP=0

cleanup() {
  # 先杀客服服务:它是唯一持有会话锁的,留着会挡住下一次运行。
  for pid in "$CS_PID" "$MCP_LOG_PID" "$MCP_AFTER_PID" "$TOPIC_PID"; do
    [ -n "$pid" ] || continue
    kill "$pid" 2>/dev/null
  done
  sleep 1
  for pid in "$CS_PID" "$MCP_LOG_PID" "$MCP_AFTER_PID" "$TOPIC_PID"; do
    [ -n "$pid" ] || continue
    kill -9 "$pid" 2>/dev/null
  done
  # 失败时**保留**工作目录(SSE 原文、三个服务的控制台)。
  # 判据是 `FAIL -gt 0` **或** `KEEP -eq 1` —— 不能只看 FAIL:几条 preflight
  # 分支在**一条断言都还没打**的时候就 `exit 1`(那时 FAIL 仍是 0),
  # 只认 FAIL 的话 cleanup 会把**唯一的**那份证据删掉。
  if [ "$FAIL" -eq 0 ] && [ "$KEEP" -eq 0 ]; then
    rm -rf "$WORK"
    # ① 的两次运行产物也收干净(它们在 `$TEMP` 下,不删会一次次堆在用户临时目录里)。
    rm -rf "$EV" "$EV_COPY"
  else
    [ -n "${CH10_SELF_LOG:-}" ] && [ -f "$CH10_SELF_LOG" ] && \
      cp "$CH10_SELF_LOG" "$WORK/transcript.txt" 2>/dev/null
    echo "  (证据留在 $(pwd)/$WORK/)"
    echo "  (① 的评测产物留在 $EV 与 $EV_COPY —— 失败时它们是证据)"
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

# ── 预检:四个端口必须是空的 ────────────────────────────────────────────
# ⚠️ **8103 也在这里**(T16 起):旁路服务由本脚本自己起,端口上留着旧进程的话
#    `wait_topic` 会连到**那个旧进程**并「通过」—— 于是 ②③ 测的是别的代码,而
#    本仓在 8000 上已经栽过这个形状(「起服务前先查端口」)。
for p in "$PORT" "$MCP_LOGISTICS_PORT" "$MCP_AFTERSALES_PORT" "$TOPIC_PORT"; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://localhost:$p/")
  if [ "$code" != "000" ]; then
    echo "预检失败:端口 $p 上已经有服务在监听(返回 $code)。本脚本要自己起服务,请先清掉:"
    echo "  netstat -ano | grep ':$p'                                    # 记下 PID"
    echo "  powershell -NoProfile -Command \"Stop-Process -Id <PID> -Force\""
    exit 1
  fi
done

# ── 预检:三份产物/数据必须在(**①③ 全靠它们**,缺了就是装置故障,不是产品红)──
# ⚠️ 这里**只查「在不在」**,不查内容 —— 内容由 ① 的检查脚本逐条判(那些判据在
#    `topic_helpers.py` 里,不在这一段)。
for f in "$TOPIC_MODEL_DIR/labels.json" "$TOPIC_MODEL_DIR/inference_config.json" \
         "$TOPIC_MODEL_DIR/config.json" "$TOPIC_MODEL_DIR/model.safetensors"; do
  if [ ! -s "$f" ]; then
    echo "预检失败:训练产物 $f 不在(或 0 字节)—— ①③ 两节都跑不了。"
    echo "  先用 scripts/train_topic_clf.py 训一份,或确认 TOPIC_MODEL_DIR 指对了。"
    exit 1
  fi
done
if [ ! -s evals/topic/topic_test.jsonl ]; then
  echo "预检失败:冻结测试集 evals/topic/topic_test.jsonl 不在 —— 验收 ① 的唯一依据。"
  exit 1
fi

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
# 一组 needle 里**有没有一个在**。$1=文件;其余 = 各 needle(hex 码点)。
# 退出码沿用 `has_needle` 的三值语义:**0**=至少一个在 / **1**=都不在 / **2**=**读不了**。
# ⚠️ 2 不许当 1 —— 那是「装置故障冒充结论」(见 `has_needle` 上面那段的规矩)。
any_needle() {   # $1=文件;其余=needle
  local file="$1"; shift
  local h rc
  for h in "$@"; do
    has_needle "$file" "$h"; rc=$?
    case "$rc" in
      0) return 0 ;;
      1) : ;;
      *) return 2 ;;
    esac
  done
  return 1
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

# ── 认证(认证章 T7):三行接入,让下面每一处裸 `curl` 一个字都不用改 ──────────
#   ① 登录一次拿 token(`cinfly` 是 admin ⇒ 一个 token 同时覆盖用户面与工作台);
#   ② 用**同名函数遮蔽 `curl`** —— 此后每一处 `curl` 自动带上 Authorization 头。
# ⚠️ 登录走 python 的 `urllib` 而不是 `curl`:此刻 curl 还没被遮蔽,而「请求体不走
#    argv」是本仓的规矩(顺带不依赖 jq)。
# ⚠️ 遮蔽只在**当前 shell** 生效。七个验收脚本里 `bash -c` / `sh -c` / `command curl`
#    / 绝对路径 curl **各 0 处**(实测),而 `$(curl …)` 那类调用是**子 shell、会继承
#    函数** ⇒ 全覆盖,没有一处漏网。
# ⚠️ **登录放在 `wait_ready` 里,不放在脚本开头**:本脚本自己起服务,脚本开头那一刻
#    服务还没起 ⇒ 只会拿到空 token;而 `wait_ready` 那条 200 判据现在也要 token。
#    登录成败因此顺带就是「服务起没起」的另一半判据(拿不到 token 时它自己会打印
#    原因,而不是让下面所有断言悄悄变成 401)。
TOKEN=""
login_token() {   # $1=base ⇒ 成功时填 TOKEN 并返回 0;失败打一行原因、返回 1
  TOKEN=$(BASE="$1" "$PYTHON" - <<'PYEOF'
import json, os, sys, time, urllib.request
base, deadline, last = os.environ["BASE"], time.monotonic() + 60, None
while time.monotonic() < deadline:
    try:
        req = urllib.request.Request(
            base + "/api/auth/login",
            data=json.dumps({"username": "cinfly", "password": "123456"}).encode(),
            headers={"Content-Type": "application/json"})
        print(json.load(urllib.request.urlopen(req, timeout=10))["token"])
        sys.exit(0)
    except Exception as exc:      # 服务还没起来 ⇒ 重试;真出错也就重试这 60 秒
        last = exc
        time.sleep(1)
sys.stderr.write("登录没成(%s):%s\n" % (base, last))
sys.exit(1)
PYEOF
)
  [ -n "$TOKEN" ]
}
curl() { command curl -H "Authorization: Bearer $TOKEN" "$@"; }

wait_ready() {   # $1=秒数
  local deadline=$((SECONDS + $1))
  # 认证(见上面那段):token 只在**服务起来之后**才拿得到 ⇒ 在第一次轮询时补。
  # 放在脚本开头必然拿到空 token,而下面那条 200 判据现在也要 token。
  [ -n "$TOKEN" ] || login_token "$BASE"
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
# 旁路服务的就绪判据**比 MCP 那两个严**:`/healthz` 必须回 **200**。
# 它的 docstring 写着这个端点回的是「服务活着 + **它到底读到了什么**」——
# 也就是**权重已经加载完**(构造 `TopicClassifier` 时读的产物)。
# 用「端口非 000」会撞上一个更早的时刻:uvicorn 已经在监听、而模型还没加载完,
# 那之后的每一次 `/predict` 都可能在等加载(或者更糟,拿到半初始化状态)。
wait_topic() {   # $1=端口 $2=秒数
  local deadline=$((SECONDS + $2))
  while [ "$SECONDS" -lt "$deadline" ]; do
    [ "$(curl -s --max-time 5 -o /dev/null -w '%{http_code}' "http://localhost:$1/healthz")" = "200" ] && return 0
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
start_topic_ready() {  # $1=日志 → 成功时 pid 在 TOPIC_NEW_PID
  : > "$1"
  local attempt
  for attempt in 1 2 3; do
    # `--model` 显式给:默认值虽然也是 models/topic-clf,但把「读哪份产物」写在命令行上,
    # 排查时这一行就是第一现场(与 `train_topic_clf.py` / `eval_topic_clf.py` 同款)。
    # `>>` 不是 `>`:重试时把上一轮截掉,那条真正的错因就没了(与 `start_cs` 同款)。
    nohup "$PYTHON" -m topic_service --model "$TOPIC_MODEL_DIR" --port "$TOPIC_PORT" \
      >> "$1" 2>&1 &
    TOPIC_NEW_PID=$!
    # 120s:CPU 上加载一份 409MB 的权重量级是十几秒,给足余量(首跑还要读盘)。
    if wait_topic "$TOPIC_PORT" 120; then return 0; fi
    _kill_pid "$TOPIC_NEW_PID"
    wait_port_free "$TOPIC_PORT" 20
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
# ①–③ 用的装置(判定内核 + 判词翻译)
# ══════════════════════════════════════════════════════════════════════════

# 把 `topic_helpers.py` 落成文件。**不写 `python -c`**:含中文的源码走 argv 会被
# MSYS2 按 CP936 重编码(ch05/ch08 各栽过一次),而这里是几百行中文。
write_topic_helpers() {
  cat > "$WORK/topic_helpers.py" <<'PYEOF'
"""ch10-B 验收 ①–③ 的判定内核(权威副本在 `scripts/acceptance_ch10.sh` 的 heredoc 里)。

## 为什么是一个文件,不是 `python -c`

本仓的规矩:**含中文的源码走 argv 会被 MSYS2 按 CP936 重编码**(ch05/ch08 各栽过一次)。
所以判定逻辑写成文件(ch09 的验收也是这么做的),argv 上只走 ASCII 的路径与数字。

## 输出协议(与 bash 侧的一行一段对应)

每行一条判词,前缀四选一:

| 前缀 | bash 侧 | 含义 |
|---|---|---|
| `PASS ` | `ok` | 这条过了 |
| `FAIL ` | `bad` | 产物不对 |
| `BOOM ` | `boom` | **装置故障**:该有的东西没有 / 判不了(比 FAIL 更严重) |
| `SKIP ` | `warn` | **未复现**:这一条本轮跑不了(不是通过,也不是失败) |
| `READ ` | 读数 | 打印入账,**不进判词** |

⚠️ **bash 侧必须核「本文件有没有产出任何判词」**(`acceptance_ch10.sh` 的
`emit_results`):没有 ⇒ 判装置故障。不核的话,这里一次崩溃(语法错、traceback)
在 bash 眼里就是**一行判词都没有**,而脚本会继续往下跑并打出一个「通过」——
ch07 记过的那条(`pytest` exit 4 + `tail` 把错误切掉 ⇒ 连 `N passed` 都没有)。

## 三个模式

| 模式 | 判什么 |
|---|---|
| `report-check` | 验收 ①:评测报告跑得出来 + **两次运行逐字节相同** + 报告里的数与 `report.json` 对得上 + 测试集构成/切分算术 |
| `dist-check` | 验收 ②:跑批写库 + 分布接口的数字与**另一条独立算法**(回池子按题面数)一致 |
| `predict-check` | 验收 ③:多诉求句同时命中多类 + 标签顺序与阈值一律来自产物 |
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

# ⚠️ 本文件落在 `<repo>/.ch10_acceptance/` 下 ⇒ 上溯**一层**才是仓库根
#    (与 ch09 的 `lftrace.py` 同款;写成 `parents[2]` 会去读仓库外面)。
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

#: 评测脚本一条命令产出的五份产物(spec §8.6 是四份,`misjudged.csv` 是 §8.5 追加的)。
FIVE = ("report.md", "report.json", "matrix_confusion.csv", "matrix_flow.csv", "misjudged.csv")

#: 冻结测试集的构成(§5.5 双池分离)。**这三个数是那份已入库的冻结测试集的属性**,
#: 不是可以随实现改的东西 —— 它们变了就说明测试集换了,报告里的每个数都要重读。
TEST_ROWS = 120
TEST_REAL = 80
TEST_SYNTH = 40
#: 实测(2026-09-27):120 行里只有 12 行带人工复核痕迹(CP-2 那 84 条流过去的)。
TEST_HUMAN_REVIEWED = 12
#: `†` 的判据(spec §8.3)。**不许为了迁就数据调低它**。
SUPPORT_MIN = 15


def emit(line: str) -> None:
    """**显式钉 UTF-8 输出边界**:本机 locale 是 cp936,直接 `print` 含中文的行在
    管道/重定向时会用 GBK 编码并抛 `UnicodeEncodeError`(本仓的平台纪律)。"""
    sys.stdout.buffer.write((line + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()


def ok(msg: str) -> None:
    emit("PASS " + msg)


def bad(msg: str) -> None:
    emit("FAIL " + msg)


def boom(msg: str) -> None:
    emit("BOOM " + msg)


def skip(msg: str) -> None:
    emit("SKIP " + msg)


def read(msg: str) -> None:
    emit("READ " + msg)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tail(text: str, n: int = 6) -> str:
    lines = [l for l in text.splitlines() if l.strip()]
    return " / ".join(lines[-n:])


# ══════════════════════════════════════════════════════════════════════════
# 验收 ①:评测报告
# ══════════════════════════════════════════════════════════════════════════

def _md_row(text: str, section: str, first_cell: str) -> list[str] | None:
    """在 `report.md` 的某一节里找一行表格,按 `|` 切开。找不到返回 `None`。

    ⚠️ **必须在节内找**:`micro-F1` 这个字样在全篇出现多次(§2 那张两行表、
    §5 的三列并排、§6 的阈值表),跨节搜索会拿错行 —— 而拿错行之后两边的数
    仍然都是「报告里的数」,看不出错。节边界用 `## ` 标题切。
    """
    start = text.find(section)
    if start < 0:
        return None
    nxt = text.find("\n## ", start + 1)
    block = text[start:] if nxt < 0 else text[start:nxt]
    for line in block.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] == first_cell:
            return cells
    return None


def _as_float(cell: str) -> float:
    return float(cell.replace("*", "").strip())


def check_report(a) -> int:
    run, copy, committed = Path(a.run), Path(a.copy), Path(a.committed)
    split_dir = Path(a.split_dir)
    problems = 0

    # ── 产物存在且非空(「存在」不等于「有内容」:一个 0 字节文件也「存在」)──
    missing = [f for f in FIVE if not (run / f).is_file()]
    if missing:
        boom("评测脚本没有产出这几份产物:%s" % "、".join(missing))
        return 2
    empty = [f for f in FIVE if (run / f).stat().st_size == 0]
    if empty:
        boom("这几份产物是 0 字节:%s" % "、".join(empty))
        return 2
    sizes = {f: (run / f).stat().st_size for f in FIVE}
    ok("五份产物都写出来了(report.md %d 字节 / report.json %d 字节 / 两张矩阵 %d+%d / misjudged %d)"
       % (sizes["report.md"], sizes["report.json"], sizes["matrix_confusion.csv"],
          sizes["matrix_flow.csv"], sizes["misjudged.csv"]))

    md_bytes = (run / "report.md").read_bytes()
    md_text = md_bytes.decode("utf-8", "replace")
    # 这两条是**下限**(关键词在不在)。它的判别力很弱 —— 一份内容全错的报告也能
    # 含有这两个字样 ⇒ 真正的判据是下面「报告里的数 == report.json 里的数」那两条。
    if "micro-F1" in md_text and "# ch10-B 主题分类器" in md_text:
        ok("report.md 有抬头与 micro-F1 指标名(下限判据;数值核对在下面)")
    else:
        bad("report.md 里没有抬头或 micro-F1 字样 —— 它不像一份评测报告")
        problems += 1

    try:
        payload = json.loads((run / "report.json").read_text(encoding="utf-8"))
    except ValueError as exc:
        boom("report.json 解不开(%s: %s)" % (type(exc).__name__, exc))
        return 2

    need = ("threshold", "alt_threshold", "test_rows", "test_file_sha256_bytes", "test_provenance",
            "test_human_reviewed_rows", "test_label_slots", "labels", "grid", "per_class",
            "flagged_labels", "unflagged_labels", "support_by_stratum")
    miss = [k for k in need if k not in payload]
    if miss:
        boom("report.json 少了这些键:%s" % "、".join(miss))
        return 2

    from app.topic.taxonomy import LABELS

    # ── (1) 构成 / 三份不相交(spec §5.5 双池分离)──────────────────────────
    n_all = payload["test_rows"]
    prov = payload["test_provenance"]
    n_real, n_syn = prov.get("real", 0), prov.get("synthetic", 0)
    if (n_all, n_real, n_syn) == (TEST_ROWS, TEST_REAL, TEST_SYNTH):
        ok("测试集构成:%d = 真实 %d + 合成 %d" % (n_all, n_real, n_syn))
    else:
        bad("测试集构成不对:测得 rows=%s provenance=%s(期望 %d = 真实 %d + 合成 %d)"
            % (n_all, json.dumps(prov, ensure_ascii=False), TEST_ROWS, TEST_REAL, TEST_SYNTH))
        problems += 1
    if n_real + n_syn == n_all:
        ok("真实 + 合成 == 全体 ⇒ 没有第三种 provenance 的行被这张三列并排表静默丢掉")
    else:
        bad("真实 %d + 合成 %d != 全体 %d ⇒ 有行两列都不进" % (n_real, n_syn, n_all))
        problems += 1

    grid = payload["grid"]
    for t in sorted(grid, key=float):
        cell = grid[t]
        got = (cell["all"]["n"], cell["real"]["n"], cell["synthetic"]["n"])
        if got == (TEST_ROWS, TEST_REAL, TEST_SYNTH):
            ok("阈值 t=%s 的三个口径行数:%d = 真实 %d + 合成 %d(三份不相交且合计=输入)"
               % (t, got[0], got[1], got[2]))
        else:
            bad("阈值 t=%s 的口径行数不对:%s(期望 %s)" % (t, got, (TEST_ROWS, TEST_REAL, TEST_SYNTH)))
            problems += 1

    if list(payload["labels"]) == list(LABELS):
        ok("报告里的 %d 类标签与顺序 == app.topic.taxonomy.LABELS(spec §2.7 的契约)" % len(LABELS))
    else:
        bad("报告里的标签与权威表不一致 —— 先修标签顺序再看指标")
        problems += 1

    # ── (2) 测试集标签的来源(订正 10 ④ / 13-I)────────────────────────────
    # ⚠️ 这一条同时是**进口守卫**:`prepare_topic_data.py import-test` 一旦被执行,
    #    120 行会**全部**带上 `human_reviewed`,而其中 108 行没有人看过 ⇒ 这里判红。
    hr = payload["test_human_reviewed_rows"]
    if hr == TEST_HUMAN_REVIEWED and 0 < hr < n_all:
        ok("测试集标签的来源已入账:只有 %d/%d 行带人工复核痕迹 ⇒ 这个 F1 是「模型 vs 预标」的一致度"
           % (hr, n_all))
    else:
        bad("带 human_reviewed 的行数是 %s(实测基准 %d / 共 %d)—— 若它变成了 %d,"
            "说明 `import-test` 被跑过:那会把「人核过」打在 108 行没人看过的数据上"
            % (hr, TEST_HUMAN_REVIEWED, n_all, n_all))
        problems += 1

    # ── (3) 每类表 / † 清单 / support 与「算术上装不下」────────────────────
    per = payload["per_class"]
    if len(per) == len(LABELS):
        ok("每类表有 %d 行(= 类目数)" % len(per))
    else:
        bad("每类表有 %d 行,权威类目是 %d 个" % (len(per), len(LABELS)))
        problems += 1
    slots = sum(r["support"] for r in per)
    if slots == payload["test_label_slots"]:
        ok("每类 support 合计 %d == 报告里的标签槽总数" % slots)
    else:
        bad("每类 support 合计 %d != 标签槽总数 %s" % (slots, payload["test_label_slots"]))
        problems += 1
    flagged = [r["label"] for r in per if r["support"] < SUPPORT_MIN]
    unflagged = [r["label"] for r in per if r["support"] >= SUPPORT_MIN]
    if flagged == list(payload["flagged_labels"]) and unflagged == list(payload["unflagged_labels"]):
        ok("† 清单与每类 support 一致:%d/%d 类带 †(判据 support < %d)"
           % (len(flagged), len(LABELS), SUPPORT_MIN))
    else:
        bad("† 清单与每类 support **对不上** —— 报出去的清单不是这张表算出来的")
        problems += 1
    if len(LABELS) * SUPPORT_MIN > slots:
        ok("算术:%d 类 × %d = %d 个标签槽 > 全表 %d 个槽 ⇒ † 铺满整张表是数据规模,不是模型不行"
           % (len(LABELS), SUPPORT_MIN, len(LABELS) * SUPPORT_MIN, slots))
    else:
        bad("「算术上装不下」不成立了(%d × %d <= %d)—— 这句话要么该删,要么数据换了"
            % (len(LABELS), SUPPORT_MIN, slots))
        problems += 1
    # support 与 † 清单**只打出来入账**(订正 10 ①:不进判词)。
    read("各类 support(入账,不进判词):%s"
         % "、".join("%s=%d" % (r["label"], r["support"]) for r in per))
    read("带 † 的类(%d 个):%s" % (len(flagged), "、".join(flagged) or "(无)"))
    read("够 %d 的类(%d 个):%s" % (SUPPORT_MIN, len(unflagged), "、".join(unflagged) or "(无)"))

    # ── (4) 三个口径各自够 support 的类数(计划订正 10 ② / 订正 13-J)────────
    sbs = payload["support_by_stratum"]
    try:
        rows_ok = (sbs["all"]["rows"], sbs["real"]["rows"], sbs["synthetic"]["rows"]) == \
                  (TEST_ROWS, TEST_REAL, TEST_SYNTH)
    except KeyError:
        boom("report.json 的 support_by_stratum 少了口径键:%s" % list(sbs))
        return 2
    if rows_ok:
        ok("support_by_stratum 的三个口径行数与上面一致(%d/%d/%d)"
           % (TEST_ROWS, TEST_REAL, TEST_SYNTH))
    else:
        bad("support_by_stratum 的口径行数与测试集构成不一致:%s" % json.dumps(sbs, ensure_ascii=False))
        problems += 1
    # ⚠️ **`†` 的判据是「每一层各自」的 support**:同一个类在「全体」里够 15、
    #    在「只看真实 80」里可能只有 2 条 ⇒ 权威列(§8.4 指定的 real)会挂**多得多的** †。
    #    这三个数是那份冻结测试集的属性(实测 3 / 1 / 0),不写明的话验收的人会把它读成缺陷。
    enough = {k: sbs[k]["enough"] for k in ("all", "real", "synthetic")}
    if enough == {"all": 3, "real": 1, "synthetic": 0}:
        ok("每层各自够 support>=%d 的类数:全体 3 / 只看真实 1 / 只看合成 0"
           "（权威列 real 上挂 16/17 个 † —— 那是语料稀薄,不是模型不行）" % SUPPORT_MIN)
    else:
        bad("每层够 support 的类数实测 %s(基准 全体 3 / 真实 1 / 合成 0)—— "
            "报告里那张 † 表与测试集对不上" % json.dumps(enough, ensure_ascii=False))
        problems += 1
    read("每层标签槽数:%s"
         % "、".join("%s=%d" % (k, sbs[k]["slots"]) for k in ("all", "real", "synthetic")))

    # ── (5) 报告里的数 == report.json 里的数(「能断数值就断数值」)──────────
    # ⚠️ 这是本节**真正的**「报告内容对不对」判据:上面那两条只断「micro-F1 这个字样在不在」,
    #    而一份**每个数都错**的报告照样含这几个字样。数字从 markdown 里**重新解析**一遍,
    #    与机器可读的那份逐个比 —— 两者分家就是「给人看的那份在撒谎」。
    row = _md_row(md_text, "## 5.", "micro-F1")
    # ⚠️ **两种失败分开报**(订正轮 1,复审 M8):旧写法一律说「找不到那一行」,
    #    而实测过「**行找到了、只是单元格数不对**」—— 红是对的、**诊断是错的**,
    #    读的人会去查表标题而不是查那一行的形状。
    if row is None:
        bad("report.md 的 §5 表里**找不到** micro-F1 那一行(表还在不在?节标题换了?)"
            "—— 数值交叉核对判不了")
        problems += 1
    elif len(row) < 4:
        bad("report.md 的 §5 里 micro-F1 那一行只有 %d 个单元格(期望 >= 4:指标名 + 三列)"
            "—— 数值交叉核对判不了" % len(row))
        problems += 1
    else:
        got = [_as_float(c) for c in row[1:4]]
        if len(got) == 3 and all(abs(g - w) < 5e-5 for g, w in
                                 zip(got, [grid[str(payload["threshold"])][k]["micro_f1"]
                                           for k in ("all", "real", "synthetic")])):
            ok("report.md 里印的 micro-F1(全体/真实/合成)= %.4f / %.4f / %.4f,"
               "与 report.json 逐位吻合" % tuple(got))
        else:
            bad("report.md 印的 micro-F1 %s 与 report.json 的全体/真实/合成 %s 对不上"
                % (got, [round(grid[str(payload["threshold"])][k]["micro_f1"], 4)
                         for k in ("all", "real", "synthetic")]))
            problems += 1
    row2 = _md_row(md_text, "## 2.", "**micro-F1**")
    if row2 is None:
        bad("report.md 的 §2 里**找不到** micro-F1 那一行(与 M8 同款:先分清「没找到」"
            "与「找到但形状不对」)")
        problems += 1
    elif len(row2) < 2:
        bad("report.md 的 §2 里 micro-F1 那一行只有 %d 个单元格(期望 >= 2)" % len(row2))
        problems += 1
    else:
        got2 = _as_float(row2[1])
        want2 = grid[str(payload["threshold"])]["all"]["micro_f1"]
        if abs(got2 - want2) < 5e-5:
            ok("report.md §2 的 micro-F1 = %.4f,与 report.json 的全体值一致" % got2)
        else:
            bad("report.md §2 的 micro-F1 %.4f != report.json 的 %.4f" % (got2, want2))
            problems += 1

    # ── (6) 两次运行逐字节相同(spec §8.6)──────────────────────────────────
    if not copy.is_dir():
        boom("第一次运行的五份产物没有被复制到 %s ⇒ 「跑两次逐字节相同」这一条判不了" % copy)
        return 2
    for f in FIVE:
        p1, p2 = run / f, copy / f
        if not p2.is_file():
            boom("第一次运行的 %s 没被复制过来 ⇒ 这一条判不了" % f)
            problems += 1
        elif sha256(p1) == sha256(p2):
            ok("两次运行逐字节相同:%s" % f)
        else:
            bad("两次运行**不是**逐字节相同:%s(spec §8.6 的「不参与任何随机」不成立 —— "
                "先查 device 是不是 cuda,再查 model.eval() 与 torch.no_grad())" % f)
            problems += 1

    # ── (7) 入库的那份报告 == 本次运行的那份(报告没被手改 / 没落后于权重)──
    for f in FIVE:
        c = committed / f
        if not c.is_file():
            boom("入库的 evals/topic/%s 不在本机" % f)
            problems += 1
            continue
        if f == "report.json":
            left = json.loads(c.read_text(encoding="utf-8"))
            right = json.loads((run / f).read_text(encoding="utf-8"))
            # 产物路径那一个字段**必然**不同(两次运行写在不同的目录),其余必须逐键相同。
            left.pop("misjudged_file", None)
            right.pop("misjudged_file", None)
            diff = sorted(k for k in set(left) | set(right) if left.get(k) != right.get(k))
            if not diff:
                ok("入库的 evals/topic/report.json 与本次运行一致(除产物路径字段)")
            else:
                bad("入库的 evals/topic/report.json 与本次运行不同(差异字段:%s)—— "
                    "要么权重换过而报告没重生成,要么那份报告被手改过" % "、".join(diff))
                problems += 1
        elif sha256(c) == sha256(run / f):
            ok("入库的 evals/topic/%s 与本次运行逐字节相同" % f)
        else:
            bad("入库的 evals/topic/%s 与本次运行不同 ⇒ 盘上那份报告已经不描述当前权重"
                % f)
            problems += 1

    # ── (8) 报告绑定的测试集 == 盘上那份(报告描述的到底是哪份数据)─────────
    test_path = Path(a.test)
    if not test_path.is_file():
        boom("冻结测试集 %s 不在本机" % test_path)
        return 2
    if sha256(test_path) == payload["test_file_sha256_bytes"]:
        ok("报告记录的测试集 sha256 与盘上的 %s 逐字节吻合" % test_path.as_posix())
    else:
        bad("测试集变过:报告记的是 %s,盘上是 %s ⇒ 这份报告描述的已经不是这份数据"
            % (payload["test_file_sha256_bytes"][:16], sha256(test_path)[:16]))
        problems += 1
    test_rows = load_rows(test_path)
    comp = {p: sum(1 for r in test_rows if r.get("provenance") == p) for p in ("real", "synthetic")}
    if comp == {"real": TEST_REAL, "synthetic": TEST_SYNTH}:
        ok("测试集文件自己的 provenance 计数也是 真实 %d / 合成 %d(与报告一致)"
           % (comp["real"], comp["synthetic"]))
    else:
        # ⚠️ **红话里必须印期望的常量**(订正轮 1,复审 M7):判据是拿 `comp` 比**常量
        #    80/40**,而旧红话印的是 `comp` 与 `prov` ⇒ 实测会印出
        #    `{"real": 80, "synthetic": 41} != {"real": 80, "synthetic": 41}`
        #    (**两个操作数一模一样**,读起来像脚本自己坏了)。
        bad("测试集文件的 provenance 计数 %s != 期望的 真实 %d / 合成 %d(报告里记的是 %s)"
            % (json.dumps(comp, ensure_ascii=False), TEST_REAL, TEST_SYNTH,
               json.dumps(prov, ensure_ascii=False)))
        problems += 1

    # ── (9) 切分算术:合计 = 输入 − 不可信靶子 / 三份互不相交 ───────────────
    splits = {}
    for name in ("train", "val", "topic_test"):
        p = split_dir / (name + ".jsonl")
        if not p.is_file():
            boom("切分产物 %s 不在本机 ⇒ 切分算术这一条判不了" % p)
            return 2
        splits[name] = load_rows(p)
    ids = {name: {r["id"] for r in rows} for name, rows in splits.items()}
    total_rows = sum(len(rows) for rows in splits.values())
    total_ids = sum(len(s) for s in ids.values())
    if total_ids == len(set().union(*ids.values())):
        ok("三份切分互不相交:train %d + val %d + test %d 行,id 一个都不重复"
           % (len(splits["train"]), len(splits["val"]), len(splits["topic_test"])))
    else:
        bad("三份切分有 %d 个 id 重复出现 ⇒ 「互不相交」不成立"
            % (total_ids - len(set().union(*ids.values()))))
        problems += 1
    pre_path = Path(a.prelabeled)
    if not pre_path.is_file():
        # 输入语料**刻意不入库**(与 corpus.jsonl / synthetic.jsonl 同例)⇒ 全新 clone 上
        # 这一条跑不了。**不静默**:报成「未复现」并写进局限清单。
        skip("输入语料 %s 不在本机(它刻意不入库)⇒ 「合计 = 输入 − 不可信靶子」这一条本轮判不了"
             % pre_path.as_posix())
    else:
        from app.topic.labeling import is_unusable_target
        pre = load_rows(pre_path)
        usable = {r["id"] for r in pre if not is_unusable_target(r)}
        unusable = {r["id"] for r in pre if is_unusable_target(r)}
        if total_rows == len(pre) - len(unusable):
            ok("合计 = 输入 − 不可信靶子:%d = %d − %d"
               % (total_rows, len(pre), len(unusable)))
        else:
            bad("切分合计 %d != 输入 %d − 不可信靶子 %d ⇒ 有行既没被写出、也不是被谓词排除的"
                % (total_rows, len(pre), len(unusable)))
            problems += 1
        got = set().union(*ids.values())
        if got == usable:
            ok("落进三份的 id 集合 == 输入里「靶子可信」的那些(没有行被别的理由静默丢掉)")
        else:
            lost, extra = sorted(usable - got), sorted(got - usable)
            bad("落进三份的 id 与「靶子可信」的输入集合差 %d 个(丢了 %s / 多出 %s)"
                % (len(lost) + len(extra), lost[:3], extra[:3]))
            problems += 1
        if ids["topic_test"] <= {r["id"] for r in pre}:
            ok("测试集的每一行都来自输入语料(没有凭空造出来的行)")
        else:
            bad("测试集里有 %d 个 id 不在输入语料里"
                % len(ids["topic_test"] - {r["id"] for r in pre}))
            problems += 1
        read("输入语料 %d 行,其中不可信靶子 %d 条;三份 %d / %d / %d"
             % (len(pre), len(unusable), len(splits["train"]), len(splits["val"]),
                len(splits["topic_test"])))

    return 1 if problems else 0


# ══════════════════════════════════════════════════════════════════════════
# 验收 ②:跑批 + 分布接口
# ══════════════════════════════════════════════════════════════════════════

def _independent_distribution(limit):
    """分布接口那几个数的**另一条算法**:把 `labels` 拉回 Python 数一遍。

    ⚠️ 这正是 `app/api/topics.py` **刻意不做**的那件事(它在 SQL 里用 `JSON_TABLE` 展开)。
    拿它当交叉核对,核的才是「SQL 里那次展开对不对」——
    若两边都跑 SQL,一处写错两边一起错,断言照样绿。

    「不同问题数」按**池子的题面**数(`low_confidence_questions.question`),这正是
    计划订正 17-A 的语义:按 `low_confidence_question_id` 去重会与行数**恒等**
    (那一列上有唯一键),得到的是个没意义的数。⇒ 回归到旧写法时,这里会与接口分家。

    ⚠️ 订正轮 1 起它**还担两件事**(复审 F1/F2:那两条判据原先只读跑批脚本**自己印的数**,
    于是一次「落库变空操作」与一次「打印为 0、实际写空标签」都能全绿跑过整节):

    | 读什么 | 用来核什么 |
    |---|---|
    | `empty_this_batch` | 本轮处理的那 `limit` 条池子行里,**库里真写着空标签的有几条** ⇒ 与跑批自报的那个数比 |
    | `max_classified_at` | 全表 `MAX(classified_at)` ⇒ 与**本轮开始前的库时钟**比,证「这一轮真的写过库」 |
    """
    import asyncio

    from sqlalchemy import text as sql_text

    from app.db.base import get_engine

    async def run():
        engine = get_engine()
        async with engine.connect() as conn:
            rows = (await conn.execute(sql_text(
                "SELECT t.low_confidence_question_id, t.labels, q.question, t.classified_at "
                "FROM topic_classifications t "
                "LEFT JOIN low_confidence_questions q ON q.id = t.low_confidence_question_id"
            ))).all()
            # 跑批读的就是池子**最旧的 limit 条**(`ORDER BY id LIMIT n`,那个 ORDER BY
            # 由 `tests/test_classify_topics.py` 对着编译出的 SQL 钉住)⇒ 这一轮写的是
            # 这 n 个 id。**必须是同一把尺子**:不然拿「全表空标签数」去比「本轮自报数」
            # 会在「上一轮留下过空标签」时得到一条假红。
            first_ids = {r[0] for r in (await conn.execute(
                sql_text("SELECT id FROM low_confidence_questions ORDER BY id LIMIT :n"),
                # `limit` 缺省(没传 `--limit`)时按「全池」算 —— 用一个大数走同一条 SQL,
                # 而不是选一条不同的语句(两条语句 = 两套语义,早晚漂开)。
                {"n": int(limit) if limit else 10 ** 9},
            )).all()}
            return rows, first_ids

    try:
        rows, first_ids = asyncio.run(run())
    except Exception as exc:                      # noqa: BLE001 —— 连不上库是装置故障
        return None, "%s: %s" % (type(exc).__name__, exc)
    counts: dict[str, int] = {}
    distinct: dict[str, set] = {}
    questions = set()
    empty_all = 0
    empty_this_batch = 0
    newest = None
    for pool_id, labels_value, question, classified_at in rows:
        if isinstance(labels_value, str):
            labels_value = json.loads(labels_value or "[]")
        labels_value = list(labels_value or [])
        if question:
            questions.add(question)
        if not labels_value:
            empty_all += 1
            if pool_id in first_ids:
                empty_this_batch += 1
        if classified_at is not None and (newest is None or classified_at > newest):
            newest = classified_at
        for label in labels_value:
            counts[label] = counts.get(label, 0) + 1
            if question:
                distinct.setdefault(label, set()).add(question)
    return {"rows": len(rows), "counts": counts, "questions": len(questions),
            "distinct": {k: len(v) for k, v in distinct.items()},
            "empty_all": empty_all, "empty_this_batch": empty_this_batch,
            "batch_ids": len(first_ids), "max_classified_at": newest}, None


def db_now(a) -> int:
    """打印**库自己的时钟**(`SELECT NOW()`,格式 `YYYY-MM-DD HH:MM:SS`)。

    为什么不用客户端时钟:实测本机 Docker 里的 MySQL 跑在 **UTC**,而主机是本地时区
    —— 两者差 8 小时,拿客户端的「现在」去比 `classified_at` 会得到一条**永远为真
    或永远为假**的断言(而那两样都不报错)。⇒ **同一把钟才可比。**
    """
    import asyncio

    from sqlalchemy import text as sql_text

    from app.db.base import get_engine

    async def run():
        engine = get_engine()
        async with engine.connect() as conn:
            return (await conn.execute(sql_text("SELECT NOW()"))).scalar()

    try:
        value = asyncio.run(run())
    except Exception as exc:                      # noqa: BLE001
        emit("!!! 读库时钟失败:%s: %s" % (type(exc).__name__, exc))
        return 2
    emit(value.strftime("%Y-%m-%d %H:%M:%S") if hasattr(value, "strftime") else str(value))
    return 0


def check_dist(a) -> int:
    problems = 0
    log_path = Path(a.batch_log)
    if not log_path.is_file():
        boom("跑批的日志文件 %s 不在 ⇒ 这一节判不了" % log_path)
        return 2
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if a.batch_rc != 0:
        bad("跑批退出码 %s(非 0)⇒ 它**没有**把归类写进库。日志末尾:%s" % (a.batch_rc, tail(text)))
        return 1
    ok("跑批退出码 0")
    if "\n!!! " in "\n" + text:
        bad("跑批日志里有脚本自己喷的失败行:%s" % tail(text))
        problems += 1
    else:
        ok("跑批日志里没有脚本自己喷的失败行")
    m = re.search(r"池子里读到 (\d+) 行", text)
    if not m:
        boom("跑批日志里找不到「池子里读到 N 行」⇒ --limit 的语义判不了")
        return 2
    got_n = int(m.group(1))
    # ⚠️ 判据是 `<= limit` **不是 `== limit`**:`--limit` 的上限语义只有在池子比它大时
    #    才会打满。写成「必须等于」的话,池子变小的那天(或全新库上池子为空)这条会红,
    #    而报出来的话是「上限没生效」—— 一条**指错方向**的红。
    #    「上限整个没生效」仍然抓得住:那时读的是**全池**(今天 65 > 20)。
    if a.limit is None or got_n <= a.limit:
        ok("跑批读了池子里最旧的 %d 条(<= --limit %s;语句里有 ORDER BY id,"
           "由 tests 对着编译出的 SQL 钉住)" % (got_n, a.limit))
    else:
        bad("--limit %s 却读到 %d 条池子行 ⇒ 上限没生效" % (a.limit, got_n))
        problems += 1
    empties = [int(x) for x in re.findall(r"空标签 (\d+) 条", text)]
    if not empties:
        boom("跑批日志里找不到「空标签 N 条」⇒ 这一条判不了")
        return 2
    # ⚠️ **订正轮 1(复审 F3)把这一条翻过来了。** 旧写法是「有空标签 ⇒ FAIL」,
    #    而那是**错的**:`labels: []` **是合法的模型输出**(一个低信号问题全部低于阈值),
    #    `scripts/classify_topics.py` 的订正 15-C 与
    #    `tests/test_classify_topics.py::test_empty_labels_are_allowed_but_counted`
    #    明写着「**允许写,但必须被数出来**」。照旧写法,一次**合法**的模型输出会让
    #    承重验收判红,而且红话给出的诊断(「而那不是业务结果」)与代码明写的行为相反。
    #    ⇒ 现在判的是**两个数一不一致**(脚本印的 vs 库里真写的),不是「必须是 0」。
    reported_empty = empties[-1]
    if len(set(empties)) == 1:
        ok("跑批报的空标签条数前后一致(%d 条)—— 它是不是真的,下面拿库核" % reported_empty)
    else:
        bad("跑批前后两行报的空标签条数不一致:%s(同一轮里两个数对不上)" % empties)
        problems += 1
    m = re.search(r"总行数 (\d+) / 已归类的池子行数 (\d+)", text)
    batch_total, batch_distinct = (int(m.group(1)), int(m.group(2))) if m else (None, None)
    if m is None:
        boom("跑批日志里找不到「总行数 N / 已归类的池子行数 M」⇒ 与接口的交叉核对判不了")
        return 2

    if a.curl_rc != 0:
        boom("curl 分布接口失败(退出码 %s)⇒ 这一节判不了" % a.curl_rc)
        return 2
    dist_path = Path(a.dist)
    if not dist_path.is_file() or dist_path.stat().st_size == 0:
        boom("分布接口的响应体是空的(%s)" % dist_path)
        return 2
    try:
        payload = json.loads(dist_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        boom("分布接口返回的不是 JSON(%s):%s" % (type(exc).__name__, tail(dist_path.read_text(
            encoding="utf-8", errors="replace"), 3)))
        return 2

    need = ("total", "distinct_questions", "last_classified_at", "model_versions", "buckets")
    miss = [k for k in need if k not in payload]
    if miss:
        boom("分布接口的响应少了这些键:%s" % "、".join(miss))
        return 2

    from app.topic.taxonomy import LABELS

    total = int(payload["total"] or 0)
    buckets = payload["buckets"]
    if total > 0:
        ok("分布接口有数据:total=%d" % total)
    else:
        bad("分布接口的 total 是 0 ⇒ 跑批的写没有落到 topic_classifications")
        problems += 1
    if batch_total is not None and batch_total == total:
        ok("跑批自己数的总行数 %d == 接口的 total(两条独立实现同源)" % batch_total)
    else:
        bad("跑批数到 %s 行,接口 total=%d ⇒ 两处对同一张表的数法分家了" % (batch_total, total))
        problems += 1
    if len(buckets) == len(LABELS) and {b["label"] for b in buckets} == set(LABELS):
        ok("17 个类目一个不少(没有数据的类补 0,页面不会「少几类」)")
    else:
        bad("bucket 的类目集合与权威表不一致:%d 个" % len(buckets))
        problems += 1
    s = sum(int(b["count"]) for b in buckets)
    if s >= total:
        ok("各 bucket 条数合计 %d >= total %d(多标签行在每个标签下各算一次)" % (s, total))
    else:
        bad("各 bucket 条数合计 %d < total %d ⇒ 有行的标签没被展开" % (s, total))
        problems += 1
    if payload["model_versions"]:
        read("model_versions=%s / last_classified_at=%s"
             % ("、".join(payload["model_versions"]), payload["last_classified_at"]))
    else:
        bad("model_versions 是空的 —— 分布页说不清这张图是谁算的")
        problems += 1

    indep, err = _independent_distribution(a.limit)
    if indep is None:
        boom("独立复算连不上库 ⇒ 分布数字的交叉核对判不了(%s)" % err)
        return 2
    if indep["rows"] == total:
        ok("独立复算(topic_classifications 全表 %d 行)与接口 total 一致" % indep["rows"])
    else:
        bad("独立复算是 %d 行,接口 total=%d" % (indep["rows"], total))
        problems += 1
    dq = int(payload["distinct_questions"] or 0)
    if dq == indep["questions"]:
        ok("不同问题数 %d == 回池子按题面数的 %d(计划订正 17-A 的语义)"
           % (dq, indep["questions"]))
    else:
        bad("不同问题数接口给 %d,回池子按题面数是 %d ⇒ 接口很可能退回了按 "
            "low_confidence_question_id 去重(那一列有唯一键 ⇒ 它会与行数恒等,是个没意义的数)"
            % (dq, indep["questions"]))
        problems += 1
    mismatch = []
    for b in buckets:
        label = b["label"]
        want_c = indep["counts"].get(label, 0)
        want_d = indep["distinct"].get(label, 0)
        if int(b["count"]) != want_c or int(b["distinct"]) != want_d:
            mismatch.append("%s: 接口 %s/%s vs 复算 %s/%s"
                            % (label, b["count"], b["distinct"], want_c, want_d))
    if not mismatch:
        ok("17 个 bucket 的「条数 / 不同问题数」逐个与独立复算一致")
    else:
        bad("bucket 与独立复算对不上的:%s(最多列 3 个)" % " ; ".join(mismatch[:3]))
        problems += 1
    # ── 空标签:**拿库里的实际行数**核跑批自己报的那个数(订正轮 1,复审 F1)──────
    # ⚠️ 复审的实测:把跑批脚本的 `"labels": labels` 改成 `[] if written == 0 else labels`
    #    (**计数逻辑一字不动**)⇒ 一行空标签**真的落进了库**,而旧判据只看脚本印的数
    #    ⇒ **`通过 54 / 失败 0`** 照样报出来,还打印「本次没有空标签行」。
    #    库侧的独立证据当时就在同一份转录里(`各 bucket 条数合计` 79 → 78)。
    #    ⇒ 判据改成**两个数比**:跑批自报的 vs 库里那一批行真写着的。
    if indep["empty_this_batch"] == reported_empty:
        ok("本轮处理的那 %d 条池子行在库里写着空标签的有 %d 条 == 跑批自报的 %d 条"
           "(空标签是合法模型输出,但**必须被数出来** —— 订正 15-C)"
           % (indep["batch_ids"], indep["empty_this_batch"], reported_empty))
    else:
        bad("跑批自报空标签 %d 条,而库里本轮那 %d 条行里**真写着空标签的有 %d 条** ⇒ "
            "那个读数是脚本自报的、不来自库 —— **故障会被读成业务结果**"
            % (reported_empty, indep["batch_ids"], indep["empty_this_batch"]))
        problems += 1
    read("空标签:全表 %d 条 / 本轮那 %d 条里 %d 条 / 跑批自报 %d 条"
         % (indep["empty_all"], indep["batch_ids"], indep["empty_this_batch"], reported_empty))

    # ── 这一轮**真的写过库**吗(订正轮 1,复审 F2)────────────────────────────
    # ⚠️ 复审的实测:把跑批脚本的 `await session.execute(stmt)` 换成 `return len(values)`
    #    (**落库变空操作**)⇒ 日志照印「已写 20 行」,整节 **54/0、退出码 0**。
    #    ⇒ 一个「批处理再也不写库」的回归可以全绿地跑过整个验收,因为这一节其余判据
    #    看的都是表的**总量**,而池子早就全归过类 ⇒ 总量不变。
    #    判据:`classified_at` 的**最大值**必须 >= **这一轮开始前取的库时钟**。
    #    ⚠️ 两头都用**库的**时钟:实测本机 Docker 里的 MySQL 跑的是 **UTC**(与主机差 8 小时),
    #    拿客户端的「现在」比会得到一条永远为真(或永远为假)的断言 —— 而那两样都不报错。
    if got_n == 0:
        skip("池子这一轮读到 0 行 ⇒ 「这一轮真的写过库」无从谈起(它本来就该什么都不写)")
    elif not (a.db_start or "").strip():
        boom("拿不到库时钟(--db-start 是空的)⇒ 「这一轮真的写过库」判不了")
        problems += 1
    else:
        try:
            t0 = datetime.strptime(a.db_start.strip(), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            boom("库时钟的格式不是 YYYY-MM-DD HH:MM:SS([%s])⇒ 这一条判不了" % a.db_start)
            t0 = None
            problems += 1
        newest = indep["max_classified_at"]
        if t0 is not None and newest is not None and newest >= t0:
            ok("库里最新的 classified_at=%s >= 本轮开始前的库时钟 %s ⇒ 这一轮真的写过库"
               % (newest, t0))
        elif t0 is not None:
            bad("库里最新的 classified_at=%s **早于**本轮开始前的库时钟 %s ⇒ "
                "跑批说它写了 %d 行,而库里一条新写都没有(**落库是个空操作**)"
                % (newest, t0, got_n))
            problems += 1

    read("分布读数:total=%d / 不同问题数=%d / bucket 条数合计=%d / 前 3 个 bucket=%s"
         % (total, dq, s, "、".join("%s:%s" % (b["label"], b["count"]) for b in buckets[:3])))
    # ⚠️ **同一个中文名,曾经两个意思**(订正轮 1 起名字分开,语义一个字没改):
    #    `scripts/classify_topics.py` 印的那个是 `COUNT(DISTINCT low_confidence_question_id)`
    #    = 「**已归类的池子行数**」(唯一键保证与行数恒等),而**接口**那个是
    #    「池子里有多少条**不同题面**」(计划订正 17-A 改过的语义)。今天 65 vs 33。
    #    名字是复审 M10 钉的:改**印出来的名**,不动任何语义(接口那个名是 17-A 刚订正、
    #    页面正在读的)。这里仍然**只打出来入账、不断言** —— 断言成哪一个都会把另一个说成错的。
    if batch_distinct is not None and batch_distinct != dq:
        read("跑批印的「已归类的池子行数」=%s 与接口的「不同问题数」%d **不是同一个数**"
             "(前者是「有多少条池子行已被归类」,后者是「池子里多少条不同题面」)"
             % (batch_distinct, dq))
    return 1 if problems else 0


# ══════════════════════════════════════════════════════════════════════════
# 验收 ③:多诉求句同时命中多类
# ══════════════════════════════════════════════════════════════════════════

def check_predict(a) -> int:
    problems = 0
    for name, path in (("healthz", a.healthz), ("predict", a.predict), ("request", a.request)):
        if not Path(path).is_file():
            boom("%s 的响应/请求体文件 %s 不在本机 ⇒ 这一节判不了" % (name, path))
            return 2
    try:
        health = json.loads(Path(a.healthz).read_text(encoding="utf-8"))
        payload = json.loads(Path(a.predict).read_text(encoding="utf-8"))
        request_body = json.loads(Path(a.request).read_text(encoding="utf-8"))
    except ValueError as exc:
        boom("响应不是 JSON(%s: %s)⇒ 这一节判不了" % (type(exc).__name__, exc))
        return 2

    from app.topic.model import load_artifacts
    from app.topic.taxonomy import LABELS

    art = load_artifacts(a.model_dir)
    threshold = float(art["threshold"])
    if health.get("status") == "ok":
        ok("旁路服务 /healthz 说自己是 ok")
    else:
        bad("旁路服务 /healthz 的 status 不是 ok:%s" % health.get("status"))
        problems += 1
    # ⚠️ **标签顺序是本章要防的第二个静默陷阱**(spec §2.7):整体反序之后服务照常起、
    #    num_labels 还是 17、scores 全在 0–1、分布页画得出来 —— 每一类都错位而无人报错。
    if list(health.get("labels") or []) == list(LABELS):
        ok("服务的标签顺序 == app.topic.taxonomy.LABELS(17 类,逐位)")
    else:
        bad("服务的标签顺序与权威表不一致 —— 分布图会整张错位,而每个组件都工作正常")
        problems += 1
    if float(health.get("threshold")) == threshold:
        ok("服务的阈值 %s == 产物 inference_config.json 里的 %s" % (threshold, threshold))
    else:
        bad("服务的阈值 %s != 产物里的 %s ⇒ 它没有从产物读阈值"
            % (health.get("threshold"), threshold))
        problems += 1
    if int(health.get("max_length") or -1) == int(art["max_length"]):
        ok("服务的 max_length %s == 产物里的(训练/推理同一种截断)" % art["max_length"])
    else:
        bad("服务的 max_length %s != 产物里的 %s"
            % (health.get("max_length"), art["max_length"]))
        problems += 1

    texts = request_body.get("texts") or []
    results = payload.get("results")
    if isinstance(results, list) and len(results) == len(texts) and texts:
        ok("返回条数 == 输入条数(%d)" % len(texts))
    else:
        boom("返回的 results 不是列表、或条数与输入不等(输入 %d 条,返回 %s)"
             % (len(texts), len(results) if isinstance(results, list) else type(results).__name__))
        return 2
    eps = 1e-9
    for i, (text, res) in enumerate(zip(texts, results)):
        labels = res.get("labels")
        scores = res.get("scores")
        if not isinstance(scores, dict) or set(scores) != set(LABELS):
            bad("第 %d 条的 scores 不是「17 个类目各一个概率」的形状" % (i + 1))
            problems += 1
            continue
        if any(not (0.0 <= float(v) <= 1.0) for v in scores.values()):
            bad("第 %d 条的 scores 有落在 [0,1] 之外的值" % (i + 1))
            problems += 1
        expect = {label for label, p in scores.items() if float(p) >= threshold - eps}
        if set(labels or []) == expect:
            ok("第 %d 条「%s」的标签 == 逐标签过阈值的解码结果(p >= %s)"
               % (i + 1, text, threshold))
        else:
            bad("第 %d 条「%s」的标签与逐标签过阈值的结果不一致:实得 %s,按阈值应是 %s "
                "⇒ 解码不是「每个 logit 独立过阈值」(取 argmax 就退化成单标签了)"
                % (i + 1, text, "、".join(labels or []) or "(空)",
                   "、".join(sorted(expect)) or "(空)"))
            problems += 1
        read("第 %d 条「%s」→ %s" % (i + 1, text, "、".join(labels or []) or "(空)"))

    first = results[0].get("labels") or []
    if len(first) >= 2:
        ok("多诉求句「%s」同时命中 %d 个类目:%s"
           % (texts[0], len(first), "、".join(first)))
    else:
        bad("多诉求句「%s」只命中 %d 个类目(期望 >= 2)%s ⇒ 多标签退化了"
            % (texts[0], len(first), ":%s" % "、".join(first) if first else "(一个都没命中)"))
        problems += 1
    return 1 if problems else 0


# ══════════════════════════════════════════════════════════════════════════

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ch10-B 验收 ①–③ 的判定内核")
    sub = ap.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("report-check")
    p.add_argument("--run", required=True, help="本次运行的五份产物所在目录")
    p.add_argument("--copy", required=True, help="**第一次**运行的五份产物的字节副本目录")
    p.add_argument("--committed", required=True, help="入库的那五份产物的目录(evals/topic)")
    p.add_argument("--split-dir", required=True, help="三份切分产物所在目录(evals/topic)")
    p.add_argument("--prelabeled", required=True, help="切分端的输入(corpus 合并产物)")
    p.add_argument("--test", required=True, help="冻结测试集 jsonl")

    p = sub.add_parser("dist-check")
    p.add_argument("--batch-log", required=True)
    p.add_argument("--batch-rc", type=int, required=True)
    p.add_argument("--dist", required=True)
    p.add_argument("--curl-rc", type=int, required=True)
    p.add_argument("--limit", type=int, default=None)
    #: 跑批**开始之前**取的**库时钟**(`db-now` 模式)⇒ 用来证「这一轮真的写过库」。
    p.add_argument("--db-start", default="")

    sub.add_parser("db-now", help="打印库自己的时钟(给 --db-start 用)")

    p = sub.add_parser("predict-check")
    p.add_argument("--predict", required=True)
    p.add_argument("--healthz", required=True)
    p.add_argument("--request", required=True)
    p.add_argument("--model-dir", required=True)

    args = ap.parse_args(argv)
    handler = {"report-check": check_report, "dist-check": check_dist,
               "predict-check": check_predict, "db-now": db_now}[args.mode]
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
PYEOF
}
write_topic_helpers

# 把内核的一行一段判词翻成 ok/bad/boom/warn/读数。$1=结果文件 $2=内核的退出码
emit_results() {
  local rc="$2"
  # ⚠️ **先核「有没有判词」**:内核崩了(语法错、import 失败、traceback)时这个文件里
  #    一行判词都没有 —— 不核的话它在 bash 眼里就是「什么都没说」,脚本接着往下跑
  #    并打出一个「通过」,而那一节根本没被验。ch07 记过这个形状(`pytest` 的 exit 4
  #    被 `tail` 切掉之后,连 `N passed` 都没有)。
  if ! grep -qE '^(PASS|FAIL|BOOM|SKIP) ' "$1"; then
    boom "装置故障:检查脚本没有产出任何判词(退出码 $rc)⇒ 这一节一条都没验。它的输出:"
    sed 's/^/      /' "$1"
    return 0
  fi
  local line
  while IFS= read -r line; do
    case "$line" in
      "PASS "*) ok "${line#PASS }" ;;
      "FAIL "*) bad "${line#FAIL }" ;;
      "BOOM "*) boom "${line#BOOM }" ;;
      "SKIP "*) warn "${line#SKIP }" ;;
      "READ "*) echo "  读数  ${line#READ }" ;;
      *) echo "      $line" ;;
    esac
  done < "$1"
  return 0
}

# 跑一次内核并把它的判词转进计数。$1=结果文件(留在 $WORK 里当证据);其余=内核 argv
topic_results() {
  local out="$1"; shift
  "$PYTHON" "$WORK/topic_helpers.py" "$@" > "$out" 2>&1
  emit_results "$out" "$?"
}

# 验收 ① 的两次运行。**两次都写 $EV 这同一个目录** —— 理由见变量区那段注释。
eval_run() {   # $1=日志文件
  "$PYTHON" scripts/eval_topic_clf.py --report "$EV/report.md" > "$1" 2>&1
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
  #    判红 = **三类 artefact 一个都没有**(工号 / 位次·时长 / 只说「接入」)。
  # ⚠️⚠️ **这一类判据是关键词匹配,不是语义判定**(订正轮 1 记):它的假红已经发生过一次
  #    —— 「排在您前面还有 1 位,预计 7 分钟左右接入」不含「排队」「等待」两词而被判红。
  #    放宽的办法是**加词**(「排在」「预计」「接入」,见文件头),而**加词就等于放松判据**;
  #    更严的做法(对着 `tool_result` 的字段断)会改成「工具给了什么」而不是
  #    「用户拿到了什么」—— 那不是这一层要问的问题。⇒ 口径就是这样,如实写在局限里。
  agent_no=$(printf '%s' "$reply" | grep -oE 'A[0-9]{3}' | head -1)
  # 位次·时长那一类(等待 / 排队 / 排在 / 预计):`any_needle` 三值分开 ——
  # 0=至少一个在 / 1=都不在 / 2=**读不了**(装置故障,不许当 1)。
  has_wait=no; has_join=no
  any_needle "$reply_file" "$H_WAIT" "$H_QUEUE" "$H_LINEUP" "$H_ETA"; rc_wait=$?
  case "$rc_wait" in
    0) has_wait=yes ;;
    1) : ;;                                     # 都确实不在 ⇒ 这一类的确没给
    *) boom "$label:装置故障:读不了回复文件(any_needle 返回 $rc_wait)—— 这一轮判不了"; return 4 ;;
  esac
  any_needle "$reply_file" "$H_JOIN"; rc_join=$?
  case "$rc_join" in
    0) has_join=yes ;;
    1) : ;;
    *) boom "$label:装置故障:读不了回复文件(any_needle 返回 $rc_join)—— 这一轮判不了"; return 4 ;;
  esac
  if [ -z "$agent_no" ] && [ "$has_wait" = "no" ] && [ "$has_join" = "no" ]; then
    "$report_bad" "$label ③ 调用成功了(ok=true),但回复里**工号 / 等待位次·时长 / 接入 三类一个都没有** —— 模型没把这次转接的结果转述给用户(这是**模型没转述**,不是产品没转出去)。回复前 200 字:$(head_chars "$reply_file" 200)"
    return 3
  fi
  ALL3_OK=$((ALL3_OK+1))
  # **三个子数分开记**(见下面读数区的说明):报出工号 / 只给了位次·时长 / 只说了「接入」。
  # 合成一个数会把「模型时时报位次、时时只报一句套话」这个不一致**平均掉**。
  if [ -n "$agent_no" ]; then
    AGENT_NO_OK=$((AGENT_NO_OK+1)); artefact="工号=$agent_no"
  elif [ "$has_wait" = "yes" ]; then
    WAIT_ONLY=$((WAIT_ONLY+1)); artefact="只给了等待位次/时长(**没报工号**)"
  else
    HANDOFF_ONLY=$((HANDOFF_ONLY+1)); artefact="只说了「接入」这类话(**既没工号也没位次·时长**)"
  fi
  "$report_ok" "$label 三层全过:intent=转人工,调了 transfer_to_human(ok=true),${artefact}"
  return 0
}

# ══════════════════════════════════════════════════════════════════════════
# 起环境
# ══════════════════════════════════════════════════════════════════════════
echo "== 起环境:客服服务($PORT)+ 两个 MCP Server(尽力而为)+ 旁路推理服务($TOPIC_PORT)=="
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
# 旁路服务**不是「尽力而为」**:②③ 两节全靠它。起不来时不 exit —— 那会让 ④
# 也跟着跑不了、把一条装置故障报成「整轮什么都没验」。改成记一个标记,由 ②③
# 各自报**装置故障**(boom),④ 照常跑完。
TOPIC_OK=1
if start_topic_ready "$WORK/topic_service.log"; then
  TOPIC_PID="$TOPIC_NEW_PID"
  echo "  旁路推理服务已就绪($TOPIC_PORT,pid=$TOPIC_PID,$TOPIC_MODEL_DIR)"
else
  TOPIC_OK=0
  show_console_head_tail "$WORK/topic_service.log"
fi

# 预热(BGE-M3)。等不到不致命(本节链路**不检索**),但要说清楚。
wait_warmup 180
case $? in
  0) echo "  BGE-M3 预热完成" ;;
  2) warn "BGE-M3 预热**失败**(日志里有那条)—— 本节不检索,不影响本节的结论" ;;
  *) warn "180s 内没等到预热完成 —— 本节不检索,不影响本节的结论" ;;
esac

# ══════════════════════════════════════════════════════════════════════════
# 验收 ①:冻结测试集上的评测报告(spec §12 验收 ①;§8.6 的可复现)
# ══════════════════════════════════════════════════════════════════════════
# ⚠️ **判据不止「报告生成了」**:一个 0 字节文件也「存在」。这一节断四组东西:
#    (a) 五份产物都写出来了(report.md/json + 两张矩阵 + misjudged.csv);
#    (b) **两次运行逐字节相同**(§8.6 说「不参与任何随机」—— 这句要么被验,要么删);
#    (c) 报告里**印出来的数** == `report.json` 里的数(一份每个数都错的报告
#        照样含 `micro-F1` 这四个字母 ⇒ 只 grep 关键词是零判别力的);
#    (d) 测试集构成 / 三份切分不相交 / 合计 = 输入 − 不可信靶子(订正 10 ①)。
# 具体判据全在 `$WORK/topic_helpers.py report-check` 里(那边一条一段,可单独重跑)。
echo ""
echo "== 验收 ①:冻结测试集上的评测报告(两次运行逐字节相同)=="
rm -rf "$EV" "$EV_COPY"; mkdir -p "$EV"
if ! eval_run "$WORK/eval_run1.log"; then
  boom "评测脚本第一次运行就失败(退出码非 0)—— 这一节判不了。日志:"
  show_console_head_tail "$WORK/eval_run1.log"
elif [ ! -s "$EV/report.md" ] || [ ! -s "$EV/report.json" ]; then
  # ⚠️ 这一支专治 19-A 那类**跨工具路径**陷阱:退出码 0,而产物落在了别处
  #    (bash 的 /tmp 与 Python 的 /tmp 不是一个地方)。报成装置故障并**把路径打出来**,
  #    否则会表现成「报告没生成」而真因是路径。
  boom "评测脚本退出码 0,但 $EV 下没有产物 —— 报告没写到这个路径(跨工具路径陷阱?)"
  show_console_head_tail "$WORK/eval_run1.log"
else
  mkdir -p "$EV_COPY"
  cp "$EV"/report.md "$EV"/report.json "$EV"/matrix_confusion.csv \
     "$EV"/matrix_flow.csv "$EV"/misjudged.csv "$EV_COPY"/
  if ! eval_run "$WORK/eval_run2.log"; then
    boom "评测脚本第二次运行失败 —— 「跑两次逐字节相同」这一节判不了。日志:"
    show_console_head_tail "$WORK/eval_run2.log"
  else
    topic_results "$WORK/report_results.txt" report-check \
      --run "$EV" --copy "$EV_COPY" --committed evals/topic \
      --split-dir evals/topic --prelabeled evals/topic/prelabeled.jsonl \
      --test evals/topic/topic_test.jsonl
  fi
fi

# ══════════════════════════════════════════════════════════════════════════
# 验收 ②:跑批归类 + 分布接口能读到(spec §12 验收 ②)
# ══════════════════════════════════════════════════════════════════════════
# ⚠️ **这一节会写库**(`topic_classifications` 的 upsert)。写是幂等的(唯一键
#    `uk_pool_question` 保证「重跑 = 覆盖」),池子今天已全部归过类 ⇒ 这一轮写的是
#    同一批行的覆盖,分布页的数不会变。
# ⚠️ 断言的核心**不是**「接口返回了 200」:那对一个 `total=0`、bucket 全 0 的响应
#    同样成立。这里断的是**接口的每个数与另一条独立算法逐个吻合** —— 独立那一半
#    在 Python 里回池子按题面数(正是 `app/api/topics.py` 刻意不在 SQL 里做的那种数法),
#    两边分家就是「SQL 里那次展开错了,而页面照画」。
echo ""
echo "== 验收 ②:跑批归类 + 分布接口能读到 =="
if [ "$TOPIC_OK" -eq 0 ]; then
  boom "装置故障:旁路服务($TOPIC_PORT)没起来 ⇒ 跑批与分布这一节判不了(不是产品红)"
else
  # ⚠️ **跑批开始之前**取一次**库自己的时钟**(订正轮 1,复审 F2):用来证「这一轮真的
  #    写过库」。取不到就是空串,检查脚本会据此报**装置故障**(而不是静默跳过那一条)。
  DB_T0=$("$PYTHON" "$WORK/topic_helpers.py" db-now 2>/dev/null)
  echo "  ── 本轮开始前的库时钟:$DB_T0"
  "$PYTHON" scripts/classify_topics.py --limit 20 > "$WORK/classify.log" 2>&1
  CLASSIFY_RC=$?
  echo "  ── 跑批输出:"
  sed 's/^/     /' "$WORK/classify.log"
  # `-o` 写文件而不是把 JSON 塞进变量:本仓记过 `'''$DIST'''` 内插进 `python -c`
  # 有多脆(引号/转义)。文件同时是检查脚本的输入与证据。
  curl -s --max-time 60 "$BASE/api/topics/distribution" -o "$WORK/dist.json"
  DIST_RC=$?
  topic_results "$WORK/dist_results.txt" dist-check \
    --batch-log "$WORK/classify.log" --batch-rc "$CLASSIFY_RC" \
    --dist "$WORK/dist.json" --curl-rc "$DIST_RC" --limit 20 --db-start "$DB_T0"
fi

# ══════════════════════════════════════════════════════════════════════════
# 验收 ③:多诉求句同时命中多个类目(spec §12 验收 ③)
# ══════════════════════════════════════════════════════════════════════════
# ⚠️ **不走对话链路**:直接打旁路服务的 `/predict`。走聊天的话测的是「模型会不会调
#    工具」,而那与分类器无关(而且那一路非确定,§5.3 的 ≥30% 多标签配额正是为它准备的)。
# 中文请求体**走文件**,不走 curl 的 argv(MSYS2 会按 CP936 重编码):那个文件同时是
# 检查脚本的「输入」—— 两侧看到的是同一份字节,不靠各自的转义。
echo ""
echo "== 验收 ③:多诉求句同时命中多个类目(不走对话链路)=="
if [ "$TOPIC_OK" -eq 0 ]; then
  boom "装置故障:旁路服务($TOPIC_PORT)没起来 ⇒ 多标签这一节判不了(不是产品红)"
else
  cat > "$WORK/predict_req.json" <<'JSON'
{"texts":["买大了想退","发票怎么开","退货运费谁出"]}
JSON
  curl -s --max-time 120 -X POST "http://127.0.0.1:$TOPIC_PORT/predict" \
    -H 'Content-Type: application/json' --data-binary @"$WORK/predict_req.json" \
    -o "$WORK/predict.json"
  curl -s --max-time 30 "http://127.0.0.1:$TOPIC_PORT/healthz" -o "$WORK/healthz.json"
  topic_results "$WORK/predict_results.txt" predict-check \
    --predict "$WORK/predict.json" --healthz "$WORK/healthz.json" \
    --request "$WORK/predict_req.json" --model-dir "$TOPIC_MODEL_DIR"
fi

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
echo "       其中 **只给了等待位次/时长**(没报工号)的:$WAIT_ONLY/$H_ROUNDS   ← 模型没报工号,但用户拿到了结果"
echo "       其中 **只说了「接入」这类话**(既没工号也没位次·时长)的:$HANDOFF_ONLY/$H_ROUNDS   ← 这一类是**最弱**的 artefact(订正轮 1 加的)"
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
echo "  ① 评测报告   $WORK/report_results.txt(判词)/ $EV 与 $EV_COPY(两次运行的五份产物)"
echo "  ② 分布接口   $WORK/dist_results.txt / classify.log(跑批输出)/ dist.json(接口原文)"
echo "  ③ 多标签     $WORK/predict_results.txt / predict.json / predict_req.json / healthz.json"
echo "  ④ 转人工     $WORK/handoff_main.sse(承重句)/ handoff_b.sse / handoff_c.sse(读数)"
echo ""
echo "== 局限(「通过」不等于这些也被验过)—— 如实列在这里,不许读成全绿 =="
# ── ①–③(ch10-B)的局限 ────────────────────────────────────────────────
echo '  * **① 那个 F1 是在「预标标签」上测的,不是在人工标注的黄金集上**:120 行里只有'
echo '    12 行带人工复核痕迹(用户 2026-09-26 拍板跳过测试集人工复核)。**逐类的 F1'
echo '    一条都不能当结论引用**(14/17 类带 †);能引用的只有全体/分层那几个汇总数,'
echo '    而 §8.4 指定的权威列是「只看真实 80」(那一列上 16/17 类带 †)。'
echo '    根因是**真实语料按类稀薄**(尺码 4 / 发票 14)—— 解救办法是补真实语料,'
echo '    不是调阈值、也不是把测试集做大(做大只改善合并列)。'
echo '  * **① 不覆盖「预标自己错了多少」**:**模型错 还是 标签错**要人看'
echo '    `evals/topic/misjudged.csv`(判错 39 条):若多数是标签错,该修的是数据不是训练。'
echo '    本脚本**不预判**这一点(§8.5 那份 CSV 的全部价值就是由人来做这个区分)。'
echo '  * **① 的「合计 = 输入 − 不可信靶子」依赖 `evals/topic/prelabeled.jsonl`,'
echo '    而它刻意不入库** ⇒ 全新 clone 上那一条报**「未复现」**(不是通过)。'
echo '  * **① 不改阈值、更不重训**:报告里 `t=0.3` 那一行只是冻结测试集上的无偏读数'
echo '    (测试集没参与任何选择),本轮采用的仍是产物里的 0.5。'
echo '  * **① 的「入库报告 == 本次运行」是一条硬断言**:它把验收与**当前这份权重**绑在'
echo '    一起 —— 换了权重而没重生成报告 ⇒ 本节红。那是「报告落后了」,不是产品坏了。'
echo '  * **② 写的是共享表**:池子今天已全部归过类 ⇒ 这一轮 `--limit 20` 是**覆盖写**'
echo '    (upsert 幂等),本脚本**没有**断言「本次新增了几行」(那要另造一条新池子行)。'
echo '  * **② 只验接口,不验页面**:分布页(admin.html 的标签页)是 Vibe Coding 的产物,'
echo '    「图有没有画对」本脚本看不到 —— 它只保证**接口那几个数的来源是对的**。'
echo '  * **② 的「不同问题数」是题面(未清洗)的不同数**:`clean()` 是 Python 侧的唯一'
echo '    实现,SQL 里那一次展开**没有**用清洗后的文本。'
echo '  * **③ 判的是「≥2 个类目」,不判具体是哪两个**:三个探针命中的类目**只打出来入账**。'
echo '    换一份权重它们可能合理地变 ⇒ 把具体类目写进判词会得到一条随训练摆动的断言。'
echo '  * **③ 走的是 `/predict`,不是对话链路** ⇒ 「用户说话之后分类器用没用上」不在这里验'
echo '    (主链路零调用分类器是 §9.4 的设计,由两条**源码扫描**测试守着)。'
echo '  * **错别字鲁棒性没有任何测试覆盖**(spec §13 第 6 条):保留 + 注入错别字是设计'
echo '    选择,而本脚本这三个探针**一个错别字都没有**。这是本章的已知盲区,不许写成「已解决」。'
# ⚠️ 下面这一条是复审的残余风险 #4,也是**最该写进去**的一条:读验收的人容易把
#    「54/0」当成**质量门**,而 ① 只断「报告跑得出来、数与数据对得上」。
echo '  * **⚠️ 这四节验收对「分类质量」零上界**:① 断的是「报告跑得出来、印出来的数与'
echo '    report.json 对得上、测试集/切分的算术成立」——**它不断 F1 有多高**。'
echo '    一份 F1 掉到 0.2 的权重,只要 ③ 那句「买大了想退」还能命中 ≥2 类、解码仍自洽,'
echo '    照样会得到「通过 N / 失败 0」。**读这个数的时候别把它当成质量门。**'
# ── ④(转人工)的局限 ─────────────────────────────────────────────────
# ⚠️ **这几行一律用单引号**:里面有反引号(在双引号里会被 bash 当**命令替换**执行 ——
# ch09 实测踩过,喷了一屏 `import: command not found` 而脚本照样打「通过」)。
echo '  * **承重的是 1 句**。另外 2 句是读数(不进判词)⇒ 「承重句过了」**不等于**'
echo '    「转人工稳定」,后者要看上面那几个数(3 句里调了几次、全过几次、报了几次工号)。'
echo '  * **第 ② 层失败的成因是结构性的,不是本脚本能修的东西**:转人工走主力 Agent'
echo '    意味着「发生不发生」取决于模型记不记得调工具,而模型只被要求决定**意图标签**。'
echo '    这是 ch06「模型只决定标签、不决定走向」的唯一一处例外(spec §11.3/§13.5),'
echo '    **没有结构保证,只有这个比例**。'
echo '  * **③ 的口径是「工号**或**等待位次·时长**或**只说「接入」」(spec 的字面是前两类,'
echo '    第三类是本轮放宽加的)—— 它证明的是「结果被转述了」**;'
echo '    「模型报不报工号」是**另一个**读数,已单列(报出工号 / 只给位次·时长 / 只说「接入」),'
echo '    **没有合成一个数**。'
echo '  * ⚠️ **③ 是关键词匹配,不是语义判定**:它认的是五串字(等待/排队/排在/预计/接入)与'
echo '    工号正则。放宽过一次(订正轮 1):「排在您前面还有 1 位,预计 7 分钟左右接入」'
echo '    曾因不含「排队」「等待」被判红 —— 而**加词就等于放松判据**。'
echo '    ⇒ 一条**措辞完全不同但确实转述了**的回复仍可能判红(已知,未再放宽)。'
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
  echo "四节全过(① 评测报告 / ② 分布接口 / ③ 多标签 / ④ 转人工)—— **局限见上,不许读成全绿**"
else
  echo ""
  echo "有失败项 —— 证据留在 $WORK/,**不许把它读成全绿**"
fi
[ "$FAIL" -eq 0 ] || exit 1
