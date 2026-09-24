#!/usr/bin/env bash
# ch09 验收 1–6(端到端)。前置:**MySQL + Milvus + 真实 key + Langfuse 可达**
# (比 ch08 多了 Milvus 与 Langfuse;`.env` 里那三个 `LANGFUSE_*` 缺一个,整套观测
# 就是 no-op,验收 ①/⑤ 会红 —— 那是**配置缺失**,不是代码坏了)。
#
# **本脚本自己起三样东西**:两个 MCP Server(8101 物流 / 8102 售后)+ 客服服务(8000)。
#   * 客服服务必须自己起:①⑤ 要的是**本轮**产生的 trace 与 intent tag,而
#     「端口被旧进程占着」时 `wait_ready` 会连到**那个旧进程**上并「通过」,
#     于是整套验收测的是别的代码(本仓记过的「起服务前先查端口」)。
#   * 两个 MCP Server 是**尽力而为**(起不来只 WARN):ch09 的前置里没有它们 ——
#     ⑤ 那一轮业务问题只要求**意图标签**,而标签打在 `classify_intent` 之后,
#     与工具能不能执行无关。起了它们只是让那一轮更像真的(少 4.8s 的发现等待,
#     见 CLAUDE.md 里那条「本机对一个已关闭的回环端口 ~2.05s」)。
#
# ── 沿用的七条(ch08 那份 1230 行的脚本是模板)────────────────────────────
#   1. **`KEEP` 与 `FAIL` 分离**,`fail_exit()` 先置 `KEEP=1` **再**退出 ——
#      失败路径不许删自己的证据;
#   2. **`EXIT` 与 `INT TERM HUP` 分开 trap**(关掉终端那条路径不经过 `EXIT`);
#   3. **中文 needle 用十六进制码点构造**,不写进源码字面量 —— 它们要当**命令行
#      参数**传出去,而 MSYS2 按 CP936 重编码 ⇒ 沉默地匹配不上,永远判「没有」。
#      请求体里的中文可以写字面量(走 stdin heredoc 或内部字符串,不经过 argv);
#   4. `wait_ready` 用墙钟 + `curl --max-time`,不用固定 sleep;
#   5. **断言前先 `join_tokens` 把 SSE 帧拼回**(逐 token 推送会把一个词切碎);
#   6. 端口可用 `PORT` / `MCP_*_PORT` 覆盖;
#   7. 凡是对 `low_confidence_questions` / `review_queue` / `tool_audit_logs` 的
#      断言,**一律按 `conversation_id` 或本轮生成的主键过滤**(这些表共享、只追加:
#      2026-09-24 T16 走查之前池子 30 行、队列 0 行;2026-09-25 T19 那次跑之前
#      池子 **56 行**、`matched_review_id IS NOT NULL` **56**(⇒ 未处理 **0**)。
#      ⚠️ **这几个数一律是当时的读数**,随运行次数只增不减;**断言不要拿它们当基线**
#      (本仓已编目过「被上次运行的数据污染」那一类假红/假绿)。
#
# ── 本章特有的四条(每条都是被实测逼出来的)──────────────────────────────
#   A. **验收 ③ 的选题必须是「商品咨询」类**(商品功能/规格/价格/库存)。知识路径
#      **只挂在那一个意图上**(`app/agent/routing.py:INTENT_TO_ROUTE`),服务/政策类
#      问法实测被判成 `其他 → 兜底`,**根本不检索** —— 那样的题面即使知识已经
#      入库、也能召回来,重问照样答不对,而红的原因与飞轮无关(T16 走查实测 6 种
#      说法:5 次兜底 + 1 次弹订单卡片)。C1。
#   B. **验收 ② 的「召回片段快照」只能是 JSON null —— 而这条断言的判别力到此为止。**
#      ⚠️ **先说清「闸挡下 ⟺ 检索为空」是什么**:它**今天成立,但它是两个旋钮默认值的
#      巧合,不是结构性质**(本条在复审 F1 里被订正过,原来这里写的是一条假定律):
#        * 闸的判据是 `bool(evidence) and evidence_confidence >= evidence_confidence_threshold`;
#        * **单条块**的合成分是 `0.8*top1 + 0.0667`(只有一块时 `gap == top1`,
#          两个权重合起来是 0.8;`count` 项是 `0.2*(1/3)`)⇒ **过闸临界 `top1 ≥ 0.1667`**
#          (实测于 2026-09-24,生产默认值:0.16 → 0.1947 被拦、0.1667 → 0.2000 过闸、
#          0.19 → 0.2187 过闸。复核脚本见 `.superpowers/probe_gate_formula.py`);
#        * 而 `retrieval_score_threshold = 0.25`(`app/tools/registry.py` 传给检索器)
#          在**进闸之前**就把低于 0.25 的块筛掉了 ⇒ 证据非空时 `top1 ≥ 0.25 > 0.1667`,
#          **必然过闸**;
#        * `app/config.py:74` 自己就写着那两个旋钮是**刻意避开**的 ⇒ **把它们调近
#          (一次很自然的调参)就会让闸开始拦「手里有块」的行**,那时这条「定律」立刻失效,
#          而本文下面那些判据的前提也跟着变。
#      ⇒ 由此:② 问的是**召回为空**的问题 ⇒ 没有任何片段可快照 ⇒ 那一行只能是 JSON `null`。
#      脚本对这一半的判据是「**null 且 `reject_reason` 就是「检索为空」**」,并且
#      **先用独立探针 `GET /api/kb/search` 证明这条问题确实召不到** —— 否则一个编程错误
#      会产生一模一样的行(C7)。
#      ⚠️ **并且如实说:这条断言对它本该抓的那个 bug 是不变的** —— 零召回那一支里
#      「闸**写了**这一列」与「闸**漏了** `evidence_snapshot=` 这个 kwarg」落出来的
#      都是 JSON `null`(闸今天传的是 `_snapshot(evidence) if evidence else None`)。
#      「闸的快照列有内容」那一支(手里有块却被拦)在本链路**结构上够不到**,由
#      `tests/test_agent_gate_ch09.py` 守着 —— 见脚本尾部的「局限」。**不为了凑覆盖
#      去构造弱召回场景**(副本仓规矩:一条会自带前提的断言比没有更坏)。
#      **「快照里有内容」那一半由验收 ④ 覆盖**:👎 那条路的 `evidence_snapshot` 是
#      **回捞重跑**的产物,选一条召得到的问题就有内容,而且能按内容回溯到本轮的问题。
#   C. **选题从候选表里现挑**:每跑一轮,② 会往知识库**真的写进一条**并核准,
#      于是那条问题**下一轮就变得召得到了**。脚本因此**每次运行时挑第一条「当前
#      召不到」的候选**(独立探针判的),并在候选用尽时**响亮地报**「请往
#      `CAND_*` 里加一条」—— 这比把题面写死、第二轮起悄悄变红要诚实。
#      ⇒ **这是一个「候选预算」,不是「可以无限重跑」的脚本。**
#      实测轨迹(**逐次可核,转录里都印着「选中候选 N」**):T18 的八次运行消耗 `#1–#7`
#      (run8 收在「只剩 `#8` 一条」);**`#8` 死在 2026-09-25 T19 的第一次跑**;
#      T19 随后补了 `#9–Q16`(8 条);三次跑又消耗 `#9` / `#10`
#      ⇒ **T19 收工时可用的是 `#11–#16`,共 6 条**。
#      ⚠️ **消耗是「被跳过」不是「被复用」**:用过的候选,其问题已被 ③ 核准进知识库 ⇒
#      下一轮**召得到** ⇒ 选题判据(「此刻召不到」)**永远跳过它**
#      ⇒ **剩余 = 总数 − 已消耗,编号不重置**(此处一度被算成「剩 13」= 拿 16 减 3,
#      漏了 T18 吃掉的那 7 条;T19 复审抓到的)。
#      补候选时**先用生产同款探针核 `hits == 0`** —— 实测起草的 22 条里**有 10 条召得到**,
#      靠眼睛判不出来。
#   D. **Langfuse ingestion 有延迟**(spec §3.5:首次读回 0 条、隔一会儿才有)
#      ⇒ ①/⑤ 一律**轮询**,不读一次就断言。
#
# ── 两条**不许绕过**的工程约束 ──────────────────────────────────────────
#   * 「界面能点开」那一半**靠人**:脚本能断的只有「数据在不在」(观测存在、
#     名字齐、同一条 trace)。① 的输出里明写了这一点,不装作脚本验过 UI(C10)。
#   * 飞轮是**单槽**的:手动触发拿到 **409 不是失败**(C8)。三处落池钩子会在
#     `classify_intent` 之后 fire-and-forget 起同一张槽的任务,而那个任务有 300s
#     寿命上界;脚本的轮询因此**容忍 409**、并可以靠 `GET /api/kb/jobs` 看它在不在跑。
set -uo pipefail

cd "$(dirname "$0")/.." || exit 2
[ -f app/main.py ] || { echo "请在项目根目录运行本脚本(找不到 app/main.py)" >&2; exit 2; }

# ── 自我转录(收尾那条「输出干净性自检」的输入,见文件后半 `output_cleanliness_check`)──
# **为什么要转录**:本脚本的绿/红只由显式的 `ok/bad/boom/warn` 喂,**没有任何东西校验
# 脚本自己的输出**。实测踩过(fix round 1):收尾那段「局限」原用反引号写在双引号里 —
# bash 把它当**命令替换**执行,喷出一屏 `import: command not found` / `syntax error`,
# **而脚本照样打「6/6 通过」**。⇒ 本轮输出必须能被脚本自己再看一遍。
#
# 形状:**父进程**`tee` 一份转录并等子进程(自己的另一份)跑完,用 `PIPESTATUS[0]` 原样
# 传回它的退出码。`tee` 是**流式**的 ⇒ 人照样实时看到输出,而转录逐行落盘。
# ⚠️ 转录放在 `log/` 里(已被 .gitignore 忽略),**不放 `$WORK`** —— `$WORK` 开头会被
# `rm -rf` 重建,放进去等于让 tee 写一个已经被删掉的文件。
SELF_LOG="${CH09_SELF_LOG:-$PWD/log/acceptance_ch09_self.log}"
if [ -z "${CH09_SELF_LOG:-}" ]; then
  mkdir -p log
  export CH09_SELF_LOG="$SELF_LOG"
  bash "$0" "$@" 2>&1 | tee "$SELF_LOG"
  exit "${PIPESTATUS[0]}"
fi

PYTHON="${PYTHON:-.venv/Scripts/python.exe}"
PORT="${PORT:-8000}"
BASE="http://localhost:$PORT"
MCP_LOGISTICS_PORT="${MCP_LOGISTICS_PORT:-8101}"
MCP_AFTERSALES_PORT="${MCP_AFTERSALES_PORT:-8102}"
LOG="log/app.log"
WORK=".ch09_acceptance"

# ── needle 一律十六进制码点(规矩 3)─────────────────────────────────────
H_FALLBACK="62b1 6b49 2c 6211 6ca1 592a 7406 89e3 60a8 7684 610f 601d 2c 53ef 4ee5 518d 8bf4 5f97 5177 4f53 4e00 4e9b 5417 3f"   # 兜底话术
H_PRODUCT="5546 54c1 54a8 8be2"            # 商品咨询
H_LOGISTICS="7269 6d41"                    # 物流
H_ORDER="8ba2 5355"                        # 订单
H_EMPTY_RETRIEVAL="68c0 7d22 4e3a 7a7a"    # 检索为空
H_USER_FEEDBACK="7528 6237 53cd 9988"      # 用户反馈
H_WARMUP_OK="9884 70ed 5b8c 6210"          # 预热完成
H_WARMUP_FAIL="9884 70ed 5931 8d25"        # 预热失败
H_HOTTEST="6700 70e7"                      # 最烧(「最烧 token 的意图」那一行)
# ④ 快照的内容回溯用。⚠️ **不能用「饮水机」** —— 那三个字**就在 ④ 的问题里**
# (`Q_FEEDBACK`),拿它当 needl 只能抓到「回复/快照里复述了问题」这种恒真的东西。
# 用**只该出现在被召回片段里**的串:饮水机那几块的 `section_path` 与正文都带型号,
# 而那几块正是这个问题召回来的(实测 `624 / 0.4604 / 商品规格手册 > 自动饮水机 Plus(型号 MH-W40)`)。
H_WATER="4d 48 2d 57"                      # MH-W(型号前缀)

# ── ② 的候选(可答性由运行时的独立探针判,见 Header C)──────────────────
# 每条:题面 / 核准答案 / 答案里的特征串(hex)。特征串是**中文四字词组**:不带空格、
# 不带数字 —— 实测模型转述知识块时会逐字抄这类词组,而 `12 毫米` 那种写法模型可能
# 写成 `12毫米`,于是断言红在一个**与缺陷无关**的排版差异上。
CAND_Q1="自动喂食器基础版能放冻干粮吗?"
CAND_A1="可以,基础版粮桶支持混合投放冻干粮,建议每次投喂量不超过一百二十克。"
CAND_M1="6df7 5408 6295 653e"              # 混合投放
CAND_Q2="全景看护摄像头的云存储怎么收费?"
CAND_A2="云存储按月订阅,七天循环录制每月九元,三十天循环录制每月十九元;也可用 TF 卡本地存储。"
CAND_M2="5faa 73af 5f55 5236"              # 循环录制
CAND_Q3="智能猫砂盆 Pro 的废砂盒可以水洗吗?"
CAND_A3="可以,废砂盒支持整盒水洗,水温不要超过四十摄氏度,建议每五到七天清洗一次。"
CAND_M3="6574 76d2 6c34 6d17"              # 整盒水洗
CAND_Q4="大型猫爬架的层板能承重多少公斤?"
CAND_A4="每层平台最大承重十公斤,顶层吊床承重八公斤,请勿超载使用。"
CAND_M4="9876 5c42 540a 5e8a"              # 顶层吊床
CAND_Q5="全景看护摄像头支持红外夜视吗?"
CAND_A5="支持,夜视靠红外补光实现,全黑环境下也能看清,画面是黑白的。"
CAND_M5="7ea2 5916 8865 5149"              # 红外补光
CAND_Q6="自动饮水机的滤芯可以自己换吗?"
CAND_A6="可以自己换,滤芯是耗材,打开上盖取出旧芯换新即可,建议按提示周期更换。"
CAND_M6="6253 5f00 4e0a 76d6"              # 打开上盖
CAND_Q7="大型猫爬架能拆洗吗?"
CAND_A7="能拆洗:绒布平台可拆下机洗,剑麻柱身不能水洗,日常用干布擦拭即可。"
CAND_M7="5e72 5e03 64e6 62ed"              # 干布擦拭
CAND_Q8="大号三档加热垫的电源线有多长?"
CAND_A8="电源线长约一点八米,线身有防抓咬套管,插座离得近就够用。"
CAND_M8="4e00 70b9 516b 7c73"              # 一点八米
# ⚠️⚠️ **CAND_Q8 / M8 留在原样,是一个已知会红的样本** —— 见下面「特征串两条选材规矩」
# 的第二条。**别把它「修好」**:它记的是 2026-09-25 T19 那次真暴露出来的失败形态。
#
# ── 特征串两条选材规矩(第一条原有,第二条 2026-09-25 实测补上)────────────
# 1. **特征串必须真的出现在它自己的答案里**(否则 ③ 那条断言从构造上就不可能成立)。
#    加候选时**用脚本自己的解码器核一遍**(`chr(int(h,16))`,不是 UTF-8 字节 ——
#    这两件事极易写混,2026-09-25 就混过一次)。读数:16 条里 **15 条**成立,
#    唯一的例外是 #8(M8 含数字,见第 2 条,刻意留着)。
# 2. ⚠️ **特征串里不许出现数字 —— 阿拉伯数字与中文数字都不行。**
#    实测(T19 那次跑,候选 #8):核准答案写的是「电源线长约**一点八米**」,而模型
#    转述成「**1.8 米**」 ⇒ `has_needle` 判红,而**产品那一侧每一步都是对的**
#    (写了块、向量化了、召得回来、闸过、回复带引用 `[1]`、不再是兜底话术)。
#    ⇒ **数字是这一族断言唯一不稳定的一类 token**(模型会把中文数字写成阿拉伯数字,
#    带一个空格),原注释只防了反方向(`12 毫米` → `12毫米`),**那次是它没防住的方向**。
#    **判据**:特征串取**实词词组**,不取任何形态的数字。CAND_M9 起的八条都按这条选的。
# 3. ⚠️ **核准答案必须真的回答那个问题。**(2026-09-25 第二次跑实测补上,**这条最要命**。)
#    那一轮我写的 A9 是「滚筒为匀速缓转设计,清理时噪音较低」,而题面问的是
#    「**转速是多少**」——**答案没回答题面**。真机结果(trace 逐帧):
#    `retrieve_knowledge:1 hits top=0.81 > confidence_gate:pass > agent:step1
#     tool=query_product > agent:self_assess_insufficient > agent:converged`
#    ⇒ 检索中了、闸也过了,**模型自己判「证据不足以回答这个问题」并走了兜底** ——
#    **它是对的**。于是 ③ 两条断言**同时**红(含那条承重的「重问不再是兜底话术」),
#    而**产品每一环都正常**。**这不是模型不听话,是夹具自相矛盾。**
#    **判据**:写完答案先自问「**单看这条答案,题面被回答了吗?**」——
#    尤其当题面问的是「多少 / 多长 / 多久 / 多少个」这类**必须给量**的问题时,
#    答案里就得**真的给出那个量**(数字可以出现在答案里,只是**不许进特征串**)。
#
#: 下面 Q9–Q16 是 T19 补的一批(T18 消耗了 `#1–#7`、`#8` 死在 T19 第一次跑,2026-09-25)。**每条都先用生产同款检索
#: 探针核过 `hits == 0`**(此刻召不到,② 的选题前提才成立;探针走
#: `build_retriever(session, settings).search(q)`,与 `/api/kb/search` 同一个调用点)。
#: 实测:起草 **22** 条,**其中 10 条其实召得到** ⇒ 「看起来没写进语料」**靠眼睛判不出来**;
#: 剩下 12 条 `hits == 0`,取其中 **8** 条列在这里(Q9–Q16)。
#: ⚠️ 那 10 条里有 3 条只比阈值 0.25 高一点点(0.2507 / 0.2615 / 0.2757)——
#: **贴着阈值过**的那些最容易在语料小幅变动后翻过来,加候选时优先挑探针读数干净的。
CAND_Q9="智能猫砂盆 Max 的滚筒转速是多少?"
CAND_A9="Max 款滚筒转速约每分钟 6 转,采用低速匀速缓转设计,清理时噪音较低。"
CAND_M9="5300 901f 7f13 8f6c"                       # 匀速缓转
CAND_Q10="自动饮水机 Plus 的水泵噪音是多少分贝?"
CAND_A10="Plus 款工作噪音约 30 分贝,用变频水泵,夜里放卧室也不影响休息。"
CAND_M10="53d8 9891 6c34 6cf5"                      # 变频水泵
CAND_Q11="大型猫爬架整箱包装的尺寸和重量是多少?"
CAND_A11="整箱约 120 × 40 × 30 厘米、毛重约 12 公斤,建议两人搬运;箱内零件按编号分组,照着图解装即可。"
CAND_M11="6309 7f16 53f7 5206 7ec4"                 # 按编号分组
CAND_Q12="大号三档加热垫的定时关机是定多久?"
CAND_A12="该型号可设 2 / 4 / 8 小时后自动关机,到点自动断电,睡前开也不担心。"
CAND_M12="5230 70b9 81ea 52a8 65ad 7535"            # 到点自动断电
CAND_Q13="自动饮水机 Pro 的无线水位提醒怎么配对?"
CAND_A13="打开 App 后长按配对键几秒,等指示灯闪烁就连上了。"
CAND_M13="957f 6309 914d 5bf9 952e"                 # 长按配对键
CAND_Q14="智能猫砂盆 Lite 的物理除臭盖板是什么材质?"
CAND_A14="盖板内芯是活性炭滤棉,可单独取出晾晒,建议定期更换。"
CAND_M14="6d3b 6027 70ad 6ee4 68c9"                 # 活性炭滤棉
CAND_Q15="全景看护摄像头的移动侦测灵敏度能调吗?"
CAND_A15="灵敏度支持自行调节,还可以框选侦测区域,减少窗帘飘动的误报。"
CAND_M15="6846 9009 4fa6 6d4b 533a 57df"            # 框选侦测区域
CAND_Q16="自动喂食器基础版断电后定时设置会丢吗?"
CAND_A16="断电后定时设置会自动保存,来电后继续按原计划投喂,不用重新设。"
CAND_M16="6309 539f 8ba1 5212 6295 5582"            # 按原计划投喂

# ── ① / ④ 的题面(可答的商品咨询;可答性由运行时探针判)──────────────────
Q_TRACE="智能猫砂盆 Lite 和 Pro 有什么区别?"
Q_FEEDBACK="自动饮水机 Pro 的滤芯多久换一次?"

# ── ⑤ 的业务问题(要拿到一个**非商品咨询**的 intent tag)────────────────
BUSINESS_QS=("订单 1001 的物流到哪了?" "订单 1011 什么时候能送到?" "帮我查一下订单 20240915 的物流信息")

PASS=0; FAIL=0; WARN=0
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
  # 失败时**保留**工作目录(SSE 原文、两个服务的控制台、Langfuse 的观测清单)。
  # 判据是 `FAIL -gt 0` **或** `KEEP -eq 1` —— 不能只看 FAIL:几条 preflight
  # 分支在**一条断言都还没打**的时候就 `exit 1`(那时 FAIL 仍是 0),
  # 只认 FAIL 的话 cleanup 会把**唯一的**那份证据删掉。
  if [ "$FAIL" -eq 0 ] && [ "$KEEP" -eq 0 ]; then
    rm -rf "$WORK"
  else
    # 本轮转录也留一份进证据目录:它是「输出干净性自检」的输入,失败时最该看的东西
    # 之一(那一屏 shell 错误行就在里面)。`cp` 而不是 `mv` —— 转录文件此刻还被
    # 文件头那个 `tee`(父进程)开着写,搬走它在 Windows 上可能失败。
    [ -n "${CH09_SELF_LOG:-}" ] && [ -f "$CH09_SELF_LOG" ] && \
      cp "$CH09_SELF_LOG" "$WORK/transcript.txt" 2>/dev/null
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
# 辅助脚本(都写成文件,不走 `python -c`:里面有中文,而 `-c` 的源码走 argv 会被
# 重编码 —— ch08 记过的那条)
# ══════════════════════════════════════════════════════════════════════════

# ── ① Langfuse:v2 observations 端点 ────────────────────────────────────
# ⚠️ 用 `GET /api/public/v2/observations`(v1 的 `/traces` 已弃用)。两种取法都实测
# 可用:`?sessionId=`(一次就拿到整条请求的全部观测,本脚本用它)与 `?traceId=`
# (拿到 traceId 之后再核一遍 —— 两种过滤必须指向同一批观测)。
# ⚠️ 行里的 `tags` 字段**是 None**:单体观测读不回 tag,只有 Metrics 聚合看得见
# (spec §3.5)⇒ ⑤ 只能靠 `scripts/intent_cost.py` 的按 tag 分组,不能靠这里。
cat > "$WORK/lftrace.py" <<'PYEOF'
"""按 sessionId / traceId 读 Langfuse v2 观测。只读、不打印凭据。

模式:
  session     按 sessionId 轮询,**等到那三个关键观测都到齐**(chat / retrieval /
              ChatOpenAI)或超时为止
  session-any 按 sessionId 轮询,拿到**第一批非空**就停(诊断用)
  trace       按 traceId 读一次

⚠️ **为什么要等「到齐」而不是「非空」**(本机实测,2026-09-24 15:0x):ingestion 是
**分批**落库的 —— 同一个会话第一次查到 **7** 条(只到 `classify_intent`/`resolve_references`),
二十秒后再按 traceId 查同一个 trace 是 **11** 条。照「非空就断言」写会拿到一份
**残缺的观测集**,于是「没有 retrieval 观测」「根观测不是 chat」两条断言在一条
**完全正常**的链路上红掉 —— 一次纯粹由脚本造的假红。

退出码:0 = 有观测;3 = 查得到、但一行都没有;4 = LANGFUSE_* 没配齐;
5 = 请求失败(网络/凭据/端点)。
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx

# ⚠️ 本文件落在 `$WORK`(`<repo>/.ch09_acceptance/`)下 ⇒ 上溯**一层**才是仓库根。
# 写成 `parents[2]` 会去读 `D:\agent\.env`(实测报 FileNotFoundError)。
REPO = Path(__file__).resolve().parents[1]


def dotenv(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    for line in (REPO / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def emit(text: str) -> None:
    sys.stdout.buffer.write(text.encode("utf-8") + b"\n")
    sys.stdout.buffer.flush()


async def fetch(base, auth, param, key):
    async with httpx.AsyncClient(timeout=40, auth=auth) as http:
        resp = await http.get(f"{base}/api/public/v2/observations",
                              params={param: key, "limit": 100})
        resp.raise_for_status()
        return resp.json().get("data", [])


async def main() -> None:
    mode, key, out = sys.argv[1], sys.argv[2], sys.argv[3]
    max_wait = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    base = dotenv("LANGFUSE_BASE_URL").rstrip("/")
    auth = (dotenv("LANGFUSE_PUBLIC_KEY"), dotenv("LANGFUSE_SECRET_KEY"))
    if not (base and auth[0] and auth[1]):
        emit("!!! LANGFUSE_* 没配齐 —— 观测整套 no-op(验收 ①/⑤ 必然红)")
        raise SystemExit(4)
    param = "traceId" if mode == "trace" else "sessionId"
    wait_complete = mode == "session"
    expected = ("chat", "retrieval", "ChatOpenAI")

    deadline = time.monotonic() + max_wait
    while True:
        try:
            rows = await fetch(base, auth, param, key)
        except Exception as exc:                      # noqa: BLE001
            emit("!!! 读 Langfuse 失败:%s:%s" % (type(exc).__name__, exc))
            raise SystemExit(5)
        names = {str(r.get("name")) for r in rows}
        complete = bool(rows) and all(n in names for n in expected)
        if complete or (rows and not wait_complete) or time.monotonic() >= deadline:
            break
        await asyncio.sleep(10)

    Path(out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    names = {str(r.get("name")) for r in rows}
    emit("ROWS=%d" % len(rows))
    emit("COMPLETE=%s" % ("yes" if all(n in names for n in expected) else "no"))
    emit("MISSING=%s" % ",".join(n for n in expected if n not in names))
    for row in rows:
        emit("%s | %s | trace=%s | parent=%s | level=%s" % (
            row.get("name"), row.get("type"), str(row.get("traceId"))[:12],
            str(row.get("parentObservationId"))[:12], row.get("level")))
    raise SystemExit(0 if rows else 3)


asyncio.run(main())
PYEOF
lftrace() { PYTHONPATH="$PWD" "$PYTHON" "$WORK/lftrace.py" "$@"; }

# ── DB 探针(只读;**每个 mode 都按 conversation_id 或主键过滤**)──────────
cat > "$WORK/dbq.py" <<'PYEOF'
"""验收脚本的只读 DB 探针。

理由(本章硬要求):`low_confidence_questions` / `review_queue` 是**共享、只追加**
的表 —— 跑本脚本之前池子里已有 35 行(全部已归并)、队列 11 行。按「最近这几条」
查会变成偶尔红、偶尔绿(本仓记过的「被上次运行的数据污染」)。
"""

import asyncio
import sys

from sqlalchemy import text

from app.db.base import get_engine


async def main() -> None:
    mode = sys.argv[1]
    out = sys.stdout.buffer
    engine = get_engine()
    try:
        async with engine.connect() as conn:
            if mode == "pool":
                # 本会话落下的池子行。列序:id|入口|快照 JSON 类型|快照长度|归并到的行|落池理由
                r = await conn.execute(
                    text(
                        "SELECT id, entry_point, JSON_TYPE(evidence_snapshot), "
                        "JSON_LENGTH(evidence_snapshot), matched_review_id, reject_reason "
                        "FROM low_confidence_questions "
                        "WHERE source_conversation_id = :s ORDER BY id"
                    ),
                    {"s": sys.argv[2]},
                )
                for row in r.fetchall():
                    out.write(("|".join("" if c is None else str(c) for c in row)
                               + "\n").encode("utf-8"))
            elif mode == "eval-max":
                # 最后落的一轮(按 (created_at, id) —— `eval_trend.py` 用的是同一个序)。
                r = await conn.execute(
                    text("SELECT id, created_at, case_count, "
                         "JSON_EXTRACT(metrics, '$.top_k') FROM eval_runs "
                         "ORDER BY created_at DESC, id DESC LIMIT 1")
                )
                row = r.first()
                # ⚠️ **逐列取,不用 `% row`** —— SQLAlchemy 的 `Row` 不是元组,
                # `"%s|%s" % row` 会抛 `TypeError`(实测:探针输出变成
                # `DBQ-ERROR TypeError`,而下游只会看到「这一轮时间戳是 DBQ-ERROR」,
                # 读起来像「那一轮没落库」)。
                if row:
                    out.write(("%s|%s|%s|%s\n" % (row[0], row[1], row[2], row[3]))
                              .encode("utf-8"))
                else:
                    out.write(b"\n")
            elif mode == "chunks":
                r = await conn.execute(
                    text("SELECT COUNT(*), SUM(vectorize_status='done') FROM knowledge_chunks"))
                row = r.first()
                out.write(("%s|%s\n" % (row[0], row[1])).encode("ascii"))
            else:
                out.write(("DBQ-ERROR unknown-mode %s\n" % mode).encode("ascii"))
                raise SystemExit(3)
    except SystemExit:
        raise
    except Exception as exc:                                # noqa: BLE001
        # **不许静默** —— 探针跑不动时下面所有 grep 都会落空,而「落空」在 grep 的
        # 世界里与「确实没有那一行」长得一模一样。
        out.write(("DBQ-ERROR %s\n" % type(exc).__name__).encode("ascii"))
        raise SystemExit(3)


asyncio.run(main())
PYEOF
# `PYTHONPATH="$PWD"` 是**必须的**:按**路径**跑脚本时 Python 把**脚本所在目录**
# (`$WORK`)塞进 sys.path,而不是当前目录 ⇒ `from app.db.base import ...` 直接
# ModuleNotFoundError。`python -c` 不会这样(CWD 在 sys.path 上)。
dbq() { PYTHONPATH="$PWD" "$PYTHON" "$WORK/dbq.py" "$@"; }

# 探针自检:跑得动吗?(跑不动 ⇒ 下面所有断言恒假,必须现在就说)
if ! dbq chunks > /dev/null 2>"$WORK/dbq.err"; then
  echo "预检失败:DB 探针跑不动($WORK/dbq.err):"; cat "$WORK/dbq.err"; fail_exit
fi

# ── 待审详情的核对(中文比较必须在 Python 里做,不能走 argv)──────────────
cat > "$WORK/reviewcheck.py" <<'PYEOF'
"""`GET /api/review/{id}` 的返回体 → 几条 ASCII 判据。

用法:`reviewcheck.py <detail.json> <question.txt> [snapshot_needle_hex]`

`question.txt` 是本轮那句**原话**(由 heredoc 落盘,UTF-8)。中文比较只在这里做
—— 走 argv 会被 MSYS2 按 CP936 重编码,于是「原话在不在」这条断言**恒假**。

打印(bash 侧照样只 grep ASCII 的形态):
    RAWS=<归并进来的原话条数>
    MATCH=yes|no                     ← 有一条的 question 与 question.txt **逐字相等**
    ENTRY=<那一行的 entry_point>
    REASON=<那一行的 reject_reason>
    SNAP=none|<n>                    ← evidence_snapshot 是 JSON null / 有 n 条
    SNAP_NEEDLE=yes|no|n/a           ← 有内容时,内容里有没有那个特征串
"""

import json
import sys


def emit(text: str) -> None:
    sys.stdout.buffer.write(text.encode("utf-8") + b"\n")
    sys.stdout.buffer.flush()


def main() -> None:
    detail = json.loads(open(sys.argv[1], encoding="utf-8").read())
    want = open(sys.argv[2], encoding="utf-8").read().strip()
    needle = ("".join(chr(int(h, 16)) for h in sys.argv[3].split())
              if len(sys.argv) > 3 and sys.argv[3] else None)
    raws = detail.get("raw_questions") or []
    emit("RAWS=%d" % len(raws))
    hit = next((r for r in raws if (r.get("question") or "") == want), None)
    emit("MATCH=%s" % ("yes" if hit else "no"))
    if hit is None:
        # 失败时把**实际**拿到的原话打出来(是别的行时,一眼能看出是归并错了
        # 还是这一条压根没进来)。
        for r in raws:
            emit("ACTUAL=%s" % (r.get("question") or ""))
        raise SystemExit(3)
    emit("ENTRY=%s" % hit.get("entry_point"))
    emit("REASON=%s" % hit.get("reject_reason"))
    snap = hit.get("evidence_snapshot")
    if snap is None:
        emit("SNAP=none")
        emit("SNAP_NEEDLE=n/a")
        raise SystemExit(0)
    emit("SNAP=%d" % len(snap))
    if needle is None:
        emit("SNAP_NEEDLE=n/a")
        raise SystemExit(0)
    texts = " ".join(
        "%s %s" % (e.get("section_path") or "", e.get("answer") or "") for e in snap)
    emit("SNAP_NEEDLE=%s" % ("yes" if needle in texts else "no"))
    for e in snap:
        emit("SNAP_ROW=%s | %s | %s" % (
            e.get("chunk_id"), e.get("score"), (e.get("section_path") or "")[:40]))


main()
PYEOF
reviewcheck() { PYTHONPATH="$PWD" "$PYTHON" "$WORK/reviewcheck.py" "$@"; }

# ── ⑥ 趋势表的核对(定位**我们那两轮**,而不是数「有两轮」)─────────────
cat > "$WORK/trendcheck.py" <<'PYEOF'
"""`scripts/eval_trend.py` 的输出 → 闭式判据。

用法:`trendcheck.py pair <trend.txt> <ts1> <ts2>` / `trendcheck.py full <trend.txt>`

⚠️ **不许在整份输出上 grep 箭头** —— 图例那行里就写着 `↑/↓`、横幅那行以 `=` 开头,
整份 grep 在**一条增量行都没有**时照样通过(假绿)。判据必须**锚在行首**,而且锚在
**我们那两轮**的下一行上(我们那两轮是第几轮取决于 `eval_runs` 里已有几行 ——
直接数到 2 是错的)。
"""

import re
import sys


def emit(text: str = "") -> None:
    sys.stdout.buffer.write(text.encode("utf-8") + b"\n")
    sys.stdout.buffer.flush()


#: ↓ ↑ = 后面必须跟空白,才算一条增量行。不能只判首字符:横幅 `===== 评估趋势 =====`
#: 的首字符正是 `=`;也不含 `≠` —— 那行的意思恰恰是「本轮不打印增减」。
DELTA = re.compile("^[" + chr(0x2193) + chr(0x2191) + r"]\s|^=\s")
#: 轮次行的前缀:`  4 ` + 两空格 + `2026-09-24 22:50:00` + 两空格 + `     5` + 两空格。
#: 分桶表的行**同样匹配**它 ⇒ 只取每个轮次号的**第一次**出现(头条表在前)。
#: ⚠️ 它依赖 `eval_trend.py` 的**列宽**(`W_ROUND/W_TIME/W_COUNT` + `GAP`,即
#: 「数字 + 空白 + 19 字时间戳 + 空白 + 条数 + 两个空格」)。**那里改了宽度,
#: 这个正则就一条都匹配不上** —— 那种情况在本文件里**响亮地失败**:`parse()` 会返回
#: 空 dict,两个模式都当场报错(`ROUNDS=0` / `趋势表里找不到本轮`),不会静默判绿。
ROUND = re.compile(r"^\s*(\d+)\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(\d+)\s\s")


def parse(text: str) -> dict:
    rounds: dict[str, tuple] = {}
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = ROUND.match(line)
        if not m:
            continue
        n, ts, count = m.group(1), m.group(2), m.group(3)
        if n not in rounds:
            rounds[n] = (ts, count, lines[i + 1] if i + 1 < len(lines) else "")
    return rounds


def main() -> None:
    mode, path = sys.argv[1], sys.argv[2]
    text = open(path, encoding="utf-8", errors="replace").read()
    rounds = parse(text)
    emit("ROUNDS=%d" % len(rounds))
    for n in sorted(rounds, key=int):
        emit("ROUND=%s %s 条数=%s NEXT=%s" % (n, rounds[n][0], rounds[n][1],
                                              rounds[n][2].strip()[:40]))
    if not rounds:
        # 一条轮次行都解析不出来 ⇒ **装置坏了**,不是被测对象坏了:多半是
        # `eval_trend.py` 的列宽被改过(见 `ROUND` 上面那条注释),或输出被截断。
        # 这时下面所有「找不到本轮」的判据都会命中,必须先说清是哪一种。
        emit("!!! 趋势表的输出里解析不出任何轮次行(列宽改过?输出被截断?ROUNDS=0)")
        raise SystemExit(8)

    if mode == "pair":
        ts1, ts2 = sys.argv[3][:19], sys.argv[4][:19]
        found = {ts: n for n, (ts, _, _) in rounds.items()}
        for ts in (ts1, ts2):
            if ts not in found:
                emit("!!! 趋势表里找不到本轮(%s)—— 说明这一轮没落进 eval_runs" % ts)
                raise SystemExit(3)
        n1, n2 = found[ts1], found[ts2]
        emit("PAIR=%s->%s" % (n1, n2))
        nxt = rounds[n2][2]
        emit("TS2_NEXT=%s" % nxt.strip()[:40])
        # 我们这两轮**同规模同 top_k** ⇒ 必须打增减(而不是 `≠ 不可比`)。
        if not DELTA.match(nxt.lstrip()):
            emit("!!! 第 %s 轮(本轮第二跑)的下一行不是增减行" % n2)
            raise SystemExit(4)
        emit("DELTA_ROW=yes")
        if not any(DELTA.match(ln.lstrip()) for ln in text.splitlines()):
            emit("!!! 全表里一条增减行都没有")
            raise SystemExit(5)
        raise SystemExit(0)

    # full:最后一轮必须是 300 条的**全量**轮,且它相对上一轮打的是「不可比」
    # (条数 5→300)—— C4:⑥ 必须收在一轮全量上,否则 admin 评测页看到的是那 5 条
    # 的结果;而且 `≠` 那行本身是有价值的输出。
    last = max(rounds, key=int)
    ts, count, nxt = rounds[last]
    emit("LAST=%s %s 条数=%s" % (last, ts, count))
    emit("LAST_NEXT=%s" % nxt.strip()[:60])
    # ⚠️ 这里的 `300` 是**用例集的规模**(`evals/测试集.md`),与本脚本无关地写死:
    # 用例集扩容/缩容之后这条会**响亮地红**(「最后一轮不是全量(条数=N)」),
    # 而那正是想要的 —— 它守的是「⑥ 收在一轮全量上、admin 评测页看到的不是 5 条」,
    # 换规模时把这里(以及下面 `latest.json` 那条的 300)一起改掉。
    if count != "300":
        emit("!!! 最后一轮不是全量(条数=%s;用例集规模若不是 300,改本文件的这个常量)" % count)
        raise SystemExit(6)
    if not nxt.lstrip().startswith(chr(0x2260)):
        emit("!!! 全量轮那一行没有打「不可比」标记")
        raise SystemExit(7)
    raise SystemExit(0)


main()
PYEOF
trendcheck() { PYTHONPATH="$PWD" "$PYTHON" "$WORK/trendcheck.py" "$@"; }

# ── SSE 收发与解析(与 ch08 逐字同款)────────────────────────────────────
ask() {    # $1=sid $2=消息 $3=SSE 输出文件(中文请求体走 stdin heredoc,不走 argv)
  curl -s -N --max-time 300 -X POST "$BASE/api/chat/stream" \
    -H "Content-Type: application/json" \
    --data-binary @- > "$3" <<JSON
{"session_id":"$1","message":"$2"}
JSON
}
has_event() { grep -q "^event: $2\$" "$1"; }
# 帧的自检。**每个断言块都先跑它** —— 「没有这一帧」这类断言在一份空文件上恒真。
frames_sane() {   # $1=文件 → 0/1
  [ -s "$1" ] && has_event "$1" meta && return 0
  return 1
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
# done 帧里的 intent(诊断 + ②/⑤ 的选题判据)。
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
# 回复(拼回后)里有没有这个 needle。**needle 用十六进制码点传**(规矩 3)。
# ⚠️ **退出码有三个,别把 1 与 2 混成一个非零**(fix round 2,复审 F2):
#   0 = 针在;1 = 文件读得动、**针确实不在**;2 = **文件读不了/解不开**(装置故障)。
# 混成一个非零的后果很具体:`has_needle ... || ok "......不再是兜底话术"` 这种**否定**断言
# 会把一次 `OSError`/解码失败**静默判绿** —— 而它本该是「装置坏了,这条判不了」。
# **凡否定断言一律走下面的 `assert_needle_absent`**,不要自己写 `else` 分支。
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
# 这一句的意图是不是某个值(②/⑤ 的选题判据;hex needle 走 argv)。
intent_is() {    # $1=SSE 文件 $2=意图的 hex
  intent_label "$1" > "$WORK/last_intent.txt"
  has_needle "$WORK/last_intent.txt" "$2"
}
# 直接量一次检索(独立探针):这条问题**此刻**召得到几块。
kb_hits() {      # $1=问题(内部字符串) $2=输出文件 → 打印块数(ERR = 探针坏了)
  # ⚠️ `?q=` 要放进 URL ⇒ 必走 argv ⇒ 中文会被 CP936 重编码。这里**故意**用 URL
  # 编码:%XX 全是 ASCII(argv 安全),服务端解出来还是那句原话。
  local enc
  enc=$("$PYTHON" -c '
import sys, urllib.parse
sys.stdout.buffer.write(urllib.parse.quote(sys.stdin.buffer.read().decode("utf-8")).encode("ascii"))' <<< "$1")
  curl -s --max-time 120 "$BASE/api/kb/search?q=$enc" > "$2"
  "$PYTHON" -c '
import json, sys
try:
    rows = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    print("ERR"); raise SystemExit(3)
print(len(rows) if isinstance(rows, list) else "ERR")' "$2"
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
# 等预热。**看日志而不是打检索**:`?q=` 里的中文走 argv 会被重编码(见 kb_hits),
# 而预热完成那句话是现成的信号。等不到不致命 —— 首次检索会现场加载(十几秒)。
wait_warmup() {   # $1=秒数 → 0=预热完成 1=没等到 2=预热**失败**
  local deadline=$((SECONDS + $1))
  while [ "$SECONDS" -lt "$deadline" ]; do
    has_needle "$LOG" "$H_WARMUP_OK" && return 0
    has_needle "$LOG" "$H_WARMUP_FAIL" && return 2
    sleep 2
  done
  return 1
}
# 读一次 /api/kb/stats 的 milvus_count(None = Milvus 不可达)。
milvus_count() {
  "$PYTHON" -c '
import json,sys
try:
    print(json.loads(sys.stdin.buffer.read().decode("utf-8")).get("milvus_count"))
except Exception:
    print("ERR")' <<< "$(curl -s --max-time 30 "$BASE/api/kb/stats")"
}
# Langfuse 的**写**侧(OTel exporter)与**读**侧(本脚本用的 REST 查询端点)是两条
# 不同的链路:**读得通不代表写得进去**。实测(2026-09-24 22:49,本机)服务端每 11s 打
# 一条
#   `opentelemetry.exporter.otlp.proto.http.trace_exporter Failed to export span batch
#    code: None, reason: ... Read timed out. (read timeout=4.99999)`
# —— 那几分钟里的请求**一条 trace 都没落上去**,而同一时刻读侧一切正常(查询回
# 200 + 空数组)。所以这条错误行是 ①(以及 ⑤ 的 intent tag)唯一能指名道姓的错因,
# 失败时必须打出来,否则现象只是一个「0 条观测」。
export_health() {
  local n; n=$(grep -c 'opentelemetry.exporter' "$LOG" 2>/dev/null || true)
  echo "  服务端 OTLP 导出失败行数:${n:-0}(写侧链路;读侧正常时它也可能非零)"
  grep 'opentelemetry.exporter' "$LOG" 2>/dev/null | tail -2 | cut -c1-160 | sed 's/^/    /'
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
else
  warn "物流 MCP Server($MCP_LOGISTICS_PORT)起不来 —— 继续(⑤ 那一轮只要意图标签,工具少一个不影响)"
fi
if start_mcp_ready mcp_servers.aftersales "$WORK/mcp_aftersales.log" "$MCP_AFTERSALES_PORT"; then
  MCP_AFTER_PID="$MCP_NEW_PID"
else
  warn "售后 MCP Server($MCP_AFTERSALES_PORT)起不来 —— 继续"
fi

# 预热(BGE-M3)。等不到不致命,但**要说清楚**,否则第一个检索请求会慢十几秒。
wait_warmup 180
case $? in
  0) echo "  BGE-M3 预热完成" ;;
  2) warn "BGE-M3 预热**失败**(日志里有那条),首次检索会现场加载或直接 502" ;;
  *) warn "180s 内没等到预热完成 —— 首次检索可能要现场加载十几秒" ;;
esac

# Milvus 在不在(ch09 的前置之一)。`/api/kb/stats` 的 milvus_count 是真去问 Milvus 的。
MILVUS_COUNT=$(milvus_count)
if [ "$MILVUS_COUNT" = "ERR" ] || [ "$MILVUS_COUNT" = "None" ]; then
  echo "预检失败:Milvus 不可达(/api/kb/stats → milvus_count=$MILVUS_COUNT)。ch09 的前置包含 Milvus ——"
  echo "  起它:docker start milvus-standalone(集合与权重都还在,别删数据)"
  fail_exit
fi
echo "  Milvus 可达(milvus_count=$MILVUS_COUNT)"

# Langfuse 在不在(①②⑤ 的前置)。用一次**必定为空**的查询探活:退出码 3 说明
# 「查得到、零行」——那正是探活想要的;4/5 说明配置或网络不对。
lftrace trace "probe-selfcheck" "$WORK/lf_selfcheck.json" > "$WORK/lf_selfcheck.txt" 2>&1
LF_RC=$?
if [ "$LF_RC" != "0" ] && [ "$LF_RC" != "3" ]; then
  echo "预检失败:Langfuse 读不通(rc=$LF_RC):"
  cat "$WORK/lf_selfcheck.txt"
  echo "  —— 检查 .env 的 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL"
  fail_exit
fi
echo "  Langfuse 可达"

# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "== 验收 ① Langfuse 里能点开一条请求看到完整链路 =="
echo "   (脚本断的是「数据在不在」;**「界面能点开、点开之后是什么样」靠人** —— C10)"
# ══════════════════════════════════════════════════════════════════════════
# 最多两轮,每轮**一个新会话** + 240s 轮询(等到 chat/retrieval/ChatOpenAI **到齐**)。
# 为什么要重试:① 依赖的是**写入**那条链路(服务端 OTel exporter → Langfuse Cloud),
# 而本机实测它会成片失败几分钟(见 `export_health` 的注释)、随后自己恢复;
# ingestion 本身也有 ≥45s 的延迟、而且是**分批**落库的(`lftrace.py` 的 docstring)。
# 两轮都读不回才判红 —— 那时把导出侧的错因一并打出来。
TRACE_OK=0
for attempt in 1 2; do
  SID1=$("$PYTHON" -c "import uuid;print(uuid.uuid4().hex)")
  echo "  第 $attempt 轮:会话=$SID1"
  ask "$SID1" "$Q_TRACE" "$WORK/acc1.sse"
  if ! frames_sane "$WORK/acc1.sse"; then
    boom "① 的 SSE 不正常(没有 meta 帧)—— 后面所有回复断言都会恒真"
    show_console_head_tail "$WORK/acc1.sse"
    break
  fi
  TOOLN=$(grep -c '^event: tool_call' "$WORK/acc1.sse" || true)
  echo "    意图=$(intent_label "$WORK/acc1.sse") 工具调用=${TOOLN:-0} 次"
  # 轮询:ingestion 有延迟(≥45s 是常态)且**分批**落库 ⇒ 等「到齐」为止。
  lftrace session "$SID1" "$WORK/lf_obs_$attempt.json" 240 > "$WORK/lf_obs_$attempt.txt" 2>&1
  LF_RC=$?
  if [ "$LF_RC" != "0" ]; then
    echo "    240s 内没读回观测(rc=$LF_RC)"
    continue
  fi
  grep '^COMPLETE=\|^MISSING=' "$WORK/lf_obs_$attempt.txt" | sed 's/^/    /'
  TRACE_OK=1
  cp "$WORK/lf_obs_$attempt.json" "$WORK/lf_obs.json"
  cp "$WORK/lf_obs_$attempt.txt" "$WORK/lf_obs.txt"
  echo "  读到观测的是第 $attempt 轮,会话=$SID1"
  ROWS=$("$PYTHON" -c '
import json,sys
print(len(json.load(open(sys.argv[1], encoding="utf-8"))))' "$WORK/lf_obs.json")
  TRACE1=$("$PYTHON" -c '
import json,sys
rows = json.load(open(sys.argv[1], encoding="utf-8"))
print(rows[0].get("traceId") if rows else "")' "$WORK/lf_obs.json")
  echo "  观测数=$ROWS traceId=$TRACE1"
  sed -n '2,12p' "$WORK/lf_obs.txt" | sed 's/^/    /'
  if [ "$ROWS" -ge 3 ]; then ok "① 观测数 $ROWS ≥ 3"; else bad "① 观测数只有 $ROWS(<3)"; fi
  LFNAMES=$("$PYTHON" -c '
import json,sys
rows = json.load(open(sys.argv[1], encoding="utf-8"))
print("\n".join(str(r.get("name")) for r in rows))' "$WORK/lf_obs.json")
  if printf '%s\n' "$LFNAMES" | grep -qx 'retrieval'; then
    ok "① 有 retrieval 观测(手工 span —— 它不在 LangChain 的 run 树里)"
  else
    bad "① 没有名为 retrieval 的观测(知识路径的检索 span 没发出来?)"
  fi
  if printf '%s\n' "$LFNAMES" | grep -qx 'ChatOpenAI'; then
    ok "① 有 ChatOpenAI 观测(模型调用)"
  else
    bad "① 没有 ChatOpenAI 观测"
  fi
  # 「完整链路」在本仓的硬判据:**同一条 trace**。修复前手工 span 各自开一条
  # (traceId 互不相同、parentObservationId 全是 null),spec §15.4 记着那次实测。
  TRACES=$("$PYTHON" -c '
import json,sys
rows = json.load(open(sys.argv[1], encoding="utf-8"))
print(len({r.get("traceId") for r in rows}))' "$WORK/lf_obs.json")
  if [ "$TRACES" = "1" ]; then
    ok "① 全部 $ROWS 条观测落在**同一条** trace 上"
  else
    bad "① 观测分属 $TRACES 条 trace —— 树是断的(spec §15.4 那个缺陷回来了)"
  fi
  # 根观测必须是 `chat`、且没有父(手工 span 挂进请求那条 trace 的判据)。
  ROOT=$("$PYTHON" -c '
import json,sys
rows = json.load(open(sys.argv[1], encoding="utf-8"))
roots = [r for r in rows if r.get("parentObservationId") in (None, "")]
print("%s|%s" % (len(roots), roots[0].get("name") if roots else "-"))' "$WORK/lf_obs.json")
  echo "  根观测:$ROOT(期望 1|chat)"
  if [ "$ROOT" = "1|chat" ]; then
    ok "① 根观测恰好一条、名字是 chat"
  else
    bad "① 根观测不是「恰好一条 chat」($ROOT)"
  fi
  # 用 traceId 再取一遍:两种过滤必须指向同一批观测(brief 写的是 traceId 那条路)。
  lftrace trace "$TRACE1" "$WORK/lf_obs_by_trace.json" > "$WORK/lf_obs_by_trace.txt" 2>&1
  TRACE_RC=$?
  if [ "$TRACE_RC" = "0" ]; then
    ROWS_BY_TRACE=$("$PYTHON" -c '
import json,sys
print(len(json.load(open(sys.argv[1], encoding="utf-8"))))' "$WORK/lf_obs_by_trace.json")
    if [ "$ROWS_BY_TRACE" = "$ROWS" ]; then
      ok "① 按 traceId 取回同样 $ROWS 条(两种过滤一致)"
    else
      # 差**通常只是** ingestion 还在分批到货(这两次查询之间又落了一批),
      # 不是「两种过滤指向不同集合」——所以这条只是 WARN,并且把两个数都打出来。
      warn "① 按 sessionId 取到 $ROWS 条、按 traceId 取到 $ROWS_BY_TRACE 条(差 $((ROWS_BY_TRACE - ROWS)) 条,多半是这两次查询之间又到了一批)"
    fi
  else
    warn "① 按 traceId 取回失败(rc=$TRACE_RC;按 sessionId 那一路已经验过)"
  fi
  break
done
if [ "$TRACE_OK" != "1" ]; then
  boom "① 两轮(每轮 240s)都没从 Langfuse 读回观测 —— 写侧没落上去,或 ingestion 迟迟不到"
  export_health
fi

# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "== 验收 ② 知识库没有的问题 → 兜底话术 + 待审队列 =="
echo "   (选题:从候选里挑第一条**当前召不到**的 —— 每跑一轮会消耗一条,见 Header C)"
# ══════════════════════════════════════════════════════════════════════════
# T19 起批量 = **16 条**(T18 消耗 `#1–#7`、T19 第一次跑消耗 `#8`,随后补了 Q9–Q16;
# 三次跑又消耗 `#9`/`#10` ⇒ **现在可用的是 `#11–#16`**)。加候选只需往上面那一段追加
# `CAND_Qn / CAND_An / CAND_Mn`,**这里一行都不用动**(数组与循环都按长度自适应)——
# 写死 `0..7` 的后果是「加了候选但脚本不看它」,而那种失败**看起来像候选被跳过**。
CAND_Q=("$CAND_Q1" "$CAND_Q2" "$CAND_Q3" "$CAND_Q4"
        "$CAND_Q5" "$CAND_Q6" "$CAND_Q7" "$CAND_Q8"
        "$CAND_Q9" "$CAND_Q10" "$CAND_Q11" "$CAND_Q12"
        "$CAND_Q13" "$CAND_Q14" "$CAND_Q15" "$CAND_Q16")
CAND_A=("$CAND_A1" "$CAND_A2" "$CAND_A3" "$CAND_A4"
        "$CAND_A5" "$CAND_A6" "$CAND_A7" "$CAND_A8"
        "$CAND_A9" "$CAND_A10" "$CAND_A11" "$CAND_A12"
        "$CAND_A13" "$CAND_A14" "$CAND_A15" "$CAND_A16")
CAND_M=("$CAND_M1" "$CAND_M2" "$CAND_M3" "$CAND_M4"
        "$CAND_M5" "$CAND_M6" "$CAND_M7" "$CAND_M8"
        "$CAND_M9" "$CAND_M10" "$CAND_M11" "$CAND_M12"
        "$CAND_M13" "$CAND_M14" "$CAND_M15" "$CAND_M16")
Q2=""; A2=""; M2=""; SID2=""; REV2=""
for ((i = 0; i < ${#CAND_Q[@]}; i++)); do
  q="${CAND_Q[$i]}"
  hits=$(kb_hits "$q" "$WORK/acc2_search_$i.json")
  if [ "$hits" = "ERR" ]; then
    boom "② 独立检索探针跑不动(候选 $((i+1)) 的 /api/kb/search 不是 JSON)"
    break
  fi
  echo "  候选 $((i+1)):此刻召回 $hits 块"
  if [ "$hits" != "0" ]; then
    echo "    (跳过:它已经召得到了 —— 多半是上一次运行核准进去的知识)"
    continue
  fi
  SID2=$("$PYTHON" -c "import uuid;print(uuid.uuid4().hex)")
  echo "  选中候选 $((i+1)):会话=$SID2"
  ask "$SID2" "$q" "$WORK/acc2.sse"
  join_tokens < "$WORK/acc2.sse" > "$WORK/acc2_reply.txt"
  got_intent=$(intent_label "$WORK/acc2.sse")
  echo "    意图=$got_intent"
  # 两条**同时**成立才算这一轮可用:①兜底话术 ②真的进过知识路径(意图=商品咨询)。
  # 少了第 ② 条的话,一次「其他 → 兜底」也会让「回复含兜底话术」为真 ——
  # 那条路**根本不落池**,于是断言会在下一段以「池子里没落行」的样子红掉,
  # 而错因在**分流**不在飞轮(C1 记的就是这个形状)。
  # ⚠️ 先分开 `has_needle` 的两种非零(复审 F2):**读不了文件是装置故障**,
  # 不是「这一轮没走兜底」—— 后者会让我们**换下一条候选**并一路把候选耗光,
  # 最后报的是「八条候选都没造出」,而真正的原因在第一条候选那次读盘就坏了。
  has_needle "$WORK/acc2_reply.txt" "$H_FALLBACK"; fb_rc=$?
  if [ "$fb_rc" = "2" ]; then
    boom "装置故障:读不了 $WORK/acc2_reply.txt —— ② 的选题判据跑不动"
    break
  fi
  if [ "$fb_rc" = "0" ] && intent_is "$WORK/acc2.sse" "$H_PRODUCT"; then
    Q2="$q"; A2="${CAND_A[$i]}"; M2="${CAND_M[$i]}"
    ok "② 兜底话术 + 意图=商品咨询(这一轮真的走了知识路径)"
    break
  fi
  echo "    这一轮不满足(兜底=$( [ "$fb_rc" = "0" ] && echo y || echo n)、意图=$got_intent),换下一条候选"
  SID2=""
done
if [ -z "$SID2" ]; then
  boom "② ${#CAND_Q[@]} 条候选都没造出「知识库没有的商品问题 → 兜底 + 落池」—— 请往脚本的 CAND_Q* 里加一条(每跑一轮消耗一条)"
else
  # 本轮落下的池子行(**按会话过滤**)。落池是紧跟在请求里的,给几秒余量。
  POOL2=0
  for attempt in 1 2 3 4 5 6; do
    dbq pool "$SID2" > "$WORK/acc2_pool.txt"
    POOL2=$(grep -c '^[0-9]' "$WORK/acc2_pool.txt" || true)
    [ "${POOL2:-0}" -ge 1 ] && break
    sleep 3
  done
  echo "  池子行(按会话过滤;列=id|入口|快照JSON类型|快照长度|归并到的行|落池理由):"
  sed 's/^/    /' "$WORK/acc2_pool.txt"
  if [ "${POOL2:-0}" -ge 1 ]; then
    ok "② 这一轮落了 $POOL2 条池子行(按 source_conversation_id 过滤,不是查全表)"
  else
    boom "② 兜底话术有了,但池子里没落行(置信度闸没落池?)"
  fi
  # 飞轮:手动触发一次(**409 不是失败** —— 落池那三处 fire-and-forget 占同一张槽)。
  FLYWHEEL_CODE=$(curl -s -o "$WORK/acc2_flywheel.json" -w '%{http_code}' --max-time 30 \
    -X POST "$BASE/api/kb/jobs/flywheel")
  echo "  POST /api/kb/jobs/flywheel → $FLYWHEEL_CODE"
  case "$FLYWHEEL_CODE" in
    201) echo "    (拿到槽了)" ;;
    409) echo "    (409 = 已有任务在跑,**不是失败** —— 单槽;它会吃到本行)" ;;
    *)   warn "② 手动触发飞轮返回 $FLYWHEEL_CODE(继续轮询待审队列)" ;;
  esac
  # 轮询到**我们的**那一行被归并(按会话过滤出来的池子行看 matched_review_id)。
  for attempt in $(seq 1 40); do
    dbq pool "$SID2" > "$WORK/acc2_pool.txt"
    REV2=$(awk -F'|' 'NF>=5 && $5 != "" {print $5; exit}' "$WORK/acc2_pool.txt")
    [ -n "$REV2" ] && break
    # 还没轮到:每 6 次(≈24s)再催一次(忙的时候是 409,正常)。
    [ $((attempt % 6)) -eq 0 ] && curl -s -o /dev/null --max-time 30 -X POST "$BASE/api/kb/jobs/flywheel"
    sleep 4
  done
  if [ -z "$REV2" ]; then
    boom "② 200s 内那一行还没被归并(matched_review_id 仍是 NULL)—— 飞轮没跑,或跑了没吃到它"
    curl -s --max-time 20 "$BASE/api/kb/jobs" > "$WORK/acc2_jobs.json"
    echo "  当前任务列表:"; head -5 "$WORK/acc2_jobs.json"
  else
    echo "  归并到的待审行 id=$REV2"
    curl -s --max-time 30 "$BASE/api/review/$REV2" > "$WORK/acc2_detail.json"
    # 原话逐字比 —— 中文比较在 Python 里做(question.txt 由 heredoc 落盘,UTF-8)。
    cat > "$WORK/q2.txt" <<EOF
$Q2
EOF
    if reviewcheck "$WORK/acc2_detail.json" "$WORK/q2.txt" > "$WORK/acc2_check.txt" 2>&1; then
      sed 's/^/    /' "$WORK/acc2_check.txt"
      ok "② 详情里有**本轮那句原话**(逐字相等)"
      # 快照:先看独立探针的结论(候选是「此刻召回 0 块」才被选中的),再断那一行自洽。
      # ⚠️ **这条断言判据很窄,而且如实说:它对「闸漏写 `evidence_snapshot=`」是不变的**
      # —— 零召回那一支两种写法都落 JSON `null`(闸今天传的是
      # `_snapshot(evidence) if evidence else None`)。能断的只有「这一行是自洽的:
      # 快照为 `null` **且** 理由就是「检索为空」**且** 独立探针也召不到」。见脚本尾部「局限」。
      #
      # ⚠️ **`null` 与空数组不是同一个值**(fix round 2 订正):本仓约定**`None` = 「当轮确实
      # 零召回」**(`app/kb/assess.py:88-92`、`app/agent/nodes.py:304-306`、
      # `tests/test_agent_gate_ch09.py:135-139` 三处同款),空数组 `[]` 是**另一个值**。
      # 所以这一支只放行 `SNAP=none`;真看到 `SNAP=0` 要**红** —— 那说明「零召回」与
      # 「记了、零召回」在这条链路上被混成了一个值,而这条断言存在的全部理由就是它们分不开。
      SNAP_SHAPE=$(grep '^SNAP=' "$WORK/acc2_check.txt" | head -1)
      if grep -q '^SNAP=[1-9]' "$WORK/acc2_check.txt"; then
        warn "② 快照居然有内容($SNAP_SHAPE)—— 与独立探针(0 块)不同,记一笔"
      elif grep -q '^SNAP=0$' "$WORK/acc2_check.txt"; then
        bad "② 快照是空数组($SNAP_SHAPE):「记了、零召回」与「漏记」在这一列上分不开 —— 本仓约定 None 才是零召回的值"
      elif grep -q '^SNAP=none$' "$WORK/acc2_check.txt"; then
        if has_needle "$WORK/acc2_check.txt" "$H_EMPTY_RETRIEVAL"; then
          ok "② 快照为 null 且落池理由就是「检索为空」—— 与独立探针(0 块)一致,不是漏记"
        else
          bad "② 快照为 null,但落池理由不是「检索为空」—— 与独立探针矛盾,查落池那一步"
        fi
      else
        bad "② 快照那一列的形状认不出来($SNAP_SHAPE)—— 先看 reviewcheck.py 的判据"
      fi
      echo "    $(grep '^ENTRY=' "$WORK/acc2_check.txt" | head -1)  $(grep '^REASON=' "$WORK/acc2_check.txt" | head -1)"
    else
      boom "② 详情里没有本轮那句原话(归并到了别的行,或这一行还没带上它)"
      sed 's/^/    /' "$WORK/acc2_check.txt"
    fi
  fi
fi

# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "== 验收 ③ 审核页点通过 → 同一个问题再问就答对 =="
echo "   (通过**之前**先断 Milvus 条数增长 —— 否则「答对了」可能原来就答得对,spec §13-8)"
# ══════════════════════════════════════════════════════════════════════════
if [ -z "$REV2" ]; then
  echo "  (跳过:② 没有拿到待审行 id —— ③ 依赖它)"
else
  MILVUS_BEFORE=$(milvus_count)
  echo "  核准前 milvus_count=$MILVUS_BEFORE"
  # 通过 = 写知识库 + **立刻**向量化(同步)。首次嵌入要现场加载权重(实测 3.3–5.4s),
  # 而权重正在加载时来一个会冻住 24–28s ⇒ `--max-time 120` 是给那段留的,不是给挂死的。
  APPROVE_CODE=$(curl -s -o "$WORK/acc3_approve.json" -w '%{http_code}' --max-time 120 \
    -X POST "$BASE/api/review/$REV2/approve" \
    -H "Content-Type: application/json" --data-binary @- <<JSON
{"approved_answer":"$A2"}
JSON
)
  echo "  POST /api/review/$REV2/approve → $APPROVE_CODE"
  sed 's/^/    /' "$WORK/acc3_approve.json"
  if [ "$APPROVE_CODE" != "200" ]; then
    boom "③ 审核通过返回 $APPROVE_CODE(502 = 向量化没成,队列行留待重试)"
  else
    CHUNKS=$("$PYTHON" -c '
import json,sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print("%s|%s" % (d.get("chunks_added"), d.get("vectorized")))' "$WORK/acc3_approve.json")
    echo "    chunks_added|vectorized = $CHUNKS"
    # ⚠️ 只有**真的是 `1|1`** 才说「并向量化」:`1|0`(= 新写了块但一条都没向量化)与
    # `None|None`(响应里没这两个键)都是**另一回事**,归进同一句绿话里等于把
    # 「通过了、但检索里没有它」这件事说成了成功(spec §13-9 要的正是别这么读)。
    case "$CHUNKS" in
      1\|1) ok "③ 通过写了知识块并向量化($CHUNKS)" ;;
      0\|0) bad "③ 通过写了 0 块、也没向量化 —— spec §13-9:那时「答不对」的原因不是这一条" ;;
      *)    warn "③ 通过返回 chunks_added|vectorized = $CHUNKS(写块与向量化的步数不等,核对上面那份响应)" ;;
    esac
    # Milvus 条数必须**增长**(spec §12.4 末尾那条)。
    GREW=0
    for attempt in $(seq 1 15); do
      NOW=$(milvus_count)
      echo "    第 $attempt 次读 milvus_count=$NOW(核准前 $MILVUS_BEFORE)"
      if [ "$NOW" != "None" ] && [ "$NOW" != "ERR" ] && [ "$NOW" -gt "$MILVUS_BEFORE" ] 2>/dev/null; then
        GREW=1; break
      fi
      sleep 2
    done
    if [ "$GREW" = "1" ]; then
      ok "③ milvus_count 增长($MILVUS_BEFORE → $NOW)——「答对」不可能是原来就答得对"
    else
      bad "③ milvus_count 没有增长(仍是 $MILVUS_BEFORE)—— 检索里没有这条新知识,「答对」无从谈起"
    fi
    # 独立探针:这条问题此刻应该召得到了(把「检索不中」与「聊天路径不中」分开)。
    hits3=$(kb_hits "$Q2" "$WORK/acc3_search.json")
    echo "  重问前的独立探针:$hits3 块"
    if [ "$hits3" != "0" ] && [ "$hits3" != "ERR" ]; then
      ok "③ 新知识现在召得回来($hits3 块)"
    else
      bad "③ 新知识召不回来(独立探针 $hits3 块)—— 重问必然答不对,原因在检索不在飞轮"
    fi
    # 逐字重问同一句。
    SID3=$("$PYTHON" -c "import uuid;print(uuid.uuid4().hex)")
    ask "$SID3" "$Q2" "$WORK/acc3.sse"
    join_tokens < "$WORK/acc3.sse" > "$WORK/acc3_reply.txt"
    echo "  重问:会话=$SID3 意图=$(intent_label "$WORK/acc3.sse")"
    echo "  回复:$(head -c 200 "$WORK/acc3_reply.txt")"
    if has_needle "$WORK/acc3_reply.txt" "$M2"; then
      ok "③ 回复含核准答案的特征串"
    else
      # ⚠️ **先分清「内容回归」与「超时截断」**(C9:超时型失败先查旋钮,别叫回归)。
      # 这一轮的请求由 `ask()` 发,带 `--max-time 300`;而每一次模型往返各自受
      # `LLM_TIMEOUT_SECONDS`(默认 60s)约束。回复被**截断**时,现象与「模型没转述
      # 那段知识」长得一样 —— 都是「回复里没有特征串」。
      bad "③ 回复里没有核准答案的特征串(特征串取自**核准答案**,不是模型自由文本)"
      if [ ! -s "$WORK/acc3_reply.txt" ] || ! has_event "$WORK/acc3.sse" done; then
        echo "     ⚠️ 这一轮的回复不完整(没有 done 帧或回复为空)⇒ **先查旋钮再叫回归**:" \
             "LLM_TIMEOUT_SECONDS(默认 60s)/ ask() 的 curl --max-time 300 / 上游限流;"
        echo "        证据:$WORK/acc3.sse"
      fi
    fi
    # **否定断言**:「兜底话术**不在**回复里」。必须走 `assert_needle_absent` ——
    # 自己写 `else → ok` 会把「文件读不了」(退出 2)也判成绿(复审 F2)。
    assert_needle_absent "$WORK/acc3_reply.txt" "$H_FALLBACK" \
      "③ 重问不再是兜底话术" \
      "③ 重问**仍然**是兜底话术 —— 知识入库了却没被用上"
  fi
fi

# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "== 验收 ④ 聊天页点 👎 → 落池 → 进待审队列 =="
echo "   (这一条同时覆盖 ② 覆盖不到的那半:召回片段快照**有内容**、且能回溯到本轮问题)"
# ══════════════════════════════════════════════════════════════════════════
hits4=$(kb_hits "$Q_FEEDBACK" "$WORK/acc4_search.json")
echo "  👎 那一轮选的是一条**召得到**的问题(此刻 $hits4 块)"
if [ "$hits4" = "0" ] || [ "$hits4" = "ERR" ]; then
  boom "④ 这条问题此刻召不到块 ⇒ 快照必然为空,这半条性质就验不到(换题面或查 Milvus)"
else
  SID4=$("$PYTHON" -c "import uuid;print(uuid.uuid4().hex)")
  ask "$SID4" "$Q_FEEDBACK" "$WORK/acc4.sse"
  join_tokens < "$WORK/acc4.sse" > "$WORK/acc4_reply.txt"
  echo "  会话=$SID4 意图=$(intent_label "$WORK/acc4.sse")"
  FB_CODE=$(curl -s -o "$WORK/acc4_feedback.json" -w '%{http_code}' --max-time 120 \
    -X POST "$BASE/api/feedback" -H "Content-Type: application/json" --data-binary @- <<JSON
{"conversation_id":"$SID4","question":"$Q_FEEDBACK","value":"down"}
JSON
)
  echo "  POST /api/feedback → $FB_CODE"
  sed 's/^/    /' "$WORK/acc4_feedback.json"
  if [ "$FB_CODE" != "200" ]; then
    boom "④ 反馈端点返回 $FB_CODE"
  else
    POOLED=$("$PYTHON" -c '
import json,sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print("%s|%s" % (d.get("pooled"), d.get("snapshot_chunks")))' "$WORK/acc4_feedback.json")
    echo "    pooled|snapshot_chunks = $POOLED"
    REV4=""
    for attempt in $(seq 1 40); do
      dbq pool "$SID4" > "$WORK/acc4_pool.txt"
      REV4=$(awk -F'|' 'NF>=5 && $5 != "" {print $5; exit}' "$WORK/acc4_pool.txt")
      [ -n "$REV4" ] && break
      [ $((attempt % 6)) -eq 0 ] && curl -s -o /dev/null --max-time 30 -X POST "$BASE/api/kb/jobs/flywheel"
      sleep 4
    done
    echo "  池子行(按会话过滤):"; sed 's/^/    /' "$WORK/acc4_pool.txt"
    if [ -n "$REV4" ]; then
      curl -s --max-time 30 "$BASE/api/review/$REV4" > "$WORK/acc4_detail.json"
      cat > "$WORK/q4.txt" <<EOF
$Q_FEEDBACK
EOF
      if reviewcheck "$WORK/acc4_detail.json" "$WORK/q4.txt" "$H_WATER" > "$WORK/acc4_check.txt" 2>&1; then
        sed 's/^/    /' "$WORK/acc4_check.txt"
        ok "④ 待审行里有**那一轮的问题原话**(逐字相等)"
        # **入口串味是 FAIL,不是 WARN**:三个入口(`置信度闸` / `生成自评` /
        # `用户反馈`)存在的**全部意义**就是「这一行进池的原因分得开」——
        # 一个 👎 落的行进成了「置信度闸」,审核人读到的原因就是错的,
        # 而它在上一条断言(原话逐字相等)上照样绿。
        if has_needle "$WORK/acc4_check.txt" "$H_USER_FEEDBACK"; then
          ok "④ 它的入口是「用户反馈」(三个入口分得开)"
        else
          bad "④ 入口不是「用户反馈」($(grep '^ENTRY=' "$WORK/acc4_check.txt" | head -1))—— 三入口串味了"
        fi
        if grep -q '^SNAP=[1-9]' "$WORK/acc4_check.txt"; then
          ok "④ 召回片段快照有内容($(grep '^SNAP=' "$WORK/acc4_check.txt"))"
        else
          bad "④ 快照没有内容($(grep '^SNAP=' "$WORK/acc4_check.txt" || echo 'SNAP=?')）—— 👎 的回捞没拿到片段"
        fi
        if grep -q '^SNAP_NEEDLE=yes$' "$WORK/acc4_check.txt"; then
          ok "④ 快照的内容与那句问题对得上(片段里出现了该商品的字样)"
        else
          bad "④ 快照的内容与那句问题对不上(可能捞回了不相干的块)"
        fi
      else
        boom "④ 待审行里没有那一轮的原话"
        sed 's/^/    /' "$WORK/acc4_check.txt"
      fi
    else
      boom "④ 200s 内 👎 那一行没进待审队列"
    fi
  fi
fi

# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "== 验收 ⑤ 按意图的 token 花销 =="
echo "   (①②③④ 都是商品咨询;这一个单独造一个**业务意图**的轮次,否则分不出发)"
# ══════════════════════════════════════════════════════════════════════════
BIZ_OK=0
for q in "${BUSINESS_QS[@]}"; do
  SID5=$("$PYTHON" -c "import uuid;print(uuid.uuid4().hex)")
  ask "$SID5" "$q" "$WORK/acc5_biz.sse"
  got=$(intent_label "$WORK/acc5_biz.sse")
  echo "  业务轮:「$q」→ 意图=$got"
  if intent_is "$WORK/acc5_biz.sse" "$H_LOGISTICS" || intent_is "$WORK/acc5_biz.sse" "$H_ORDER"; then
    BIZ_OK=1
    ok "⑤ 造出了一个业务意图的轮次(意图=$got)—— 与商品咨询不同,⑤ 才有了第二个意图行"
    break
  fi
done
if [ "$BIZ_OK" = "0" ]; then
  warn "⑤ 三次业务问法都没落进「物流/订单」—— 「两个意图」这条可能靠不住(见下)"
fi
# `intent_cost.py --minutes 30` 打真实 Metrics API。**轮询**到有数据为止:
# ingestion 有延迟,读一次就断言会得到「没有找到任何带 intent:* 标签的观测」。
INTENT_OK=0; BODY=0
for attempt in $(seq 1 12); do
  PYTHONPATH="$PWD" "$PYTHON" scripts/intent_cost.py --minutes 30 > "$WORK/acc5_intent.txt" 2>&1
  BODY=$(awk '/^-+$/{f=1;next} f&&/^[^ ].*  *[0-9]+  *[0-9]+$/{print}' "$WORK/acc5_intent.txt" | wc -l)
  if [ "${BODY:-0}" -ge 2 ]; then INTENT_OK=1; break; fi
  echo "  第 $attempt 次读:带 intent 标签的行数=$BODY(ingestion 还没到)"
  sleep 20
done
sed 's/^/    /' "$WORK/acc5_intent.txt"
if [ "$INTENT_OK" = "1" ]; then
  N_INTENTS=$(awk '/^-+$/{f=1;next} f&&/^[^ ].*  *[0-9]+  *[0-9]+$/{print $1}' \
    "$WORK/acc5_intent.txt" | sort -u | wc -l)
  ok "⑤ 输出里有 $BODY 个意图行(去重后 $N_INTENTS 个不同意图)≥ 2"
  # ⚠️ **口径提醒**:这条断的是「窗口内至少两个 `intent:*` 行」——**窗口里的外来流量
  # 也能满足它**(本脚本跑之前 30 分钟内若有别人打过聊天接口,一样算数)。
  # 所以这里**标出**:表里有没有本轮造的那个业务意图(物流/订单)那一行。
  # ⚠️ 措辞收着说(复审已判「不用返工、改措辞即可」):这是在**全窗口的输出上** grep,
  # 命中只能说明「窗口里出现过这个意图」,**不等于**「这一行就是本轮那次请求挣来的」——
  # 归因做不了(观测端点的 `tags` 是 `None`,单体观测读不回 tag)。
  has_needle "$WORK/acc5_intent.txt" "$H_LOGISTICS" || has_needle "$WORK/acc5_intent.txt" "$H_ORDER"
  ours_rc=$?
  if [ "$ours_rc" = "0" ]; then
    ok "⑤ 窗口内有**本轮那个业务意图**(物流/订单)的意图行 —— 注意这是全窗口 grep,不是归因"
  else
    warn "⑤ 窗口内没看到物流/订单意图行(本轮那句业务问题还没进 ingestion?)⇒ 这条断言的判别力这一轮打折"
  fi
  has_needle "$WORK/acc5_intent.txt" "$H_HOTTEST"; hot_rc=$?
  if [ "$hot_rc" = "0" ]; then
    ok "⑤ 输出里能看出 token 最多的那个意图"
  elif [ "$hot_rc" = "2" ]; then
    boom "装置故障:读不了 $WORK/acc5_intent.txt"
  else
    bad "⑤ 没打出「最烧 token 的意图」那一行"
  fi
else
  boom "⑤ 240s 内 Metrics 里带 intent 标签的行不足 2 个(ingestion 延迟 / 那一轮没打上 tag)"
  echo "     手工复核方向:spec §3.4 的「tag 边界落在 classify_intent 上」"
  export_health
fi

# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "== 验收 ⑥ 评估两轮趋势 =="
echo "   (两轮 --limit 5;**不看「总共有几轮」** —— eval_runs 里本来就有 3 行)"
# ══════════════════════════════════════════════════════════════════════════
EVAL_OK=1
TS1=""; TS2=""
for round in 1 2; do
  # ⚠️ **跑前跑后各取一次 `MAX(id)`**:只取跑后那一次的话,把「这一轮落的那一行」
  # 与「上一次跑留下的那一行」当成同一个东西 —— `run_eval.py` 今天插不进去是
  # 非零退出(`_record_eval_run` 在 `main` 里最后一步),所以眼下是**潜伏**;
  # 一旦哪天它改成「落库失败只警告」,⑥ 就会拿旧行的时间戳去趋势表里找,
  # 红在「找不到本轮」而不是「这一轮没落库」。
  EVAL_ID_BEFORE=$(dbq eval-max | tr -d '\r' | cut -d'|' -f1)
  PYTHONPATH="$PWD" "$PYTHON" scripts/run_eval.py --limit 5 --trigger manual \
    > "$WORK/acc6_eval$round.log" 2>&1
  rc=$?
  LAST_ROW=$(dbq eval-max | tr -d '\r')
  LAST_ID=$(echo "$LAST_ROW" | cut -d'|' -f1)
  echo "  第 $round 轮 --limit 5:退出码=$rc,eval_runs 的 MAX(id):${EVAL_ID_BEFORE:-空} → ${LAST_ID:-空}"
  echo "    新落的一轮:$(echo "$LAST_ROW" | tr '|' ' ')"
  if [ "$rc" != "0" ] || [ -z "$LAST_ID" ] || [ "$LAST_ID" = "$EVAL_ID_BEFORE" ]; then
    EVAL_OK=0
    boom "⑥ 第 $round 轮没有在 eval_runs 里留下新行(退出码=$rc,id ${EVAL_ID_BEFORE:-空} → ${LAST_ID:-空})"
    echo "    控制台尾:"; tail -20 "$WORK/acc6_eval$round.log" | sed 's/^/      /'
  fi
  if [ "$round" = "1" ]; then TS1=$(echo "$LAST_ROW" | cut -d'|' -f2); else TS2=$(echo "$LAST_ROW" | cut -d'|' -f2); fi
done
if [ "$EVAL_OK" != "1" ]; then
  boom "⑥ 两轮 --limit 5 评估没跑成(见上面的控制台尾)"
else
  echo "  我们的两轮时间戳:$TS1 / $TS2"
  PYTHONPATH="$PWD" "$PYTHON" scripts/eval_trend.py > "$WORK/acc6_trend.txt" 2>&1
  if trendcheck pair "$WORK/acc6_trend.txt" "$TS1" "$TS2" > "$WORK/acc6_pair.txt" 2>&1; then
    sed 's/^/    /' "$WORK/acc6_pair.txt"
    ok "⑥ 趋势表里定位到我们那两轮,且后一轮那一行**打了增减**(锚在行首,不是整表 grep)"
  else
    boom "⑥ 趋势表里定位不到我们那两轮的增减行"
    sed 's/^/    /' "$WORK/acc6_pair.txt"
    sed -n '1,25p' "$WORK/acc6_trend.txt" | sed 's/^/      /'
  fi
  # C4:⑥ 必须收在**一轮全量**上 —— `--limit 5` 会把 `evals/results/latest.json`
  # 写成那 5 条的结果,而 admin 评测页读的就是它。**不手工还原那个文件**(手工还原
  # 会与 `eval_runs` 里最后那一轮对不上);跑一轮不带 `--limit` 的。
  FULL_BEFORE=$(dbq eval-max | tr -d '\r' | cut -d'|' -f1)
  PYTHONPATH="$PWD" "$PYTHON" scripts/run_eval.py --trigger manual > "$WORK/acc6_eval_full.log" 2>&1
  FULL_RC=$?
  LAST_FULL=$(dbq eval-max | tr -d '\r')
  echo "  收尾的全量轮:退出码=$FULL_RC,eval_runs 的 MAX(id):${FULL_BEFORE:-空} → $(echo "$LAST_FULL" | cut -d'|' -f1)"
  echo "    最后一行:$(echo "$LAST_FULL" | tr '|' ' ')"
  if [ "$FULL_RC" != "0" ] || [ "$(echo "$LAST_FULL" | cut -d'|' -f1)" = "$FULL_BEFORE" ]; then
    boom "⑥ 收尾的全量轮没有在 eval_runs 里留下新行(退出码=$FULL_RC)"   # 下面的趋势断言会跟着红,这里先说清原因
  fi
  PYTHONPATH="$PWD" "$PYTHON" scripts/eval_trend.py > "$WORK/acc6_trend_full.txt" 2>&1
  if trendcheck full "$WORK/acc6_trend_full.txt" > "$WORK/acc6_full.txt" 2>&1; then
    sed 's/^/    /' "$WORK/acc6_full.txt"
    ok "⑥ 最后一轮是全量(300 条),且打了「与上一轮不可比(条数 5→300)」"
  else
    bad "⑥ 收尾的全量轮没落成(或趋势表里那一行不是「不可比」)"
    sed 's/^/    /' "$WORK/acc6_full.txt"
  fi
  # ⚠️ `latest.json` **没有** case_count 这个键(它的形状是 `{top_k, strategies}` ——
  # 条数只记在 `eval_runs.metrics` 里,spec §10.3 那两个键是给**趋势表**的)。
  # 想判「这份结果是全量还是那 5 条」,只能从**各桶的 n 相加**看 —— 那也正是
  # admin 评测页显示的东西(它逐桶渲染 n 与分数)。取各策略里 n 之和最大的那个
  # (存在的策略都有全套桶;权重不在时「混合+Rerank」会是空字典)。
  LATEST_N=$("$PYTHON" -c '
import json
d = json.load(open("evals/results/latest.json", encoding="utf-8"))
strats = (d.get("strategies") or {}).values()
print(max((sum(int(b.get("n") or 0) for b in s.values()) for s in strats), default=0))')
  if [ "$LATEST_N" = "300" ]; then
    ok "⑥ latest.json 回到全量(各桶 n 之和=300,admin 评测页不会再是那 5 条的结果)"
  else
    bad "⑥ latest.json 的各桶 n 之和=$LATEST_N(应当是 300 —— 说明它还停在那 5 条那一轮)"
  fi
fi

# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "通过 $PASS / 失败 $FAIL / 未复现 $WARN"
echo "  ① Langfuse 完整链路   见 $WORK/lf_obs.txt(**「界面能点开」那半靠人**)"
echo "  ② 兜底 + 待审队列     见 $WORK/acc2_check.txt"
echo "  ③ 通过 → 再问答对     见 $WORK/acc3_reply.txt"
echo "  ④ 👎 → 落池 → 待审   见 $WORK/acc4_check.txt"
echo "  ⑤ 按意图 token 花销   见 $WORK/acc5_intent.txt"
echo "  ⑥ 评估两轮趋势        见 $WORK/acc6_trend.txt / $WORK/acc6_trend_full.txt"
echo ""
echo "== 局限(「6/6 通过」不等于这些也被验过)—— 如实列在这里,不许读成全绿 =="
# ⚠️ **这几行一律用单引号**:里面有反引号(在双引号里会被 bash 当**命令替换**执行 ——
# 实测踩过:`` `tests/test_agent_gate_ch09.py` `` 那一段让 bash 真去跑那个 py 文件,
# 于是收尾清单里混进一屏 `import: command not found`,而脚本照样打「6/6 通过」)。
echo '  * ① 的「界面能点开、点开之后是什么样」**没有自动化覆盖**:脚本断的只有数据在不在'
echo '    (观测齐、同一条 trace、根是 chat)。要看界面,请人拿上面那行 traceId 去 Langfuse 打开。'
echo '  * **闸那条 evidence_snapshot 没有端到端覆盖**:② 问的是**零召回**的问题 ⇒'
echo '    没有任何片段可快照;而零召回那一支里「闸写了这一列」与「闸漏了 evidence_snapshot=」'
echo '    落出来的**都是 JSON null** ⇒ ② 那条断言对它本该抓的那个 bug **不变**。'
echo '    「手里有块却被闸拦」那一支(需要把两个旋钮调近,见本文件 Header B)在本链路'
echo '    结构上够不到 —— 它由 tests/test_agent_gate_ch09.py 守着,不在端到端验收覆盖内。'
echo '    **不为了凑覆盖去构造弱召回场景**(那要改服务端旋钮,验的是场景不是产品)。'
echo '  * ⑤ 的口径是「窗口内至少两个 intent:* 行」—— **窗口里的外来流量也能满足它**;'
echo '    本轮自己只贡献了一个业务意图轮次(输出里会标出那一行在不在)。'
echo '  * ② 的候选是一个**预算**:每跑一轮消耗一条(CAND_Q1..CAND_Q16,现 16 条),
     用尽会**响亮地报**「请加一条」。⚠️ 消耗是「**跳过**」不是「**复用**」(用过的问题已被
     写进知识库 ⇒ 下一轮召得到 ⇒ 永远被跳过)⇒ **剩余 = 总数 − 已消耗**。实测:T18 消耗
     `#1–#7`、`#8` 死在 T19 第一次跑、三次跑又消耗 `#9`/`#10` ⇒ **现可用 `#11–#16`,共 6 条**;
     加候选时**三条选材规矩都要守**(marker ∈ answer 且用**码点**核、**marker 里不许有数字**、
     **核准答案必须真的回答题面**),理由见文件头「特征串三条选材规矩」。'
echo '  * ⚠️ **③ 那条「回复含核准答案的特征串」不是不变的**:T19 那次它就红在**数字**上
     —— 核准答案写「一点八米」、模型转述成「1.8 米」。产品那一侧每一步都对
     (写块 / 向量化 / 召回 / 过闸 / 带引用 / 不再是兜底),红的只是那句话的字面。
     本轮之后的新候选已按「marker 不含数字」选材;**别据此认为 ③ 从此不会红**。'
echo "  * 本脚本**会真的往共享表里写东西**(池子 / 待审队列 / 知识库 / eval_runs,以及"
echo "    conversations / messages / tool_audit_logs),且一次跑要好几分钟(收尾那一轮是 300 条全量评估)。"

# ══════════════════════════════════════════════════════════════════════════
# 输出干净性自检(F3,fix round 2)—— **判词之前**跑,让「装置自己喷错误行」
# 不再可能与「6/6 通过」共存。
#
# 判据只有三条 ASCII 串:`command not found` / `syntax error` / `unexpected EOF`
# (都用 grep -E,不会与中文判词撞车)。它自己**可能红**:把任意一段这样的行进
# 转录(或让脚本自己在输出里喷一行),这条就 `boom`。
# ══════════════════════════════════════════════════════════════════════════
output_cleanliness_check() {
  echo ""
  echo "== 输出干净性自检(装置自己喷错误行,就不许与「6/6 通过」共存)=="
  if [ -z "${CH09_SELF_LOG:-}" ] || [ ! -f "$CH09_SELF_LOG" ]; then
    # 只有绕过文件头那个 tee 包装才会走到这里(手工设了 CH09_SELF_LOG 之类)。
    # **不静默**:说清「这一条本轮跑不了」,而不是让它悄悄绿着。
    warn "没有转录文件(${CH09_SELF_LOG:-未设置})⇒ 这条自检本轮跑不了"
    return
  fi
  # **屏障**:先打一个哨兵行,再**轮询转录直到哨兵出现** —— 那说明 tee 已经追平,
  # 之后扫到的就是**本轮全部**输出。不靠 sleep 猜(猜短了会漏、猜长了白等)。
  local sentinel="ch09-selfcheck-sentinel-$$-$RANDOM" i hits
  echo "$sentinel"
  for i in $(seq 1 50); do
    grep -qF "$sentinel" "$CH09_SELF_LOG" && break
    sleep 0.1
  done
  if ! grep -qF "$sentinel" "$CH09_SELF_LOG"; then
    boom "转录里没有刚才那行哨兵 ⇒ tee 没追上(文件被别的东西截断?),这条自检可信度不足"
  fi
  hits=$(grep -nE 'command not found|syntax error|unexpected EOF' "$CH09_SELF_LOG" | head -5)
  if [ -n "$hits" ]; then
    boom "输出里出现了 shell 级错误行(装置自己喷的)—— 不许与「6/6 通过」共存:"
    printf '%s\n' "$hits" | sed 's/^/      /'
  else
    ok "输出干净(转录里没有 command not found / syntax error / unexpected EOF)"
  fi
}
output_cleanliness_check

if [ "$FAIL" -eq 0 ]; then
  echo ""
  echo "6/6 通过"
else
  echo ""
  echo "有失败项 —— 证据留在 $WORK/,**不许把它读成全绿**"
fi
[ "$FAIL" -eq 0 ] || exit 1
