# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目

电商智能客服系统。

- **ch01(纯对话)** 已合并到 `main`:SSE 流式对话 + 结构化抽取。
- **ch02(Function Calling 查数据)** 交付:模型单轮选工具 → 后端执行 → 回灌 → 作答。含四张 MySQL 表、五个 `@tool`、工具执行器、评估集、端到端验收、聊天页。
- **ch03(知识库 + 向量检索)** 交付(分支 `ch03-kb`):`query_faq` 内部实现从关键词查 `faq` 表换成 **BGE-M3 + Milvus 的语义检索**(工具契约一字未改)。含结构感知切分、语料导入、双写幂等、对话挖知识、检索评估集、端到端验收 7 项。设计源见 ch03 spec(§12 订正最多的一章)。
- **ch04(知识库管理台)** 交付(分支 `ch04-kb-console`):文档查看/在线上传、后台触发向量化与从会话挖知识,独立管理页 `admin.html`。核心是 `app/kb/jobs.py`(JobStore)+ `app/kb/orchestrate.py`(后台任务:专用线程 + **自建独立 engine**)+ `app/api/kb.py`(7 端点)。
- **ch04 增补(混合检索 + 重排 + 评估)** 交付:Milvus BM25(`text` 字段 jieba analyzer + BM25 Function)+ `hybrid_search` RRF + bge-reranker-v2-m3 重排(候选池 20,CPU 性能约束);生成 QC(自评 json_mode → 拒答落 `low_confidence_questions` 池 + `citations` 帧 + 负面知识 prompt);四策略评估 `scripts/run_eval.py`。设计源见 ch04 增补 spec §13。
- **ch05(生产级架构:LangGraph 确定性编排 + 主力 ReAct Agent)** 已合并 `main`:把 `/api/chat/stream` 从「模型单轮选工具」换成 **确定性图骨架** —— `resolve_references → classify_intent → route_by_intent`(纯函数,写死在代码里)→ 五出口;知识类走强制预检索 + 置信度闸再进 Agent,业务类直交 Agent;Agent 是骨架里的**一个节点**(手写 ReAct,第二轮**不绑 tools** 是结构保证)。含 `POST /api/ticket`、聊天页两个独立按钮。设计源见 ch05 spec。
- **ch06(分流器正式版)** 交付(分支 `ch06-intent-router`):把 ch05 里占位的前两个节点做成正式版 —— **九类意图**(含「其他」;**第九类「转人工」是 2026-09-25 ch10-A 补入的,走主力 Agent**)+ `{intent, confidence}` 强制 JSON;**指代消解 + Query 改写**(失败原样透传);**退款退货/售后走一条确定性子流程**(取订单 → Query 扩写 + 强制检索政策 → 主力 Agent 判「这一单能不能退」→ 给退款表单或说明原因);**缺订单号时 `interrupt()` 真挂起**,前端渲染订单卡片,点选后 `Command(resume=...)` 同 thread 续跑;`POST /api/refund` + `refund_requests` 表。设计源见 ch06 spec。

- **ch07(上下文管理:三层滑窗 + 后台摘要 + 多会话)** 交付(分支 `ch07-context`):把 ch01 那条「按整轮裁到 token 预算」的单层裁剪升级成**三层结构** —— 最近**原文**(层 1,预算七成)/ 中间**截短**(层 2,三成)/ 最远**梗概**(后台摘要),两个锚点(`summary_upto_msg_id` / `layer1_from_msg_id`,都是 `messages.id`)划边界,**降级只挪 id、不搬数据**;token 预算**从模型窗口倒推**(`memory/budget.py`),不写死常量;每轮打两行 JSON 上下文日志(`model_ctx` / `history_ctx`);**工具结果本章起落表**(`role='tool'`);前端加**会话侧栏 + 切换回载**。设计源见 ch07 spec。

- **ch08(工具系统:注册中心 + MCP + 写操作确认流)** 交付(分支 `ch08-tool-registry`):把写死的五个 `@tool` 换成**即插即用的工具系统** —— `ToolSpec`(名 / 用途描述 / **原始 JSON Schema** / `read`|`write` / 来源)进注册中心,内置工具**包内自动发现**(在 `app/tools/builtin/` 里新增一个文件就是一个新工具),**全章唯一**的 JSON Schema 校验器,三态权限闸,唯一执行点 `execute_tool`,新表 `tool_audit_logs`;两个**自建业务 MCP Server**(`mcp_servers/logistics.py` → 8101 / `aftersales.py` → 8102,Streamable HTTP)+ 一个**每请求现问现拿、单 Server 连不上就降级**的客户端(`app/mcp/client.py`,**刻意不缓存**);并把建工单改成**确认流** —— `agent` 撞到未确认的写调用就**停循环** → `interrupt()` 弹卡片 → `Command(resume=…)` 同 thread 续跑 → 执行或拒绝,**挂起的那一轮完全不落库**。设计源见 ch08 spec(§15 订正最多的一章)。

- **ch09 · `.superpowers/` 的入库规则**(最终修复轮补,此前它是**随机的**):
  `.superpowers/` **不是 gitignore 的**,只有 `.superpowers/sdd/` 自带一个内容为 `*` 的
  `.gitignore`。判据是「**有没有被 git 跟踪的文档引用为某个数/某条修复的凭据**」——
  是 ⇒ **入库**(探针脚本、变异日志、验收转录、pytest 全量转录都算);否 ⇒ 留在本机
  (一次性运行产物,与 `evals/results/` 被 gitignore 同类;`.superpowers/t17/latest.json.bak`
  就是这么退出版本控制的)。`.superpowers/sdd/**`(SDD 账本与逐任务报告)**一律本机 workspace、
  不入版本控制** —— 文档引用它时按这个读法读,别指望 clone 里有。引用必须**逐条解析得开**
  (或落在一张写明理由的例外表里),判据由 `.superpowers/probe_final_citations.py` 复查
  (实测 20 条引用:16 条解析得开 + 4 条已记账例外 + **落空 0**)。**入库前先扫密钥**
  (实测已入库的那些只有 `LANGFUSE_*` 的**变量名**,没有值)。
  将来有人再往文档里写一条 `.superpowers/` 路径时,**先把它 `git add` 进去**再引用。

- **ch09(数据观测 + 低置信度数据飞轮)** 交付(分支 `ch09-observe-flywheel`)。两件事:
  **① 观测** —— Langfuse(Cloud,配在 `.env`)经 `app/observability.py` 接入,那是**全章唯一**的
  langfuse 边界(其余模块只认 `trace_scope` / `intent_scope` / `span` / `make_handler` 四个名字,
  有源码扫描测试守着);每请求一个**根观测**(`chat`)+ 一个手动 `retrieval` span + 意图 tag
  (`intent:<x>`,在 `classify_intent` 跑完后**中途进入**);三个 `LANGFUSE_*` 任一为空 ⇒ 整套观测
  no-op 且**不 import langfuse**(单测「全程不联网」靠它守)。`app/agent/json_stream.py` 是**三态
  增量解码器**(`lead` / `protocol` / `plain`,纯状态机、零 IO),让**知识轮**边收 token 边解出
  `useful` / `answer`,而违约时**逐字节退回 ch08 的纯文本行为**。
  **② 数据飞轮** —— 低置信度池有**三个入口**(`置信度闸` / `生成自评` / 用户点 👎 走
  `POST /api/feedback`),每行带 `evidence_snapshot`(落池当轮的召回片段);`app/flywheel/`
  (normalize → dedupe → pipeline)把池子变成 `review_queue`;`app/api/review.py` 让人工**通过**
  (通过 ⇒ 立刻写知识库并**同步向量化**,否则「重问就答对」要等下次 `build_kb`)或驳回;
  审核页在 `admin.html` 的「待审」标签页;`scripts/run_eval.py`(加 `--trigger`)+ `scripts/eval_trend.py`
  把每轮评估记进 `eval_runs` 并打趋势表。设计源见 ch09 spec(**§15 订正最多的一章**,15.1–15.15)。

- **ch10-A(「转人工」补成第九类意图)** 交付(分支 `ch10-topic-classifier`;ch10 是两章,
  **A 支(转人工)先做、B 支(微调多标签主题分类器)在后**)。ch06 交付时「转人工」**既不是意图、
  也不是出口** —— 它只是投诉出口发的 `choices` 帧,由前端**纯前端模拟**。ch10-A 把它补成
  **真正的第九类**:`app/agent/routing.py` 的 `INTENT_TO_ROUTE["转人工"] = HANDOFF`(9 键)、
  `INTENT_LABELS` 9 元组;`app/agent/graph.py` 把 `HANDOFF` 指向**已有的 `agent` 节点**
  (**不开新出口**,`_OUTLETS` 仍 5 个),由 Agent 调内置的**模拟**工具 `transfer_to_human`
  (`app/tools/builtin/handoff.py`,不接真人系统)。该工具在 `app/tools/policy.py` 里
  **显式声明为 `read`**、不在 `WRITE_TOOLS` 里 ⇒ `kind_of` 放行直调、**不过 ch08 的确认流**。
  投诉出口那个「转人工」按钮同时改成**发一条真实消息**(消除「同一个词两套行为」)。
  ⚠️ **一处与 ch06 立身之本的冲突,已知情接受**:ch06 的原则是「模型只决定意图标签、
  不决定走向」(`routing.py` 模块 docstring),而转人工走 Agent 之后**它发生不发生取决于
  模型记不记得调工具** —— 这是全仓**唯一一处**例外,结构上拦不住,只能由
  `scripts/acceptance_ch10.sh` 把它测成一个**比例读数**(三层:意图 / 工具被调 / 回复含工号或等待)
  并把读数如实写进 `dev-notes/ch10.md`(见 ch10 spec §11.4)。

- **ch10-B(微调 17 类多标签主题分类器)** 交付(分支 `ch10-b-topic-classifier`,ch10 的第二半;
  **ch10-A 合并到 `main` 之后**做的)。它把「低置信度池里那堆问题该先补哪块知识」变成
  一个**能算的数**:17 类权威类目表(`app/topic/taxonomy.py`;9→17 是**有损投影**)+
  语料合流 / 配额合成 / 大模型预标 / 分层切分(`scripts/prepare_topic_data.py`)+
  **本地全参微调**(`scripts/train_topic_clf.py`,基座 hfl/chinese-roberta-wwm-ext,
  产物 `models/topic-clf` —— 409MB,**不入库**)+ 冻结测试集评测
  (`scripts/eval_topic_clf.py` → `evals/topic/report.md` + report.json + 两张矩阵 + misjudged.csv,
  **都入库**)+ **旁路推理服务**(`topic_service/`,独立进程 **8103**,**不在对话链路上**)+
  离线跑批(`scripts/classify_topics.py` → 表 `topic_classifications`)+
  只读分布接口(`GET /api/topics/distribution`,标签在 SQL 里用 `JSON_TABLE` 展开)+
  管理台「主题分布」页。设计源见 ch10 spec(§15 现有 15.1–15.4 四条实现订正)。

- **ch10 跟进(2026-09-27,不在任何一章的范围内)三件事**:① `admin.html` 从
  **「知识库管理」改名为「工作台」**(`index.html` 顶栏那个入口同步改;
  ch04 spec §6 那一行加了后记 —— 它描述的是 ch04 交付时的样子);② 工作台加了
  **「首页」**(第一个目录),给入库 / 待审 / 评测 / 主题分布 / 链路各一张卡:
  **一行摘要 + 一个入口**,六个目录 `flex:1` **等宽铺满**;③ 评测页加了
  **「运行评测」按钮** ⇒ `POST /api/kb/jobs/eval` → `app/kb/eval_job.py`
  (**子进程**,跑的就是 `scripts/run_eval.py --trigger manual`;
  实测一轮全量 **52 秒**,写 `latest.json` + 追加一行 `eval_runs`;**不接 `--limit`**,
  理由见那个端点的 docstring)。
  另:三条前端缺陷(#113 引用弹层 × 关不掉 / #114 历史回载丢齿轮与文档链接 /
  #115 工作台加「链路」页)也已修完,全过程见 `dev-notes/ch10.md` 的「阶段 9」。

- **ch10-B 的五条命门**(每条都对应一次「报错指向别处」或「静默出错」):
  1. **`problem_type` 不是配置项,是从 `labels.dtype` 猜出来的**(transformers 5.17,
     spec §2.3)。单标签(dtype=long)与多标签(dtype=float + `BCEWithLogitsLoss`)走**两条
     不同的头**,猜错时报错指向别处 ⇒ 训练侧**显式设** `problem_type` 并当场核对。
  2. **标签顺序是契约**(spec §2.7)。模型第 N 个 logit 对应哪个类目**靠产物里那张表的下标**;
     服务若自己硬编码一份顺序 ⇒ **分布图整张错位,而每一个组件都工作正常**(scores 全在
     0–1、写库成功、页面画得出来、没有任何东西报错)。三道防线**全落在产物上**
     (`labels.json` / `inference_config.json` / `train_meta.json`),服务与评测脚本
     **只走 `app.topic.model.load_artifacts` 这一个读侧**;`/healthz` 把 `labels` **整张表**
     印出来(`num_labels` 对**重排**是瞎的 —— 反序之后它还是 17)。
  3. **train / serve 同源**:清洗(`app/topic/clean.py`)与类目表(`app/topic/taxonomy.py`)
     两侧**import 同一份**,不是各写一遍 —— 写成测试守着(spec §5.1 / §9.1)。
  4. **跑批整批原子**:`classify_topics.py` 一次运行**一个事务**,服务故障 ⇒ **什么都不写**。
     ⚠️ **但「服务连不上」与「服务返回空」是两件事,别写成一件**(订正轮 1 修:
     这句话此前写成「绝不写空标签」,而 **`labels=[]` 是合法的模型输出**,照写)——
     **连不上 ⇒ 一行都不写;返回空 ⇒ 照写 + 必须把条数印出来**。
     不数出来的后果才是那条命门:**分布页会把「服务每次都返回空」读成
     「这些问题没有主题」—— 故障被读成业务结果**(见 `scripts/classify_topics.py` §b 的
     订正 15-C 与 `tests/test_classify_topics.py::test_empty_labels_are_allowed_but_counted`)。
  5. **`labels` 是 JSON 列,读它要用 `JSON_TYPE` 而不是 `IS NULL`**:ORM 的 `JSON` 类型
     (`none_as_null=False`)把 Python 的 `None` 落成**字面 JSON `null`**,它在 SQL 上
     **不是 NULL** ⇒ `WHERE labels IS NULL` 一行都筛不出来。同一族的账 ch09 在
     `evidence_snapshot` 上已经付过一次(见下面「已知问题与未达成项」)。

**ch03 不做**:关键词召回、混合检索(BGE-M3 的 sparse/colbert)、重排 —— 只跑 dense 单路。**ch04 不做**:文档删除/编辑、任务持久化、并发任务队列。**ch07 不做**:跨会话长期记忆、用户画像、语义检索捞历史、主题重要度、摘要淘汰清理(表只追加)。**ch08 不做**:Skill 机制、接更多外部系统、工具的**热重载**(改完**我们自己的代码**不重启 —— §3.2 的界线只到「新增一个内置文件」为止)。**ch09 不做**(spec §1 非目标):低置信度问题**按主题归类的微调分类器**(用户点名「下一步的事」);**`Faithfulness` 之类的生成段 LLM-as-judge 指标**(用户 2026-09-23 订正:需求里那半个词指的是**置信度兜底机制**,`eval_runs` 只落**检索段**指标);**不改 ch08 的工具系统 / 确认流 / MCP 接入**;跨会话长期记忆与用户画像照旧不做。另:Langfuse 用 **Cloud** 不自部署(用户 2026-09-23 拍板,spec §2.1),prompt 版本管理 / 数据集与实验那一半没接。**全程不做**:多轮 Agent Loop。
(**认证**原本也在此列,由**用户 2026-09-27 要求**补上 —— 见
`docs/superpowers/specs/2026-09-27-ecommerce-cs-auth-design.md` 与
`dev-notes/ch10.md`。)

文档即设计源:`docs/superpowers/specs/` 下的 spec 是权威设计文档(内有「实现订正」小节,记录代码与最初设计的偏离及原因);`dev-notes/chNN.md` 是按阶段实时记录的开发留痕。改行为前先读 spec 对应章节。

## 高频命令

```bash
.venv/Scripts/python.exe -m pytest                             # 全部测试(含 db,需 MySQL)
.venv/Scripts/python.exe -m pytest -m "not db"                 # 只跑不需要数据库的
.venv/Scripts/python.exe -m pytest tests/test_trim.py          # 单个文件
.venv/Scripts/python.exe -m pytest tests/test_trim.py::test_rounds_never_start_with_a_tool_message

.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000    # 起服务;浏览器开 http://localhost:8000
.venv/Scripts/python.exe evals/run_tool_selection_eval.py       # 工具选择评估集,需真实 key + MySQL
bash scripts/acceptance.sh                                      # 端到端验收 1–9,需服务已启动 + 真实 key
# 工作台(ch04 起叫「管理台」,ch10 跟进改名):浏览器开 http://localhost:8000/admin.html
# 六个目录:首页 / 入库 / 待审 / 评测 / 主题分布 / 链路。首页五张卡是各页的摘要 + 入口;
# 「评测」页的「运行评测」按钮跑的是与命令行逐字相同的那条命令(见下面的 ch10 一段)。

# 认证(2026-09-27;前置同 ch01:MySQL + 真实 key)
.venv/Scripts/python.exe scripts/seed_users.py                   # 预置演示账号(cinfly / demo-user;幂等)
# ↑ 登录页那两个账号从这儿来;改密码 / 加账号都改这个脚本的 ACCOUNTS 再跑一次。
#   ⚠️ 账号**不是配置项**(不在 `.env` 里):scrypt 的盐是随机的,写进配置就得把某一轮的盐焊死。
#   `.env` 里只有 `JWT_SECRET` / `JWT_EXPIRE_MINUTES` 两个设置项,演示值见入库的 `.env.example`。

# ch07(前置:MySQL + Milvus + 真实 key)
.venv/Scripts/python.exe scripts/run_summary_eval.py            # 摘要标注样例评估(11 条,打网络)
bash scripts/acceptance_ch07.sh                                 # ch07 验收 1–5;⚠️ **自己起服务**(8000 演示配置 / 8001 默认配置)
# ↑ 跑之前先清空 8000/8001;它自己截断 log/app.log(不截断的话断言会命中旧行而恒真)

# ch08(前置:MySQL + 真实 key;**不需要** Milvus)
.venv/Scripts/python.exe -m mcp_servers.logistics                 # 手动起物流 Server:127.0.0.1:8101/mcp
.venv/Scripts/python.exe -m mcp_servers.aftersales                # 手动起售后 Server:127.0.0.1:8102/mcp
bash scripts/acceptance_ch08.sh                                   # ch08 验收 1–6
# ↑ **验收脚本自己起三样东西** —— 两个 MCP Server(8101/8102)+ 客服服务(8000),
#   跑完自己收干净;上面那两条 `-m` 是给**手工演示/手工核验**用的,验收不需要先跑它们。
#   它在 8000 上**反复起停客服服务**(验收 1 装完新工具起一次、还原后再起一次、
#   验收 6 换一套 env 再起一次),所以跑之前先清掉 8000 的残留进程:
#   否则会 curl 到旧代码,得到本仓记过的那类**假红**。端口可用 `PORT` / `MCP_*_PORT` 覆盖。
#   ⚠️ 验收 3 重启的是 **MCP Server**,不是客服服务 —— 它的断言正是「客服服务的进程号没变」。

# ch09(前置:MySQL + **Milvus** + 真实 key + **Langfuse 可达**)
.venv/Scripts/python.exe scripts/calibrate_evidence.py            # 置信度阈值标定(读 evals/测试集.md 300 条,走真实链路)
# ↑ 改 `evidence_confidence_threshold` 之前**先跑它**;`--dump` 出逐条置信度(定「平台段」靠它)
.venv/Scripts/python.exe scripts/intent_cost.py --minutes 30      # 按意图的 token 花销(打 Langfuse Metrics API)
.venv/Scripts/python.exe scripts/run_eval.py --limit 5 --trigger manual   # 只跑前 5 条(给验收用)
.venv/Scripts/python.exe scripts/eval_trend.py                    # 评估趋势(每轮相对上一轮的增减)
.venv/Scripts/python.exe scripts/run_flywheel_eval.py             # 飞轮评估集(标准化 + 查重两半,打网络)
bash scripts/acceptance_ch09.sh                                   # ch09 验收 1–6
# ↑ **验收脚本自己起客服服务(8000)+ 两个 MCP Server(尽力而为:起不来只 WARN)**,
#   跑完自己收干净 ⇒ 跑之前先清掉 8000/8101/8102 的残留进程(否则会 curl 到旧代码 —— 本仓记过的那类假红)。
#   ⚠️ **一次跑要好几分钟**(⑥ 收尾那一轮是 300 条用例的全量评估),而且它**会往共享的表里
#   真的写东西**:② 落一条低置信度问题,③ 把它核准进知识库(写 1 块 + 向量化 1),④ 再落一条 👎,
#   ⑥ 往 `eval_runs` 加 3 行(两轮 5 条 + 收尾一轮 300 条全量)。**每一轮聊天还会连带涨**
#   `conversations` / `messages` / `tool_audit_logs`(本机 2026-09-24 读数:`tool_audit_logs`
#   共 **8730** 行、**光这一天写了 2024 行**;`conversations` 468 / `messages` 2284 ——
#   这几个数只是当时的读数,随用例与运行次数增长,别当常量引用)。
#   它自己也会在结尾打一份**「局限」**清单(哪些东西 6/6 并不覆盖)—— 读之前先看那几行。
#   ⚠️ ③ 跑完**那条问题就答得对了** ⇒ ② 的选题是**运行时现挑**的:脚本从 `CAND_Q*` 里挑
#   第一条「此刻 `GET /api/kb/search` 召不到」的候选用,用尽就**响亮地报**「请加一条」
#   (每跑一轮消耗一条)。这比写死题面、第二轮起悄悄变红要诚实。
#   ⚠️ **所以这个脚本不是「可以无限重跑」的**:它有一个**候选预算**。
#   ⚠️ **已消耗的候选是「被跳过」,不是「被复用」**(这一条最容易算错 —— T19 复审就抓到一次):
#   候选一旦被用过,那条问题**就进了知识库**(③ 会核准它)⇒ 下一轮它**召得到**了 ⇒
#   脚本的选题判据(「此刻召不到」)**永远跳过它**。⇒ **剩余数 = 总数 − 已消耗数**,
#   而 `CAND_Qn` 的编号**不重置**。
#   实际耗用轨迹(逐次可核,三份转录里都印着「选中候选 N」):
#   **T18 的八次运行消耗了 `#1–#7`**(run8 结束时「只剩 `#8` 一条」);
#   **`#8` 死在 T19 的第一次验收跑**(夹具缺陷,那次 5/6);
#   **T19 随后补了 `#9–#16`(8 条)**;三次跑又消耗了 **`#9`(#2 跑)与 `#10`(#3 跑)**
#   ⇒ **做完 T19,可用的是 `#11–#16`,共 6 条**。
#   预算用尽时的唯一正解是**往 `CAND_Q*` 里加候选**,不是把题面写死 ——
#   写死之后第二轮起会**悄悄变红**(那条问题已经被上一轮写进知识库了)。
#   ⚠️ **加候选有三条选材规矩**(都在 T19 用真跑换来的,写在脚本头的「特征串三条选材规矩」里):
#   ① marker 必须真的出现在它自己的答案里,**且要用脚本自己的解码器核**(那是
#   `chr(int(h,16))` **码点**,不是 UTF-8 字节);② **marker 里不许出现数字**(中文与阿拉伯
#   都不行 —— 实测「一点八米」被模型写成「1.8 米」,③ 那条回声断言因此红,而**产品每一环都对**);
#   ③ **核准答案必须真的回答那个题面**(实测:一条没回答「转速是多少」的答案让模型
#   **正确地**自评不足走了兜底 ⇒ ③ 那条**承重**断言也红)。
#   ⇒ **③ 不是一条不变的断言**:它断的是「模型**逐字**复述了核准答案里的某个词组」,
#   而模型会改写(尤其数字)。红的判据是「先看产品那几步(写块 / 向量化 / 召回 / 过闸 /
#   带引用)是不是都绿,再判是不是代码坏了」。
#   ⚠️ ①⑤ 依赖 Langfuse,而**读**侧(REST 查询)通了**不代表写**侧(服务端 OTel exporter)通 ——
#   实测本机导出会成片读超时几分钟而读侧一切正常;那种情况下 ① 会**重试两轮**,仍失败就把
#   服务端日志里那行 `opentelemetry.exporter ... Read timed out` 打出来指认错因。
#   ⚠️ ① 的「界面能点开、点开之后是什么样」**靠人看**:脚本断的只有「数据在不在」(观测齐、同一条 trace)。
#   ⚠️ 它会把**本轮自己的输出**转录一份到 `log/acceptance_ch09_self.log`(已 gitignore),
#   并在判词**之前**扫它一遍:**命中 `command not found` / `syntax error` / `unexpected EOF`
#   就判红** —— 让「装置自己喷错误行」不再可能与「6/6 通过」共存(踩过:收尾文案里的反引号
#   被 bash 当命令替换执行,喷了一屏错误而脚本照样报 6/6)。失败时那份转录会被复制进证据目录。

# ch10(前置:MySQL + 真实 key;**不需要 Milvus** —— 分类器这条线不检索)
#   起服务之前**先起旁路推理服务**(8103):②③ 两类验收全靠它,`classify_topics.py` 也要它。
.venv/Scripts/python.exe -m topic_service --model models/topic-clf --port 8103
.venv/Scripts/python.exe scripts/prepare_topic_data.py collect   # 语料合流(三源 → corpus.jsonl)
.venv/Scripts/python.exe scripts/train_topic_clf.py              # 训练(约十几分钟;产物不入库)
.venv/Scripts/python.exe scripts/eval_topic_clf.py --report evals/topic/report.md  # 验收 ① 的落点
.venv/Scripts/python.exe scripts/classify_topics.py --dry-run    # 先看会写哪些行
.venv/Scripts/python.exe scripts/classify_topics.py --limit 20   # 跑批(写 topic_classifications)
bash scripts/acceptance_ch10.sh                                  # ch10 验收 ①–④
# ↑ 检索评测那一轮也可以**从工作台点**(「评测」页的「运行评测」):它跑的是与
#   `.venv/Scripts/python.exe scripts/run_eval.py --trigger manual` 逐字相同的命令,
#   只是包在一个**子进程**里(`app/kb/eval_job.py`)。⚠️ 它**需要 Milvus**,
#   而且**每跑成一**轮就往 `eval_runs` **多写一行**(撤不掉)。
# ↑ **它自己起四样东西**:客服服务(8000)+ 两个 MCP Server(尽力而为)+ **旁路服务(8103)**
#   ⇒ 跑之前先清掉 8000/8101/8102/8103 的残留进程(否则 curl 到旧代码 —— 本仓记过的那种假红)。
#   ⚠️ **一次实测 100–110 秒**:① 会把评测脚本**跑两遍**(比两次运行的产物是否逐字节相同)。
#   (原先这里写「约 2–4 分钟」—— **那是个没量过的数**,订正轮 1 按实测改。)
#   ⚠️ **必须在 Git Bash 里跑**:从 cmd/PowerShell/Python 的 `subprocess` 直接调 `bash` 会解析到
#   **WSL 的 bash**(CreateProcess 把 System32 排在 PATH 之前),那里 `localhost` 与 `/tmp`
#   都不是 Windows 这边的 ⇒ 一屏假红。脚本自己有一道 `OSTYPE` 自检把这种情况拦在开头。
#   ⚠️ 它**会写库**:② 那一节跑 `classify_topics.py --limit 20`(upsert,幂等:唯一键
#   `uk_pool_question` 保证「重跑 = 覆盖,不是追加」)。① 的两次运行产物落在 `$TEMP` 下
#   (**不写 `/tmp`**:bash 的 `/tmp` 是 MSYS 的,Python 的 `/tmp` 是 `D:\tmp`,不是同一个地方)。
#   ⚠️ ① 有一条硬断言是「**入库的五份报告 == 本次运行的那五份**」⇒ 换了权重必须重生成报告。

# 建库 / 升级(Milvus 另需 docker start milvus-standalone;BGE-M3 等权重由 main.py 预热)
.venv/Scripts/python.exe scripts/init_db.py                     # 建表:create_all,只建**不存在的表**
# ⚠️ **`init_db.py` 永不加列。** `create_all` 对已存在的表是**空操作** —— 它不会
#    比对形状,也不会报错。所以**每个带 `db/chNN.sql` 的章都必须在 init_db 之外
#    再执行那份 DDL**,升级一个老库时尤其:`db/ch03.sql`(knowledge_chunks)、
#    `db/ch04.sql`(low_confidence_questions)、`db/ch06.sql`(refund_requests)、
#    `db/ch07.sql`(新表 conversation_summaries + conversations 的两个锚点列)、
#    **`db/ch08.sql`(新表 tool_audit_logs)**、
#    **`db/ch09.sql`(新表 `review_queue` + `eval_runs`,外加
#    `low_confidence_questions` 的两列 `evidence_snapshot` / `matched_review_id`)**、
#    **`db/ch10.sql`(新表 `topic_classifications`,ch10-B 的归类结果)**、
#    **`db/auth.sql`(新表 `users`,认证功能的账号表,2026-09-27)**、
#    **`db/followup_messages_citations.sql`(`messages` 加一列 `citations`)**
#    ⚠️ 后两份**都不带章号,因为它们都不是一章的产物** —— `db/auth.sql` 是 2026-09-27
#    那次「加登录」的功能追加(`db/followup_messages_citations.sql` 是它的先例);
#    followup 那份的起因是「聊天页历史回载
#    丢掉了工具齿轮与文档链接」这个缺陷(2026-09-27 修)。
#    它的走法**与 ch09 那份同款(是一条 ALTER)**:**旧库必须手动跑它**
#    (`init_db.py` 永不加列,`create_all` 对已存在的 `messages` 是空操作);
#    **全新库不要跑它** —— ORM 侧有这一列,`create_all` 会连它一起建出来,
#    那份 DDL 再跑一遍就在 ALTER 上报 **1060**(与 ch08/ch10 的 1050 同族,
#    **刻意不幂等**)。漏掉它的后果**不是「少个功能」**:`messages` 缺这一列 ⇒
#    **每一个请求**的落库(`append_turn`)与**每一次回载**都在 `Unknown column`
#    那一步炸。两条路径的差异只有**列序**(ALTER 追加在表末、create_all 按 ORM
#    声明序;纯文本差异,没有一条 SQL 按位置取值),写在那个文件头里。
#    ⚠️ `db/ch10.sql` 与 `db/ch08.sql` **同款**:它只有 `CREATE TABLE`,而 ORM 侧有同名模型
#    (`TopicClassification`)⇒ `init_db.py` 的 create_all **已经把它建出来了** ⇒
#    **在全新库上跑它会在那条 CREATE 上响亮地报 `ERROR 1050`(表已存在)。
#    那是刻意的、不是脏库** —— 与 `db/ch06.sql` 的 refund_requests、`db/ch08.sql` 的
#    tool_audit_logs 是同一个已知取舍。**想让 DDL 成为权威形状**才需要
#    `DROP TABLE topic_classifications;` 再跑一遍(头部的注释写着这条,T12 就是这么核的);
#    两条路径的形状差异逐条记在 `app/db/models.py` 的 `TopicClassification` docstring 里。
#    漏掉它的后果是**功能性的**:`/api/topics/distribution` 与 `classify_topics.py` 直接
#    `Unknown table`(这张表**没有别的写方**,批处理是唯一入口)。
#    ⚠️ **`db/auth.sql` 与 `db/ch08.sql` / `db/ch10.sql` 同族**(**不是**与 ch09 那份同族):
#    它只有一句 `CREATE TABLE users`,而 ORM 侧有同名模型(`app/db/models.py` 的 `User`)
#    ⇒ `init_db.py` 的 create_all **已经把它建出来了** ⇒ **在全新库上跑它会在那条
#    CREATE 上响亮地报 `ERROR 1050`(表已存在)。那是刻意的、不是脏库** ——
#    与 `db/ch06.sql` 的 refund_requests、`db/ch08.sql` 的 tool_audit_logs、
#    `db/ch10.sql` 的 topic_classifications 是**同一个**已知取舍(本仓的第四处)。
#    ⇒ **正常路径只需要 `init_db.py`,不需要跑这一份**;想让**这份 DDL 成为形状的权威**
#    才需要 `DROP TABLE users;` 再跑一遍,然后 `SHOW CREATE TABLE users\G` 核对。
#    两条路径的形状差异**只有文本**(列定义逐字一致;DDL 那份多了表级 `COMMENT`)——
#    逐条写在 `db/auth.sql` 的文件头里。
#    漏掉它的后果是**功能性的**:`users` 表不存在 ⇒ `POST /api/auth/login` 与
#    `scripts/seed_users.py` 直接 `Unknown table`,**一个账号都登不进去**。
#    ⚠️ **走法别照抄 ch09 那份**:`db/ch09.sql` 第一句就是 ALTER ⇒
#    **老库升级要「先 ch09.sql、再 init_db.py」**;这份的走法与 ch08/ch10 一致
#    (**先 `init_db.py`**;1050 之后 `DROP TABLE users;` 再跑)。
#    账号数据**不在这份 DDL 里**(scrypt 的盐是随机的),由 `scripts/seed_users.py` 幂等写入。
#    漏掉 ch09 那份的后果**不是「少个功能」**:飞轮每条 `WHERE matched_review_id IS NULL`
#    的选择谓词、审核页的每一行、趋势表的每一轮都读那两列/两张表 ⇒ 一进 `/api/review/*`
#    或 `eval_trend.py` 就是 `Unknown column`。⚠️ 走法与其余几份**相反**(DDL 头部写着,
#    实测于 2026-09-23):它**第一句就是 ALTER `low_confidence_questions`** ⇒
#    **老库升级:先 `db/ch09.sql`、再 `init_db.py`**;**全新库:只跑 `init_db.py`,
#    不要跑那份 DDL**(先跑它在 ALTER 上 1146,先跑 init_db 再跑它 1060 + 1050 三条全红)。
#    漏掉 ch07 那份的后果不是「少个功能」:`conversations` 缺两列、`conversation_summaries`
#    整张表不存在 ⇒ **每一个请求**都在 `ensure_conversation` 或分层读锚点那一步炸,
#    而报错指向 SQL 列名,读起来像「ORM 写错了」。五份文件都不幂等(重复执行**响亮地失败**,
#    这是刻意的:静默跳过会让「表已存在但形状不对」永远补不上);顺序上 `db/ch07.sql`
#    自己**先 ALTER 后 CREATE**,必须在它内部的次序就是那样,别再调。
#    ⚠️ `db/ch08.sql` 在**全新**库上会响亮地报 `ERROR 1050`(表已存在)—— 因为 ORM 侧有
#    同名模型,`init_db.py` 的 create_all 已经顺带把这张表建了出来。**这是刻意的**,
#    与 `db/ch06.sql` 的 refund_requests 是同一个已知取舍,不是脏库。两条路径建出来的
#    表**形状仍有三处不同**(T5 修复轮把行为差异都对齐了,**剩下的全是文本差异**;
#    逐列编译 `ToolAuditLog.__table__` 的 `CreateTable` 与 `db/ch08.sql` 对着数出来的):
#    ① **两个索引名** —— DDL 的 `idx_conv` / `idx_created` 对 SQLAlchemy 自动生成的
#       `ix_tool_audit_logs_conversation_id` / `ix_tool_audit_logs_created_at`;
#    ② **表的 `COMMENT`** —— DDL 有 `COMMENT='工具调用审计(ch08)'`,ORM 侧为 None;
#    ③ **七个列级 `COMMENT`** —— DDL 里 `conversation_id` / `tool_call_id` / `source` /
#       `args` / `result_summary` / `status` / `retry_count` 各带 `COMMENT '…'`,
#       而 ORM 侧全文没有任何 `comment=`。
#    **其余都已对齐**:`id` 两条路都是 `BIGINT`;带 `DEFAULT ''` 的字符串列**恰好两个**
#    (`result_summary`、`error_detail`),`args` 与 `status` 两条路**都没有** `DEFAULT`;
#    `created_at` 两条路**都有索引**(只是名字不同,即上面 ①)。
#    ⚠️ **③ 曾被写小成「六个」**(审查员列清单时漏了 `tool_call_id`),而 CLAUDE.md 一度
#    只写了 ①② —— **「只剩两处」是个源不支持的绝对断言**。数一遍再引用。
#    所以新库上仍然应当**让 DDL 建表**:要么先跑
#    `db/ch08.sql` 再跑 `init_db.py`,要么 1050 之后 `DROP TABLE tool_audit_logs;` 再跑一遍
#    那份 DDL,然后用 `SHOW CREATE TABLE tool_audit_logs\G` 核对(见 `dev-notes/ch08.md`)。
#    全新 checkout 的顺序:`init_db.py` → 依次 `db/ch03.sql` / `ch04` / `ch06` / `ch07` / `ch08`
#    (**`db/ch09.sql` / `db/ch10.sql` / `db/auth.sql` /
#    `db/followup_messages_citations.sql` 都不在此列**
#    —— ch09 那份第一句是 ALTER(空库上 1146)、ch10 与 auth 只有 CREATE(会 1050)、
#    followup 那份是 ALTER(空库上 1060,列已存在);那四份的走法各自写在上面
#    与它们自己的文件头里)。

# ch03(前置:docker start milvus-standalone)
.venv/Scripts/python.exe scripts/build_kb.py                    # 建库;重跑=幂等补齐(中断了直接再跑)
.venv/Scripts/python.exe scripts/build_kb.py --reindex          # 全表打回 pending + 删集合 + 重算
.venv/Scripts/python.exe scripts/mine_qa.py                     # 对话挖知识(独立脚本 + 外部调度)
.venv/Scripts/python.exe evals/run_retrieval_eval.py            # 检索评估,需 Milvus + BGE-M3
.venv/Scripts/python.exe evals/run_retrieval_eval.py --dist     # 打印相似度分布,用于定阈值
```

`pytest.ini` 已设 `testpaths = tests`、`addopts = -q`。异步测试用 `@pytest.mark.anyio`(backend 由 `tests/conftest.py` 固定为 asyncio),不用 pytest-asyncio。

**不要再往命令行加 `-q`。** `addopts` 里已有一个,叠加成 `-qq` 后 pytest 在 `verbosity < -1` 时**整行不打印 `N passed`** —— 失败仍会报,但通过数没了,等于把「我验过」的证据抹掉。要过滤用 `-m "not db"`。

## 架构

依赖方向严格单向:`api → services → {tools, db, memory, prompts, llm}`。

```
app/config.py     pydantic-settings 读 .env;四个必填字段(三个 OPENAI_* + DATABASE_URL),数值项带下界
app/llm.py        ChatOpenAI 工厂,_build 收口全部硬约束
app/prompts.py    System/抽取 Prompt + 消息组装;Message -> BaseMessage 转换的**唯一**出口
app/schemas.py    纯数据模型,唯一被到处引用的类型源
app/auth.py       认证功能(2026-09-27):**全仓唯一**的鉴权边界(写法对齐 `app/observability.py`)。
                  别的模块只认 10 个名字:`AuthError` / `AuthenticatedUser` / `USER` / `ADMIN` /
                  `hash_password` / `verify_password` / `create_token` / `decode_token` /
                  `require_user` / `require_admin`。身份(用户名 + role)**放在 token 的 claims 里**
                  ⇒ 每个受保护请求**零次库往返**(代价:改 role 后旧 token 到过期前仍带旧 role)。
app/db/           base(引擎/会话工厂)、models(**13 张表**:conversations / messages / tickets /
                  knowledge_chunks / low_confidence_questions / qa_extraction_staging /
                  refund_requests / conversation_summaries / tool_audit_logs /
                  review_queue / eval_runs / topic_classifications / users)、session(FastAPI 依赖)
app/tools/        ch08 起是**工具系统**,不再是「五个 @tool 放一个文件」:
                  spec.py(ToolSpec + **唯一**的 JSON Schema 校验器 validate_args)、
                  policy.py(权限声明表 kind_of,未声明 = 只读)、
                  audit.py(审计**唯一写口** record_audit,永不抛)、
                  builtin/(**包内自动发现**,一文件一工具)、mock_data.py(种子数据源,内置与 MCP 共用)、
                  registry.py(每请求组装 name → ToolSpec)、executor.py(**唯一执行点**,超时/重试/六类分诊)
app/mcp/          ch08 在线:client.py —— 每请求现问现拿两个业务 Server 的工具,单 Server 挂了降级
mcp_servers/      ch08 两个**独立进程**的业务 Server(logistics 8101 / aftersales 8102,Streamable HTTP)
app/memory/       ch01-06:store.py(锁注册表)、trim.py(token 计数与按整轮切轮);
                  ch07 新增:budget.py(窗口→历史预算→层1/层2)、layers.py(三层切分 + 层2 截短)、
                  summarize.py(摘要 prompt + 触发判定 + 原子落库)、tasks.py(后台摘要执行体)、
                  journal.py(两个上下文日志)—— **全部不依赖 LangChain**
app/services/     chat.py(纯校验的 prepare_turn)、extract.py(抽取)、history.py(会话历史读写)
app/api/          chat.py、extract.py、conversations.py(ch07 两个只读端点)、
                  feedback.py(ch09 飞轮入口 ③:`POST /api/feedback`)、
                  review.py(ch09 待审队列的四个端点)、
                  topics.py(ch10-B 只读分布)、traces.py(ch10 跟进:Langfuse 只读代理)、
                  auth.py(2026-09-27:登录 `POST /api/auth/login` + `GET /api/auth/me`)
                  ⚠️ **鉴权挂在各 router 的 `APIRouter(dependencies=[…])` 上**(不逐个端点写):
                  用户面 `require_user`、工作台 `require_admin`,**唯一公开**的是登录端点。
                  全矩阵 **27 个操作**,权威清单是 spec §6.4 的三张表;
                  `tests/test_auth_wiring.py` 把那张表**整张抄进测试断全等**(集合 + 守卫类型)——
                  只断「有没有守卫」是瞎的:把某个端点的守卫换成 admin 也照样全绿(复审实测)。
app/static/       聊天页 + **工作台** admin.html(单页,无构建工具链;
                  ch10 跟进起有**六个目录**:首页 / 入库 / 待审 / 评测 / 主题分布 / 链路);
                  auth.js(2026-09-27:两页共用的**唯一**「401 怎么办」出口 ——
                  `authFetch` 带 token、401 ⇒ 清 token + 弹浮层 + **登录后重放那一次调用**,
                  ⚠️ 静态目录挂在 `/` ⇒ 页面上引用的是 **`/auth.js`,不是 `/static/auth.js`**)
app/retrieval/    ch03 在线检索:embedder.py(BGE-M3 懒加载)、milvus.py、search.py(KnowledgeRetriever)
app/kb/           ch03 离线管线(不在请求路径上):chunker / ingest / writer / mining;
                  ch04 的 jobs.py(JobStore)与 orchestrate.py(后台任务);
                  ch10 跟进增 **eval_job.py**:评测**子进程**任务(工作台的「运行评测」
                  按钮;四条后台任务共用同一个 JobStore 槽);
                  ch09 增 **evidence.py**(三信号置信度 `evidence_confidence`,纯函数);
                  ch09 改 **assess.py**(落池写口 `record_low_confidence` 多收一个
                  `evidence_snapshot=`;`None` 与 `[]` 是**两个不同的值**)
app/observability.py  ch09:**全章唯一**的 Langfuse 边界(四个名字:`trace_scope` / `intent_scope` /
                  `span` / `make_handler`)。关掉时全 no-op 且不 import langfuse
app/agent/json_stream.py  ch09:三态增量 JSON 解码器(纯状态机,零 IO/零 await/零依赖)
app/flywheel/     ch09 数据飞轮(**池子的下游**):normalize.py(口语 → 标准问法 + 示例答案)、
                  dedupe.py(对 `review_queue` 做**语义**查重)、pipeline.py(编排:归并或新建)、
                  tasks.py(后台任务,带**整条任务的墙钟上界**)
knowledge/        知识语料(3 份 Markdown,首行带 <!--type: ...--> 类型标记)
scripts/          build_kb.py、mine_qa.py(离线建库与挖知识)、calibrate_evidence.py、
                  intent_cost.py、eval_trend.py、run_flywheel_eval.py
app/topic/        ch10-B 分类器的**纯函数内核**(⚠️ **除 `model.py` 之外**零 torch、零 IO ——
                  `model.py` 既 `import torch` 又读写产物目录,别把这一行读成整包的属性):taxonomy.py(17 类权威表 +
                  BOUNDARY + 9→17 投影)、clean.py(脱敏/格式,训练与推理**同源**)、
                  labeling.py(分层切分 + 证据校验 + `is_unusable_target` + 错别字注入)、
                  metrics.py(四个指标 + 两张矩阵,**零第三方依赖**)、model.py(产物的**唯一读侧**
                  `load_artifacts`)、synth.py(配额合成的形态约束)
topic_service/    ch10-B 的**旁路推理进程**(8103,`python -m topic_service`):model.py(**产物 → logits
                  → sigmoid → 阈值 → 标签**,标签顺序/阈值/长度一律读产物)+ server.py(`/predict` 与
                  `/healthz`)。**不在对话链路上** —— 主链路零调用分类器(源码扫描测试守着)
app/api/topics.py ch10-B:只读分布接口(标签在 SQL 里用 `JSON_TABLE` 展开;**回池子按题面**数
                  「不同问题数」,不是按池子行 id —— 那一列有唯一键,数与行数恒等)
scripts/          ch10-B 加:prepare_topic_data.py(合流/切分)、train_topic_clf.py(微调)、
                  eval_topic_clf.py(冻结测试集评测,验收 ① 的落点)、classify_topics.py(批量归类)
```

ch03 把依赖方向扩展为 `tools → retrieval → db` 与 `kb → {db, llm, retrieval}`,仍是单向。

ch08 又加了两条边,方向分别是:`tools → {db}`(审计落库,本来就有)与 `mcp → tools`
(`client.py` 把 MCP 的工具**转成 `ToolSpec`** 再交给注册表),以及一个**不在 `app/` 下的**
`mcp_servers/ → {tools.mock_data}`(两个 Server 与内置工具**共用同一份种子数据**,
否则同一个订单号会在两边说两套话)。

**ch09 加的四条边**(都不是新方向,但有一条值得单独记):

- **`app/agent/nodes.py` → `app.flywheel.tasks`** —— 知识轮自评不足时,**落池之后**
  fire-and-forget 起一轮飞轮(`start_flywheel_job_safely`,自己吞装配故障)。
  这是**请求路径第一次直接依赖离线管线那一侧**;`app/flywheel/` **不反向引用 `app.agent`**
  ⇒ 图仍无环。
- `app/api/feedback.py` → `{app.kb.assess, app.flywheel.tasks, app.retrieval.search,
  app.tools.registry}`(**重跑一次检索**回捞片段,见 §6.2 的语义偏差)与
  `app/api/review.py` → `{app.kb.writer, app.kb.chunker, app.retrieval.*}`(通过 ⇒ 写库 + 立刻向量化).
- 两个手工观测点(`retrieval`)落在 `app/agent/nodes.py` 与 `app/agent/refund_nodes.py`。

> **`app/observability.py` 是全章唯一的 langfuse 边界**:**`app/` 下零处**别的模块 import
> langfuse(一条**源码扫描**测试守着,见模块 docstring —— 它抓的是「有没有人写下这行 import」)。
> 别的模块只认 `trace_scope` / `intent_scope` / `span` / `make_handler` 四个名字 ⇒
> 「换观测后端」改的是这一个文件。

> **`app/tools/` 不依赖 `app/mcp/`,两者靠 `ToolSpec` 交接。** 这是本章最关键的一条接缝:
> 内置工具与 MCP 工具在注册表里**长得一模一样**(同一种 `ToolSpec`),执行器、权限闸、
> 校验器、审计**都不知道**手里这条是从哪来的。所以 `app/tools/` 里**没有一行** import
> `langchain_mcp_adapters` 或 `app.mcp` —— 想加 MCP 的话,改的是 `app/mcp/client.py`,
> 不是执行器。反过来说:`mcp_servers/` **不在请求路径上、也不 import `app/mcp/`**,
> 它是「别人的进程」在本仓里的模板。

**ch07 又加了两条反向边**(上面那条「严格单向」因此不再完整,而它是本仓最容易被
后来人当成公理的一条):`app/memory/summarize.py` → `app/services/history.py`
(记账在 ch07 spec §12.2①),以及 `app/memory/tasks.py` → `app/services/history.py`
与 `app.db.models`(模块 docstring 也记了一份)。**图仍然无环** ——
`app/services/history.py` 只依赖 `app.db.models` 与 `app.schemas`,不回头引 `memory`,
所以这不是缺陷,不需要重构。两条的理由相同:那段读历史的 I/O(`load_history`)
与「落库 + 推锚点」的**原子动作**只有一份实现,在 `memory/` 里重写一遍就是同一条
规则两处实现。**代价已记账**:日后若 `services/history.py` 反过来 import `memory`
就会成环,届时把 `append_summary_and_advance` 换成一个传入的回调即可,
改动局限在 `summarize_range` 的签名。

两条贯穿性的结构约定:

- **`memory/` 与 `services/history.py` 不依赖 LangChain**,只碰 `app.schemas.Message` 纯数据类。转 `BaseMessage` 是 `prompts.py:to_lc_messages` 一处的职责。
- **`services/` 的函数接收 llm 实例作为参数**,不在模块层建全局单例;FastAPI 侧靠 `Depends` 注入,测试用 `dependency_overrides` 替换。

### ch03 的检索链路

```
离线:knowledge/*.md ──chunker──→ write_chunks(MySQL,pending)
                                    ↓ vectorize_pending(嵌入 + Milvus upsert)
                                MySQL: vector_id + status=done
在线:query_faq(keyword) → KnowledgeRetriever.search
        = 嵌入 → Milvus Top-K → 阈值过滤 → 按 id 回查 MySQL → 按相似度序组装
```

> **2026-09-20:`faq` 表已废弃删除。** 原先离线管线还有一路
> 「faq 表 12 条 ──迁移──→ write_chunks」,那 12 条**已经迁完**并成为
> `knowledge_chunks` 里 `content_type="faq"` 的块(在线检索查的是它 + Milvus)。
> 该表此后没有读写方,**用户已删表**;`Faq` 模型、`faq_migration()`、
> `build_kb.py` 的迁移分支、seed 里的 FAQ_ROWS 一并删除。
> 注意:`content_type="faq"` 与上传类型 `"faq"` 是**取值**,与那张表无关,保留。

- **Milvus 只当索引,不存文本**:集合只有 `id`(= `str(MySQL id)`)与 `vector`;原文一律回 MySQL 取,所以集合可随时 drop 重建(`build_kb --reindex`)。
- **`retrieval/search.py` 是错误语义的翻译边界**:Milvus/嵌入故障 → `ToolInfrastructureError`(502),绝不降级成「没搜到」。`tools/errors.py` 是零依赖的错误词汇表,retrieval 反向引用它已记账(ch03 spec §12)。

### ch07 的上下文管理链路

```
messages 表(按 id 升序)
├───────────────┬──────────────────────┬─────────────────────────┤
│  已被梗概覆盖  │        层 2           │         层 1            │
│  (不再逐条读)  │   中间,截短,预算 30%  │  最近,原文,预算 70%    │
└───────────────┴──────────────────────┴─────────────────────────┘
                ▲                      ▲
     summary_upto_msg_id      layer1_from_msg_id        ← 都是 messages.id
```

每请求(`api/chat.py`,流开始**之前**):预算推导 → `layers.degrade`(层 1 超预算就往后挪锚点,
**只挪 id**;挪过的轮次自动落进层 2)→ `layers.split`(层 2 **按截短后的版本计数**)→
层 2 超预算 ⇒ **起后台摘要任务**(`memory/tasks.py`,不 await)→ 组装 `history`(层2 截短段 + 层1 原文段)
→ 打 `history_ctx` → 播种进 state。Agent 节点组装前再打一行 `model_ctx`(带**分段** token 与锚点)。

- **三档压缩强度递增、代价也递增**:层 1 原文(代价 0)→ 层 2 截短(有损但**可逆**,原文还在 MySQL)
  → 梗概(提炼,**不可逆**,原文从此不进上下文)。**先用便宜的,不够了才用贵的。**
- **两个动作看的版本不同**:降级看层 1 **原文**,摘要看层 2 **截短后**,而摘要**读的是原文**
  (拿截短文本去提炼 = 把截断损失焊进梗概)。
- 摘要的「落库 + 推锚点」是**一个原子动作**(`services/history.append_summary_and_advance`):
  只成一半 ⇒ **梗概重复压一遍** 或 **一段历史永久消失**,两者都不报错。
- 两个只读端点:`GET /api/conversations`(侧栏,`preview` 取第一条 `role='user'` 行的前 30 字)
  与 `GET /api/conversations/{id}/messages`(回载**用户看见过的**对话:滤 `role='tool'` 与空 `content` 行)。

### 两条链路

**对话**:`api/chat.py` → `store.lock_for()` 取锁 → `ensure_conversation` + `load_history`(MySQL) → `prepare_turn` 校验预算 → 显式构造 `EventSourceResponse` → `stream_turn` 单轮编排 → **流完整走完后**才 `append_turn` 落库 → `finally` 释放锁。

**抽取**:`/api/extract` 独立于对话,本章未改动。

### 单轮工具编排(本章核心)

```
第一轮:model.bind_tools(TOOLS).astream(msgs)
         ├ 文本 chunk     → 推 SSE token 帧
         └ tool_call chunk → 累积,不外推
     └ 有 tool_calls → tool_call 帧 → 执行 → tool_result 帧 → 回灌 ToolMessage → 第二轮
        无 tool_calls → 已经流完,单次调用即 done
第二轮:model.astream(msgs)   ← **不绑 tools**
```

**「只做单轮」是结构保证,不是提示词约定。** 第二轮不绑 tools,模型在结构上无法再调。实测过「第二轮仍绑 tools 时模型这次没再调」—— 但那是**模型行为,不是保证**,不采用。

注意:**「单轮」不等于「单工具」**。模型可以在同一轮里并发发多个 `tool_call`,执行器**全部执行**并逐个回灌 —— 这是**必须的**,少回灌一个就构成「有 `tool_calls` 没有对应 tool 消息」,上游直接 400。

SSE 事件协议:`meta` → `token` / `tool_call` → `tool_result` → `done` / `error`。

## 若干不读多文件就会踩的硬约束

每一条都对应一个「报错指向别处、极难定位」的故障。

**`use_responses_api=False`**(`app/llm.py`)。LangChain 1.x 的 OpenAI provider 默认走 Responses API,DeepSeek 等兼容网关只实现 Chat Completions。不关掉会调用失败,而报错指向「模型不存在」。

**`tool_call` 条目必须带 `"type": "tool_call"` 键**(`app/tools/`、所有测试替身)。`BaseTool.ainvoke` 判「这是不是工具调用」**只看** `x.get("type") == "tool_call"` 这一个条件;缺键时它把整个 dict 当成**参数**去校验工具 schema,于是每次调用都返回「参数不合法」的可恢复失败。真实链路上模型流出的 chunk 经 LangChain 解析后**是带这个键的**。测试里的 `FakeChunk` 必须补上。

**抽取只能用 `method="json_mode"`**(`app/services/extract.py`)。本项目端点上 `function_calling` 与 `json_schema` 均返回 400,换模型也一样。连带:提示词里**必须出现字面 `JSON` 字样**,且 `EXTRACT_SYSTEM_PROMPT` 描述结构时**不得使用裸花括号**(`ChatPromptTemplate` 按 f-string 解析)。

**流式取文本用 `chunk.text`,不是 `chunk.content`**。1.x 里后者是 content block 列表。

**`create_ticket` 的 `conversation_id` 用闭包工厂,不用 `InjectedToolArg`**。后者对模型隐藏了字段,但值必须另行注入,而该机制在 langchain-core 1.6.3 上无文档 —— 直接用 `tool_call` 调用会抛 `ValidationError`。闭包让参数**根本不在签名里**。

**工具伪随机必须用 `hashlib.sha256` 种子,不能用内置 `hash()`**。`hash()` 对 str 每进程随机化,会让「同一订单号永远返回同样数据」在重启后失效,而**同进程内的测试完全测不出来**(`test_tools_random.py` 里有一条跨进程测试专钉这个)。

**重试用白名单,不是黑名单**。只重试幂等的 `query_*`;**`create_ticket` 永不重试**(超时后重试会建出两张工单)。`ToolNotFound` 与 `ValidationError` 也都不重试 —— 重放同样的参数只会同样失败。

**错误语义边界**:`422` 只表示「模型输出无法解析为约定结构」;上游故障(401/超时/限流/连接失败)一律 `502` + 固定文案。因为 openai SDK 的 `str(exc)` 是**上游响应体原文**,原样回显会泄漏密钥。所有出站错误文本(SSE `error` 帧、`tool_result` 的失败 `summary`、422/502 的 detail)必须过 `app/sanitize.py:redact_api_key`。

**`ToolInfrastructureError` 必须向上抛,不能回灌给模型** —— 数据库故障绝不能被伪装成「你的订单号查不到」。

**锁必须在所有非流式退出路径上释放**,包括 `except BaseException`(`CancelledError` 是 `BaseException`)。**每一条路径都要有具名测试** —— 漏掉的话持锁会话**永久** 409(持锁会话不被 TTL 也不被 LRU 淘汰)。

**预算校验必须在流开始前完成**。SSE 一旦 yield 过第一帧,响应头就发出了,状态码再也改不了 —— 所以端点是普通 `async def` 手工构造 `EventSourceResponse`。

**ch03 · Milvus/嵌入的七条命门**(每条都对应一次实测故障,细节见 ch03 spec §12 与 `dev-notes/ch03.md`):

**VARCHAR 主键必须显式传 `max_length`**。pymilvus 的快捷建法不会替你补,漏了直接 `1101 type param(max_length) should be specified` 拒建。**单测用假 client 测不出这条**(假 client 只验证"我按我以为的形状调用了"),所以凡是对外部系统的调用,单测绿了还要真机冒烟一次。

**`upsert` 后必须 `flush` 才立查**(默认 Bounded 一致性下 search 返回空);**行数一律用 `query(count(*))`**,`get_collection_stats` 的 `row_count` 数的是 insert 操作、未扣 delete,同 pk 三遍 upsert 会报 9 而真值是 3。

**写 Milvus 成功之后才改 MySQL 状态并提交**。顺序反了,写 Milvus 失败的行会被记成 done、重跑永不补齐 —— 静默的永久缺失,不报错,只让某些知识永远检索不到。

**挖知识的 prompt 必须写明「什么不算知识」**(非答案、个案数据、客套、对话状态)。首版没写,把客服「抱歉查不到运费」这种**非答案**挖成了知识,直接把验收 1 的正确答案挤下 top-1 —— 模型的失败被挖进知识库,再教它下次继续失败。

**去重基准必须在抽取之前取快照**。之后取的话,本轮刚写进 staging 的产物会把自己全判成"已存在",结果是 **0 条保留** —— 而这个错误看起来就像"这轮确实没抽出新东西",没有任何报错。

**冷启动的模型加载不在请求路径上**:2.2GB 权重首次加载十几秒 > `tool_timeout_seconds`(10s),冷进程第一个检索请求必然超时;超时取消协程后会话停在未收尾的事务上,而 `query_faq` **在重试白名单里**,重试复用同一 session 直接撞 `PendingRollbackError` → 用户看到 502。两处保障:`app/main.py` lifespan 起后台线程预热(**pytest 下跳过**,单测不加载模型是硬规矩);retriever 在取消路径 `rollback()` 再抛(`except BaseException` —— `CancelledError` 是 BaseException,只抓 `Exception` 正好漏掉)。

**`_ensure_model` 必须有 `threading.Lock`**:没锁时预热线程与首请求并发会加载**两份 2.2GB** 且两边都成功、不报任何错。

**ch04 · 后台任务的三个命门**(细节见 ch04 spec §12 与 `dev-notes/ch04.md`):

**后台任务必须自建 engine,不能用 `get_engine()` 的 lru_cache 单例** —— 那单例绑在首次使用它的主事件循环上,后台线程 `asyncio.run` 里复用会出跨循环的异步连接问题。`orchestrate.py` 每任务 `create_async_engine` + 任务结束 `dispose()`。

**encode 也要锁(`_encode_lock`)**:后台任务与聊天检索共享同一个 torch 模型实例,并发前向跨线程不保证安全。与 `_load_lock` 分开,串行 encode 的代价是任务批量编码时聊天查询可能等当前一批(数秒)。

**验收/测试的确定性**:聊天里的工具选择与关键词抽取非确定(deepseek 在 temperature=0 下亦然),凡是「召回」「挖知识幂等」这类断言**不要经过 LLM 聊天** —— 召回直接查 retriever、幂等直接查「知识库有无重复三元组」。

**裁剪按 `user` 边界切轮,不按 `assistant`**(`memory/trim.py`)。OpenAI 兼容 API 要求 `tool` 消息前面紧跟着带对应 `tool_call_id` 的 `assistant` 消息;按 `assistant` 收轮会把这对切开,而那**只在历史长到触发裁剪时偶发 400**。

**ch07 · 上下文管理的命门**(细节见 ch07 spec §12 与 `dev-notes/ch07.md`):

**`add_messages` 是 append-only,而且会给没有 id 的消息**当场赋一个 uuid4**(`langgraph.graph.message` 源码逐字:`if m.id is None: m.id = str(uuid.uuid4())`)。**后果**:每轮都「从 MySQL 读全量历史 → 塞进 `state.messages`」的话,重新构造的消息**没有 id** ⇒ **一个都匹配不上** ⇒ 整段历史被**再追加一遍**;第三轮历史就是三份,而**每一轮的回复看起来都完全正常**。两道防线缺一不可:① **只在 `state["messages"]` 为空时播种**;② `to_lc_messages` 给每条消息带上 `str(MySQL id)`(同一个 id 再次并入是**替换**不是追加 ⇒ 重播种幂等)。只有 ① 时「state 非空但库里有更多行」仍会追加;只有 ② 时每轮都要白读一次全量历史。

**`InMemorySaver` 是纯内存字典,「落盘」不成立**(它的 docstring 自己写着 only for debugging or testing;本环境只装了 `langgraph.checkpoint.{base,memory,serde}`,没有任何持久化 saver)。⇒ 需求里那句「State 里的完整历史靠 checkpoint **落盘**留着」**是错的**,已作为与需求的偏离记在 ch07 spec §2.2。**MySQL 才是跨会话/跨进程的权威源**;服务重启后 checkpoint 全空 ⇒ 下一次请求自动从 MySQL 重新播种(这就是「重启自愈」)。

**日志必须显式 `encoding="utf-8"`**(`app/logging_setup.py`)。本机 locale 是 **cp936**,不给 encoding 时 Python 用 `locale.getpreferredencoding()`,**中文日志行直接抛 `UnicodeEncodeError`** —— 而它发生在**写日志的时候**,报错位置指向与业务毫无关系的地方。连带:`setup_logging()` 必须在 **pytest 下跳过**,否则整套测试往仓库根写 108KB 的 `log/app.log`,**而验收 4b 就是 `grep log/app.log`** ⇒ 陈旧行让那条断言恒真。

**层 2 必须按「截短后的版本」计数**(`layers.split`)。按原文数的话,截短就退化成纯渲染装饰 —— 摘要该什么时候触发还是什么时候触发,**截短对级联零影响**,而所有输出看起来都正常。这是全章最容易静默失效的一处(它同时是 T4 那条「不可能满足的断言」要守护的性质)。

**两个锚点只能挪 id,不能搬数据;且 `0` 是有含义的值**(`summary_upto_msg_id=0` = 尚无梗概,`layer1_from_msg_id=0` = 层 1 起于最早、**层 2 为空**)。把 `0` 当成「到末尾」会让**新会话的每条消息同时落在层 1 与层 2**(组装是 `layer2 + layer1`)⇒ 上下文凭空翻倍,**不报错、不丢消息**。同理:`model_ctx` 的锚点**不能**由调用方传、更不能带 `0` 默认值 —— 那会打出一对长得像真值、却什么也没说的 `bounds`。

**层 2 截短只截 `content`,结构字段(`tool_calls` / `tool_call_id`)原样保留**。截断 `tool_calls` 就是把 `tool` 消息与它父亲拆开 ⇒ 上游 400,且**只在历史长到触发分层时复现**。

**摘要失败等于什么都没发生**(不重试、边界不动):「落库 + 推锚点」的原子性保证了两者要么都成、要么都不成。**模型返回空/纯空白时也不许落库、不许推锚点** —— 写一条空梗概**再**推锚点等于**把那段历史静默删除**(层 2 不再读它,而摘要表里那一段是空的)。`memory/tasks.py` 的 `summary skip` 因此把「区间为空」与「模型吐空」**分开记**。

**后台摘要任务必须自建 engine**(专用线程 + 线程内 `asyncio.run` + 任务结束 `dispose()`),理由与 ch04 的 `orchestrate.py` 完全相同:`get_engine()` 的 lru_cache 单例绑在**首次使用它的事件循环**上。**在跑标记的摘除必须在 `finally` 里** —— 漏掉不是「多跑一次」,而是那个会话**再也压不了**(每次都被当成「已有任务在跑」),而用户侧每一轮看起来都完全正常。

**ch08 · 工具系统与 MCP 的命门**(细节见 ch08 spec §15 与 `dev-notes/ch08.md`):

**`mcp>=1.24,<2` 是被 `langchain-mcp-adapters==0.3.2` 的 `Requires-Dist` 钉死的**(轮子 `METADATA` 逐字:`mcp<2.0.0,>=1.24.0`;实际解析到 **1.30.0**)。不要装 2.x。**Context7 整站已经迁到 v2**(连标着 v1 的 library id 返回的也是 v2 内容),所以**这一处以锁定版本的轮子源码为准,不以文档为准**:1.x 的入口是 **`mcp.server.FastMCP`**(不是 `MCPServer`),传输参数是**直接关键字参数** `FastMCP(name, *, host=…, port=…, streamable_http_path=…, stateless_http=…, json_response=…)`,**不是** `FastMCP(..., settings=Settings(...))` —— 那个模块里**另一个**也叫 `Settings` 的 pydantic 模型**字段全无默认值**,照猜会踩进去(写实现计划时逐字核对 `__init__` 才挖出来)。

**`convert_mcp_tool_to_langchain_tool` 必须传 `connection=`,不是 `session=`**(`app/mcp/client.py`)。传 session 的话那个 session 一关,造出来的工具就废了;传 connection 则每次调用自建连接。

**`handle_tool_errors=False` 必须显式关**(同上)。默认 `True` 会把 MCP 的调用故障**包成一条正常的工具返回** —— 于是在执行器眼里「物流服务连不上」是**成功**,直接违反本仓那条「基础设施故障绝不伪装成查不到」。关掉后 adapters 抛 `ToolException`,执行器分诊成 **502**(`ToolInfrastructureError("工具执行失败")`)。

> ⚠️ **下面这句一度写错、由 T12 实测订正**:关掉之后**不会**走 `TransientToolError` 那条可重试分支。实测(单测内注入 `ToolException`,`tool_retry_attempts=2`):工具**只被调用了一次**、直接 `ToolInfrastructureError` ⇒ **MCP 传输故障今天不重试**。原因是 `app/tools/errors.py` 的 `TransientToolError` **全仓没有任何生产抛出点**(只有 `errors.py` 的定义、`executor.py` 的 `except`、和 `tests/test_executor_gate.py` 的两处注入)⇒ 那条分支在生产上是**死代码**,而 spec §6.3 设计的「MCP 传输类故障可重试」**没有落地**。详见 spec §15 ㊱。

**写操作决议是三态,不是布尔**(`pending` / `approved` / `denied`,`executor.py`)。用 `approved=False` 一个值表达「没问过」与「问过、用户说不」的话,**取消路径会再拿到一次 `confirmation_required`**,于是取消永远不会被记成「权限拒绝」(验收 5 落空)。同理:认不出的决议值**响亮地抛**、且**不审计** —— 不许用 `!= APPROVED` 当拒绝处理,那会在 `tool_audit_logs` 里写一条**谎报用户点了取消**的行,而那张表正是验收 5 读的表。

**`retry_count` 记的是真实发生过的重试次数(`attempts_made − 1`),不是配置值**(`executor.py`)。写成配置值的话,一个**第一次就成功**的查询会被审计成「重试了 2 次」,而**没有任何断言会红**(验收 6 那条在全超时的一轮里两者都是 2,区分不了;真正守住它的是 `tests/test_executor_gate.py`)。

**`turn_messages` 是覆写通道,承载本轮产生的全部消息**。续跑(`apply_write_decision`)与决议节点都必须「**读旧值再追加**」;只返回新那一条会**丢掉带 `tool_calls` 的 AIMessage**,而那一轮**看起来一切正常**(`log_turn` 只拿它落库 —— 少一条就少落一条)。它与 `pending_write` / `write_decision` 一起进 `resolve_references` 的逐轮重置。

**`agent` 不再是 `_OUTLETS` 的成员**(`app/agent/graph.py`)—— 它现在是**条件出口**(`route_after_agent`:有 `pending_write` 就 `confirm_write`,否则 `log_turn`)。把它**无条件**接回 `log_turn` 会让**挂起的那一轮一半写库、一半没写**(单测看不出来:要显式 resume 才走得到那儿)。同理 `refund_pick_order` 也不在 `_OUTLETS` 里。

**`pending_write` / `write_decision` 必须连同它们的每轮清零一起落地**(通道在 `app/agent/state.py`,清零在 `nodes.make_resolve_references_node`)。checkpointer 是**进程级单例**、`thread_id = session_id`,未写的通道**保留上一轮的值** —— 漏了清零的后果是**上一轮批准过的写操作,这一轮自动放行**。(本仓「通道与它的清零必须同处一地」的第四次应用。)

**`FastMCP.call_tool()` 返回的是 2-tuple `(list[ContentBlock], dict)`,而它的返回注解写的是 `Sequence[ContentBlock] | dict[str, Any]`** —— **注解与实测不符,是个陷阱**。照注解写 `result[0].text` 会得到 `TypeError: Object of type TextContent is not JSON serializable`。正确写法是 `result[0][0].text`(T6 的实现者实测到并订正)。

**`isError: true` 在 mcp 1.30.0 里是通用的,不能当「业务性未找到」的同义词**(`mcp/server/lowlevel/server.py` 的 `_make_error_result`):**工具名不存在**、**入参校验失败**、**出参 schema 不匹配**、**返回类型不认识**全都汇进它,**形状一模一样**(单条 `TextContent` + `isError`)。要分辨只能靠**文案**(或先查 `spec is None`);判错的代价是**把「这一单查不到」变成 502** —— 正是本仓那条「不许拿服务端故障指责用户输入」的反面。

**「内置 vs MCP」的差别不在热重载能力,而在工具从哪来**(`app/tools/builtin/__init__.py`):**新增**一个内置模块(**新文件名**)**当场生效、不用重启客服服务**(`discover()` 每请求重跑 `pkgutil.iter_modules`,FileFinder 的目录缓存按 mtime 失效);**修改**一个**已有**模块**仍必须重启**(`importlib.import_module` 直接返回 `sys.modules` 的缓存项)。两条都实测过。MCP 那两个方向都只要重启**它自己**。

**审计写口一落地,整套测试每轮都往真库的 `tool_audit_logs` 写记录(只追加)**(2026-09-23 实测:一轮全量 `1569 → 1624`,即 **+55 行**;T5 当时的读数是 ~150 行 —— **两个数都是当时的读数,随用例数增长,不是恒定的**)。⇒ **凡是对那张表的「查最近这几条」式断言,必须按 `conversation_id` 过滤**(`scripts/acceptance_ch08.sh` 的 `dbq.py` 每个 mode 都过滤),否则会变成**偶尔红、偶尔绿**,而那正是本仓编目过的「被上次运行的数据污染」那一类。

**本机对一个已关闭的回环端口调裸 `socket.connect()` 要 ~2.05s 才拿到拒绝**(2026-09-23 复测:端口 1 / 9 / 65500 / 54321 / 8101 / 8102 分别是 2.036 / 2.055 / 2.050 / 2.055 / 2.039 / 2.055 秒),而**在监听**的端口是毫秒级(19530 0.4ms、3307 22ms)。⇒ 「两个 MCP Server 都没起时每请求白等 ≈4.8s」**是**本机**的性质,不是这条链路的性质**(两个 Server ≈ 2×2.05 + adapters 的 0.35×2 ≈ 4.83s,与实测吻合)。**引用任何具体秒数都必须带「本机实测」四个字**;可移植的上界只有一个:`mcp_discovery_timeout_seconds × 2 = 10s` —— 而**那个上界本身未实测**(它对应「只吞 SYN 不回」那条路径,实测走的是「立刻拒绝」那条,且一轮发现里未必只有一次请求,所以它未必紧)。

**ch09 · 观测(Langfuse)与飞轮的命门**(细节见 ch09 spec §15 与 `dev-notes/ch09.md`):

**`LangfuseSpan.update(**kwargs)` 的 kwargs 被静默丢弃** —— 源码 docstring **逐字**写着
`**kwargs: Additional keyword arguments (ignored)`。想在中途改 trace 属性只有一个口子:
`propagate_attributes`,而它返回的 `_AgnosticContextManager` **没有 `__aenter__`** ⇒
中途进入只能用**同步** `__enter__`(那是 `observability.TagScope` 存在的理由)。
**这是本仓「静默无效」家族的第四个成员**(前三个:ch06 未声明通道写入被丢弃、ch07
`add_messages` 给无 id 消息赋 uuid4、ch09 §2.2 的 `response_format` 走 beta 路径)——
共同的形状是「一个看起来会生效的赋值,什么都没做,而且不报错」。

**只调 `propagate_attributes` 而不开「当前 span」⇒ 每个观测各自成一条 trace**。
`start_as_current_observation` 的父级取自 **OTel 当前 span**,而 Langfuse 的 LangChain
回调**不把观测挂成 current**(它靠 LangChain 的 run tree 定父子)。⇒ `trace_scope` 里的
**根观测(`chat`)是必需品,不是装饰**(T3 真机实测:修复前同一次请求 **2 个 `traceId`**、
手工 span 的 `parentObservationId` 全是 `null`;修复后每个请求**只有 1 个**)。

**`response_format` 与 `bind_tools` 在本网关上互斥**(langchain-openai 1.6.2 走 beta 路径,
而那条路只接受 strict 工具)⇒ 知识轮的作答协议是**纯提示词驱动**、**没有硬保证**,
解码器**fail-open**:模型不守协议 ⇒ 逐字节退回 ch08 的纯文本行为(`json_stream` 的 `plain` 态)。
违约率**未被量成一个率** —— 真机 6 个样本里 **0 次**违约(T10),那是「这 6 轮守了协议」,
不是「违约率是 0」。**别把 0/6 引成 0%。**

**Metrics v2 的三条实测**:① `query` 是**一个 JSON 字符串参数**
(`params={"query": json.dumps({...})}`),**不是把字段平铺进 query string**;
② 按 `tags` 过滤要 `arrayOptions`(该组合法算子是 `any of` / `none of` / `all of`,
**`contains` 是 string 那一组的、用了直接 400**);③ **`sum_totalCost` 恒为 0**
(本项目模型没在 Langfuse 里配价格)⇒ `scripts/intent_cost.py` **只报 token,不报钱**
(且 `totalCost` 必须**显式请求**,不请求时那个键**根本不在响应里** ⇒ 那句「恒为 0」
会退化成空值兜底,成了断言而不是读数)。⚠️ 同族的静默坑:`dimensions` 清空时行里
**没有 `tags` 键** ⇒ `r.get("tags")` 恒判 False ⇒ 脚本**安静地打印「没有找到任何 intent
观测」**并退出码 0 —— 它比 400 危险得多(400 逼你查,它只给你一个自信的错答案)。

**`app/llm.py` 必须显式传 `timeout`(本章最贵的一条)**。不传的话 langchain-openai(1.6.2)
把 `request_timeout=None` **原样**交给 openai SDK,而 SDK 对「显式给的 `None`」的处理是
**不设超时** —— 不是它自己的 `Timeout(connect=5, read=600)`;实测
`model.root_async_client._client.timeout` 是 `Timeout(timeout=None)`,**四相全 None**,
连 DNS 与握手都没上界。**后果不是「慢」,是永不返回**:对端一个字节都不回时那次 `await`
谁也等不回来,而**挂起不是异常** ⇒ `finally` 永不执行、**飞轮的单槽被永久占死**,
唯一解法是重启客服服务(T16 走查实测:三个任务分别盯到 **666 / 245 / 382 秒**仍是
`running`,此后每次手动触发都是 409)。两道防线都是设置项:`llm_timeout_seconds`
(每次往返,默认 60)与 `flywheel_job_timeout_seconds`(整条任务,默认 300)。
⚠️ **标量会连带把 SDK 的 `connect=5` 换成 60**(连接阶段反而放松 12 倍;修前是 ∞,
所以不是回归);要保住它就用**四元组**(**二元组会让 write/pool 落回 `None` = 又没上界了**)。

**请求路径上还有两处「不走 `execute_tool`」的检索/向量化,它们的上界是
`retrieval_timeout_seconds`(默认 10,最终修复轮加的)** —— `POST /api/feedback` 的
尽力回捞与 `POST /api/review/{id}/approve` 的同步向量化。`tool_timeout_seconds`
**够不着它们**(它们不是工具调用),不加界时 Milvus 接了 TCP 不回话会让这两条请求
**永远不返回**,而它们占的是**整个事件循环**(不是只占那一个用户)。⚠️ **这道界只圈得住
`await` 的那一半**:两处内部大头是同步调用(torch 前向、pymilvus 往返),循环在它们
里面跑不到定时器(ch07 实测)⇒「Milvus 接了 TCP 不回话」**今天仍会拖住循环**,
被圈住的是收尾那次 `commit` / MySQL 回查;**别把它读成「这两条路已经不会卡了」**。

**那个单槽只关了一半 —— 不许写成「已修复」**。`JobStore`(**ch04** 的)只有**一个**
`running` 槽,`vectorize` / `mine`(ch04 的 `app/kb/orchestrate.py`)**至今没有任何死线**,
ch04 管理台的 `pollJob` 也没有轮询上界 ⇒ 「任务卡住 ⇒ 槽位永久占死」这条路**在那两个任务上
依然敞着**。本章只给**三个任务里的一个**(`flywheel`)加了整条任务的墙钟上界。

**`evidence_confidence_threshold` 是标定值,但标定只定出了「平台段」。** 读
`scripts/calibrate_evidence.py` 的表:`(0, 0.2894]` 内**任何**阈值在那 300 条上读数**逐位相同**
(300 条里没有一条置信度落在这个开区间内)⇒ 现取值 `0.2` 是**平台内的一次判断,不是最优解**;
**改它之前先重跑那个脚本**。两条更值钱的读数:(a) 误杀率 **0.175 与阈值无关** —— 被误杀的
42 条正常问题**检索返回空**,闸的 `bool(evidence)` 在**任何**阈值下都拦它们(连 0 也拦),
集中在口语桶(`C_colloquial` **21/60 = 35%**,`B_model` 只有 3/60);(b) 拦截率 0.967
**几乎全部由「检索为空」挣来**,公式的判别带**零覆盖** ⇒ **本章验证的是检索器,不是那个三信号公式。**

**`evidence_min_score` 必须同时用在 `top1` / `count` / `gap` 三个信号上**(spec §15.8)。
只用在「有效证据数」上是一条**看起来更严、实际更松**的写法:一条低于下限的块**不进条数、
却占住 `top2` 的位子**,把分差压小 ⇒ 换个路影响置信度。全部块低于下限时,
与「空证据」走**同一个出口**、五个键全 0。

**闸不是「召回为空才拦」(`iff` 是错的)。** 两个旋钮各管一段:单条 `score = 0.16` 的块
**过得了** `evidence_min_score`(0.15)、**过不了**合成分(`0.6×0.16 + 0.2×(1/3) ≈ 0.163 < 0.2`)⇒
**「手里有块、却被拦」真实可达**。那一刻丢掉的是这块的原文与得分,审核页上
「知识库真缺这块」与「有、但没检到」长得**一模一样** —— 分开它俩正是快照的用途。

**`evidence_snapshot` 有三个取值,不是两个**(最终修复轮收紧了本条的措辞):
**`None` = 当轮确实零召回**;空列表是另一个、会读错的值(三处写着这件事:
`app/kb/assess.py`、`app/agent/nodes.py`、`tests/test_agent_gate_ch09.py`);
而 👎 那条路上**回捞失败**落的是**哨兵** `{"error": "recall_failed"}`
(`app/api/feedback.py:RECALL_FAILED_SNAPSHOT`,顶层类型与真快照不同:对象 vs 数组)。
⚠️ **哨兵是这一列唯一一处「故障」标记,别把它读成第三个业务值**:它存在的全部理由是
「回捞失败」与「零召回」原先**逐字节相同** ⇒ 审核页会把**服务不可用**读成
**「知识库缺这块」这个诊断**(事故现场:Milvus 挂着,`POST /api/feedback` 照回
`200 {"pooled": true}`)。审核页对它单画一句「召回失败(检索服务不可用)」;
`scripts/acceptance_ch09.sh` 的 `reviewcheck.py` 把它印成 `SNAP=failed`(不能印成 `1` ——
对象也有长度,而那会被验收 ② 读成「快照居然有内容」)。
⚠️ 而且:**`NULL` 不等于「知识库没有这条」** ——
一个编程错误(漏传 kwarg)会产生**逐字节相同**的行。要分开只能靠当轮检索的独立读数。

**`matched_review_id` 一列担两个语义**(spec §7.1):既记「这条池子行归并到了 `review_queue`
的哪一行」,**又**是流水线的**待处理标记**(`WHERE matched_review_id IS NULL`)。
⇒ 它是**可空且有含义**的(NULL = 尚未处理);谁把它写成 `NOT NULL`,谁就抹掉了「待处理」
这个状态,**而且不会有任何东西报错**。流水线的幂等**只靠它**,不需要第二个状态列。
池子行是**共享且只追加**的:飞轮按 `ORDER BY id LIMIT batch_size` 吃**最旧的**未处理行 ⇒
「我这次跑只处理了自己那几行」**不成立**(T16 走查实测:那 30 条 ch09 之前的老行
**全部**被消费掉、`matched_review_id` 全填上了)。

**`db/ch09.sql` 是给「已有 `low_confidence_questions`、但缺那两列」的库升级用的**:
**升级老库:先跑它、再跑 `init_db.py`**;**全新库:只跑 `init_db.py`、不要跑它**
(空库上先跑它在 ALTER 上报 **1146**;先 `init_db.py` 再跑它 **1060 + 1050** 三条全红)。
它**刻意不幂等** —— 与 db/ch03/04/06/07/08 同规矩。核对:
`SHOW CREATE TABLE low_confidence_questions\G` 里要出现 `evidence_snapshot`。

**`mount("/")` 必须在 `include_router` 之后**(`app/main.py`),否则静态目录会抢走 `/api/*`。

## 写测试的规矩(本项目血的教训)

**本项目的头号风险是「假绿测试」,不是错误代码。**

ch01 抓到 4 类;ch02 又抓到 **7 条「在它本该禁止的实现下依然通过」的断言**,其中 4 条出自主控写的计划文本、3 条是实现者**先测了一下**才发现的。**两次全章复盘,没有一处缺陷是实现者引入的。**

写断言前先反问:**如果实现改错了,这条断言的输出会不会不同?** 具体到本仓库:

- **`Settings(...)` 构造必须传 `_env_file=None`**(`tests/test_*` 里统一如此)。仓库根有真实 `.env`,`pydantic-settings` 会自动读它 —— 不传的话,「缺字段应报错」的测试会因为 `.env` 把值补上而**静默通过**。
- **db 测试读真实 `.env`,不加 `_env_file=None`**;它们用 `@pytest.mark.db` 标记。
- **别为自由文本写字符串断言**。`deepseek-flash` 在 `temperature=0` 下依然非确定。
- **计数类断言必须实测能区分**:`@tool` 的参数校验发生在**函数体之外**,所以「这个工具被调用了几次」的计数器**必须放在 `ainvoke` 边界**,写在函数体里在参数非法时恒为 0、区分不出任何实现。
- **变量名不等于语义 —— 断言一个字段之前,先去读它是怎么被赋值的(ch05 最刺眼的一处)**。ch05 的验收 5 写了 `[ "$STEPS" -ge 2 ]` 来断「ReAct 不止一步」,`STEPS` 取自 done 帧的 **`agent_steps`**。那个名字读作「步数」,但 `app/agent/nodes.py` 里它是**绑工具轮次的序号且把收敛那一轮也算进去**,**任何一次工具调用都得到 2** —— 于是这条断言在 `tool_calls >= 2` 之外**零判别力**,而且**漏得掉真回归**:把循环改成单轮、让模型在一轮里并发发两个 `tool_call`,照样绿。更刺眼的是:**计划里那段注释自己就写了「收敛那一轮也被算了一步」** —— 事实写下来了,结论却没用到它。改断 trace 里的 `agent:step2`(那个字符串**只在第 2 轮真的发了工具调用时才追加**)。
- **「在**处理之后**注入」是本项目最高频的假绿形态 —— ch06 一章之内中了三次**。三次的形状完全一样:**测试把「已经被处理好的值」喂给被测对象,于是「处理」那一步永远不被验**。
  - T3:API 端点测 502 时,patch 的函数**直接抛 `ToolInfrastructureError`** —— 跳过了「把 `SQLAlchemyError` 翻译成它」那一步,而**那一步正是缺陷所在**(翻译根本不存在,真故障返回 500)。
  - T4:单测断言 `log_turn` 的帧里有 `confidence`,但它是**直接塞进 state 字典**的 —— LangGraph 全程没参与,而 `ChatState` **根本没有这个通道**,真机上写入被静默丢弃、恒为 `None`。
  - T6:检索的「基础设施故障必须上抛」用例注入的也是**已翻译好**的错误;而 `_load_rows` 的裸 `SQLAlchemyError` 会被 `except Exception` 吞成「这条查询失败」。
  **判据**:写这类测试前先问「我注入的这个值,**在被测对象内部还会被处理一次吗**?」——会,就注入**处理之前**的形态。对照写法见 `tests/test_api_ticket.py:151-212`(注入裸 `OperationalError`,走真分类路径)。
- **「语言/库 X 在情况 Z 下表现 Y」这类断言,要么带可复现证据,要么显式标注「未验证」**。ch05 一个 brief 里写过 `confidence_gate:pass`,实际是 `fail`;另一处断言「修好阈值验收 5 就会稳」,实测那个题面在阈值 0 时 top-1 只有 0.089(知识库**根本没覆盖**)。**两条都是先写结论、后没跑**。裁定「这不归本章管」时同理:**不能只看文本授权,还要算这条缺陷会不会卡住本章自己的验收**。ch07 又中一次:验收脚本的注释里写「冷进程里第一次 `query_faq` 必然超时 502」,而实测是**一次 24 秒的静默停顿**(同步调用阻塞事件循环 ⇒ `asyncio.wait_for` 的定时器**根本没机会跑**;同机实测:套 `to_thread` 会超时、直接阻塞不会)。**结论错了,现象也就描述错了。**
- **(e) 测试输入小到触发不了被测行为**(ch07 形态 ⑤)。`test_render_turns_includes_tool_rows` 里的工具内容只有 **29 字**,短于 `layer2_tool_chars=60` ⇒ **截短根本不触发**,「原文版」与「截短版」在那个输入上**完全一样**,三条断言对一个错误实现**同样成立** —— 而它守护的正是「`render_turns` 必须喂原文」这条本章最容易错的性质。同类:端点 `preview` 那条用例用了 5 个字的短消息 ⇒ **截不截到 30 字完全不可观测**。**判据:构造输入时先算一遍「这个输入真的会走到那条分支吗」**(阈值、长度、条数都要够)。
- **(f) 断言一个变量,而它的名字与语义不符** —— 在 ch05 已记过(`agent_steps` 读作「步数」,实际是绑工具轮次的序号),**ch07 又长出同一个形状**:`tokens.layer1` 在 **`model_ctx`** 里是「层 1 的 token 数」,在 **`history_ctx`** 里却是「**整个扁平窗口**的 token 数」—— **同一个键名,两个意思**,而两行日志长得一样。**判据不变:断言一个字段之前,先去读它是怎么被赋值的。**
- **(g) 替身替被测对象完成了语义**(ch07 形态 ⑦)。`FakeSession` 自己 `sorted(...)`,于是**端点删掉 `order_by` 照样绿**;同根两处:user 过滤退到 Python、`summarized: true` 在真实库里**从无实例**。**判据:被测对象是否把这件事委托给了替身?** 是 ⇒ 这条断言测的是替身。**修法**:凡「SQL 传没传对」这类语义,补一组 `@pytest.mark.db` 在真实库上验(见 `tests/test_api_conversations_db.py`),**并把该进 SQL 的过滤挪回 SQL,扩替身而不给实现加兜底**。

**两条元教训(ch07 全章复盘)**:

- **验证装置自己会产假绿 —— 而且它有两种长相,必须先分清**。ch07 的变异脚本一共出了**五次**事故:锚点打在**同一文件的另一处**(`replace(...,1)` 命中第二个 `.order_by`)、**node id 过期**(pytest **exit 4**,而 `tail` 把错误切掉后**连 `N passed` 都没有**)、**改注释让锚点失配**(而改的正是被锚着的那段注释)、**输出管道把「锚点没打上」的行过滤掉**(差一点被报成「全绿」)、正则吃掉闭括号让 JS 语法崩而被当成 RED。⇒ **「变异后没红」有两个互斥的解释:断言无判别力,或变异压根没生效。** 判据与动作:① 锚点**锚在代码上,不锚在注释上**(注释正是 diff 最容易改到的东西);② 每次变异后**断言命中数恰好为 1**;③ 看不到 `N passed|failed` 就打印 `!!!`(那是「不要再加 `-q`」那条的自动化版本);④ **绝不把证据输出接进任何截断/过滤管道**。
- **不变量要放在唯一写口上,不要靠每个调用方自觉**。ch07 的前端竞态修法是把「代际令牌递增」收进 `setSessionId`(当前会话的唯一写口),一次关掉三种顺序;后端同款:消息 id 由 `to_lc_messages` **一处**给、锚点由 `layers` **一处**产出、层 2 的计数口径由 `_assemble` **一处**算、`MAX(seq)+1` 只该有**一个**实现。**每多一个「调用点自己记得做」,就多一种静默漂移。**
- **读回数据库的值要用新 session**:SQLAlchemy 身份映射持**弱引用**,同 session 重读是否打到库取决于还有没有东西引用着那个 ORM 对象 —— 会变成「靠 refcount 走运」的断言。
- **复述类断言要对着真实来源验**:让替身**真的把密钥写进异常文本**,否则「响应里没有密钥」是恒真的。
- 单测**全程不联网**;评估集与验收脚本才允许打真实网络。

## 数据与产物

- `evals/tool_selection_cases.jsonl` —— 15 条工具选用例,`expected` 是工具名(闭式精确匹配)或 `null`(不该调工具)。
- **工具选择准确率 13/15 = 86.7% 可引用**(闭式枚举,与 ch01 那个被样本拟合的 `expected_solution` 关键词口径不同)。但引用时须一并说明:**该数字是在没有生产 system prompt 的条件下测得的**,且**用例集偏弱**(非 null 的 13 条里 11 条从不失误,信息量主要来自 2 条诱饵)。
  - **⚠️ 追加限定(2026-09-22,ch08 —— 比上一条更弱,因为它有两层)**:**这个 13/15 是在旧配置下测得的,而且测它的脚本在 ch08 改过之后从未被执行过。**
    - ① **工具定义的顺序变了。** ch08 起由手写的 `[query_order, query_product, query_logistics, query_faq, create_ticket]` 变成 **`(模块名, 工具名)` 排序**(`app/tools/builtin/` 的自动发现,spec §3.4)⇒ 发给模型的工具定义块**逐字节不同**。
    - ② **`query_logistics` 从内置下线、改由物流 MCP Server 提供**(spec §8.3)。而 `evals/run_tool_selection_eval.py` 原先只用 `build_tools`(**内置那一半**)⇒ 那 **3 条物流用例在结构上不可能通过**。「口径变了」这句话**描述不了「有 3 条根本跑不了」**。
    - T7 因此把该脚本改成与 `app/api/chat.py` **同款的两步**(`await discover_mcp_specs` → `build_registry(extra=…)`,含 MCP 发现,发现失败时同样降级),否则它测的是一个**与生产不再对应**的工具集。**改的是被测量的配置,用例集一字未动。**
    - **⚠️ 而那次改动之后脚本没有被重跑**(要真实 key + MySQL + 两个 MCP Server),所以现在的状态是:**一个描述旧配置、且其测量脚本已改而从未执行的数**。引用它时必须把这两层都说出来,或者重跑一次。
- `evals/extract_cases.jsonl`(ch01)的 `expected_solution` 分数**不可引用** —— 关键词是看到输出措辞后才放宽的。
- `evals/retrieval_cases.jsonl`(ch03)—— 23 条(19 换说法正例 + 4 干扰项),闭式口径(期望片段取自语料**逐字原文**且须在**同一块**里全部出现,不掺主观判断)。**⚠️「23/23」那一版是 ch03 的 dense 单路;ch04 换混合+重排后从未复核过,现链路是 13/23** —— 引用时必须说清是哪条链路。用例自造、4 条干扰项里 3 条离阈值很远、不构成压力;**「卖手机」是唯一有信息量的近域硬负例**。
  - **⚠️ 追加限定(2026-09-22,ch07 终审):这个 13/23 是 `rerank_top_k = 3`(旧的 `retrieval_top_k`)下测的**。ch07 把该旋钮的默认值改成 **5**(spec §9.1 授权),也就是说**现在放行的块更多**、这条链路的口径与那次测量**不是同一个配置**,而**从未在新默认值下复核过**。引用它时必须一并说明这一点 —— 与上面「dense 单路 vs 混合+重排」是同一类限定:**一个看起来已经验过、其实描述的是另一套配置的数**。
- 阈值 `retrieval_score_threshold` = **0.25**(2026-09-20 重定,原 0.58)。**0.58 是在 dense 余弦分数上标定的**(正例最低 0.609 / 干扰最高 0.560,区间仅 0.049 宽),ch04 换混合+重排时**原值沿用**,而重排器输出的是 **sigmoid** 分数 —— 两把尺子不可通约,0.58 比可用区间上界还高。现链路实测可用区间 **`(0.114, 0.358]`**(宽 0.244),取中点。**⚠️ 这两个数同样是 `rerank_top_k = 3` 下测的**(见上一条的追加限定)。上界一度被写成 `0.389`:那是按 **top-1 代理量**算的,偏乐观 —— 实测有一条正例靠**第 3 名**的 0.358 命中,而它的 top-1 是 0.766(`_rerank` 返回 Top-K,阈值判的是「含齐期望片段那一块」的分)。订正过程见 `dev-notes/ch05.md` 阶段 21 与 ch03/ch04/ch05 spec 的后记。`dedupe_threshold` = 0.95 **仍是未实测值** —— 真实数据上从未被触发过,不要当成已验证的。
- **`retrieval_score_threshold` 改一次要动两处,它们是一致的**:`app/tools/registry.py` 传给 retriever(过滤块)与 `app/agent/nodes.py` 的置信度闸(取 max 比阈值)。因为 `retrieve_knowledge` 拿到的块**已经**过同一阈值,闸的 `max(scores) >= threshold` 在有 evidence 时几乎恒真 —— **闸的实际效果约等于「检索是否返回非空」**。
- `evals/summary_cases.jsonl`(ch07)—— 11 条**摘要**标注样例,四类:正例 3 / 负例 2 / **幻觉探针** 2 / 四样提炼物(product、identifier、request、unresolved)各 1。口径**闭式**(关键词、字数上限、`\d{4,32}` 正则)。**实测 9/11**:两条负例(纯寒暄)判 MISS —— 模型**不返回空串**,而是吐约 40 字的**元叙述**(「本次对话未涉及任何商品…」)。**它没有编事实,但那句正是 prompt 点名的「对话状态一律不留」** ⇒ 这条既是「prompt 遵从度不满」的读数,也说明 T8 的「空输出退路」在真实模型上**很难触发**。**引用时必须带上这句**,别把 9/11 读成「实现坏了」。
  - **幻觉探针的判别力靠一个前提**:该用例的对话里**本来就没有** `\d{4,32}` 形态的数字。脚本对每条探针**自动核对这个前提**(用例自检),不成立就单独报 `!!!` 而不混进 MISS。另有**探针自检**:`\d{4,32}` 必须能匹配 `20240915`/`13800138000`(真会出现的形态)、**不能**匹配 `99` —— 后者正是 ch06 T1 那条**同义反复断言**(用 `\d{4,32}` 匹配「99」,长度对不上 ⇒ 恒真)的反面教材。
- `evals/results/` 被 gitignore,是历史运行产物。
- `evals/flywheel_cases.jsonl`(ch09)—— **两半合一个文件**,靠 `kind` 字段区分:
  `normalize` 那半 **10 条**、`dedupe` 那半 **9 对**(`kind` 缺省按 `normalize` 处理,
  所以旧 10 条一字未动)。**读数(normalize)**:最终判据(C 口径)**13 轮里 3 轮各挂 1 条**
  (约每 4 轮 1 次、挂的不是同一条),逐条 **127/130**;D 口径 5 轮全对。
  **读数(dedupe)**:**9/9**,给两个分母 —— 原始 **9** 与**带信息 6**(近域硬负例 2/2)。
  ⚠️ **这个分数不是稳定量,而且是四次判据订正之后的读数**;单次「10/10」**不许当门禁**、
  也不代表跑一遍永远 10/10(细节见「已知问题与未达成项」里 ch09 那一段与 T13 报告 §2.0/§2.3)。
  跑它:`.venv/Scripts/python.exe scripts/run_flywheel_eval.py`(打网络)。
- `eval_runs`(ch09 新表)+ `scripts/eval_trend.py`:一行一轮、按 `(created_at, id)` 连成趋势;
  **条数或 `top_k` 与上一轮不同的两轮标「不可比」并整行不打箭头**(分母不同,差值不是「变化」)。
  ⚠️ **真实两轮读数逐位相同 ⇒ 真实数据上不会有 `↓`/`↑`**,那是**确定性链路的性质**,
  **不是「模型稳定」**(箭头只有合成数据验过;详见 ch09 已知问题那一段)。
- 置信度阈值的**读数**放在上面的硬约束里(平台段 `(0, 0.2894]` + 拦截率 0.967 / 误杀率 0.175),
  改它之前先跑 `scripts/calibrate_evidence.py`。

## 已知问题与未达成项(如实记账,不许读成「全绿」)

**ch08 · 未达成项:「缺必填项就主动追问」没有实现。** 用户对本章的原始要求里有一条「建工单缺必填项时**主动追问补齐、不许瞎编**」。实测下来**这一条做不到**,而「撞到写调用就卡住 → 弹卡片确认 → 落库 → 带工单号回来」那三段**都是通的**。

- 验收脚本里那条题面**用单轮提示直奔卡片**,「Agent 先追问补齐」那半**从未跑过**。T11 的实现者随后去探**九种说法**,结论是**结构性**的:没有业务落点的建单请求全部走 **其他→兜底** 或 **投诉→固定话术出口**(那个出口的按钮走 `POST /api/ticket`,**不是**确认流);而**能**走到 Agent 的说法都自带业务落点,模型会**从 `query_order` / `query_logistics` 的返回里合成一个描述**直接调 `create_ticket` —— **它不追问**。
- 换句话说:**「不许瞎编」这一半今天是靠「模型没瞎编」侥幸成立的,不是被守住的性质**。
- 这条需要用户拍板(接受现状 / 在提示词或图里补一个追问节点)。**引用本章时说「确认流已交付」是可以的,说「缺参追问已实现」是错的。**

**ch08 照出来的 pre-existing 缺陷:写路径超时会往用户可见的 `error` 帧里吐裸的 SQLAlchemy 内部文本。**

- **现象**:验收 6 的写路径(`TOOL_TIMEOUT_SECONDS=0.001` + `create_ticket`)那次续跑会推一条 `error` 帧,文案是 `This Session's transaction has been rolled back due to a previous exception during flush. To begin a new transaction with your Session, first issue Session.rollback(). Original exception was: …`。
- **两个成因,缺一不成**:① **ch01 起的那个通道** —— `app/api/chat.py` 的 `except Exception as exc:` 直接 `redact_api_key(str(exc))` 推 `error` 帧,它只抹**密钥**,不抹**内部实现细节**;② `create_ticket` **没有 ch03 那种取消路径的 `rollback()`** —— 超时把协程取消在 SQL 中间,session 停在**待回滚**状态,后续任何一次用它都抛 `PendingRollbackError`,而 `str()` 就是上面那一整段。(抛点未逐行定位 —— 现场只留了 error 帧的文案。)
- **⚠️ 它不是本章引入的回归**:两个成因都在 ch08 之前就在,是**本章的 A6 把它照出来的**(A6 只断审计行,所以它既没判过也没判红)。方向上与本仓「所有出站错误文本必须过 sanitize」「基础设施故障一律固定文案」那两条**不一致**(泄漏的不是密钥,是内部实现细节)。
- **两条候选修法(择一或都做,未实施)**:
  1. **给 API 层兜底**:那个 `except Exception` 里不再直接 `str(exc)`,改成固定文案 + 把原文 `logger.error` 出去(与 502 那条路径同款)。改一处,覆盖面最大。
  2. **给写工具的取消路径补 `rollback()`**:照 ch03 `retrieval/search.py` 在 `except BaseException` 里先 `rollback()` 再抛的样子,给 `create_ticket`(以及任何将来直接用调用方 session 的写工具)补上。**治因**,但只治这一条路径。

**ch09 · 那个单槽只关了一半 —— 不许写成「单槽问题已修复」。** `JobStore`(ch04 的,内存注册表)
只有**一个** `running` 槽,**三个任务共用**:`vectorize` / `mine`(ch04)/ `flywheel`(ch09)。
本章只给**其中一个**(`flywheel`)加了整条任务的墙钟上界(`flywheel_job_timeout_seconds=300`);
**`vectorize` 与 `mine` 至今没有任何死线**,ch04 管理台的 `pollJob` 也**没有轮询上界**
⇒ 「任务卡住 ⇒ 槽位永久占死 ⇒ 只能重启服务」这条路在那两个任务上**依然敞着**。
(T16 走查的原始现象:三个卡住的任务分别盯到 **666 / 245 / 382 秒**仍是 `running`,
此后每次手动触发都是 409 —— 连「手动那根杠杆」也拿不到槽。)

**ch09 · 验收对「置信度闸那一列的 `evidence_snapshot`」零覆盖 —— ② 那条断言对它本该抓的
bug 是不变的。** ② 问的是**零召回**的问题 ⇒ 没有片段可快照;而零召回那一支里,
「闸**写了**这一列」与「闸**漏了** `evidence_snapshot=` 这个 kwarg」落出来的**都是 JSON `null`**
⇒ 断言在两种实现下都成立。守它的是 `tests/test_agent_gate_ch09.py`,**不在端到端覆盖内**;
脚本自己在结尾的「局限」里打这条。

⚠️ **而这一段最容易被一个错的读数骗过去 —— 它自己被骗过一次(T19 复审抓的)。**
「闸那一列有没有被填充」**必须用 JSON 感知的判据**读,`IS NOT NULL` **不算数**:
这一列的 ORM 类型是 `JSON`(默认 `none_as_null=False`)⇒ Python 的 `None` 落库是
**字面 JSON `null`**,它 **SQL 上不是 NULL**。T19 实测闸行(**46** 条)按类型分组:

| SQL NULL | 字面 JSON `null` | **`JSON_TYPE='ARRAY'`(真的带快照)** |
|---|---|---|
| **29** | **17** | **0** |

⇒ **闸那一列至今一条真快照都没有过**。那 29 条 SQL-NULL 是**加列之前就存在的行**
(ALTER 补列时给的 NULL);17 条 JSON-`null` 是「**没记快照**」在库里长出来的样子。
**T18b 把 kwarg 接上了,但那条改动对零召回的行产不出快照 ⇒ 它在生产数据上从未被行使过。**

⚠️ **这条陷阱 `app/kb/assess.py` 的 docstring 早就写着**(2026-09-23 实测),原文:
「**它落的是 JSON 的 `null`,不是 SQL 的 NULL** …… 别拿 `WHERE evidence_snapshot IS NULL` 筛
『这条没记快照』—— **一行都筛不出来**」。⇒ **T19 这次错在只读了这道警告的一半**:
警告说「`IS NULL` 筛不出东西」,我拿 `IS NOT NULL` 去数「有快照的行」,
**而那正是同一个陷阱的另一面**(它把 JSON `null` 数成了非空)。
本仓那条「引用了一条教训 ≠ 免疫于它」在这里换了个壳:**读过的那半条也照样能踩。**
**判据**:这一列一律用 `JSON_TYPE(evidence_snapshot)` 读;也不要拿 `IS NULL` / `IS NOT NULL`
任何一边当「有没有快照」。同理「池子里有行 ≠ 那条断言有判别力」,两件事别混。
**没有为了凑覆盖去构造弱召回场景**(那要改服务端旋钮,验的是场景不是产品)。

**ch09 · 验收脚本有一个「候选预算」,不是可以无限重跑的。** ② 每次运行会往知识库**真的写进一条**
并核准 ⇒ 那条问题下一轮就召得到了 ⇒ 脚本每次从 `CAND_Q*` 里挑**第一条此刻召不到**的,
**每跑一轮消耗一条**,用尽就**响亮地报**「请加一条」。
**⚠️ 消耗是「被跳过」,不是「被复用」**(这一条最容易算错,复审抓过一次):用过的候选,
其问题已被 ③ 核准进知识库 ⇒ 下一轮它**召得到**了 ⇒ 选题判据(「此刻召不到」)**永远跳过它**
⇒ **剩余数 = 总数 − 已消耗数,而编号不重置**。
实际轨迹(逐次可核,三份转录里都印着「选中候选 N」):**T18 的八次运行消耗 `#1–#7`**
(run8 收在「只剩 `#8` 一条」);**`#8` 死在 T19 的第一次跑**(夹具缺陷,那次 5/6);
T19 补了 `#9–#16`(8 条);三次跑又消耗 `#9` / `#10`
⇒ **T19 收工时可用的是 `#11–#16`,共 6 条**。
(此处曾写成「剩 13」—— 那是**拿 16 去减 3**,漏了 T18 早已吃掉的 7 条。)
**加候选有三条规矩**(都吃过教训,详见脚本头):① marker 必须真的出现在它自己的答案里,
**并且要用脚本自己的解码器核**(那是 `chr(int(h,16))` **码点**,不是 UTF-8 字节 —— 极易写混);
② **marker 里不许出现数字(阿拉伯与中文都不行)**;
③ **核准答案必须真的回答那个题面** —— 见下面那两条。
用尽后的唯一正解是**加候选**,不是把题面写死(写死之后第二轮起会**悄悄变红**)。

**ch09 · ③ 的「回复含核准答案的特征串」不是一条不变的断言(2026-09-25 实测红在数字上)。**
核准答案写的是「电源线长约**一点八米**」,模型转述成「**1.8 米**」⇒ 判红;而**产品那一侧
每一步都是对的**(写了块、向量化了、召得回来、过闸、回复带引用 `[1]`、不再是兜底话术 ——
③ 下面那条否定断言就是绿的)。**数字是这一族「回声断言」唯一不稳定的一类 token**
(模型会把中文数字写成阿拉伯数字,还带一个空格);原注释只防了反方向(`12 毫米` → `12毫米`)。
⇒ **判据**:特征串取**实词词组**,不取任何形态的数字。修复只是**补了选材规矩**(断言一个字没动),
`CAND_Q8 / M8` **原样留在文件里当证据**。

**ch09 · 评估流水线是确定性的 ⇒ 趋势表上的 `= 0.000` 不是「模型稳定」。** 两轮真实评估
(各 5 条)在四个策略、八个指标上**逐位相同**,所以真实数据上**一个 `↓` / `↑` 都不会出现**
(箭头只有喂 `trend_synthetic.py` 的造数才验过)。要读成「**同一批用例走了同一条确定性检索链路**」。

**ch09 · 飞轮自己的评估集是不稳定的 ⇒ 不许拿「10/10」当门禁。** `evals/flywheel_cases.jsonl`
在最终判据(C 口径)下:**13 轮里 3 轮各挂 1 条**(约每 4 轮 1 次),**挂的不是同一条**
(第 4 / 6 / 8 条各一次),逐条 **127/130**;D 口径 5 轮全对。⇒ **单次「10/10」不代表稳定**,
引用时必须带上「四次判据订正之后的读数 + 23 轮采样」这个限定。**查重那一半要报两个分母**:
原始 **9/9** 与**带信息 6/6**(9 对里 3 对是「明显同义/完全同一」之外的弱用例,其中近域硬负例 2/2)。
另有 T13 的一条**判据缺陷**记账:`evals/flywheel_cases.jsonl` 的用例集**偏弱**,
四次订正**全部是判据缺陷、不是模型缺陷**。

**ch09 · `NULL` 快照 ≠ 「知识库没有这条」。** 一个**编程错误**(漏传 kwarg)会产生**逐字节相同**的
行 ⇒ 要分开「知识库真缺这块」与「有、但没检到」,只能靠**当轮的独立检索读数**。同族地,
本仓约定 **`None` = 当轮确实零召回**,空列表 `[]` 是**另一个**值(三处代码写着这件事),
而 **👎 的回捞失败**落的是**哨兵** `{"error": "recall_failed"}`(最终修复轮加的**第三个**
取值 —— 它是**故障**标记,不是业务值:别把它读成「零召回」的另一种写法,也别拿
`len()` 去数它,对象**有长度**)。

**ch09 · 三条如实记账的既有/前端缺陷(都不阻塞本章验收)**:
- **`admin.html` 的 `showResult` 读 `r.body || {}` 里的 `info.op.chunks_added`** ⇒ 一个 **2xx 但
  body 不是 JSON** 的响应会渲染成「新增知识块 **undefined**」(同一处还有 `undefined 块`)。
- **`app/static/index.html` 的 `makeCitesClickable` 读 `textContent`、写回 `innerHTML`**
  —— **先于本章**(ch04 T9 起就在),不是 ch09 引入的。
- **验收脚本「输出干净」自检有一条残留**:那条**绿判词的文案自己就含**它要扫的三个串
  ⇒ 同一份转录上**再跑第二次会自咬**(进程内只调用一次,外观级;候选也只剩很少,没为它再烧一轮全量)。

**ch09 · 池子那 30 行 ch09 之前的行已被飞轮消费掉 —— 任何「池子未动」的旧基线都不再成立。**
飞轮每批吃 `WHERE matched_review_id IS NULL ORDER BY id LIMIT batch_size`(最旧的先),
而池子是共享、只追加的 ⇒ 「我这次只处理了自己那几行」**不成立**。T19 实测(**两个时点都在这里,
别混**):**T19 开工时 56 行 / `IS NOT NULL` 56 / 未处理 0**;
**第一次跑完之后 58 行 / 58 / 0**(中间那个数是**跑过一轮之后**的,不是开工时的);
**T19 收工时 63 行 / 63 / 0**。
**这些数一律是当时的读数**,随运行次数只增不减 —— **断言不要拿它们当基线**。

**ch09 · Langfuse 的写侧会成片失败几分钟,而读侧一切正常。** 服务端 OTel exporter 打
`Read timed out`(`read timeout=4.99999`)而 REST 查询端点(①⑤ 用的那条)**照常返回**。
⇒ ① 因此**重试两轮**,仍失败就把服务端日志里那行 `opentelemetry.exporter ... Read timed out`
打出来指认错因 —— 别把它读成「代码坏了」。连带:`sum_totalCost` **恒为 0**(没配模型价格),
`scripts/intent_cost.py` **只报 token、不报钱**。

**ch10-B · 本章测不出**结论性的 per-class 指标 —— 能引用的只有汇总数,而权威列上一条都没有。**
根因是**真实语料按类稀薄**,不是模型不行、也不是阈值没调好:

- `†` 的判据是 spec §8.3 的 `support < 15`,而 **17 类 × 15 = 255 个标签槽 > 全表 169 个槽**
  ⇒ **算术上装不下**「17 类各 ≥15」。
- 实测(`evals/topic/report.json`,`scripts/acceptance_ch10.sh` 验收 ① 会把这三个数断成判词):
  **全体 120 → 够 15 的 3/17**(退换货 18 / 运费 15 / 保修维修 15);
  **只看真实 80(§8.4 指定的权威列)→ 1/17**;**只看合成 40 → 0/17**。
  ⚠️ `†` 是**每一层各自**重判的 ⇒ **权威列挂的 † 比全体更多**(16/17 vs 14/17),
  别把它读成「那一列更差」(见 ch10 spec §15.3)。
- ⇒ **本章没有任何一条可以按类下的结论**;能引用的只有全体/分层那几个汇总数
  (micro-F1 0.8285 / macro-F1 0.7866 / 整条一致率 0.675 @ t=0.5,三口径见 report.md)。
- **解救办法只有一个:补真实语料**(真实池里 `尺码` 只有 4 条 ⇒ 它在权威列上的上限**永远是 4**)。
  **不是**调低阈值 15(那是拿判据迁就数据)、**不是**把测试集做大(做大只改善合并列)、
  **也不是**给头四类配额(真实 80 里要为 4 个类抽掉 ≈48 个名额,其余 13 类各剩 ~2.5 条 ——
  拿一个更差的仪器换一个被合成数据污染的列)。三条裁定与其理由见 ch10 spec §15.2 与
  `dev-notes/ch10.md` 阶段 8 补。
- ⚠️ **报告里那个 F1 是在「预标标签」上测的,不是在人工标注的黄金集上**:120 行里只有
  **12 行**带 `human_reviewed`(用户 2026-09-26 拍板跳过测试集人工复核;那 12 行是从 CP-2
  复核过的 84 条流过去的:`train 60 + val 12 + test 12`)。**`import-test` 刻意没有执行** ——
  跑了它会给 120 行**全部**打上「人核过」,而其中 **108 行没有人看过**。
  验收 ① 因此**把它断成判词**:`test_human_reviewed_rows` 一旦变成 120,那一节就红。

## 平台陷阱(Windows + Git Bash)

本机 locale 是 **cp936**,这个陷阱在 ch02 咬过**三次**,属**复发型**:

- **含中文的请求体不能走 `curl` 的 argv**。MSYS2 会按 CP936 重编码,服务端只回 `error parsing the body`。一律走 stdin heredoc。
- **子进程输出要显式钉编码**。跨进程测试给子进程加 `-X utf8`,否则管道上的 stdout 按 GBK 编码而父进程按 UTF-8 解码,报错表现为 `proc.stdout is None`。
- **脚本打印非 ASCII 要钉输出边界**,用 `sys.stdout.buffer.write(...encode("utf-8"))`,不要依赖控制台 codec(`✓`/`✗` 不在 GBK 里,`print` 会直接崩)。
- **验收断言不能直接 grep 原始 SSE 流**。回复逐 token 推送,`20240915` 会被切成三个独立帧。用 `join_tokens` 拼回后再比对。
- **不要用 `grep '[一-龥]'` 检查中文完好性**:C locale 下 bracket expression 退化成字节区间,对真实 UTF-8 和 mojibake 全部匹配,是个恒真的假断言。脚本里的 `has_cjk` 按 Python 码点判断。
- **起服务前先查端口**:8000 上残留的僵尸进程会让你 curl 到旧代码,从而得出「新代码坏了」的**假红**。ch02 的最终验证就差点栽在这上面。ch05 又遇到一次,且**一次开了两个 uvicorn** —— 见到多个就全部清掉再起,别猜哪个是新的。
- **⚠️ 从 Python(或 cmd / PowerShell)调 `bash` 拿到的是 WSL 的 bash,不是 Git Bash(ch10-B 实测,2026-09-27)**。`CreateProcess` 的搜索顺序把 **System32 排在 PATH 之前** ⇒ `subprocess.run(["bash", …])` 解析到 `C:\Windows\System32\bash.exe`(**哪怕 `shutil.which("bash")` 明明返回的是 Git 那一份** —— 两者查的不是同一张表)。后果是**一屏假红**:WSL 里 `curl http://localhost:8000` 到不了 Windows 上跑的服务(WSL2 有自己的网络命名空间)⇒ 起来像是「客服服务起不来」;`/tmp` 也是 Linux 的 /tmp,而 **Python 的 `/tmp` 是 `D:\tmp`**。**判据:`bash -c 'uname -s'` 必须是 `MINGW*`/`MSYS*`(Git Bash 的 `OSTYPE` 实测是 `cygwin`,WSL 是 `linux-gnu`)**;固定走绝对路径(`D:\kit\Git\usr\bin\bash.exe`),别让 PATH 决定。`scripts/acceptance_ch10.sh` 开头有一道 `OSTYPE` 自检把这种情况**拦在跑之前**。
- **LangGraph 的两条实测硬约束(ch06,都是「报错指向别处」的类型)**:
  - **`astream(stream_mode="custom")` 会把 `interrupt()` 整个吞掉** —— 一个帧都不吐、run 直接结束、`state.next` 停在待续节点、**不报任何错**。interrupt 只从 **`updates`** 模式浮出(`{'__interrupt__': (Interrupt(value=…),)}`)。ch06 的订单卡片差点因此「永远不出现且不报错」;端点的流模式因此是 `["custom","updates"]`。
  - **`resume` 时节点会从头重跑**(`interrupt()` 之前的代码再执行一遍)。所以**放 `interrupt()` 的节点里不能有别的事** —— 取订单那类有副作用的活必须在它**之后**的节点。ch06 有一条「取订单恰好一次」的用例,计数器放在 **tool 的 `ainvoke` 边界**上(那正是本项目栽过的边界)。
  - **未在 `ChatState` 里声明的通道,写入被静默丢弃**(只 `logger.warning`,不抛)。ch06 因此丢过一整个交付物:节点写了 `confidence`、通道不存在、**单测全绿而生产恒为 `None`**。加通道时**连带加它的每轮清零**(checkpointer 是进程级单例、thread_id=session_id,未写通道**保留上一轮的值**)。
- **引用一条陷阱 ≠ 免疫于它(ch05 实测)**。ch05 的编排者在给子代理的 dispatch 里**逐字引用过**上面那条「含中文的请求体不能走 curl argv」,几分钟后**自己**用 `curl --data-binary "{\"message\":\"退货政策是什么\"}"` 发中文,拿到 `{"detail":"There was an error parsing the body"}`,12 个样本全部没有 done 帧。**结论不是「下次记得」,而是换掉通道**:含非 ASCII 的请求一律走 **httpx 这类替你处理编码的客户端**(或 stdin heredoc),不要依赖「我记得要避开 argv」。

## 工作方式要求

用户对本项目开发有固定要求(见 memory,勿自行放宽):

1. **全程走 Superpowers 流程**,技能自动触发不跳步。
2. **非可单测产出(Prompt 模板、数据类)把 TDD 换成评估集/标注样例验证**,其余步骤照走;纯 UI 页面例外,用 Vibe Coding 直做。
3. **涉及具体库/框架/API 的用法,先用 Context7 MCP 查最新官方文档再动手**,不许凭记忆写。
4. **用户点名的技术选型是定死的** —— 走不通就停下来问,不许自行换方案。spec §9 中预先标注为「待实测」的参数,按实测结果改动属设计授权,但仍需记账并告知用户可一句话回退。
5. **过程实时留痕到 `dev-notes/chNN.md`**,每完成一个阶段就补一段,记四样:用户关键原话、关键产出、被拒绝/被纠偏了什么、翻车与返工。**明确不允许收尾时一次性补记。**
