from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置。四个字段必填,其余有默认值。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 必填:无默认值
    openai_base_url: str
    openai_api_key: str
    openai_model: str
    database_url: str

    # 可选:有默认值
    chat_temperature: float = 0.7
    extract_temperature: float = 0.0
    safety_margin_tokens: int = 512
    # 会话存储的两个数都必须为正,否则 `SessionStore` 会**静默**失去互斥:
    # max_sessions<=0 时 lock_for 刚建的锁会在同一次调用里被容量淘汰掉,
    # session_ttl_seconds<=0 时 _purge 每次都把所有未持锁条目判为过期,
    # 两者都让"同一会话两次 lock_for 拿到同一把锁"不成立。
    session_ttl_seconds: int = Field(default=1800, gt=0)
    max_sessions: int = Field(default=1000, gt=0)
    session_lock_timeout_seconds: float = 60.0
    brand_name: str = "本店"

    # 工具执行。三个数都加了界:配置写错要在启动时炸,不能等到运行时
    # 变成"重试循环一次都不跑、空错误文案交给模型"这种静默故障。
    tool_timeout_seconds: float = Field(default=10.0, gt=0)
    # ch08 把默认值从 1 提到 2(共 3 次尝试,最坏 20.3s → 30.3s)。
    # **这是跨章行为变更** —— 它是全局旋钮,ch03–ch07 的耗时一并变了。
    # 一句话回退:`.env` 里 `TOOL_RETRY_ATTEMPTS=1`。
    # ⚠️ 写操作**永不重试**是结构保证(由 kind 推出),不受这个数影响。
    tool_retry_attempts: int = Field(default=2, ge=0)
    tool_retry_delay_seconds: float = Field(default=0.3, ge=0)

    # ---- ch09 · Langfuse 观测。三个值任一为空 ⇒ 整套观测 no-op ----
    # **单测"全程不联网"这条硬约束就靠它守**:测试的 Settings 不传这三个键,
    # 于是 observability 全部走空壳,一个字节都不出网。
    # ⚠️ 默认值是 **Langfuse Cloud(美国区)**;要"链路数据不出自家服务器"
    #    就换成自部署地址 —— 只改这一个值,代码不动。
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "https://us.cloud.langfuse.com"

    # ---- ch09 · 置信度闸(spec §4)----
    # 判据 = w_top1*top1 + w_count*min(条数/max_count,1) + w_gap*clamp(top1-top2,0,1)
    # 三个权重之和为 1 ⇒ 输出天然落在 0–1。
    #
    # `evidence_confidence_threshold` = **0.2**,而它**不是**那次标定测出来的
    #    **最优点** —— 标定测出来的是**一段平台**,`0.2` 是**平台内的一次判断**。
    #    (来源与读数:`scripts/calibrate_evidence.py` 在 `evals/测试集.md`
    #    —— 300 条 / D_absent 应拒答桶 60 条 —— 上跑**真实检索链路**。)
    #    ⚠️ **只读这一段的人请把上面那句当结论**:别把 0.2 引成"最优"或"标定得
    #    0.2";要引用就引**平台**与下面的两个实测数字。
    #
    #    ⚠️ **标定定出来的是"一个平台",不是"这一个点"。** 300 条里**没有一条**的
    #    置信度落在 `(0, 0.2894)` 这个开区间内 —— 正常类非零最小 **0.2894**、
    #    D_absent 非零最小 **0.5420**,其余(42 + 58 = 100 条)全是 **0.0**。
    #    ⇒ `(0, 0.2894]` 内的**任何**阈值在这 300 条上读数**逐位相同**:
    #       阈值任意 ∈ (0, 0.2894] ⇒ 拦截率 0.967(D_absent 拦下 58/60)
    #                              误杀率 0.175(正常桶误杀 42/240)
    #    **所以"0.2"不是测出来的最优点,是平台内的一次判断** —— 别读成"标定得 0.2",
    #    也别拿它去引"最优"。取 0.2(而不是脚本按规则吐出的网格起点 0.05)是为了
    #    离 0 与离正常桶下沿(0.2894)各留一段余量;同值的另外两个旋钮
    #    (`retrieval_score_threshold` 0.25 / `evidence_min_score` 0.15)刻意避开。
    #    整张表与选值理由见 spec §15.9 / dev-notes/ch09.md 阶段 4。
    #
    #    ⚠️ **误杀率 0.175 与这个阈值无关**:那 42 条是**检索返回空**的正常问题,
    #    闸的 `bool(evidence)` 在**任何**阈值下都拦它们(连 0 也拦)。
    #    所以"17.5% 的正常问题会拿到兜底话术"是**检索覆盖率**的问题,不是这个旋钮
    #    (集中在口语桶:`C_colloquial` 21/60 = 35%)。见 spec §13 风险表。
    #    ⚠️ 反过来说,拦截率能到 0.967 也**几乎全是"检索为空"挣的**(60 条 D_absent
    #    里 58 条空证据)⇒ **公式的判别带在这份评估集上根本没被行使**:
    #    本章验证的是**检索器**,不是那个三信号公式。同见 §13 风险表。
    #    一句话回退:改回 0.42(连同 `tests/test_config_ch09.py` 那条 `==`)。
    #    按平台段的读法,回退**不影响**本章评估集上的任何读数。
    evidence_confidence_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    evidence_min_score: float = Field(default=0.15, ge=0.0, le=1.0)
    evidence_max_count: int = Field(default=3, ge=1)
    w_evidence_top1: float = Field(default=0.6, ge=0.0, le=1.0)
    w_evidence_count: float = Field(default=0.2, ge=0.0, le=1.0)
    w_evidence_gap: float = Field(default=0.2, ge=0.0, le=1.0)

    # ---- ch09 · 召回片段快照(spec §6)----
    snapshot_top_n: int = Field(default=5, ge=1)
    snapshot_answer_chars: int = Field(default=400, ge=1)

    # ---- ch09 · 飞轮(spec §8)----
    flywheel_batch_size: int = Field(default=10, ge=1)

    # ---- ch09 · 两个**墙钟上界**(T16,2026-09-24 的真实故障)----
    #
    # 这两个数不是预防性旋钮,是一次故障的产物。现场:飞轮后台任务三次卡在
    # `running` 不放(**666s / 245s / 382s**),零日志、`message` 空串、库里一条
    # 开着却空转的事务;而运行槽是**单槽**的 ⇒ 端点此后永远 409、**连手动那个
    # 「跑一轮飞轮」也拿不到槽**(它的存在理由正是「从一次坏跑恢复」),
    # 唯一的出路是**重启客服服务**。
    #
    # 根因两半,一个旋钮各管一半(详见 `app/llm.py:_build` 与
    # `app/flywheel/tasks.py` 的模块 docstring):
    #   ① 不传 `timeout` ⇒ openai SDK 收到**显式 None** ⇒ 它的处理是「不设超时」
    #      (不是它自己的 600s 默认)⇒ httpx 客户端是 `Timeout(timeout=None)`,
    #      连接/读/写/池**四相全无上界**;
    #   ② 任务本身没有寿命上界 ⇒ 挂起**不是异常**,`finally` 永不执行 ⇒
    #      「每条出口都进终态」那句保证在有界等待之外根本不成立。
    #
    # `llm_timeout_seconds`:**每一次模型往返**的上界(四相都用它)。
    #   60s 的由来:T16 走查的日志里**最慢一次调用约 4.5s**(相邻两次 200 之间的
    #   最大间隔),一批 10 行 / 20 次调用共 29s ⇒ 60s ≈ 13 倍余量。
    #   ⚠️ 一次**逻辑调用**的最坏耗时不是这个数:SDK 自己还会重试
    #   (`max_retries=2`)—— 黑洞端口实测 `timeout=1.0` 时 **4.25s** ≈
    #   (1+2)×timeout + 两次退避 ⇒ 60s 对应最坏 ≈ 3 分钟。
    #   **重试刻意保留**:同一份日志里有 **7 次「首次连接失败 → 重试成功」**,
    #   砍掉它会让那些本来能成的调用开始报错。
    #   ⚠️ 这是**跨章行为变更**:ch01–ch08 的全部模型调用(对话/抽取/摘要/评估脚本)
    #   同时有了上界 —— 刻意的,它们今天同样会被一次静默拖死。
    #   一句话回退:`LLM_TIMEOUT_SECONDS=86400`(等价于「基本不设上界」)。
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    #
    # `flywheel_job_timeout_seconds`:**整条任务**的上界(一条任务 = 一批)。
    #   300s 对应「一批 10 行 ≈ 30s」的 10 倍余量;它与 `flywheel_batch_size` 是一对,
    #   把批量调大要一起看这个数。
    #   ⚠️ 被它杀掉的任务**这一批一行都不会落地**(`run_flywheel` 整批只提交一次)。
    #   这不是缺陷 —— 池子行靠 `matched_review_id IS NULL` 幂等,下一轮会重新吃到;
    #   它保的是「槽**永远**能被下一次拿到」。
    flywheel_job_timeout_seconds: float = Field(default=300.0, gt=0)
    #
    # `retrieval_timeout_seconds`:**请求路径上两处直接调用检索/向量化组件**的墙钟上界
    #   (ch09 最终修复轮复审 D5)。那两处**不经 `execute_tool`**,所以
    #   `tool_timeout_seconds` 够不着它们 —— 收窄成「工具超时」那一个旋钮,
    #   结果是「同一个 BGE-M3 + Milvus 往返,走工具那条有上界、走端点那条没有」:
    #     - `app/api/feedback.py` 👎 那一步的**尽力回捞**(`retriever.search`);
    #     - `app/api/review.py` 审核通过之后的**同步向量化**(`vectorize_rows`)。
    #   **为什么另立一个而不是复用 `tool_timeout_seconds`**:后者是「工具执行器的
    #   **每一次尝试**」的界(执行器还会按 `kind` 乘上重试次数),两个含义混在一个
    #   旋钮上正是本仓删掉 `reranker_use_fp16` 的那个形状。10s 与它取同值,是因为
    #   两处**干的就是同一件事**(一次嵌入 + 一次 Milvus 往返),不是随手抄的。
    #
    #   ⚠️ **它只圈得住 `await` 的那一半,如实记账**:两处内部的大头都是**同步调用**
    #   (torch 前向、pymilvus 往返)—— 事件循环在它们里面根本跑不到定时器
    #   (ch07 实测的「`wait_for` 的定时器在循环被阻塞时不触发」)。被圈住的是
    #   两者的**收尾 `await`**(feedback 的 MySQL 回查 / review 的 `commit`)。
    #   「Milvus 接了 TCP 但不回话」那一半**今天仍然会拖住事件循环**;要连它一起圈住
    #   得把同步段挪进线程,而 `_load_rows` 用的是调用方的 `AsyncSession`(不可跨线程)
    #   ⇒ 不是一次修复轮能顺手改的。详见 `app/api/feedback.py:_search_bounded`。
    #   一句话回退:.env 里 `RETRIEVAL_TIMEOUT_SECONDS=86400`(等价于「基本不设上界」)。
    retrieval_timeout_seconds: float = Field(default=10.0, gt=0)

    # ch05 编排。两个数都加了界:写错要在启动时炸,不能等运行时变成
    # 「ReAct 循环一次都不跑」或「预算恒超 → 第一步就强制收敛」这种静默故障。
    max_agent_steps: int = Field(default=5, ge=1)
    agent_token_budget: int = Field(default=20000, ge=1)

    # ch06 退款子流程。Query 扩写的条数上限(spec §9)。
    #
    # **`ge=1` 是硬边界,不是防呆**:实测两种越界都不报错、只静默变坏 ——
    # `0` → `expand_queries` 返回**空列表**(检索空转,而它与「库里没有这条
    # 知识」长得一模一样);负数更糟,`[-1]` 是 Python 的负切片语义
    # 「去掉最后 N 条」,于是**静默少一条**,看起来完全正常。
    # `expand_queries` 自己在 `max_queries < 1` 时直接抛(第二道),这里在
    # **启动时**就拒(第一道)。
    query_expansion_max_queries: int = Field(default=3, ge=1)

    # ---- ch03:知识库与向量检索。全部可选带默认值,检索组件懒初始化,----
    # ---- 配置有值不等于启动就连接 Milvus / 加载 BGE-M3。           ----
    embedding_model_path: str = "models/bge-m3"
    embedding_max_length: int = Field(default=1024, gt=0)
    embedding_batch_size: int = Field(default=16, gt=0)
    # ch04 重排(懒加载,权重由用户放 models/ 下)
    # 精度不再用配置项控制:`Reranker` 按 `torch.cuda.is_available()` 自己决定
    # (有 CUDA 就 cuda:0 + fp16,否则 CPU + fp32)。原来的 `reranker_use_fp16`
    # 在 GPU 分支落地后已无人读 —— 留着会让「改了没反应」的配置项存在。
    reranker_model_path: str = "models/bge-reranker-v2-m3"
    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_collection: str = "knowledge"
    # 阈值越界一个方向等于永远全滤空、另一个方向等于没有阈值(不相关也硬凑答案)。
    # 都在启动时拒。
    # ⚠️ 0.25 是 **2026-09-20 在混合+重排链路上实测**得出的,不是原值。
    # 原值 0.58 由 ch03 在 **dense 余弦**分数上标定(正例最低 0.609/干扰最高 0.560,
    # 区间仅 0.049 宽),ch04 换混合+重排时原值沿用 —— 而重排器 `compute_score`
    # 输出的是 **sigmoid** 分数,两者分布不可通约。
    #
    # **生产证据(最有力的那条)**:自然长问句的重排分实测只有 **0.27~0.34**
    # (正确块就在里面),在 0.58 下**整段被滤空** → 证据为空 → 置信度闸 fail →
    # 用户拿到「抱歉,我没太理解您的意思」。**不报错、不写日志**,trace 里只看得到
    # `retrieve_knowledge:0 hits`(节点跑了、只是没命中)—— 这就是本次改动的直接原因。
    #
    # **实测区间**(`evals/run_retrieval_eval.py --dist`,ch03 那 23 条用例):
    # 能命中的正例,其阈值上界(含齐期望片段那块的最大分)最低 **0.358**;
    # 干扰项 top-1 最高 **0.114** → 可用区间 `(0.114, 0.358]`,取 **0.25**
    # (中点约 0.236,0.25 仍在区间内)。同一批用例:0.58 得 9/23,0.25 得 13/23;
    # 剩下 10 条与阈值无关(检索质量问题,单独记账)。
    #
    # **如实记下它的脆处**:① 0.25 之后,上面那些真实长问句只高出阈值
    # **0.02~0.09**(0.27 那条只高 0.02);② 区间的上下界都来自 **23 条自造用例**,
    # 其中只有「你们卖不卖手机」一条是有信息量的近域硬负例。阈值再往 0.114 那侧靠
    # 会给真实问句更宽的余量,但那需要补更近域的硬负例来支撑,不是现在能拍的。
    #
    # 回退:改回 0.58 即恢复 2026-09-20 之前的行为(会丢 4 条本可命中的正例)。
    retrieval_score_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    dedupe_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    chunk_max_chars: int = Field(default=800, gt=0)
    chunk_overlap_chars: int = Field(default=100, ge=0)
    # 挖知识批次 <= 0 → range() 空转,脚本"成功"但一行没抽,比报错更糟。
    mine_batch_conversations: int = Field(default=5, ge=1)

    # ---- ch08:MCP 接入 ----
    # ⚠️ 这一节由 ch08 T11 的 Step 1 逐字规定,**因 T7 的实现必须先能跑**而
    # 提前落到这里(`app/mcp/client.py:_connections` 读这三个字段;
    # 缺了的话**每一个**聊天请求都在 `discover_mcp_specs` 里 AttributeError)。
    # T11 执行到 Step 1 时这三个字段已在,不必重复加。
    #
    # 两个 URL 给本地演示的默认值(端口与 `mcp_servers/` 两张表一致)。
    # 发现超时**给界** —— 它挂在**请求路径上**(每请求现问现拿),
    # 写错会让每个请求都卡住。
    mcp_logistics_url: str = "http://127.0.0.1:8101/mcp"
    mcp_aftersales_url: str = "http://127.0.0.1:8102/mcp"
    mcp_discovery_timeout_seconds: float = Field(default=5.0, gt=0)

    # ---- ch07:上下文管理。全部带界 —— 写错要在启动时炸,不能等运行时 ----
    #
    # `model_context_window` / `max_output_tokens` **取代并已删除**了本章之前的
    # `context_budget_tokens` / `reserved_output_tokens`:旧的两个数是「直接给一个
    # 预算」,新的口径是「从模型窗口倒推」。两套并存时它们就是两个含义重叠的旋钮
    # —— 本项目已经吃过这个亏(`reranker_use_fp16` 在 GPU 分支落地后无人读,已删),
    # 所以这次是把旧的两个**删掉**,而不是留着「以防万一」。
    model_context_window: int = Field(default=18000, ge=1024)
    max_output_tokens: int = Field(default=2000, ge=1)
    max_user_input_tokens: int = Field(default=2000, ge=1)
    # 单个工具结果的上限,同时也是「单轮 ReAct 峰值」的一项。
    # `max_agent_steps`(已有)与它相乘就是峰值。
    tool_result_max_tokens: int = Field(default=1200, ge=1)
    # 取代 `retrieval_top_k`(同一把旋钮:读点是 app/tools/registry.py 与
    # evals/run_retrieval_eval.py —— 后者常被漏掉,删旋钮时要一起改)。
    rerank_top_k: int = Field(default=5, ge=1)
    # 历史预算 = min(keep_rounds × per_round_steady, 窗口匀得出来的)。
    keep_rounds: int = Field(default=20, ge=1)
    #
    # ⚠️ `per_round_steady` 是**估算,不是实测** —— 与 `dedupe_threshold=0.95`
    # 同族:写下来但还没验证过,**不要当成已验证的**。
    # 它只要 < 演示配置下的 窗口/keep_rounds,「想留住的轮数」那一支就会胜出,
    # 历史预算会远小于窗口能匀出来的量,层 1 会**每轮都降级**。
    # 校准方法:跑一轮真实对话,从 `model_ctx` 日志读每轮实际占用,取中位数回填。
    per_round_steady: int = Field(default=600, ge=1)
    # 层 2 的截短:客服答复留几个字、工具结果留几个字。
    layer2_assistant_chars: int = Field(default=50, ge=1)
    layer2_tool_chars: int = Field(default=60, ge=1)
    # 梗概长度上限,**同时也是「注入梗概」这项固定开销的来源**。
    summary_max_chars: int = Field(default=200, ge=1)
    # 单块检索证据 / 五个工具定义渲染后的估算开销。同样是估算值。
    evidence_block_tokens: int = Field(default=250, ge=1)
    tool_def_tokens: int = Field(default=800, ge=1)

    @model_validator(mode="after")
    def _overlap_must_leave_room(self):
        if self.chunk_overlap_chars >= self.chunk_max_chars:
            raise ValueError(
                "chunk_overlap_chars 必须小于 chunk_max_chars,"
                "否则每块的重叠就吃掉了配额,递归切分永不收敛"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
