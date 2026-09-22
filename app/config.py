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
    tool_retry_attempts: int = Field(default=1, ge=0)
    tool_retry_delay_seconds: float = Field(default=0.3, ge=0)

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
