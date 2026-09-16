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
    context_budget_tokens: int = 8192
    reserved_output_tokens: int = 1024
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

    # ---- ch03:知识库与向量检索。全部可选带默认值,检索组件懒初始化,----
    # ---- 配置有值不等于启动就连接 Milvus / 加载 BGE-M3。           ----
    embedding_model_path: str = "models/bge-m3"
    embedding_max_length: int = Field(default=1024, gt=0)
    embedding_batch_size: int = Field(default=16, gt=0)
    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_collection: str = "knowledge"
    # top_k <= 0 → 搜索永远空,检索静默失效;阈值越界一个方向等于永远全滤空、
    # 另一个方向等于没有阈值(不相关也硬凑答案)。都在启动时拒。
    retrieval_top_k: int = Field(default=3, ge=1)
    retrieval_score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    dedupe_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    chunk_max_chars: int = Field(default=800, gt=0)
    chunk_overlap_chars: int = Field(default=100, ge=0)
    # 挖知识批次 <= 0 → range() 空转,脚本"成功"但一行没抽,比报错更糟。
    mine_batch_conversations: int = Field(default=5, ge=1)

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
