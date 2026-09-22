import pytest
from pydantic import ValidationError

from app.config import Settings

REQUIRED = {
    "openai_base_url": "https://api.deepseek.com/v1",
    "openai_api_key": "sk-test",
    "openai_model": "deepseek-chat",
    "database_url": "mysql+asyncmy://u:p@h:3306/db",
}


def test_reads_required_fields():
    settings = Settings(_env_file=None, **REQUIRED)
    assert settings.openai_base_url == "https://api.deepseek.com/v1"
    assert settings.openai_model == "deepseek-chat"


def test_optional_fields_have_defaults():
    settings = Settings(_env_file=None, **REQUIRED)
    assert settings.chat_temperature == 0.7
    assert settings.extract_temperature == 0.0
    assert settings.context_budget_tokens == 8192
    assert settings.reserved_output_tokens == 1024
    assert settings.safety_margin_tokens == 512
    assert settings.session_ttl_seconds == 1800
    assert settings.max_sessions == 1000
    assert settings.session_lock_timeout_seconds == 60.0


def test_missing_model_is_rejected():
    """OPENAI_MODEL 必填:不给默认值,避免换模型时静默用错模型名。"""
    with pytest.raises(ValidationError) as exc:
        Settings(
            _env_file=None,
            openai_base_url="x",
            openai_api_key="y",
            database_url="mysql+asyncmy://u:p@h:3306/db",
        )
    assert "openai_model" in str(exc.value)


def test_missing_base_url_is_rejected():
    with pytest.raises(ValidationError) as exc:
        Settings(
            _env_file=None,
            openai_api_key="y",
            openai_model="z",
            database_url="mysql+asyncmy://u:p@h:3306/db",
        )
    assert "openai_base_url" in str(exc.value)


def test_missing_api_key_is_rejected():
    """OPENAI_API_KEY 必填:不给默认值,避免无密钥时静默启动、首请求才炸。"""
    with pytest.raises(ValidationError) as exc:
        Settings(
            _env_file=None,
            openai_base_url="x",
            openai_model="z",
            database_url="mysql+asyncmy://u:p@h:3306/db",
        )
    assert "openai_api_key" in str(exc.value)


def test_reads_api_key():
    assert Settings(_env_file=None, **REQUIRED).openai_api_key == "sk-test"


def test_database_url_is_required():
    """DATABASE_URL 必填,不给默认值 —— 默认值会拿一个可能不对的连接串去连。"""
    missing = {k: v for k, v in REQUIRED.items() if k != "database_url"}
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **missing)


def test_database_url_is_read_from_settings():
    settings = Settings(_env_file=None, **REQUIRED)
    assert settings.database_url == "mysql+asyncmy://u:p@h:3306/db"


def test_tool_defaults():
    settings = Settings(_env_file=None, **REQUIRED)
    assert settings.tool_timeout_seconds == 10.0
    assert settings.tool_retry_attempts == 1
    assert settings.tool_retry_delay_seconds == 0.3


def test_negative_retry_attempts_is_rejected():
    """重试次数为负 → 执行器的 attempts 算成 0,循环一次都不跑、
    last_message 停在空串,最终把**空错误文案**交给模型。启动即拒。"""
    with pytest.raises(ValidationError) as exc:
        Settings(_env_file=None, **REQUIRED, tool_retry_attempts=-1)
    assert "tool_retry_attempts" in str(exc.value)


def test_non_positive_timeout_is_rejected():
    """超时 <= 0 会让每一次工具调用立即超时。"""
    with pytest.raises(ValidationError) as exc:
        Settings(_env_file=None, **REQUIRED, tool_timeout_seconds=0)
    assert "tool_timeout_seconds" in str(exc.value)


def test_negative_retry_delay_is_rejected():
    with pytest.raises(ValidationError) as exc:
        Settings(_env_file=None, **REQUIRED, tool_retry_delay_seconds=-0.1)
    assert "tool_retry_delay_seconds" in str(exc.value)


def test_zero_bounds_are_allowed():
    """下界只在负数上收:0 次重试(等价于只试一次)与零延迟都是合法配置。"""
    settings = Settings(
        _env_file=None, **REQUIRED, tool_retry_attempts=0, tool_retry_delay_seconds=0
    )
    assert settings.tool_retry_attempts == 0
    assert settings.tool_retry_delay_seconds == 0


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_max_sessions_is_rejected(bad):
    """max_sessions <= 0 → `_enforce_capacity` 会把 `lock_for` 刚建的那把锁
    自己淘汰掉(它在 LRU 末尾、且尚未被 acquire),下一次请求遂铸出一把
    **新锁** —— 同一会话的两个请求并行跑,每会话互斥静默消失,全程无报错。
    ch01 里 0 只是"不留历史",ch02 把它提成了并发正确性开关,启动即拒。"""
    with pytest.raises(ValidationError) as exc:
        Settings(_env_file=None, **REQUIRED, max_sessions=bad)
    assert "max_sessions" in str(exc.value)


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_session_ttl_is_rejected(bad):
    """session_ttl_seconds <= 0 → `_purge` 每次调用都把全部未持锁条目判为
    "已过期"并立刻回收,锁条目活不过一次调用 —— 同一类静默故障。"""
    with pytest.raises(ValidationError) as exc:
        Settings(_env_file=None, **REQUIRED, session_ttl_seconds=bad)
    assert "session_ttl_seconds" in str(exc.value)


def test_positive_store_bounds_are_accepted():
    """反面:收界不能把默认值或最小的合法值误伤。"""
    defaults = Settings(_env_file=None, **REQUIRED)
    assert defaults.max_sessions == 1000
    assert defaults.session_ttl_seconds == 1800

    tight = Settings(
        _env_file=None, **REQUIRED, max_sessions=1, session_ttl_seconds=1
    )
    assert (tight.max_sessions, tight.session_ttl_seconds) == (1, 1)


# ---- ch03:知识库与向量检索的配置 ----


def test_ch03_fields_have_defaults():
    """13 个新字段全部可选带默认值 —— 环境里没有 Milvus 的机器仍能起服务、
    跑与检索无关的单测(检索组件懒初始化,配置有值不等于启动就连接)。"""
    s = Settings(_env_file=None, **REQUIRED)
    assert s.embedding_model_path == "models/bge-m3"
    assert s.embedding_max_length == 1024
    assert s.embedding_batch_size == 16
    assert s.milvus_uri == "http://127.0.0.1:19530"
    assert s.milvus_collection == "knowledge"
    assert s.rerank_top_k == 5
    # 0.25 是实测定出来的(2026-09-20,混合+重排链路),不是随手取的数:
    # 能命中的正例 top-1 最低 0.389 / 干扰项 top-1 最高 0.114 → 区间 (0.114, 0.389]。
    # 推导细节在字段旁边(app/config.py),这里只钉默认值不被静默改掉。
    assert s.retrieval_score_threshold == 0.25
    assert s.dedupe_threshold == 0.95
    assert s.chunk_max_chars == 800
    assert s.chunk_overlap_chars == 100
    assert s.mine_batch_conversations == 5


@pytest.mark.parametrize("bad", [0, -1])
@pytest.mark.parametrize("field", ["embedding_max_length", "embedding_batch_size", "chunk_max_chars"])
def test_non_positive_encoding_and_chunking_params_are_rejected(field, bad):
    """tokenizer 的 max_length/batch_size 为 0 会直接抛进 encode 深处;
    chunk_max_chars <= 0 让递归切分产不出合法块。都是配置期该抓住的错。"""
    with pytest.raises(ValidationError) as exc:
        Settings(_env_file=None, **REQUIRED, **{field: bad})
    assert field in str(exc.value)


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_top_k_is_rejected(bad):
    """top_k <= 0 → Milvus 搜索永远返回空,query_faq 永远走「未收录」——
    检索功能静默失效且无报错。必须启动即拒。"""
    with pytest.raises(ValidationError) as exc:
        Settings(_env_file=None, **REQUIRED, rerank_top_k=bad)
    assert "rerank_top_k" in str(exc.value)


@pytest.mark.parametrize("bad", [-0.1, 1.1])
@pytest.mark.parametrize("field", ["retrieval_score_threshold", "dedupe_threshold"])
def test_out_of_range_thresholds_are_rejected(field, bad):
    """两个阈值都是 [0,1] 上的相似度:越界一个方向等于永远全滤空(检索
    静默失效),另一个方向等于没有阈值(不相关也硬凑答案)。"""
    with pytest.raises(ValidationError) as exc:
        Settings(_env_file=None, **REQUIRED, **{field: bad})
    assert field in str(exc.value)


def test_threshold_bounds_are_accepted():
    """0 与 1 本身合法(1 只在归一化向量的完全重复上达得到)。"""
    s = Settings(_env_file=None, **REQUIRED,
                 retrieval_score_threshold=0.0, dedupe_threshold=1.0)
    assert s.retrieval_score_threshold == 0.0
    assert s.dedupe_threshold == 1.0


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_mine_batch_is_rejected(bad):
    """挖知识批次 <= 0 → range() 空转,脚本「成功」但一行没抽 —— 假成功比
    报错更糟。"""
    with pytest.raises(ValidationError) as exc:
        Settings(_env_file=None, **REQUIRED, mine_batch_conversations=bad)
    assert "mine_batch_conversations" in str(exc.value)


@pytest.mark.parametrize("bad", [-1, 800, 900])
def test_overlap_not_smaller_than_max_chars_is_rejected(bad):
    """overlap >= chunk_max_chars 时递归切分永不收敛(每块重叠就吃掉了
    配额)—— 跨字段约束,在模型层统一拒绝,不让 chunker 运行时死循环。"""
    with pytest.raises(ValidationError) as exc:
        Settings(
            _env_file=None,
            **REQUIRED,
            chunk_max_chars=800,
            chunk_overlap_chars=bad,
        )
    assert "chunk_overlap_chars" in str(exc.value)


# ---- ch05:编排 ----

def test_agent_step_limit_must_be_positive():
    """<=0 会让 ReAct 循环一次都不跑 —— 必须在启动时炸,不能运行时静默。"""
    import pytest

    with pytest.raises(ValueError):
        Settings(_env_file=None, **REQUIRED, max_agent_steps=0)


def test_agent_token_budget_must_be_positive():
    import pytest

    with pytest.raises(ValueError):
        Settings(_env_file=None, **REQUIRED, agent_token_budget=0)


# ---- ch06:退款子流程 ----


def test_query_expansion_cap_defaults_to_three():
    assert Settings(_env_file=None, **REQUIRED).query_expansion_max_queries == 3


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_query_expansion_cap_is_rejected(bad):
    """两种越界**都不报错、只静默变坏**,所以必须在启动时拒:

    - `0` → `expand_queries` 返回**空列表**,检索空转 —— 而它与「库里没这条
      知识」长得一模一样(两个受害者:召回为空、以及随后那句「我没太理解」);
    - 负数 → `queries[:-1]` 是 Python 的负切片语义「去掉最后 N 条」,于是
      **静默少一条**,看起来完全正常。

    `expand_queries` 自己也会抛(第二道闸),但那时已经是运行期、且只在
    真的走到退款子流程时才响 —— 配置错误该在**启动**时就响。
    """
    with pytest.raises(ValidationError) as exc:
        Settings(_env_file=None, **REQUIRED, query_expansion_max_queries=bad)
    assert "query_expansion_max_queries" in str(exc.value)
