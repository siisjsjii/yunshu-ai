"""低置信度池 → 多标签主题归类的**离线批处理**(spec §9.2)。

```
读池(low_confidence_questions)→ 清洗(clean.py,与训练侧同源)
   → 批量 POST /predict(默认 http://127.0.0.1:8103)
   → 写 topic_classifications(唯一键 uk_pool_question ⇒ 重跑 = 覆盖)
```

用法:
    .venv/Scripts/python.exe scripts/classify_topics.py --dry-run --limit 20
    .venv/Scripts/python.exe scripts/classify_topics.py

**前置**:旁路服务在跑(`python -m topic_service --model models/topic-clf --port 8103`)、
MySQL 可达、`topic_classifications` 表存在(`scripts/init_db.py` 或 `db/ch10.sql`)。

## 三条不肯让步的性质

**① 服务故障 ⇒ 响亮失败,绝不写空标签。** 写了空标签的行会让分布页把「服务挂了」读成
「这些问题没有主题」—— 正是 ch09 那条 👎 回捞失败被读成「知识库缺这块」的同款事故。
本脚本在这一条上比别处都严:**整次运行要么全落、要么一行不落**。

**② 清洗与训练侧**同一份实现**(`app.topic.clean.clean`)。** 只在一侧清洗 =
train/serve skew:模型在真机上看到的是另一种文本,而**没有任何东西会报错**,
它只让线上准确率悄悄低于测试集读数。这条 import 由
`tests/test_topic_clean.py::test_both_sides_use_the_same_clean` 用**源码扫描**守着
(那里要的是「有没有人写下这行」,不是「跑到了没有」)。

**③ 一批一次 POST,批内条数与顺序一一对应**(服务的契约)。条数对不上 ⇒ 抛,
不许按 `zip` 配对 —— 「第 3 条的结果配给了第 4 条的问题」在分布页上完全看不出来。

## 记账:三处与本仓「不许静默」原则相关的取舍

### a. `model_version` 是一个**诚实的近似**(订正 15-A)

那一列是 `NOT NULL` 而计划里一个字都没写。`train_meta.json` 里**没有权重指纹**
(T11 复审 Q4 裁定),所以取

    f"{meta['base']}@{meta['data_fingerprint']}::{meta['trained_at_utc']}"

**不能单用 `data_fingerprint`** —— 它是**语料**的、**一次运行**的指纹
(`train_meta.json` 的 `data_fingerprint_note` 逐字写着:增强产物不可复现,
所以那个 sha256 是「一次运行」的,不是「这份语料」的)⇒ **两份权重不同的训练可以同值**,
而那正是这一列要分的东西(分布页要 `COUNT(DISTINCT model_version)`)。
补上 `trained_at_utc` 与 `base` 之后,同一次训练的重跑仍然同值(那是**对的**:
同一份权重),不同的训练几乎不会撞。
**严格的权重指纹要改 `save_artifacts`(`sha256(model.safetensors)`)并重跑训练 ——
那是 T9/T10 的账,不该在这里补。** 这里是近似,不是指纹。

### b. 空标签**允许写,但必须数出来**(订正 15-C)

`labels: []` 是**合法 JSON**(服务返回空 = 全部低于阈值,是合法的模型输出),
`NOT NULL` 拦不住。⇒ 允许写,但每次运行都会打印「本次空标签 N 条」—— 否则
「服务每次都返回空」会被分布页读成「这些问题没有主题」,那是**故障被读成业务结果**。
⚠️ 「服务故障不写」与「服务返回空照写」是**两件事**:前者那批**一行都不落**(见 ①),
后者是**真实的模型输出**,落库才有意义。

### c. 落库事务的粒度是**整次运行**,不是一批(与 §9.2 那句「一批一事务」的偏离)

`classify_batch` 的 `write` 是**同步**回调(替身是 `list.append`),所以落库发生在
批处理**跑完之后**:`run()` 收集全部行,再一次性 upsert + commit。
这与 §9.2「一个批次一个事务」**不同**,而且方向是**更严**:

- **好处**:服务跑到一半挂掉时**不会**留下半份归类结果。半份结果在分布页上
  **与一份完整结果长得一模一样**(同样的图、同样的「数据截至…」),而
  「这批池子只归了一半」这件事**没有任何东西会报错** —— 正是本节 ① 那种
  「故障被读成业务结果」。半份不落,重跑幂等,代价是那几批要重算一遍。
- **代价如实记账**:§9.2 的「已写的批次保留」在这里**不成立** —— 异常时前面成功的
  批次也不落库。它们**没有丢**(输入没动、重跑同值),但要等服务恢复后重跑。
- 单批之内仍然是「**整批形状全部通过之后才写第一行**」(见 `classify_batch`),
  所以「32 条里第 7 条失败 ⇒ 整批不写」这条**结构性**成立。

## 一条不属于本脚本的事

`topic_classifications` 与 `low_confidence_questions` 都是**共享且只追加**的表。
凡是对它们的断言**必须按 id 过滤**,不许写「查最近这几条」那种形式 ——
本仓记过「被上次运行的数据污染 ⇒ 偶尔红偶尔绿」那一类。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

# 直接 `python scripts/classify_topics.py` 时仓库根不在 sys.path 上
# (与 `scripts/prepare_topic_data.py` / `scripts/eval_trend.py` 同款)。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402
from sqlalchemy import func, select  # noqa: E402
from sqlalchemy.dialects.mysql import insert as mysql_insert  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402

from app.db.base import get_sessionmaker  # noqa: E402
from app.db.models import LowConfidenceQuestion, TopicClassification  # noqa: E402
# ⚠️ 这一行是**同源守卫**要求的(tests/test_topic_clean.py::test_both_sides_use_the_same_clean):
#    推理侧必须与训练侧(`scripts/prepare_topic_data.py`)用**同一份** `clean`。
#    它在本文件里也被真的用到(`classify_batch` 发请求之前洗一遍),不是一条只为过守卫而存在的 import。
from app.topic.clean import clean  # noqa: E402
from app.topic.taxonomy import LABELS  # noqa: E402

#: 默认服务地址。**8103,不是 8101/8102** —— 那两个是 ch08 的两个 MCP Server 的。
DEFAULT_SERVICE_URL = "http://127.0.0.1:8103"
DEFAULT_BATCH_SIZE = 32

#: 单次 HTTP 往返的上界(**显式给,绝不留 None**)。
#: 本仓为「模型客户端不设超时 ⇒ 那次 await 谁也等不回来,挂起不是异常 ⇒
#: `finally` 永不执行」付出过代价(ch09 T16,三个任务各盯到 666/245/382 秒仍是 running)。
#: 这里服务侧是 CPU 前向(32 条短文本一个量级为秒),120s 是宽裕的上界,不是目标值。
REQUEST_TIMEOUT_SECONDS = 120.0

#: `model_version` 的列宽(`db/ch10.sql` 与 ORM 都是 `VARCHAR(128)`)。
MODEL_VERSION_MAX_CHARS = 128


def _emit(text: str = "") -> None:
    """打印一行。**显式钉 UTF-8 输出边界。**

    本机 locale 是 cp936:直接 `print` 含中文的行会在**管道 / 重定向**时
    用 GBK 编码而抛 `UnicodeEncodeError`(本仓记过多次)。走 `sys.stdout.buffer`
    则字节可控;stdout 没有 `buffer` 时(某些捕获装置)退回 `print`。
    """
    stream = getattr(sys.stdout, "buffer", None)
    if stream is None:
        print(text)
        return
    stream.write((text + "\n").encode("utf-8"))
    stream.flush()


class TopicServiceProtocolError(RuntimeError):
    """服务的返回**形状**不是约定形状 —— 响亮失败,绝不猜。

    与 `ToolInfrastructureError` 同一条本仓原则:**基础设施故障不许伪装成业务结果**。
    这里防的是它的另一个面:形状漂移(键改名 / 外层不是对象 / 类目名漂开)时
    若用 `.get(..., [])` 兜底,调用方会把「读不懂」当成「没有主题」。
    """


# --------------------------------------------------------------------------
# `model_version`(订正 15-A)
# --------------------------------------------------------------------------


def model_version_from_meta(meta: Mapping[str, Any] | None) -> str:
    """`train_meta.json` → `topic_classifications.model_version`(见模块 docstring 段 a)。

    **这是个诚实的近似,不是权重指纹**:产物里没有权重哈希,能拿到的只有
    「哪份 base + 哪次运行」这一对。缺字段一律**抛** —— 拼出一个 `None@None::None`
    那样的值比报错坏得多:它是**合法字符串**,会照常落库、照常让分布页显示一个
    假的版本号,而没有任何东西会报错。
    """
    if not isinstance(meta, Mapping):
        raise ValueError(f"train_meta 不是对象({type(meta).__name__})—— 服务没给对东西")

    missing = [k for k in ("base", "data_fingerprint", "trained_at_utc") if not meta.get(k)]
    if missing:
        raise ValueError(
            f"train_meta 缺字段 {missing} —— model_version 是 NOT NULL 列,"
            "拼不出真值就该响亮失败,不许落一个看不出哪来的值"
        )

    version = f"{meta['base']}@{meta['data_fingerprint']}::{meta['trained_at_utc']}"
    if len(version) > MODEL_VERSION_MAX_CHARS:
        raise ValueError(
            f"model_version 长 {len(version)} 字符,超过列宽 {MODEL_VERSION_MAX_CHARS} "
            "—— 不许把整个 dict 塞进来(`meta` 里的超参/指标都在)"
        )
    return version


# --------------------------------------------------------------------------
# HTTP 客户端(订正 15-B):`{"results": …}` 这个形状**只许**在这里被拆
# --------------------------------------------------------------------------


class TopicServiceClient:
    """`topic_service` 的客户端。两个端点: `/healthz`(先问「你读到了什么」)、`/predict`。

    ⚠️ **拆 `["results"]` 是这一个类的职责**,不是 `classify_batch` 的:
    纯逻辑层收/发**裸 list**(替身 `_FakeClient` 就是照这个写的),否则 HTTP 形状会漏进去,
    「替身」与「生产」走的就不是同一条路了。
    """

    def __init__(self, base_url: str = DEFAULT_SERVICE_URL, *,
                 timeout: float = REQUEST_TIMEOUT_SECONDS,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        # `transport` 那个口子只给单测(挂 `httpx.MockTransport`);生产走默认。
        # 用「换 transport」而不是「patch post」,是因为被测的是**这个类**
        # —— 换 transport 才叫真的把它跑了一遍。
        self.base_url = base_url
        self._client = httpx.AsyncClient(
            base_url=base_url, timeout=timeout, transport=transport
        )

    async def __aenter__(self) -> "TopicServiceClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def healthz(self) -> dict:
        """`GET /healthz` —— **批处理先问的那一句**(「服务起没起 + 它读到了什么」)。

        形状校验照 `/predict`:不是对象、缺 `train_meta` ⇒ 抛。
        `train_meta` 缺了的话 `model_version_from_meta` 会在下游抛,但**在这里抛更好**:
        错因指向「服务没透出 train_meta」,而不是「脚本拼版本号拼错了」。
        """
        resp = await self._client.get("/healthz")
        resp.raise_for_status()          # 5xx/4xx 一律向上抛,绝不读成「一切正常」
        payload = json.loads(resp.text)
        if not isinstance(payload, dict) or not isinstance(payload.get("train_meta"), dict):
            raise TopicServiceProtocolError(
                f"/healthz 的形状不是约定形状(期望含 train_meta 的对象):{_shape_hint(payload)}"
            )
        return payload

    async def predict(self, texts: Sequence[str]) -> list:
        """`POST /predict` → **裸 list**(`{"results": [...]}` 里那一层)。

        形状漂移**响亮失败**,不许 `.get("results", [])` 兜底 ——
        兜底之后「读不懂响应」与「模型没给任何主题」在调用方眼里**一模一样**,
        而那正是 15-C 单列出来的那种事故。
        """
        resp = await self._client.post("/predict", json={"texts": list(texts)})
        resp.raise_for_status()
        payload = json.loads(resp.text)
        if not isinstance(payload, dict) or "results" not in payload:
            raise TopicServiceProtocolError(
                f"/predict 的返回里没有 results(期望 {{\"results\": [...]}}):"
                f"{_shape_hint(payload)}"
            )
        return payload["results"]


def _shape_hint(payload: Any) -> str:
    """给出错信息一个**能定位**的形状描述(只印顶层类型与键名,不印内容)。"""
    if isinstance(payload, dict):
        return f"顶层是对象,键 = {sorted(payload)[:10]}"
    if isinstance(payload, list):
        return f"顶层是数组,长 {len(payload)}"
    return f"顶层是 {type(payload).__name__}"


# --------------------------------------------------------------------------
# 纯逻辑层:批处理
# --------------------------------------------------------------------------


def _validate_batch_results(results: Any, batch: Sequence[Mapping[str, Any]]) -> None:
    """一批的返回**全部**检查完才准写第一行 —— 这是「整批原子」的落点。

    四样,缺一不可(`tests/test_classify_topics.py` 各有一条用例):

    1. 外层是 list;2. **条数与这一批的输入一致**(服务的契约);
    3. 每条是对象且 `labels` / `scores` 形状对;4. **`labels` 里每个名字都在
       `taxonomy.LABELS` 里**(订正 15-D)—— 这是「标签顺序 / 类目表两处漂移」
       这条链上**最靠近数据的那一道**:名字一旦落进 JSON 列就再也回不来了
       (数据库对它零约束),分布页会画出一个不存在的类目。
    """
    if not isinstance(results, list):
        raise TopicServiceProtocolError(
            f"服务返回的不是数组({type(results).__name__})—— 见客户端侧的形状校验"
        )
    if len(results) != len(batch):
        raise TopicServiceProtocolError(
            f"服务返回 {len(results)} 条,这一批发出去 {len(batch)} 条 —— "
            "条数对不上时按 zip 配对会静默错位(第 3 条的结果配给第 4 条的问题)"
        )
    for index, result in enumerate(results):
        if not isinstance(result, Mapping):
            raise TopicServiceProtocolError(f"第 {index} 条结果不是对象:{type(result).__name__}")
        labels = result.get("labels")
        scores = result.get("scores")
        if not isinstance(labels, list) or not isinstance(scores, Mapping):
            raise TopicServiceProtocolError(
                f"第 {index} 条结果的形状不对(labels 须为数组、scores 须为对象)"
            )
        for label in labels:
            if label not in LABELS:
                raise TopicServiceProtocolError(
                    f"第 {index} 条结果的标签 {label!r} 不在 taxonomy.LABELS 里 —— "
                    "类目表两处漂移了(分布页会画出一个不存在的类目)"
                )


async def classify_batch(
    rows: Iterable[Mapping[str, Any]],
    *,
    client: Any,
    write: Callable[[dict], Any],
    batch_size: int,
) -> dict:
    """`rows` 按 `batch_size` 切片 → 清洗 → `client.predict(texts)` → `write(行)`。

    - `rows` 每项是 `{"id": int, "question": str}`;
    - `write` 是**同步**回调,收到 `{"low_confidence_question_id", "labels", "scores"}`;
    - `client.predict(texts)` **返回裸 list**(见 `TopicServiceClient`);
    - **任何异常向上抛**;一批里只要有一条形状不对,**这一批一行都不写**
      (校验整批跑完才轮到 `write`,结构上做不到写一半);
    - 空标签是**合法结果**,照写 —— 但要数出来(返回值的 `empty_labels` / `empty_ids`)。

    返回 `{"rows", "batches", "empty_labels", "empty_ids"}`。
    """
    rows = list(rows)
    if batch_size < 1:
        raise ValueError(f"batch_size 必须 >= 1,实际 {batch_size}")

    written = 0
    batches = 0
    empty_ids: list[int] = []

    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        # 清洗**在发请求之前** —— 与训练侧同源(见模块 docstring ②)。
        texts = [clean(row["question"]) for row in batch]

        results = await client.predict(texts)
        _validate_batch_results(results, batch)      # ← 整批校验,先于任何一次 write

        for row, result in zip(batch, results):      # noqa: B905(两边等长已由上一行保证)
            labels = list(result["labels"])
            if not labels:
                empty_ids.append(row["id"])
            write({
                "low_confidence_question_id": row["id"],
                "labels": labels,
                "scores": dict(result["scores"]),
            })
            written += 1
        batches += 1

    return {
        "rows": written,
        "batches": batches,
        "empty_labels": len(empty_ids),
        "empty_ids": empty_ids,
    }


# --------------------------------------------------------------------------
# 落库(唯一写口)
# --------------------------------------------------------------------------


async def upsert_classifications(
    session: AsyncSession, rows: Sequence[Mapping[str, Any]], model_version: str
) -> int:
    """`topic_classifications` 的**唯一写口**:`INSERT … ON DUPLICATE KEY UPDATE`。

    唯一键 `uk_pool_question` 是「重跑 = 覆盖,不是追加」的全部保证
    (`db/ch10.sql` 与 ORM 两侧同名)。没有它,重跑会在池子旁边**静静堆出第二份结果**,
    分布页每一类的条数跟着翻倍,而没有任何东西会报错。

    `classified_at` 在覆盖时**刷新成当前时间**:它记的是「这条结果是什么时候算的」,
    而重跑之后算它的就是这一次。
    """
    if not rows:
        return 0
    values = [
        {
            "low_confidence_question_id": int(row["low_confidence_question_id"]),
            "labels": list(row["labels"]),
            "scores": dict(row["scores"]),
            "model_version": model_version,
        }
        for row in rows
    ]
    stmt = mysql_insert(TopicClassification).values(values)
    stmt = stmt.on_duplicate_key_update(
        labels=stmt.inserted.labels,
        scores=stmt.inserted.scores,
        model_version=stmt.inserted.model_version,
        classified_at=func.now(),
    )
    await session.execute(stmt)
    return len(values)


def pool_statement(limit: int | None):
    """读池子的那条 SELECT。**抽出来单独一个函数**,是为了让它可被编译、可被断言。

    ⚠️ `ORDER BY id` **不能省**:`--limit N` 的语义是「最旧的 N 条」,
    而这**只有**在有序时才成立 —— 去掉它,InnoDB today 通常仍按主键返回
    (所以删掉 `order_by` 的变异在真库上**不会红**,见 T13 报告的 M12),
    但那是**实现细节,不是契约**:换个索引、加个覆盖索引、换 MySQL 版本都可能变。
    契约这一半就靠 `tests/test_classify_topics.py::test_the_pool_statement_is_ordered_by_id`
    对着**编译出来的 SQL** 钉住(行为断言在这里是瞎的,理由写在那个用例里)。
    """
    stmt = (
        select(LowConfidenceQuestion.id, LowConfidenceQuestion.question)
        .order_by(LowConfidenceQuestion.id)
    )
    if limit is not None:
        return stmt.limit(limit)
    return stmt


async def load_pool(session: AsyncSession, limit: int | None) -> list[dict]:
    """读池子(语句见 `pool_statement`)。"""
    found = (await session.execute(pool_statement(limit))).all()
    return [{"id": row.id, "question": row.question} for row in found]


async def distribution_numbers(session: AsyncSession) -> tuple[int, int]:
    """分布页抬头那两个数:**总行数 / 不同问题数**。

    「不同问题数」按 `low_confidence_question_id` 去重 —— 唯一键保证一行一条,
    所以它等于「有多少条池子行已被归类」。
    """
    total = (
        await session.execute(select(func.count()).select_from(TopicClassification))
    ).scalar_one()
    distinct = (
        await session.execute(
            select(func.count(func.distinct(TopicClassification.low_confidence_question_id)))
        )
    ).scalar_one()
    return int(total), int(distinct)


# --------------------------------------------------------------------------
# 编排
# --------------------------------------------------------------------------


async def run(
    *,
    client: Any,
    model_version: str,
    limit: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dry_run: bool = False,
) -> dict:
    """读池 → 归类 → 落库。**整次运行一个事务**(见模块 docstring 段 c)。

    服务故障 ⇒ `classify_batch` 抛出 ⇒ 这里**什么都不写**,异常继续向上抛。
    """
    async with get_sessionmaker()() as session:
        pool = await load_pool(session, limit)
        _emit(f"池子里读到 {len(pool)} 行" + (f"(--limit {limit})" if limit else ""))

        collected: list[dict] = []
        stats = await classify_batch(
            pool, client=client, write=collected.append, batch_size=batch_size
        )

        _emit(f"分批 {stats['batches']} 次,归类 {stats['rows']} 行,"
              f"其中**空标签 {stats['empty_labels']} 条**")
        if stats["empty_ids"]:
            # ⚠️ 空标签要**数出来并打印**(15-C):「服务每次都返回空」否则会被分布页
            #    读成「这些问题没有主题」—— 故障被读成业务结果。
            _emit(f"  空标签的池子 id:{stats['empty_ids']}")

        if dry_run:
            for row in collected:
                _emit(f"  [dry-run] id={row['low_confidence_question_id']} "
                      f"labels={row['labels']}")
            written = 0
        else:
            written = await upsert_classifications(session, collected, model_version)
            await session.commit()

        total, distinct = await distribution_numbers(session)

    return {
        "pool_rows": len(pool),
        "classified": stats["rows"],
        "written": written,
        "empty_labels": stats["empty_labels"],
        "empty_ids": stats["empty_ids"],
        "batches": stats["batches"],
        "total_rows": total,
        "distinct_questions": distinct,
        "model_version": model_version,
        "dry_run": dry_run,
    }


async def _main(args: argparse.Namespace) -> int:
    async with TopicServiceClient(args.service_url) as client:
        health = await client.healthz()
        model_version = model_version_from_meta(health.get("train_meta"))
        _emit(f"服务 {args.service_url}:labels={len(health.get('labels') or [])} "
              f"threshold={health.get('threshold')} max_length={health.get('max_length')}")
        _emit(f"model_version = {model_version}")

        summary = await run(
            client=client,
            model_version=model_version,
            limit=args.limit,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )

    verb = "会写" if summary["dry_run"] else "已写"
    _emit("")
    _emit(f"{verb} {summary['written']} 行;本次空标签 {summary['empty_labels']} 条")
    _emit(f"分布页会看到:总行数 {summary['total_rows']} / 不同问题数 "
          f"{summary['distinct_questions']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="classify_topics.py",
        description="低置信度池 → 多标签主题归类(离线批处理,spec §9.2)",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="最多处理多少条池子行(按 id 升序,即最旧的先)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"每次 POST 发几条(默认 {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--service-url", default=DEFAULT_SERVICE_URL,
                        help=f"旁路推理服务地址(默认 {DEFAULT_SERVICE_URL})")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印「会写哪些行」,不落库")
    args = parser.parse_args(argv)

    if args.batch_size < 1:
        _emit(f"!!! --batch-size 必须 >= 1,实际 {args.batch_size}")
        return 2

    try:
        return asyncio.run(_main(args))
    except Exception as exc:  # noqa: BLE001 —— 这里就是要**响亮**收口,见下
        # 退出码非 0(§9.2)。**不吞异常**:文案里带类型与原文,排查从这一行开始。
        # 注意不打印整个 traceback 不是「掩盖」——它是给跑批用的稳定出口;
        # 想看栈就 `python -m pdb` 或直接 import 调 `run()`。
        _emit(f"!!! 跑批失败:{type(exc).__name__}: {exc}")
        _emit("!!! 本次**未落库**(服务故障绝不写空标签;DB 故障整次运行回滚)。")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
