"""飞轮编排:把低置信度池里的问题逐行「标准化 → 查重 → 归并或新建」。

    low_confidence_questions(未处理)
        └─ normalize ─→ dedupe(对着 review_queue.status='pending' 那一批)
                          ├ 命中 → occurrences += 1 + 回填 matched_review_id
                          └ 未命中 → 新建一行 review_queue + 回填 matched_review_id

## 幂等靠**一列两义**

`matched_review_id` 既是「这条池子行归并到了待审队列的哪一行」,**又是本流水线的
待处理标记**(`WHERE matched_review_id IS NULL`)。所以重跑天然不会重复归并 ——
处理过的行不再出现在选择里。⚠️ 谁把这个谓词去掉,谁就拿到了一个**看起来完全正常**
的重跑:行数不变(`review_queue` 上没建新行 —— 重吃的那几行会命中同一行),
`occurrences` 却在悄悄翻倍。它在库上唯一的痕迹就是「计数虚高」。

## 批的大小与顺序

`ORDER BY id LIMIT batch_size`:升序 = **先吃旧的**。池子是共享的、跨章的,所以
「一次跑完整个池子」不是本函数的职责 —— 一次一小批,由调度方反复调(离线脚本 /
验收脚本),和 `build_kb.py` 那种「中断了直接再跑」是同一种用法。

## 逐行失败不拖垮整批

一行炸了就记 `failed` 并**跳过**,那一行的 `matched_review_id` 留 NULL ⇒ 它仍在
「待处理」里,下一次跑会重试它。**不做重试**(同一句原话、同一个模型,重放大概率
同样失败)也不 `rollback`(见下 `for` 循环里的注释)。
"""

import logging

from sqlalchemy import select

from app.db.models import LowConfidenceQuestion, ReviewQueue
from app.flywheel.dedupe import find_duplicate
from app.flywheel.normalize import normalize_question

logger = logging.getLogger(__name__)


async def run_flywheel(*, session, model, batch_size: int) -> dict:
    """跑一批。返回 `{"processed", "merged", "created", "failed"}`。

    `processed` 是**这一批取到的行数**(含失败的),`merged + created` 是其中真的
    落了地的那些 —— 两者不等时差额就是 `failed`,审核页/验收靠这个差额发现「池子里
    有一批行一直在失败」。

    **选择在函数内部做**(而不是让调用方传一批行进来):那个 `WHERE ... IS NULL`
    的谓词就是幂等本身,把它留在调用方等于让每个调用方各自记得写对一次 —— 本仓
    的老教训是「不变量要放在唯一写口上,不要靠每个调用方自觉」。
    """
    rows = (
        await session.execute(
            select(LowConfidenceQuestion)
            .where(LowConfidenceQuestion.matched_review_id.is_(None))
            .order_by(LowConfidenceQuestion.id)
            .limit(batch_size)
        )
    ).scalars().all()

    stats = {"processed": 0, "merged": 0, "created": 0, "failed": 0}
    for row in rows:
        stats["processed"] += 1
        try:
            norm = await normalize_question(row.question, model=model)
            # ⚠️ 这次 SELECT **必须在循环里**,不能提到循环外。提到外面的话,本轮
            # 前面几行刚建出来的待审行**看不见** ⇒ 同一条缺口在本轮里被建两遍
            # (跨轮次仍然幂等,所以这个缺陷只在「一轮里出现同义的两行」时露头,
            # 而验收正好是这个形状)。`test_duplicate_hit_...` 就是钉它的。
            pending = (
                await session.execute(
                    select(ReviewQueue).where(ReviewQueue.status == "pending")
                )
            ).scalars().all()
            hit = await find_duplicate(norm["standard_question"], pending, model=model)
            if hit is not None:
                hit.occurrences += 1
                row.matched_review_id = hit.id
                stats["merged"] += 1
            else:
                queue_row = ReviewQueue(
                    standard_question=norm["standard_question"],
                    example_answer=norm["example_answer"],
                    occurrences=1,
                    # **显式写 `status="pending"`**(spec §8.2 的伪代码就是这么写的)。
                    # 模型上的 `default="pending"` 与列上的 `server_default` 本来也会补,
                    # 行为等价 —— 这里是**对齐 spec 的写法**,不是修行为:显式一行让
                    # 「新建的行一定是待审」这件事在调用处可见,而不是要读者去查
                    # `db/models.py` 的默认值(那两处默认值是给**别的**插入路径用的)。
                    status="pending",
                    first_raw_question=row.question,
                    source_conversation_id=row.source_conversation_id,
                )
                session.add(queue_row)
                await session.flush()      # 拿自增主键(下面那行要指回它)
                row.matched_review_id = queue_row.id
                stats["created"] += 1
        except Exception:  # noqa: BLE001
            # **一行失败不拖垮整批**:那一行的 matched_review_id 留 NULL,下次重跑
            # 还会处理它。失败原因进日志(exc_info)——
            # 不带 traceback 的话,池子里那行只会「一直不消失」,没人知道为什么。
            #
            # ⚠️ 这里**刻意不 rollback**:本函数只在最后提交一次,前面几行是
            # `flush`(还没提交)。中途 rollback 会把它们的写入连同 `matched_review_id`
            # 的赋值一起丢掉 —— 内存里的 `row` 对象仍带着那个 id,于是**内存与库
            # 不一致**,而那一批的返回值还会说自己 `created` 了几行。
            # 代价:若失败是**库层**的错误(session 已进入待回滚态),后面的行会
            # 连锁失败 —— 它们都记 `failed`、都留在池子里等下次,没有数据被写坏。
            #
            # `except Exception` 收的是**这一行**的失败;`CancelledError` 是
            # `BaseException`,不在这里,会原样向上抛(客户端断开时那个协程就该停)。
            stats["failed"] += 1
            logger.warning("飞轮处理失败 row id=%s", row.id, exc_info=True)

    # 整批一次提交:要么这一批的归并都落地,要么都不落 —— 半批落地的中间态会让
    # 下一次重跑拿不准哪些算处理过(而 `matched_review_id` 正是那个判据)。
    await session.commit()
    logger.info(
        "飞轮一批:processed=%s merged=%s created=%s failed=%s",
        stats["processed"], stats["merged"], stats["created"], stats["failed"],
    )
    return stats
