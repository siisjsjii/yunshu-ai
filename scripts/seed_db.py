"""灌种子数据。用法:.venv/Scripts/python.exe scripts/seed_db.py

幂等:faq 按 question 去重;样例会话与工单用固定主键,重复跑不会堆积。
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select

from app.db.base import get_sessionmaker
from app.db.models import Conversation, Faq, MessageRecord, Ticket

# 注意:本表刻意不含「邮费」「运费」相关条目 —— 验收 3 的漏召回是预期结果,
# 若此处出现该词,验收 3 会假通过。见 spec §7.3。测试 test_seed_contains_no_shipping_fee_entry 守护此约束。
FAQ_ROWS = [
    {"question": "退货政策是什么", "answer": "支持七天无理由退货,商品需保持完好、吊牌齐全。", "category": "退换货"},
    {"question": "怎么申请退货", "answer": "在订单详情页点击「申请退款」,选择退货原因并提交,审核通过后会给出寄回地址。", "category": "退换货"},
    {"question": "可以换货吗", "answer": "可以。签收后七天内,商品无使用痕迹即可申请换货,请提供订单号与原/目标规格。", "category": "退换货"},
    {"question": "退款多久到账", "answer": "退货入库验收通过后 1-3 个工作日退回原支付渠道,具体到账时间以银行为准。", "category": "退换货"},
    {"question": "发票怎么开", "answer": "下单时可在备注中填写抬头与税号;已完成的订单可联系客服补开电子发票。", "category": "发票问题"},
    {"question": "发票可以开专票吗", "answer": "可以开具增值税专用发票,请提供公司名称、税号、地址电话与开户行账号。", "category": "发票问题"},
    {"question": "物流一直没更新", "answer": "物流信息可能存在延迟,一般 24 小时内会更新。超过 48 小时未更新请联系客服为您催件。", "category": "物流异常"},
    {"question": "快递显示签收但我没收到", "answer": "请先与快递员或代收点核实。确认未收到的,联系客服,我们会向快递公司发起核查。", "category": "物流异常"},
    {"question": "发什么快递", "answer": "默认发中通/圆通,偏远地区可能改发邮政。下单后无法指定快递公司。", "category": "物流异常"},
    {"question": "商品有货吗", "answer": "商品页显示库存实时同步。若显示缺货,可点击「到货通知」,补货后会短信提醒。", "category": "商品咨询"},
    {"question": "支持哪些支付方式", "answer": "支持微信、支付宝、银行卡以及花呗分期。", "category": "商品咨询"},
    {"question": "怎么联系人工客服", "answer": "在对话框中直接说明需要人工,或拨打客服热线 400-000-0000(9:00-21:00)。", "category": "其他"},
]

SAMPLE_CONVERSATION = "seed0000000000000000000000000000"
SAMPLE_TICKET_NO = "T-SEED-0001"


async def seed() -> None:
    async with get_sessionmaker()() as session:
        existing = set(
            (await session.execute(select(Faq.question))).scalars().all()
        )
        for row in FAQ_ROWS:
            if row["question"] not in existing:
                session.add(Faq(**row))

        if not (
            await session.execute(
                select(Conversation).where(Conversation.id == SAMPLE_CONVERSATION)
            )
        ).scalars().first():
            session.add(
                Conversation(
                    id=SAMPLE_CONVERSATION, user="demo-user", status="active"
                )
            )
            session.add(
                MessageRecord(
                    conversation_id=SAMPLE_CONVERSATION,
                    role="user",
                    content="订单 20240915 的鞋码不对,我想换大一码",
                )
            )
            session.add(
                MessageRecord(
                    conversation_id=SAMPLE_CONVERSATION,
                    role="assistant",
                    content="好的,请提供原规格与目标规格,我为您登记换货。",
                )
            )
            session.add(
                Ticket(
                    ticket_no=SAMPLE_TICKET_NO,
                    conversation_id=SAMPLE_CONVERSATION,
                    description="鞋码偏小,想换成大一码",
                    ticket_type="换货",
                    status="open",
                )
            )

        await session.commit()


if __name__ == "__main__":
    asyncio.run(seed())
    print(f"种子完成:faq {len(FAQ_ROWS)} 条 + 1 组样例会话/消息/工单")
