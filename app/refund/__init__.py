"""退款子流程的纯数据与纯函数。

**不依赖 LangChain、不依赖 app.agent** —— 这样 `app/agent/refund_nodes.py`
与 `app/api/refund.py` 都能引用它而不产生反向依赖边。
"""
