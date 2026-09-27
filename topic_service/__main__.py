"""`python -m topic_service --model models/topic-clf --port 8103`

⚠️ **8103,不是 8101/8102** —— 那两个是 ch08 的两个 MCP Server 的。

跑之前先查端口有没有残留进程(本仓记过「curl 到旧代码 ⇒ 假红」的账):

    netstat -ano | grep 8103

`--device` 默认 **cpu**,与 `scripts/eval_topic_clf.py` 同一条理由(逐字节可复现);
真要上卡再显式传 `--device cuda`。
"""

from __future__ import annotations

import argparse

import uvicorn

from topic_service.model import TopicClassifier
from topic_service.server import create_app


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="topic_service", description="17 类多标签主题分类旁路服务")
    #: 默认值**写在这里是可以的**:它不是「产物里的那份配置」,而是「去哪找产物」。
    #: 真正不许写死的是 `labels` / `threshold` / `max_length` —— 那三样在 `TopicClassifier` 里
    #: 一律读 `labels.json` / `inference_config.json`(spec §9.1)。
    ap.add_argument("--model", default="models/topic-clf", help="训练产物目录")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8103)
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = ap.parse_args(argv)

    classifier = TopicClassifier(args.model, device=args.device)
    # 启动时把读到的那几个数印出来 —— 服务与产物对不上时,这一行就是第一现场。
    print(f"topic_service: model_dir={classifier.model_dir} "
          f"labels={len(classifier.labels)} threshold={classifier.threshold} "
          f"max_length={classifier.max_length} device={args.device}", flush=True)

    uvicorn.run(create_app(classifier), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
