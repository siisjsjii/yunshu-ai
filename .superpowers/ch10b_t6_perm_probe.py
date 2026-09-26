"""T6 探针:docstring 里那句「语料重排会不会换一批样本」——**先量再写**。

量三件事(全部在真实 `prelabeled.jsonl` 上):
1. 逐行同序重跑:同一批(可复现的基线);
2. 行序**整体倒过来**:抽出的集合与 1 差多少;
3. 桶的内容不变、只把**行序随机打乱**(固定种子):同上。
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.topic.labeling import pick_review_sample  # noqa: E402

rows = [json.loads(l) for l in
        (ROOT / "evals" / "topic" / "prelabeled.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()]
print(f"输入 {len(rows)} 行")

base = [r["id"] for r in pick_review_sample(rows, per_label=5)]
print(f"① 逐行同序      : {len(base)} 条")

again = [r["id"] for r in pick_review_sample(rows, per_label=5)]
print(f"①' 再跑一次      : 与①逐位相同 = {again == base}")

rev = [r["id"] for r in pick_review_sample(list(reversed(rows)), per_label=5)]
print(f"② 整体倒序      : {len(rev)} 条;与①相同的 {len(set(rev) & set(base))} 条,"
      f"不同的 {len(set(rev) ^ set(base))} 条")

shuffled_rows = list(rows)
random.Random(7).shuffle(shuffled_rows)
sh = [r["id"] for r in pick_review_sample(shuffled_rows, per_label=5)]
print(f"③ 随机重排      : {len(sh)} 条;与①相同的 {len(set(sh) & set(base))} 条,"
      f"不同的 {len(set(sh) ^ set(base))} 条")
