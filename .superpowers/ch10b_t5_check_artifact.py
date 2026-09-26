"""对 `evals/topic/prelabeled.jsonl` 做**独立复算**(不走脚本自己的读数)。

检查:行数 / id 唯一且无洞 / 标签都在 17 类里 / 证据逐条真是清洗后问句的子串 /
三个读数的独立复算 / 标签分布。
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.topic.clean import clean
from app.topic.taxonomy import LABELS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evals" / "topic" / "prelabeled.jsonl"
CORPUS = ROOT / "evals" / "topic" / "corpus.jsonl"
SYNTH = ROOT / "evals" / "topic" / "synthetic.jsonl"

rows = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()]
src = []
for p in (CORPUS, SYNTH):
    if p.exists():
        src += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

buf = []
buf.append(f"产物行数 = {len(rows)}   语料合计 = {len(src)}")
ids = [r["id"] for r in rows]
buf.append(f"id 唯一 = {len(set(ids)) == len(ids)};id 与语料顺序一致 = {ids == [r['id'] for r in src][:len(ids)]}")
missing = [r["id"] for r in src if r["id"] not in set(ids)]
buf.append(f"语料里还没预标的 id 数 = {len(missing)}  头几条 = {missing[:5]}")

bad_label = [(r["id"], lb) for r in rows for lb in r["labels"] if lb not in LABELS]
buf.append(f"不在 17 类里的标签 = {len(bad_label)} {bad_label[:5]}")

bad_ev = []
for r in rows:
    q = clean(r["question"])
    for lb, ev in r["evidence"].items():
        if not (clean(ev) and clean(ev) in q):
            bad_ev.append((r["id"], lb, ev))
        if lb not in r["labels"]:
            bad_ev.append((r["id"], lb, "!!标签不在 labels 里"))
buf.append(f"独立复算:证据不是子串(或标签对不上)的 = {len(bad_ev)} 条 {bad_ev[:5]}")

# 三个读数:这里用产物自己的列**独立复算**,与脚本打印的数对账
parse_failed = [r for r in rows if r["parse_failed"]]
flagged = [r for r in rows if r["rejected_labels"]]
zero = [r for r in rows if not r["labels"]]
buf.append(f"独立复算读数:parse_failed={len(parse_failed)}  rejected 非空={len(flagged)}  空标签={len(zero)}")
buf.append(f"  parse_failed 为真的 id(前 10)= {[r['id'] for r in parse_failed[:10]]}")
buf.append(f"  rejected 非空的 id(前 10)= {[(r['id'], r['rejected_labels']) for r in flagged[:10]]}")
buf.append(f"  空标签的 id 数 = {len(zero)};其中 parse_failed 的 = {sum(1 for r in zero if r['parse_failed'])}")

dist = Counter(lb for r in rows for lb in r["labels"])
multi = sum(1 for r in rows if len(r["labels"]) > 1)
buf.append(f"标签分布(共 {sum(dist.values())} 个标签,{multi} 条多标签 = {multi/max(1,len(rows)):.1%}):")
for lb in LABELS:
    buf.append(f"    {lb}: {dist.get(lb, 0)}")
extra = {k: v for k, v in dist.items() if k not in LABELS}
buf.append(f"  分布里的非法标签 = {extra}")
buf.append(f"form/provenance 分布 = {Counter(r.get('form', r.get('provenance')) for r in rows)}")
sys.stdout.buffer.write("\n".join(buf).encode("utf-8") + b"\n")
