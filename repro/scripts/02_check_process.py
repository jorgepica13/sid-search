"""Phase 1, step 1: verify process.py outputs against recall_lite, set by set.

Usage (from the sid-search root):
    python -I repro/scripts/02_check_process.py | tee repro/audit/02_process.txt
"""
from collections import defaultdict
import orjson

KS = "external/KuaiSearch/recall/data"
RAW = "data/raw/recall_lite/train.jsonl"

def read_queries(p):
    return dict(l.rstrip("\n").split("\t", 1) for l in open(p, encoding="utf-8"))

def read_qrels(p):
    d = defaultdict(set)
    for l in open(p, encoding="utf-8"):
        qid, _, doc, rel = l.rstrip("\n").split("\t")
        d[qid].add(int(doc))
    return d

out = {s: (read_queries(f"{KS}/{s}.queries.tsv"), read_qrels(f"{KS}/{s}.qrels.tsv")) for s in ("train", "test")}

exp = {"train": ({}, {}), "test": ({}, {})}
for l in open(RAW, "rb"):
    r = orjson.loads(l)
    rel = set(r["clicked_item_ids"] or []) | set(r["purchased_item_ids"] or [])
    if rel:
        q, qr = exp[r["split"]]
        q[str(r["session_id"])] = r["query"]
        qr[str(r["session_id"])] = rel

for s in ("train", "test"):
    (oq, orl), (eq, erl) = out[s], exp[s]
    print(f"=== {s}")
    print(f"  queries: output {len(oq):,}  expected {len(eq):,}  same qids {oq.keys() == eq.keys()}  "
          f"same text {sum(oq[k] == eq[k] for k in eq):,}/{len(eq):,}")
    print(f"  qrels:   output {sum(map(len, orl.values())):,} pairs  expected {sum(map(len, erl.values())):,}  "
          f"identical sets {sum(orl.get(k) == v for k, v in erl.items()):,}/{len(erl):,}")
    print("  3 rows (qid | query | relevant item_ids, sorted):")
    for k in list(oq)[:3]:
        print(f"    {k} | {oq[k]} | {sorted(orl[k])}")
