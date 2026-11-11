"""Phase 2, method 1: audit the BM25 run (data/runs/bm25).

- Recomputes R@k / HR@k independently from rankings.jsonl + test.qrels.tsv
- Compares with the paper and with their printed metrics
- Measures ties at the cutoff (score at rank k == score at rank k+1) and the R@k range
  that tie-breaking alone can produce *within the retrieved top-100*.

Usage (from the sid-search root):
    python -I repro/scripts/08_check_bm25.py | tee repro/audit/08_bm25.txt
"""
import json
from collections import defaultdict
import numpy as np
import orjson

RUN = "data/runs/bm25"
KS = "external/KuaiSearch"
PAPER = {"recall@10": 0.0706, "hr@10": 0.1001, "recall@20": 0.1037, "hr@20": 0.1427,
         "recall@50": 0.1564, "hr@50": 0.2088}
CUTS = [10, 20, 50, 100]

qrels = defaultdict(set)
for l in open(f"{KS}/recall/data/test.qrels.tsv", encoding="utf-8"):
    qid, _, doc, rel = l.rstrip("\n").split("\t")
    if int(rel) > 0:
        qrels[qid].add(doc)
queries = dict(l.rstrip("\n").split("\t", 1) for l in open(f"{KS}/recall/data/test.queries.tsv", encoding="utf-8"))
runs = [orjson.loads(l) for l in open(f"{RUN}/rankings.jsonl", "rb")]
theirs = json.load(open(f"{RUN}/metrics.json"))

print(f"rankings: {len(runs):,} queries   qrels: {len(qrels):,}   same qid set: {set(r['qid'] for r in runs) == set(qrels)}")
lens = np.array([len(r["doc_ids"]) for r in runs])
print(f"results per query: 0 -> {(lens == 0).sum():,}   <100 -> {(lens < 100).sum():,}   =100 -> {(lens == 100).sum():,}")
print(f"scores sorted descending in every ranking: {all(all(a >= b for a, b in zip(r['scores'], r['scores'][1:])) for r in runs)}")

m = defaultdict(list)
lo, hi, tie_q = defaultdict(list), defaultdict(list), defaultdict(int)
for r in runs:
    rel, docs, sc = qrels[r["qid"]], r["doc_ids"], r["scores"]
    for k in CUTS:
        hit = len(set(docs[:k]) & rel)
        m[f"recall@{k}"].append(hit / len(rel)); m[f"hr@{k}"].append(float(hit > 0))
        # tie block straddling the cutoff, within the retrieved list
        if len(docs) > k and sc[k - 1] == sc[k]:
            tie_q[k] += 1
            s = sc[k - 1]
            block = [d for d, x in zip(docs, sc) if x == s]
            above = sum(1 for x in sc if x > s)
            slots = k - above
            n_rel_block = sum(1 for d in block if d in rel)
            base = len(set(d for d, x in zip(docs, sc) if x > s) & rel)
            lo[k].append((base + max(0, slots - (len(block) - n_rel_block))) / len(rel))
            hi[k].append((base + min(slots, n_rel_block)) / len(rel))
        else:
            lo[k].append(hit / len(rel)); hi[k].append(hit / len(rel))

print(f"\n{'metric':<12}{'ours':>9}{'theirs':>9}{'paper':>9}{'Δ rel':>9}   tie range within top-100")
for k in CUTS:
    for name in (f"recall@{k}", f"hr@{k}"):
        ours = float(np.mean(m[name])); th = theirs["metrics"][name]; p = PAPER.get(name)
        d = f"{100 * (ours - p) / p:+.1f}%" if p else ""
        rng = f"[{np.mean(lo[k]):.4f}, {np.mean(hi[k]):.4f}]" if name.startswith("recall") else ""
        print(f"{name:<12}{ours:>9.4f}{th:>9.4f}{(p or float('nan')):>9.4f}{d:>9}   {rng}")
for k in CUTS:
    print(f"queries with a score tie across rank {k}/{k+1}: {tie_q[k]:,} ({100 * tie_q[k] / len(runs):.1f}%)")

print("\n3 queries (qid | query | relevant | top-5 retrieved with score, * = relevant):")
for r in runs[:3]:
    rel = qrels[r["qid"]]
    top = ", ".join(f"{d}{'*' if d in rel else ''}:{s:.2f}" for d, s in zip(r["doc_ids"][:5], r["scores"][:5]))
    print(f"  {r['qid']} | {queries[r['qid']]} | {sorted(rel)} | {top}")
