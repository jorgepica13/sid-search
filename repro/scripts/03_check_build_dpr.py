"""Phase 1, step 2: verify build_dpr.py output (train_dpr.json).

Rebuilds the expected positives from train.qrels.tsv (file order) + corpus titles and
compares them sample by sample; checks negatives are 1 per query and never a positive.

Usage (from the sid-search root):
    python -I repro/scripts/03_check_build_dpr.py | tee repro/audit/03_build_dpr.txt
"""
import hashlib, json, os
from collections import Counter, defaultdict
import orjson

KS = "external/KuaiSearch"
P = f"{KS}/recall/data/train_dpr.json"

title = {}
with open(f"{KS}/data/corpus.jsonl", "rb") as f:
    for l in f:
        r = orjson.loads(l); title[r["item_id"]] = r["item_title"]
title2ids = defaultdict(list)
for i, t in title.items():
    title2ids[t].append(i)

queries = dict(l.rstrip("\n").split("\t", 1) for l in open(f"{KS}/recall/data/train.queries.tsv", encoding="utf-8"))
qrels = defaultdict(list)                      # qid -> [item_id] in file order
for l in open(f"{KS}/recall/data/train.qrels.tsv", encoding="utf-8"):
    qid, _, doc, _ = l.rstrip("\n").split("\t"); qrels[qid].append(int(doc))

size = os.path.getsize(P)
sha = hashlib.sha256(open(P, "rb").read()).hexdigest()
data = json.load(open(P, encoding="utf-8"))

print(f"file: {P}  {size:,} B  sha256 {sha}")
print(f"samples: {len(data):,}   expected (train queries): {len(queries):,}")
print(f"keys per sample: {dict(Counter(tuple(sorted(d)) for d in data))}")

ok_q = ok_pos = 0
n_pos = n_neg = 0
neg_count = Counter()
neg_is_pos_title = 0
neg_title_ambiguous = 0
for d, qid in zip(data, qrels):                 # same order as qrels file
    ok_q += d["query"] == queries[qid]
    exp = [{"text": title[i]} for i in qrels[qid]]
    ok_pos += d["positive_item"] == exp
    n_pos += len(d["positive_item"])
    neg_count[len(d["negative_item"])] += 1
    n_neg += len(d["negative_item"])
    pos_titles = {title[i] for i in qrels[qid]}
    for n in d["negative_item"]:
        if n["text"] in pos_titles:
            neg_is_pos_title += 1
        if len(title2ids[n["text"]]) > 1:
            neg_title_ambiguous += 1

print(f"query text matches train.queries.tsv (in qrels order): {ok_q:,}/{len(data):,}")
print(f"positive_item == titles of qrels items, same order:    {ok_pos:,}/{len(data):,}")
print(f"positive texts: {n_pos:,}   expected qrel pairs: {sum(map(len, qrels.values())):,}")
print(f"negatives per sample: {dict(neg_count)}   total {n_neg:,}")
print(f"negatives whose title equals a positive's title: {neg_is_pos_title:,}")
print(f"negatives whose title is shared by >1 corpus item: {neg_title_ambiguous:,}")
dup_titles = sum(1 for v in title2ids.values() if len(v) > 1)
print(f"corpus: {len(title):,} items, {len(title2ids):,} distinct titles, "
      f"{dup_titles:,} titles shared by >1 item ({sum(len(v) for v in title2ids.values() if len(v) > 1):,} items)")
print("\n3 samples:")
for d in data[:3]:
    print("  " + json.dumps({"query": d["query"],
                             "positive_item": d["positive_item"][:3] + ([f"... +{len(d['positive_item'])-3}"] if len(d["positive_item"]) > 3 else []),
                             "negative_item": d["negative_item"]}, ensure_ascii=False))
