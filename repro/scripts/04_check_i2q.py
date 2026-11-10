"""Phase 1, step 3: verify construct_i2q.py output (docTquery_train_data.jsonl).

Expected: one line per positive of train_dpr.json, in order:
  {"query": q, "item": PROMPT_PREFIX + positive title}

Usage (from the sid-search root):
    python -I repro/scripts/04_check_i2q.py | tee repro/audit/04_construct_i2q.txt
"""
import hashlib, json, os
from collections import Counter

KS = "external/KuaiSearch/recall/data"
PREFIX = "为下面的商品标题生成一个用户搜索query："
P = f"{KS}/docTquery_train_data.jsonl"

dpr = json.load(open(f"{KS}/train_dpr.json", encoding="utf-8"))
expected = [{"query": d["query"].strip(), "item": PREFIX + p["text"].strip()}
            for d in dpr for p in d["positive_item"] if p["text"].strip()]
got = [json.loads(l) for l in open(P, encoding="utf-8")]

print(f"file: {P}  {os.path.getsize(P):,} B  sha256 {hashlib.sha256(open(P,'rb').read()).hexdigest()}")
print(f"lines: {len(got):,}   expected (positives in train_dpr.json): {len(expected):,}")
print(f"identical line by line: {sum(a == b for a, b in zip(got, expected)):,}/{len(expected):,}")
print(f"keys: {dict(Counter(tuple(sorted(g)) for g in got))}")
print(f"all items start with prompt prefix: {all(g['item'].startswith(PREFIX) for g in got)}")
pairs = Counter((g["query"], g["item"]) for g in got)
print(f"distinct (query, item) pairs: {len(pairs):,}   duplicated pairs: {sum(1 for c in pairs.values() if c > 1):,} "
      f"({sum(c for c in pairs.values() if c > 1):,} lines)")
print(f"distinct queries: {len({g['query'] for g in got}):,}   distinct items: {len({g['item'] for g in got}):,}")
print("\n3 rows:")
for g in got[:3]:
    print("  " + json.dumps(g, ensure_ascii=False))
