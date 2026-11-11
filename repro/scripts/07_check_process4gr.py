"""Phase 1, step 6: verify process4gr.py outputs.

- item_text_codes.json       [{"text": title_i, "output": "c0 c1 c2 dup"}] in corpus order
- query_item_code_train.json [{"text": query, "output": code}] one per train qrel line (file order)
- query_item_code_test.json  [{"text": query, "output": [codes]}] one per test qid (qrels order)

Usage (from the sid-search root):
    python -I repro/scripts/07_check_process4gr.py | tee repro/audit/07_process4gr.txt
"""
import hashlib, json, os
from collections import Counter, defaultdict
import orjson
import torch

KS = "external/KuaiSearch"
D = f"{KS}/recall/data"

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()

ic = torch.load(f"{D}/embeddings/item_code.pt", map_location="cpu").numpy()
code = [f"{a} {b} {c} {d}" for a, b, c, d in ic.tolist()]
ids, titles = [], []
with open(f"{KS}/data/corpus.jsonl", "rb") as f:
    for l in f:
        r = orjson.loads(l); ids.append(r["item_id"]); titles.append(r["item_title"])
id2row = {i: k for k, i in enumerate(ids)}
code2row = {c: k for k, c in enumerate(code)}
print(f"code strings unique (code -> item is a bijection): {len(code2row) == len(code)}")

for name in ("item_text_codes.json", "query_item_code_train.json", "query_item_code_test.json"):
    p = f"{D}/{name}"
    print(f"file {name}: {os.path.getsize(p):,} B  sha256 {sha(p)}")

# items
items = json.load(open(f"{D}/item_text_codes.json", encoding="utf-8"))
ok = sum(1 for k, r in enumerate(items) if r["text"] == titles[k] and r["output"] == code[k])
print(f"\n=== item_text_codes.json: {len(items):,} records (corpus {len(titles):,})  "
      f"text==title_i and output==item_code[i]: {ok:,}/{len(titles):,}")
print(f"  keys {dict(Counter(tuple(sorted(r)) for r in items))}   (no item_id field: row order is the link)")

def queries(split):
    return dict(l.rstrip("\n").split("\t", 1) for l in open(f"{D}/{split}.queries.tsv", encoding="utf-8"))
def qrels(split):
    out = []
    for l in open(f"{D}/{split}.qrels.tsv", encoding="utf-8"):
        qid, _, doc, _ = l.rstrip("\n").split("\t"); out.append((qid, int(doc)))
    return out

# train
q, qr = queries("train"), qrels("train")
tr = json.load(open(f"{D}/query_item_code_train.json", encoding="utf-8"))
exp = [{"text": q[qid], "output": code[id2row[doc]]} for qid, doc in qr]
print(f"\n=== query_item_code_train.json: {len(tr):,} records (qrel lines {len(qr):,})  "
      f"identical to expected, in order: {sum(a == b for a, b in zip(tr, exp)):,}/{len(exp):,}")
back = sum(1 for (qid, doc), r in zip(qr, tr) if ids[code2row[r['output']]] == doc)
print(f"  output code maps back to the qrel item_id: {back:,}/{len(qr):,}")
pairs = Counter((r["text"], r["output"]) for r in tr)
print(f"  distinct (query text, code) pairs: {len(pairs):,}   distinct query texts: {len({r['text'] for r in tr}):,}   "
      f"distinct codes: {len({r['output'] for r in tr}):,}")

# test
q, qr = queries("test"), qrels("test")
by = defaultdict(list)
for qid, doc in qr:
    by[qid].append(code[id2row[doc]])
te = json.load(open(f"{D}/query_item_code_test.json", encoding="utf-8"))
exp = [{"text": q[qid], "output": cs} for qid, cs in by.items()]
print(f"\n=== query_item_code_test.json: {len(te):,} records (test qids {len(by):,})  "
      f"identical to expected, in order: {sum(a == b for a, b in zip(te, exp)):,}/{len(exp):,}")
print(f"  codes total {sum(len(r['output']) for r in te):,} (qrel lines {len(qr):,})")
txt = Counter(r["text"] for r in te)
print(f"  distinct query texts: {len(txt):,}; texts appearing in >1 record: {sum(1 for c in txt.values() if c > 1):,} "
      f"({sum(c for c in txt.values() if c > 1):,} records)  <- no qid in the file: same text = separate records")
dup_hi = sum(1 for r in te for c in r["output"] if int(c.split()[3]) > 0)
print(f"  test relevant codes with dup > 0: {dup_hi:,}/{sum(len(r['output']) for r in te):,}")

print("\n3 records of each:")
for name, recs in (("item", items), ("train", tr), ("test", te)):
    for r in recs[:3]:
        print(f"  {name:5s} " + json.dumps(r, ensure_ascii=False))
