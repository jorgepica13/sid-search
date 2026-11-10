"""Phase 1, step 4: verify title_embedding_prepare.py output (item_emb_512.npy).

Row i must be the embedding of line i of data/corpus.jsonl (= items_lite, file order).
Checks shape/dtype/norms/NaN, re-encodes a random sample independently, and checks that
items with identical titles got identical vectors.

Usage (from the sid-search root):
    python -I repro/scripts/05_check_embeddings.py | tee repro/audit/05_title_embedding.txt
"""
import hashlib, os
from collections import defaultdict
import numpy as np
import orjson
import torch
from sentence_transformers import SentenceTransformer

KS = "external/KuaiSearch"
P = f"{KS}/recall/data/embeddings/item_emb_512.npy"

ids, titles = [], []
with open(f"{KS}/data/corpus.jsonl", "rb") as f:
    for l in f:
        r = orjson.loads(l); ids.append(r["item_id"]); titles.append(r["item_title"])

h = hashlib.sha256()
with open(P, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 24), b""):
        h.update(chunk)
E = np.load(P, mmap_mode="r")
print(f"file: {P}  {os.path.getsize(P):,} B  sha256 {h.hexdigest()}")
print(f"shape {E.shape}  dtype {E.dtype}   corpus lines {len(titles):,}   match {E.shape[0] == len(titles)}")

norms, nan = [], 0
for s in range(0, E.shape[0], 500_000):
    b = np.asarray(E[s:s + 500_000], dtype=np.float32)
    nan += int(np.isnan(b).any(axis=1).sum())
    norms.append(np.linalg.norm(b, axis=1))
norms = np.concatenate(norms)
print(f"rows with NaN: {nan}")
print(f"L2 norm min / mean / max: {norms.min():.6f} / {norms.mean():.6f} / {norms.max():.6f}")

rng = np.random.default_rng(0)
idx = np.sort(np.concatenate([[0, 1, 2], rng.choice(len(titles), 2000, replace=False)]))
m = SentenceTransformer("BAAI/bge-small-zh-v1.5", device="cuda:0")
ref = m.encode([titles[i] for i in idx], batch_size=256, convert_to_numpy=True)
got = np.asarray(E[idx], dtype=np.float32)
cos = (ref * got).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(got, axis=1))
print(f"\nindependent re-encode of {len(idx)} rows (seed 0): cosine min {cos.min():.6f}  mean {cos.mean():.6f}  "
      f"max |diff| {np.abs(ref - got).max():.2e}")
# misalignment control: compare each row against the NEXT title's embedding
cos_shift = (ref[:-1] * got[1:]).sum(1) / (np.linalg.norm(ref[:-1], axis=1) * np.linalg.norm(got[1:], axis=1))
print(f"control, row i vs title of a different sampled row: cosine mean {cos_shift.mean():.4f}")

t2i = defaultdict(list)
for i, t in enumerate(titles):
    t2i[t].append(i)
groups = [v for v in t2i.values() if len(v) > 1]
sample = [groups[j] for j in rng.choice(len(groups), 1000, replace=False)]
maxdiff = max(float(np.abs(np.asarray(E[g], dtype=np.float32) - np.asarray(E[g[0]], dtype=np.float32)).max()) for g in sample)
print(f"identical-title groups checked: 1000 (of {len(groups):,})   max |diff| within group: {maxdiff:.2e}")

print("\n3 rows (row | item_id | title | first 6 dims):")
for i in range(3):
    print(f"  {i} | {ids[i]} | {titles[i]} | {np.round(np.asarray(E[i][:6]), 4).tolist()}")
