"""Phase 1, step 5: verify faiss_kmeans_quantize.py outputs.

- codes_512[512, 512, 512]_2048.npy : [3, N] RQ-KMeans codes
- item_code.pt                       : [N, 4] = codes.T + dedup index
Recomputes the dedup index independently (occurrence number of each 3-code tuple in row
order) and reports collision rates, codebook usage and the relation with duplicate titles.

Usage (from the sid-search root):
    python -I repro/scripts/06_check_rqkmeans.py <job .out log> | tee repro/audit/06_rqkmeans.txt
"""
import hashlib, os, re, sys
from collections import Counter
import numpy as np
import orjson
import torch

KS = "external/KuaiSearch"
EMB = f"{KS}/recall/data/embeddings"
CODES = f"{EMB}/codes_512[512, 512, 512]_2048.npy"
PT = f"{EMB}/item_code.pt"

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()

if len(sys.argv) > 1:
    log = open(sys.argv[1], encoding="utf-8", errors="replace").read().replace("\r", "\n")
    objs = re.findall(r"Iteration (\d+) \([^)]*\): objective=([\d.e+]+)", log)
    levels, cur = [], []
    for it, ob in objs:
        if it == "0" and cur:
            levels.append(cur); cur = []
        cur.append(float(ob))
    levels.append(cur)
    print("=== faiss KMeans objective (sum of squared distances on the 1,048,576-point training sample)")
    for l, v in enumerate(levels):
        print(f"  level {l}: iter 0 {v[0]:,.0f} -> iter {len(v)-1} {v[-1]:,.0f}   ({len(v)} iterations)")

codes = np.load(CODES)
ic = torch.load(PT, map_location="cpu")
print(f"\n=== files")
print(f"  {CODES}  {os.path.getsize(CODES):,} B  sha256 {sha(CODES)}")
print(f"    shape {codes.shape} dtype {codes.dtype}")
print(f"  {PT}  {os.path.getsize(PT):,} B  sha256 {sha(PT)}")
print(f"    type {type(ic).__name__} shape {tuple(ic.shape)} dtype {ic.dtype} device {ic.device}")
ic = ic.numpy()
N = ic.shape[0]
print(f"  item_code[:, :3] == codes.T : {np.array_equal(ic[:, :3], codes.T)}")

print("\n=== codebook usage (codes in [0, 511])")
for l in range(3):
    c = np.bincount(codes[l], minlength=512)
    print(f"  level {l}: range [{codes[l].min()}, {codes[l].max()}]  used {np.count_nonzero(c)}/512  "
          f"items per code min {c.min():,} / p50 {int(np.median(c)):,} / max {c.max():,}")

# independent dedup: occurrence index of each (c0,c1,c2) in row order
key = (codes[0].astype(np.int64) << 18) | (codes[1].astype(np.int64) << 9) | codes[2].astype(np.int64)
order = np.argsort(key, kind="stable")
ks = key[order]
start = np.r_[True, ks[1:] != ks[:-1]]
grp_start = np.maximum.accumulate(np.where(start, np.arange(N), 0))
occ = np.empty(N, dtype=np.int64); occ[order] = np.arange(N) - grp_start
print("\n=== 4th code (dedup)")
print(f"  equals independent recomputation (occurrence # of the 3-code tuple, row order): {np.array_equal(ic[:, 3], occ)}")
d = ic[:, 3]
n_tuple = int(start.sum())
sizes = np.diff(np.r_[np.flatnonzero(start), N])
print(f"  distinct 3-code tuples: {n_tuple:,} of {N:,} items  ({100*n_tuple/N:.2f}%)")
print(f"  items with dedup > 0 (collide with an earlier item): {(d > 0).sum():,}  ({100*(d > 0).mean():.2f}%)  "
      f"<- duplicates_rate computed (not printed) by code_check()")
print(f"  items in a tuple shared by >1 item: {sizes[sizes > 1].sum():,}  ({100*sizes[sizes > 1].sum()/N:.2f}%)")
print(f"  max dedup value: {d.max():,}   (largest group size {sizes.max():,})")
print(f"  dedup value p50 / p90 / p99 / p99.9: {np.percentile(d, [50, 90, 99, 99.9]).astype(int).tolist()}")
print(f"  items with dedup >= 512 (would not fit a 512-token 4th level): {(d >= 512).sum():,}")
print(f"  group sizes: {dict(sorted(Counter(np.minimum(sizes, 10)).items()))}  (10 = 10 or more)")
print(f"  4-code tuples unique: {len(np.unique(np.c_[key, d], axis=0)) == N}")

ids, titles = [], []
with open(f"{KS}/data/corpus.jsonl", "rb") as f:
    for l in f:
        r = orjson.loads(l); ids.append(r["item_id"]); titles.append(r["item_title"])
t_first = {}
same_title_earlier = np.zeros(N, dtype=bool)
for i, t in enumerate(titles):
    if t in t_first:
        same_title_earlier[i] = True
    else:
        t_first[t] = i
print("\n=== relation with duplicate titles")
print(f"  items whose title already appeared earlier: {same_title_earlier.sum():,}  ({100*same_title_earlier.mean():.2f}%)")
print(f"    of those, dedup > 0: {(d[same_title_earlier] > 0).sum():,}  ({100*(d[same_title_earlier] > 0).mean():.2f}%)")
print(f"  collisions (dedup > 0) NOT explained by a repeated title: {((d > 0) & ~same_title_earlier).sum():,}  "
      f"({100*((d > 0) & ~same_title_earlier).sum()/N:.2f}% of items)")

big = order[grp_start[np.argmax(np.diff(np.r_[np.flatnonzero(start), N]) == sizes.max())] if False else 0]
gi = np.flatnonzero(start)[np.argmax(sizes)]
members = np.sort(order[gi:gi + sizes.max()])
print(f"\n  largest group: code {codes[:, members[0]].tolist()}  {sizes.max():,} items; 5 titles:")
for i in members[:5]:
    print(f"    row {i} | item {ids[i]} | {titles[i]}")

print("\n=== 3 rows (row | item_id | title | item_code)")
for i in range(3):
    print(f"  {i} | {ids[i]} | {titles[i]} | {ic[i].tolist()}")
