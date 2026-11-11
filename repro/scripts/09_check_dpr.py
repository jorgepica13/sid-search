"""Phase 2, method 2: audit the DPR-SDE run.

Re-encodes the test queries with the saved query encoder exactly as their test.py does
(max_length 64, padding max_length, CLS), runs an exact inner-product top-100 search on GPU
over their saved doc embeddings (doc_embedding/corpus_shard_0.npy), recomputes R@k / HR@k,
saves rankings, and measures ties (identical titles -> identical doc embeddings).

Usage (from the sid-search root):
    python -I repro/scripts/09_check_dpr.py [MODEL_DIR EMB_DIR OUT_DIR JOB_LOG] | tee repro/audit/09_dpr_sde.txt
    defaults = run 1 (batch 64). Paths relative to external/KuaiSearch except OUT_DIR and JOB_LOG.
"""
import json, os, re, sys, time
from collections import defaultdict
import numpy as np
import torch
from transformers import BertModel, BertTokenizer

KS = "external/KuaiSearch"
A = sys.argv[1:] + [None] * 4
MODEL_DIR = A[0] or "model_signle/best"
EMB_DIR = A[1] or "doc_embedding"
OUT = A[2] or "data/runs/dpr_sde"
LOG = A[3]
PAPER = {"recall@10": 0.0826, "hr@10": 0.1210, "recall@20": 0.1293, "hr@20": 0.1814,
         "recall@50": 0.2079, "hr@50": 0.2769}
THEIRS = {"recall@10": 0.0753, "hr@10": 0.1101, "recall@20": 0.1153, "hr@20": 0.1625,
          "recall@50": 0.1840, "hr@50": 0.2504, "recall@100": 0.2542, "hr@100": 0.3328}
if LOG:  # their printed metrics from the job log
    txt = open(LOG, encoding="utf-8", errors="replace").read()
    THEIRS = {k: float(v) for k, v in re.findall(r"^((?:recall|hr)@\d+): ([\d.]+)$", txt, re.M)}
CUTS, K = [10, 20, 50, 100], 100
torch.backends.cuda.matmul.allow_tf32 = False

queries = {}
for l in open(f"{KS}/recall/data/test.queries.tsv", encoding="utf-8"):
    p = l.rstrip("\n").split("\t")
    if len(p) >= 2:
        queries[p[0]] = p[1].replace("’", "'")
qrels = defaultdict(set)
for l in open(f"{KS}/recall/data/test.qrels.tsv", encoding="utf-8"):
    qid, _, doc, rel = l.rstrip("\n").split("\t")
    if int(rel) > 0:
        qrels[qid].add(doc)
qids = list(queries)

enc_dir = f"{KS}/{MODEL_DIR}/query_encoder"
print(f"model {enc_dir}  embeddings {KS}/{EMB_DIR}  out {OUT}")
tok = BertTokenizer.from_pretrained(enc_dir)
qe = BertModel.from_pretrained(enc_dir, add_pooling_layer=False).cuda().eval()
Q = []
with torch.no_grad():
    for i in range(0, len(qids), 512):
        x = tok([queries[q] for q in qids[i:i + 512]], max_length=64, truncation=True,
                padding="max_length", return_tensors="pt").to("cuda")
        Q.append(qe(**x).last_hidden_state[:, 0, :])
Q = torch.cat(Q)                                  # [8468, 768] fp32
del qe; torch.cuda.empty_cache()

t = time.time()
E = np.load(f"{KS}/{EMB_DIR}/corpus_shard_0.npy", mmap_mode="r")
ids = np.load(f"{KS}/{EMB_DIR}/corpus_shard_0_ids.npy")
print(f"doc embeddings {E.shape} {E.dtype}, ids {ids.shape}, first ids {ids[:3].tolist()}")
best_s = torch.full((len(qids), K), -float("inf"), device="cuda")
best_i = torch.zeros((len(qids), K), dtype=torch.long, device="cuda")
CH = 500_000
for s in range(0, E.shape[0], CH):
    D = torch.from_numpy(np.ascontiguousarray(E[s:s + CH])).cuda()
    sc = Q @ D.T
    v, i = sc.topk(K, dim=1)
    cs, ci = torch.cat([best_s, v], 1), torch.cat([best_i, i + s], 1)
    best_s, o = cs.topk(K, dim=1)
    best_i = ci.gather(1, o)
    del D, sc
print(f"exact GPU search: {time.time() - t:.0f} s")
S, I = best_s.cpu().numpy(), best_i.cpu().numpy()

m, lo, hi, ties = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(int)
os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/rankings.jsonl", "w") as f:
    for qi, qid in enumerate(qids):
        docs = [str(ids[j]) for j in I[qi]]
        sc = S[qi]
        f.write(json.dumps({"qid": qid, "doc_ids": docs, "scores": [float(x) for x in sc]}) + "\n")
        rel = qrels[qid]
        for k in CUTS:
            hit = len(set(docs[:k]) & rel)
            m[f"recall@{k}"].append(hit / len(rel)); m[f"hr@{k}"].append(float(hit > 0))
            if k < K and sc[k - 1] == sc[k]:
                ties[k] += 1
                v = sc[k - 1]; above = int((sc > v).sum()); slots = k - above
                block = [d for d, x in zip(docs, sc) if x == v]
                nr = sum(d in rel for d in block)
                base = len({d for d, x in zip(docs, sc) if x > v} & rel)
                lo[k].append((base + max(0, slots - (len(block) - nr))) / len(rel))
                hi[k].append((base + min(slots, nr)) / len(rel))
            else:
                lo[k].append(hit / len(rel)); hi[k].append(hit / len(rel))

print(f"\n{'metric':<12}{'recomp.':>9}{'theirs':>9}{'paper':>9}{'Δ vs paper':>12}   tie range")
res = {}
for k in CUTS:
    for name in (f"recall@{k}", f"hr@{k}"):
        v = float(np.mean(m[name])); res[name] = v; p = PAPER.get(name)
        d = f"{100 * (v - p) / p:+.1f}%" if p else ""
        rng = f"[{np.mean(lo[k]):.4f}, {np.mean(hi[k]):.4f}]" if name.startswith("recall") and k < K else ""
        print(f"{name:<12}{v:>9.4f}{THEIRS[name]:>9.4f}{(p or float('nan')):>9.4f}{d:>12}   {rng}")
for k in CUTS[:-1]:
    print(f"queries with an exact score tie across rank {k}/{k+1}: {ties[k]:,} ({100 * ties[k] / len(qids):.1f}%)")
json.dump({"metrics": res, "n_queries": len(qids)}, open(f"{OUT}/metrics.json", "w"), indent=2)

print("\n3 queries (qid | query | relevant | top-5 with score, * = relevant):")
for qi in range(3):
    qid = qids[qi]; rel = qrels[qid]
    top = ", ".join(f"{ids[j]}{'*' if str(ids[j]) in rel else ''}:{S[qi][r]:.2f}" for r, j in enumerate(I[qi][:5]))
    print(f"  {qid} | {queries[qid]} | {sorted(rel)} | {top}")
