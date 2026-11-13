"""Phase 2, generative methods: audit a DSI/LTRGR run from its saved best_model.

Re-runs their test decoding exactly (recall/GR/train.py: Trie-constrained beam search,
num_beams=50, num_return_sequences=50, max_new_tokens=30, test batch 32), saves the 50
decoded codes per test record, maps them back to item_ids and recomputes R@k / HR@k
independently. Also checks validity of decoded codes and duplicates within a ranking.

Usage (from the sid-search root):
    python -I repro/scripts/11_check_gr.py DSI/best_model data/runs/dsi | tee repro/audit/11_dsi.txt
"""
import json, os, sys, time
from collections import Counter
import numpy as np
import torch

KS = "external/KuaiSearch"
os.chdir(KS)
sys.path.insert(0, "recall/GR")
from transformers import T5Tokenizer, MT5ForConditionalGeneration  # noqa: E402
from train import Trie, TrieConstraintLogitsProcessor              # noqa: E402
from dataset import IndexingTrainDataset, QueryEvalCollator        # noqa: E402
from torch.utils.data import DataLoader                            # noqa: E402

MODEL, OUT = sys.argv[1], os.path.join("../..", sys.argv[2])
PAPER = {"DSI/best_model": {"Recall@10": 0.0623, "Hits@10": 0.0965, "Recall@20": 0.0892, "Hits@20": 0.1344,
                            "Recall@50": 0.1369, "Hits@50": 0.2018},
         "LTRGR/best_model": {"Recall@10": 0.0688, "Hits@10": 0.1049, "Recall@20": 0.0986, "Hits@20": 0.1477,
                              "Recall@50": 0.1501, "Hits@50": 0.2184}}.get(MODEL, {})
theirs = json.load(open(f"{MODEL}/test_metrics.json"))

tok = T5Tokenizer.from_pretrained(MODEL)
codes = torch.load("recall/data/embeddings/item_code.pt", map_location="cpu").numpy()
code_str = [" ".join(map(str, r)) for r in codes]
ids = []
with open("data/corpus.jsonl", encoding="utf-8") as f:
    for l in f:
        ids.append(json.loads(l)["item_id"])
code2item = dict(zip(code_str, ids))
t = time.time(); trie = Trie()
for s in set(code_str):
    trie.insert(tok.encode(s, add_special_tokens=False))
print(f"trie {len(code2item):,} codes ({time.time()-t:.0f} s)")

test = IndexingTrainDataset(path="recall/data/query_item_code_test.json", corpus_path="recall/data/item_text_codes.json",
                            max_length=32, tokenizer=tok, flag="test")
model = MT5ForConditionalGeneration.from_pretrained(MODEL).cuda().eval()
dl = DataLoader(test, batch_size=32, collate_fn=QueryEvalCollator(tokenizer=tok, padding="longest"),
                shuffle=False, drop_last=False, num_workers=1)
lp = [TrieConstraintLogitsProcessor(trie, tok)]
K_LIST = [10, 20, 50]
m = {f"{n}@{k}": 0.0 for k in K_LIST for n in ("Recall", "Hits")}
invalid = dup_rank = 0
os.makedirs(OUT, exist_ok=True)
t = time.time(); n = 0
with open(f"{OUT}/rankings.jsonl", "w") as fo:
    for inputs, labels in dl:
        with torch.no_grad():
            out = model.generate(inputs["input_ids"].cuda(), max_new_tokens=30, num_beams=50, logits_processor=lp,
                                 num_return_sequences=50, do_sample=False, early_stopping=True)
        out = out.reshape(inputs["input_ids"].shape[0], -1, out.shape[-1])
        for beams, label in zip(out, labels):
            rank = tok.batch_decode(beams, skip_special_tokens=True)
            invalid += sum(r not in code2item for r in rank)
            dup_rank += len(rank) - len(set(rank))
            rel = set(map(str, label))
            for k in K_LIST:
                hit = sum(r in rel for r in rank[:k])
                m[f"Recall@{k}"] += hit / len(rel); m[f"Hits@{k}"] += float(hit > 0)
            fo.write(json.dumps({"text": test.data[n]["text"], "relevant": sorted(rel), "codes": rank,
                                 "item_ids": [code2item.get(r) for r in rank]}, ensure_ascii=False) + "\n")
            n += 1
print(f"decoded {n:,} test records in {time.time()-t:.0f} s; invalid codes {invalid}; duplicate codes within a ranking {dup_rank}")
print(f"\n{'metric':<11}{'recomp.':>9}{'theirs':>9}{'paper':>9}{'Δ theirs vs paper':>19}")
for k in K_LIST:
    for name in (f"Recall@{k}", f"Hits@{k}"):
        v = m[name] / n; p = PAPER.get(name)
        d = f"{100 * (theirs[name] - p) / p:+.1f}%" if p else ""
        print(f"{name:<11}{v:>9.4f}{theirs[name]:>9.4f}{(p or float('nan')):>9.4f}{d:>19}")
json.dump({k: v / n for k, v in m.items()}, open(f"{OUT}/metrics.json", "w"), indent=2)

print("\n3 records (query | relevant codes | top-5 decoded, * = relevant):")
for l in list(open(f"{OUT}/rankings.jsonl", encoding="utf-8"))[:3]:
    r = json.loads(l); rel = set(r["relevant"])
    print(f"  {r['text']} | {r['relevant']} | " + ", ".join(c + ("*" if c in rel else "") for c in r["codes"][:5]))
