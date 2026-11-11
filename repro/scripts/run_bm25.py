"""Phase 2, method 1: BM25 — run KuaiSearch's own Evaluator and keep its rankings.

Calls recall/BM25/bm25.py:Evaluator.run_full_evaluation with exactly the arguments of
scripts/recall_bm25_eval.sh (same code path, no change to their code). Their script only
prints mean metrics; this wrapper additionally saves the top-k ranking of every query and
the metrics, so the run can be audited.

Must run from the external/KuaiSearch root (their relative paths). Set RETRIV_BASE_PATH to
keep the index out of ~/.retriv.

Usage:
    cd external/KuaiSearch
    RETRIV_BASE_PATH=../../data/retriv python ../../repro/scripts/run_bm25.py ../../data/runs/bm25
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "recall/BM25")          # their script dir: makes `bm25` and `BM25` importable
from bm25 import Evaluator                 # noqa: E402

# == scripts/recall_bm25_eval.sh
ARGS = dict(
    index_name="kuaisearch",
    corpus_file="data/corpus.jsonl",
    queries_file="recall/data/test.queries.tsv",
    qrels_file="recall/data/test.qrels.tsv",
    top_k=100,
    cutoffs=[10, 20, 50, 100],
    pseudo_query_file=None,
)

out = Path(sys.argv[1] if len(sys.argv) > 1 else "../../data/runs/bm25")
out.mkdir(parents=True, exist_ok=True)
print(f"RETRIV_BASE_PATH={os.environ.get('RETRIV_BASE_PATH')}  args={ARGS}  out={out}")

t0 = time.time()
metrics, results = Evaluator().run_full_evaluation(**ARGS)
elapsed = time.time() - t0

with open(out / "rankings.jsonl", "w", encoding="utf-8") as f:
    for qid, docs in results.items():
        f.write(json.dumps({"qid": qid,
                            "doc_ids": [d["doc_id"] for d in docs],
                            "scores": [float(d["score"]) for d in docs]}) + "\n")
with open(out / "metrics.json", "w", encoding="utf-8") as f:
    json.dump({"args": ARGS, "n_queries": len(results), "elapsed_s": round(elapsed, 1),
               "metrics": {k: float(v) for k, v in metrics.items()}}, f, indent=2)
print(f"saved {len(results)} rankings and metrics to {out}  ({elapsed:.0f} s)")
