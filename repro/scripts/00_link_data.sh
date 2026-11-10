#!/usr/bin/env bash
# Link KuaiSearch Lite raw tables to the paths the KuaiSearch recall scripts expect
# (data/{corpus,recall,relevance}.jsonl relative to external/KuaiSearch).
# No copy, no change to their code: data/ is gitignored in the fork (patch 1).
# corpus.jsonl = items_lite as-is: it has item_id/item_title, no empty titles, and
# contains every relevant/impressed item of recall_lite (repro/audit/01_counts.txt).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$REPO/external/KuaiSearch/data"
cd "$REPO/external/KuaiSearch/data"
ln -sfn ../../../data/raw/items_lite/train.jsonl  corpus.jsonl
ln -sfn ../../../data/raw/recall_lite/train.jsonl recall.jsonl
ln -sfn ../../../data/raw/relevance/train.jsonl   relevance.jsonl
ls -l
