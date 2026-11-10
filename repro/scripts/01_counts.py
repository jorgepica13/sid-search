"""Phase 1, step 0: read-only audit of KuaiSearch Lite raw tables.

Checks the counts against the paper (arXiv 2602.11518) and the assumptions the
KuaiSearch recall pipeline makes implicitly (unique ids, qrel items present in the
corpus, no empty titles, split field values, sessions dropped for lack of interaction).

Usage (from the sid-search root):
    python -I repro/scripts/01_counts.py data/raw | tee repro/audit/01_counts.txt
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import orjson

PAPER = {"users": 102_086, "items": 6_634_118, "queries": 555_553}


def rows(path):
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                yield orjson.loads(line)


def show(label, value, paper=None):
    extra = ""
    if paper is not None:
        extra = f"   paper {paper:,}  {'OK' if value == paper else f'DIFF {value - paper:+,}'}"
    print(f"  {label:<52s} {value:>12,}{extra}")


def pct(a, b):
    return f"{100 * a / b:.2f}%" if b else "n/a"


def sample(rs, n=3):
    for r in rs[:n]:
        r = {k: (f"<list len={len(v)}: {v[:5]}{'...' if len(v) > 5 else ''}>" if isinstance(v, list) else v)
             for k, v in r.items()}
        print("   ", orjson.dumps(r).decode())


def main(raw):
    raw = Path(raw)

    # ------------------------------------------------------------------ items
    print("=== items_lite")
    ids, title_len, first = [], [], []
    empty_title = 0
    uniq = {k: set() for k in ["brand_id", "seller_id", "category_level1_id",
                               "category_level2_id", "category_level3_id"]}
    unknown = Counter()
    for i, r in enumerate(rows(raw / "items_lite/train.jsonl")):
        if i < 3:
            first.append(r)
        ids.append(r["item_id"])
        t = r.get("item_title") or ""
        if not t:
            empty_title += 1
        title_len.append(len(t))
        for k in uniq:
            uniq[k].add(r.get(k))
        for lvl in (1, 2, 3):
            if r.get(f"category_level{lvl}_name") == "UNKNOWN":
                unknown[lvl] += 1
    ids = np.asarray(ids, dtype=np.int64)
    n_items = len(ids)
    item_set = set(ids.tolist())
    show("rows", n_items, PAPER["items"])
    show("unique item_id", len(item_set))
    print(f"  item_id range                                        [{ids.min():,}, {ids.max():,}]")
    print(f"  item_id sorted ascending in file                     {bool(np.all(ids[1:] > ids[:-1]))}")
    show("empty item_title (pipeline skips these!)", empty_title)
    tl = np.asarray(title_len)
    print(f"  title length (chars) mean / p50 / p99 / max          "
          f"{tl.mean():.1f} / {np.percentile(tl, 50):.0f} / {np.percentile(tl, 99):.0f} / {tl.max()}")
    for k, s in uniq.items():
        show(f"unique {k}", len(s))
    for lvl in (1, 2, 3):
        print(f"  category_level{lvl}_name == UNKNOWN                    "
              f"{unknown[lvl]:>12,}  ({pct(unknown[lvl], n_items)})")
    print("  first 3 rows:")
    sample(first)
    del ids, tl, title_len

    # ------------------------------------------------------------------ users
    print("\n=== users_lite")
    users, first = [], []
    for i, r in enumerate(rows(raw / "users_lite/train.jsonl")):
        if i < 3:
            first.append(r)
        users.append(r["user_id"])
    user_set = set(users)
    show("rows", len(users), PAPER["users"])
    show("unique user_id", len(user_set))
    print("  first 3 rows:")
    sample(first)

    # ----------------------------------------------------------------- recall
    print("\n=== recall_lite")
    n = 0
    sessions, rec_users = set(), set()
    dup_session = 0
    split = Counter()
    kept = Counter()             # sessions with >=1 click or purchase (what process.py keeps)
    queries = {}                 # split -> set of query strings (kept sessions)
    rel_sizes = {}               # split -> list of |clicked U purchased|
    missing_rel, total_rel = 0, 0
    missing_imp, total_imp = 0, 0
    purch_not_clicked, n_purch = 0, 0
    empty_query = 0
    t_range = {}
    first = []
    for r in rows(raw / "recall_lite/train.jsonl"):
        n += 1
        if n <= 3:
            first.append(r)
        sid = r["session_id"]
        if sid in sessions:
            dup_session += 1
        sessions.add(sid)
        rec_users.add(r["user_id"])
        s = r.get("split")
        split[s] += 1
        q = r.get("query") or ""
        if not q:
            empty_query += 1
        t = r.get("time_index")
        lo, hi = t_range.get(s, (t, t))
        t_range[s] = (min(lo, t), max(hi, t))
        clicked = r.get("clicked_item_ids") or []
        purchased = r.get("purchased_item_ids") or []
        imp = r.get("impressed_item_ids") or []
        total_imp += len(imp)
        missing_imp += sum(1 for x in imp if x not in item_set)
        cs = set(clicked)
        n_purch += len(purchased)
        purch_not_clicked += sum(1 for x in purchased if x not in cs)
        rel = cs | set(purchased)
        if not rel:
            continue
        kept[s] += 1
        queries.setdefault(s, set()).add(q)
        rel_sizes.setdefault(s, []).append(len(rel))
        total_rel += len(rel)
        missing_rel += sum(1 for x in rel if x not in item_set)

    show("rows (= sessions)", n, PAPER["queries"])
    show("unique session_id", len(sessions))
    show("duplicate session_id rows", dup_session)
    show("unique user_id", len(rec_users))
    show("user_id not in users_lite", len(rec_users - user_set))
    show("users_lite users with no recall session", len(user_set - rec_users))
    show("rows with empty query", empty_query)
    print(f"  split values                                         {dict(split)}")
    for s in sorted(split, key=str):
        print(f"  [{s}]")
        show("  rows", split[s])
        show("  kept by process.py (>=1 click or purchase)", kept[s])
        show("  dropped (no interaction)", split[s] - kept[s])
        print(f"    kept share                                         {pct(kept[s], split[s])}")
        if s in queries:
            show("  unique query strings (kept)", len(queries[s]))
            rs = np.asarray(rel_sizes[s])
            print(f"    relevant items/session mean / p50 / p99 / max      "
                  f"{rs.mean():.2f} / {np.percentile(rs, 50):.0f} / {np.percentile(rs, 99):.0f} / {rs.max()}")
            show("  qrel pairs (session, item)", int(rs.sum()))
        lo, hi = t_range[s]
        print(f"    time_index range                                   [{lo:,}, {hi:,}]")
    if "train" in queries and "test" in queries:
        inter = queries["train"] & queries["test"]
        show("test query strings also seen in train", len(inter))
        print(f"    share of unique test queries                       {pct(len(inter), len(queries['test']))}")
    show("relevant items NOT in items_lite", missing_rel)
    print(f"    of all relevant (session,item) pairs               {pct(missing_rel, total_rel)}")
    show("impressed items NOT in items_lite", missing_imp)
    print(f"    of all impressions                                 {pct(missing_imp, total_imp)}")
    show("purchased items not in clicked", purch_not_clicked)
    print(f"    of all purchases                                   {pct(purch_not_clicked, n_purch)}")
    print("  first 3 rows:")
    sample(first)

    # -------------------------------------------------------------- relevance
    print("\n=== relevance")
    n, split, score, first = 0, Counter(), Counter(), []
    for r in rows(raw / "relevance/train.jsonl"):
        n += 1
        if n <= 3:
            first.append(r)
        split[r.get("split")] += 1
        score[(r.get("split"), r.get("score"))] += 1
    show("rows", n)
    print(f"  split values                                         {dict(split)}")
    print(f"  (split, score) counts                                {dict(sorted(score.items(), key=str))}")
    print("  first 3 rows:")
    sample(first)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
