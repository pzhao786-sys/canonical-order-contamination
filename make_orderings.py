"""Build control orderings of a benchmark file with FRESH randomness.

Because the order is generated here, no model can have seen it during training:
any preference a model shows for it cannot be memorised example order.

  shuffled : uniformly random order            -> exchangeable (null control)
  sorted   : examples grouped by topic         -> non-exchangeable (structure control)
             (TF-IDF + k-means clusters; random cluster order, random order
              inside each cluster, so the structure has no direction)

Example:
  python make_orderings.py --src data/arc_easy.jsonl --out data/controls \
      --n 2000 --cluster_size 20 --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import os
import random

from contam.core import load_examples


def topic_sorted(examples, cluster_size: int, seed: int):
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    rng = random.Random(seed)
    k = max(2, len(examples) // cluster_size)
    X = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2).fit_transform(examples)
    labels = KMeans(n_clusters=k, n_init=3, random_state=seed).fit_predict(X)
    clusters = {}
    for idx, lab in enumerate(labels):
        clusters.setdefault(int(lab), []).append(idx)
    order = []
    keys = list(clusters)
    rng.shuffle(keys)
    for key in keys:
        members = clusters[key]
        rng.shuffle(members)
        order.extend(members)
    return [examples[i] for i in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000, help="subsample size (0 = all)")
    ap.add_argument("--cluster_size", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.src))[0]
    examples = load_examples(args.src, max_examples=10**9)
    examples = [e if e.endswith("\n") else e + "\n" for e in examples if e.strip()]
    examples = list(dict.fromkeys(examples))  # drop exact duplicates
    for seed in args.seeds:
        rng = random.Random(1000 + seed)
        pool = examples[:]
        rng.shuffle(pool)
        if args.n:
            pool = pool[:args.n]
        with open(os.path.join(args.out, f"{stem}.shuffled{seed}.txt"), "w", encoding="utf-8") as f:
            f.writelines(pool)
        with open(os.path.join(args.out, f"{stem}.sorted{seed}.txt"), "w", encoding="utf-8") as f:
            f.writelines(topic_sorted(pool, args.cluster_size, seed))
        print(f"wrote {stem} seed {seed}: {len(pool)} examples")


if __name__ == "__main__":
    main()
