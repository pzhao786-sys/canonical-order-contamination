"""Run the sharded likelihood comparison test (+ reversed-order diagnostic).

Example:
  python run_test.py --model gpt2-xl --files data/a.jsonl data/b.jsonl \
      --tag paper50 --num_shards 50 --perms 50 --seeds 0 1 2 --out results

One JSON per (model, tag, file, seed) is written to
  <out>/<model slug>/<tag>/<file stem>__seed<seed>.json
Existing files are skipped, so an interrupted sweep can simply be restarted.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

from contam.core import load_examples, run_file


def slug(name: str) -> str:
    return name.strip("/").replace("/", "__")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--files", nargs="+", required=True, help="paths or globs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", required=True, help="name of the protocol, e.g. paper50 or k10")
    ap.add_argument("--num_shards", type=int, default=50)
    ap.add_argument("--min_shard_size", type=int, default=0)
    ap.add_argument("--perms", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--max_examples", type=int, default=5000)
    ap.add_argument("--context_len", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=0)
    ap.add_argument("--dtype", choices=["fp32", "tf32", "fp16", "bf16"], default="tf32")
    ap.add_argument("--max_batch_tokens", type=int, default=4096)
    args = ap.parse_args()

    files = []
    for pat in args.files:
        hits = sorted(glob.glob(pat))
        files.extend(hits if hits else [pat])
    out_dir = os.path.join(args.out, slug(args.model), args.tag)
    os.makedirs(out_dir, exist_ok=True)

    todo = []
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        for seed in args.seeds:
            path = os.path.join(out_dir, f"{stem}__seed{seed}.json")
            if not os.path.exists(path):
                todo.append((f, seed, path))
    print(f"[run_test] model={args.model} tag={args.tag} files={len(files)} "
          f"seeds={args.seeds} todo={len(todo)}", flush=True)
    if not todo:
        return

    from contam.scoring import LMScorer  # imported late: needs torch + GPU
    scorer = LMScorer(args.model, dtype=args.dtype, context_len=args.context_len,
                      stride=args.stride or None, max_batch_tokens=args.max_batch_tokens)
    print(f"[run_test] loaded {args.model}: {scorer.n_params/1e6:.0f}M params, "
          f"context={scorer.context_len}, stride={scorer.stride}, dtype={args.dtype}", flush=True)

    fixed_cache = {}  # file -> canonical / reversed log-probs (seed independent)
    t_start = time.time()
    for n_done, (f, seed, path) in enumerate(todo, 1):
        examples = load_examples(f, args.max_examples)
        res = run_file(scorer, examples, num_shards=args.num_shards,
                       min_shard_size=args.min_shard_size, perms=args.perms, seed=seed,
                       fixed=fixed_cache.get(f))
        fixed_cache[f] = {"ll_canon": res["ll_canon"], "ll_rev": res["ll_rev"]}
        res.update({
            "model": args.model, "n_params": scorer.n_params, "file": f, "tag": args.tag,
            "context_len": scorer.context_len, "stride": scorer.stride, "dtype": args.dtype,
        })
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(res, fh)
        os.replace(tmp, path)
        s = res["summary"]
        print(f"[{n_done}/{len(todo)}] {os.path.basename(f)} seed={seed} n={res['n_examples']} "
              f"shards={s['num_shards']} p_canon={s['p_canon']:.3g} p_rev={s['p_rev']:.3g} "
              f"eff_canon={s['eff_canon_per_ex']:+.4f} eff_rev={s['eff_rev_per_ex']:+.4f} "
              f"tok/s={res['scored_tokens']/max(res['seconds'],1e-9):.0f} "
              f"elapsed={time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
