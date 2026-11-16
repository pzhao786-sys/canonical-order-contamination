"""Aggregate result JSONs into tables (CSV) for the report.

  python analyze.py --results results --out results/summary

Writes:
  runs.csv        one row per (model, tag, file, seed)
  files.csv       seeds aggregated: median p, mean effect, seed-to-seed spread
  null_calib.csv  false-positive rate of the fake-canonical (shuffle-vs-shuffles) tests
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np
from scipy.stats import chi2

from contam.core import summarize


def fisher(pvals) -> float:
    p = np.clip(np.asarray([x for x in pvals if x == x], dtype=np.float64), 1e-300, 1.0)
    if len(p) == 0:
        return float("nan")
    return float(chi2.sf(-2 * np.log(p).sum(), df=2 * len(p)))


def load_runs(results_dir: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*", "*", "*.json"))):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        s = summarize(r["ll_canon"], r["ll_rev"], r["ll_shuf"], r["shard_sizes"])  # recompute from raw
        ll_shuf = np.asarray(r["ll_shuf"])
        n_tok = max(1, r["n_tokens"])
        row = {
            "model": r["model"], "tag": r["tag"],
            "file": os.path.splitext(os.path.basename(r["file"]))[0],
            "seed": r["seed"], "n_examples": r["n_examples"], "n_tokens": r["n_tokens"],
            "num_shards": s["num_shards"], "perms": s["perms"],
            "mean_shard_size": float(np.mean(r["shard_sizes"])),
            "n_params": r.get("n_params"), "dtype": r.get("dtype"),
            # capability proxy: per-token NLL (nats) of the shuffled file
            "nll_per_token_shuf": float(-ll_shuf.mean(axis=1).sum() / n_tok),
            "tok_per_s": r["scored_tokens"] / max(r["seconds"], 1e-9),
        }
        for k, v in s.items():
            if k != "null_p":
                row[k] = v
        null_p = np.asarray(s["null_p"], dtype=np.float64)
        row["null_n"] = int(len(null_p))
        row["null_frac_below_05"] = float((null_p < 0.05).mean()) if len(null_p) else float("nan")
        rows.append(row)
    return rows


def write_csv(path: str, rows: list[dict]):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def aggregate(rows: list[dict]) -> list[dict]:
    groups = {}
    for r in rows:
        groups.setdefault((r["model"], r["tag"], r["file"]), []).append(r)
    out = []
    for (model, tag, file), g in sorted(groups.items()):
        row = {"model": model, "tag": tag, "file": file, "n_seeds": len(g),
               "n_examples": g[0]["n_examples"], "num_shards": g[0]["num_shards"],
               "mean_shard_size": g[0]["mean_shard_size"],
               "nll_per_token_shuf": float(np.mean([x["nll_per_token_shuf"] for x in g]))}
        for name in ("canon", "rev"):
            p = np.array([x[f"p_{name}"] for x in g], dtype=np.float64)
            e = np.array([x[f"eff_{name}_per_ex"] for x in g], dtype=np.float64)
            z = np.array([x[f"z_{name}"] for x in g], dtype=np.float64)
            row[f"p_{name}_median"] = float(np.median(p))
            row[f"p_{name}_min"] = float(p.min())
            row[f"p_{name}_max"] = float(p.max())
            row[f"z_{name}_mean"] = float(z.mean())
            row[f"eff_{name}_mean"] = float(e.mean())
            row[f"eff_{name}_sd_seeds"] = float(e.std(ddof=1)) if len(e) > 1 else float("nan")
            row[f"eff_{name}_se_shards"] = float(np.mean([x[f"eff_{name}_per_ex_se"] for x in g]))
        row["p_canon_minus_rev_median"] = float(np.median([x["p_canon_minus_rev_two_sided"] for x in g]))
        out.append(row)
    return out


def null_calibration(rows: list[dict]) -> list[dict]:
    groups = {}
    for r in rows:
        if r["null_n"]:
            groups.setdefault((r["model"], r["tag"]), []).append(r)
    out = []
    for (model, tag), g in sorted(groups.items()):
        n = sum(x["null_n"] for x in g)
        hits = sum(x["null_frac_below_05"] * x["null_n"] for x in g)
        out.append({"model": model, "tag": tag, "runs": len(g), "null_tests": n,
                    "frac_p_below_0.05": hits / n})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/summary")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = load_runs(args.results)
    files = aggregate(rows)
    write_csv(os.path.join(args.out, "runs.csv"), rows)
    write_csv(os.path.join(args.out, "files.csv"), files)
    write_csv(os.path.join(args.out, "null_calib.csv"), null_calibration(rows))
    print(f"{len(rows)} runs, {len(files)} (model, tag, file) cells")
    for r in files:
        print(f"{r['model'][:34]:34s} {r['tag'][:14]:14s} {r['file'][:34]:34s} n={r['n_examples']:5d} "
              f"p_canon={r['p_canon_median']:.2e} p_rev={r['p_rev_median']:.2e} "
              f"eff_canon={r['eff_canon_mean']:+.4f} eff_rev={r['eff_rev_mean']:+.4f}")


if __name__ == "__main__":
    main()
