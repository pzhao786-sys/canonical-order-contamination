"""Model-agnostic logic for the sharded likelihood comparison test.

Follows compute_sharded_comparison_test.py from
https://github.com/tatsu-lab/test_set_contamination (Oren et al., ICLR 2024)
and adds ONE extra ordering per shard: the reversed order.

Nothing in this file imports torch, so it can be unit-tested with a mock scorer.
"""
from __future__ import annotations

import json
import random
import time

import numpy as np
from scipy.stats import t as tdist


# --------------------------------------------------------------------------- data
def load_examples(path: str, max_examples: int = 5000) -> list[str]:
    """Same loader as the authors: a .json file is a list, anything else is one
    example per raw line (trailing newline kept, it acts as the separator)."""
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            examples = json.load(f)
        examples = [e if isinstance(e, str) else json.dumps(e) for e in examples]
    else:
        with open(path, "r", encoding="utf-8") as f:
            examples = f.readlines()
    return examples[:max_examples]


def resolve_num_shards(n: int, num_shards: int, min_shard_size: int = 0) -> int:
    """Paper protocol: a fixed number of shards. Our variant: additionally require
    at least `min_shard_size` examples per shard (fewer shards for small files)."""
    r = num_shards
    if min_shard_size and min_shard_size > 0:
        r = min(r, n // min_shard_size)
    r = min(r, n // 2)  # a shard needs at least 2 examples to be shuffled
    return max(r, 1)


def shard_bounds(n: int, r: int) -> list[tuple[int, int]]:
    """Contiguous shards; the first n % r shards get one extra example (Alg. 1)."""
    counts = [n // r + (1 if i < n % r else 0) for i in range(r)]
    edges = [0] + np.cumsum(counts).tolist()
    return [(edges[i], edges[i + 1]) for i in range(r)]


def flatten(list_of_lists):
    return [x for s in list_of_lists for x in s]


# -------------------------------------------------------------------------- stats
def ttest_greater(d, ddof: int = 0) -> tuple[float, float]:
    """One-sided t-test for E[d] > 0. ddof=0 reproduces the authors' code
    (z = mean/np.std*sqrt(r), df = r-1); ddof=1 is the textbook t-test."""
    d = np.asarray(d, dtype=np.float64)
    r = len(d)
    if r < 2:
        return float("nan"), float("nan")
    sd = d.std(ddof=ddof)
    if sd == 0:
        return float("nan"), float("nan")
    z = d.mean() / sd * np.sqrt(r)
    return float(z), float(tdist.sf(z, df=r - 1))


def ttest_two_sided(d, ddof: int = 1) -> tuple[float, float]:
    z, p = ttest_greater(d, ddof=ddof)
    if np.isnan(z):
        return z, p
    return z, float(2 * tdist.sf(abs(z), df=len(d) - 1))


def summarize(ll_canon, ll_rev, ll_shuf, shard_sizes) -> dict:
    """All test statistics for one (model, file, seed) run.

    s_canon_i = LL(canonical shard i) - mean_pi LL(shuffled shard i)   [paper]
    s_rev_i   = LL(reversed  shard i) - mean_pi LL(shuffled shard i)   [ours]
    """
    ll_canon = np.asarray(ll_canon, dtype=np.float64)
    ll_rev = np.asarray(ll_rev, dtype=np.float64)
    ll_shuf = np.asarray(ll_shuf, dtype=np.float64)
    k = np.asarray(shard_sizes, dtype=np.float64)
    r, m = ll_shuf.shape
    mean_shuf = ll_shuf.mean(axis=1)
    s_canon = ll_canon - mean_shuf
    s_rev = ll_rev - mean_shuf

    out = {"num_shards": int(r), "perms": int(m)}
    for name, s in (("canon", s_canon), ("rev", s_rev)):
        z0, p0 = ttest_greater(s, ddof=0)
        z1, p1 = ttest_greater(s, ddof=1)
        out[f"z_{name}"] = z0                      # authors' statistic
        out[f"p_{name}"] = p0                      # authors' p-value
        out[f"t_{name}"] = z1                      # textbook t statistic
        out[f"p_{name}_ddof1"] = p1
        out[f"eff_{name}_per_ex"] = float((s / k).mean())   # nats per example
        out[f"eff_{name}_per_ex_se"] = float((s / k).std(ddof=1) / np.sqrt(r)) if r > 1 else float("nan")
        out[f"frac_shards_pos_{name}"] = float((s > 0).mean())
    # Is the preference specific to the exact canonical order?
    t_diff, p_diff = ttest_two_sided(ll_canon - ll_rev, ddof=1)
    out["t_canon_minus_rev"] = t_diff
    out["p_canon_minus_rev_two_sided"] = p_diff

    # Free calibration check: treat each shuffle as a fake "canonical" order and
    # test it against the remaining shuffles. Under exchangeability of the
    # *shuffles* these are draws from the null, so they should look Uniform(0,1).
    null_p = []
    if m >= 3:
        tot = ll_shuf.sum(axis=1)
        for j in range(m):
            d = ll_shuf[:, j] - (tot - ll_shuf[:, j]) / (m - 1)
            null_p.append(ttest_greater(d, ddof=0)[1])
    out["null_p"] = null_p
    # Whole-file permutation test only makes sense with a single shard.
    if r == 1:
        out["p_perm_canon"] = float((1 + (ll_shuf[0] >= ll_canon[0]).sum()) / (m + 1))
        out["p_perm_rev"] = float((1 + (ll_shuf[0] >= ll_rev[0]).sum()) / (m + 1))
    return out


# ---------------------------------------------------------------------------- run
def build_orderings(tok_examples, bounds, perms: int, seed: int):
    """Returns (sequences, index) where index[i] = (shard, kind, j)."""
    rng = random.Random(seed)
    seqs, index = [], []
    for i, (a, b) in enumerate(bounds):
        shard = tok_examples[a:b]
        seqs.append(flatten(shard)); index.append((i, "canon", 0))
        seqs.append(flatten(shard[::-1])); index.append((i, "rev", 0))
        for j in range(perms):
            # same primitive as the authors: random.sample(l, k=len(l))
            seqs.append(flatten(rng.sample(shard, k=len(shard)))); index.append((i, "shuf", j))
    return seqs, index


def run_file(scorer, examples: list[str], num_shards: int = 50, min_shard_size: int = 0,
             perms: int = 50, seed: int = 0, chunk_tokens: int = 2_000_000,
             fixed: dict | None = None) -> dict:
    """Run the test for one file. `scorer` needs .tokenize(list[str]) and
    .score(list[list[int]]) -> array of sequence log-probabilities.

    `fixed` may carry {'ll_canon','ll_rev'} from a previous seed of the same
    configuration so the deterministic orderings are not recomputed."""
    t0 = time.time()
    tok = scorer.tokenize(examples)
    n = len(tok)
    r = resolve_num_shards(n, num_shards, min_shard_size)
    bounds = shard_bounds(n, r)
    seqs, index = build_orderings(tok, bounds, perms, seed)

    if fixed is not None:  # drop canon/rev, reuse cached values
        keep = [i for i, (_, kind, _) in enumerate(index) if kind == "shuf"]
        seqs = [seqs[i] for i in keep]
        index = [index[i] for i in keep]

    # score in chunks so length-sorted batching stays efficient but memory bounded
    scores = np.zeros(len(seqs), dtype=np.float64)
    start, n_tok = 0, 0
    for i, s in enumerate(seqs):
        n_tok += len(s)
        if n_tok >= chunk_tokens or i == len(seqs) - 1:
            scores[start:i + 1] = scorer.score(seqs[start:i + 1])
            start, n_tok = i + 1, 0

    ll_canon = np.zeros(r) if fixed is None else np.asarray(fixed["ll_canon"], dtype=np.float64)
    ll_rev = np.zeros(r) if fixed is None else np.asarray(fixed["ll_rev"], dtype=np.float64)
    ll_shuf = np.zeros((r, perms))
    for (i, kind, j), v in zip(index, scores):
        if kind == "canon":
            ll_canon[i] = v
        elif kind == "rev":
            ll_rev[i] = v
        else:
            ll_shuf[i, j] = v

    shard_sizes = [b - a for a, b in bounds]
    shard_tokens = [sum(len(e) for e in tok[a:b]) for a, b in bounds]
    res = {
        "n_examples": n,
        "n_tokens": int(sum(shard_tokens)),
        "seed": seed,
        "requested_num_shards": num_shards,
        "min_shard_size": min_shard_size,
        "shard_sizes": shard_sizes,
        "shard_tokens": shard_tokens,
        "ll_canon": ll_canon.tolist(),
        "ll_rev": ll_rev.tolist(),
        "ll_shuf": ll_shuf.tolist(),
        "scored_tokens": int(sum(len(s) for s in seqs)),
        "seconds": time.time() - t0,
    }
    res["summary"] = summarize(ll_canon, ll_rev, ll_shuf, shard_sizes)
    return res
