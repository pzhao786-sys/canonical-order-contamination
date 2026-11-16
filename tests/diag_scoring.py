"""GPU sanity checks for the scorer (run on the instance):
  1. batch invariance: a sequence scored alone == scored inside a batch
  2. agreement with a plain, unbatched port of the authors' loop
  3. precision modes (tf32 / fp16 / bf16) against strict fp32, plus throughput

  python -m tests.diag_scoring --model gpt2 --file <path> --n 200 --dtypes fp32 tf32 fp16
"""
import argparse
import math
import random
import time

import numpy as np
import torch

from contam.core import flatten, load_examples, shard_bounds
from contam.scoring import LMScorer


@torch.no_grad()
def authors_logprob(model, tokens, c, s, device="cuda"):
    """Line-by-line port of compute_logprob_of_token_sequence (one sequence, no batching)."""
    inputs, targets = tokens[:-1], tokens[1:]
    logp = torch.zeros((1, 1), dtype=torch.float32).to(device)
    t = len(inputs)
    k = math.ceil(max(0, t - c) / s)
    for j in range(k + 1):
        start, end = s * j, min(s * j + c, t)
        rel_offs = max(0, c - s) if j > 0 else 0
        w_inp = torch.tensor(inputs[start:end]).to(device)
        w_trg = torch.tensor(targets[start:end]).to(device)
        out = model(torch.unsqueeze(w_inp, 0))
        logps = torch.nn.functional.log_softmax(out.logits[0].float(), dim=-1)
        logps = logps.gather(-1, w_trg.unsqueeze(-1)).squeeze(-1)
        logp += logps[rel_offs:].sum()
    return float(logp.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--file", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--shards", type=int, default=50)
    ap.add_argument("--use_shards", type=int, default=12)
    ap.add_argument("--perms", type=int, default=6)
    ap.add_argument("--dtypes", nargs="+", default=["fp32", "tf32", "fp16"])
    ap.add_argument("--max_batch_tokens", type=int, default=6000)
    args = ap.parse_args()

    ex = load_examples(args.file, args.n)
    ref32 = None
    for dtype in args.dtypes:
        rng = random.Random(0)
        sc = LMScorer(args.model, dtype=dtype, max_batch_tokens=args.max_batch_tokens)
        tok = sc.tokenize(ex)
        bounds = shard_bounds(len(tok), args.shards)
        seqs = []
        for a, b in bounds[:args.use_shards]:
            shard = tok[a:b]
            seqs += [flatten(shard), flatten(shard[::-1])]
            seqs += [flatten(rng.sample(shard, len(shard))) for _ in range(args.perms)]
        n_tok = sum(len(s) for s in seqs)
        sc.score(seqs[:2])  # warm-up
        torch.cuda.synchronize(); t0 = time.time()
        batched = sc.score(seqs)
        torch.cuda.synchronize(); dt = time.time() - t0
        alone = np.array([sc.score([s])[0] for s in seqs])
        ref = np.array([authors_logprob(sc.model, s, sc.context_len, sc.stride) for s in seqs[:16]])
        line = (f"[{dtype}] tok/s={n_tok/dt:7.0f}  max|batched-alone|={np.abs(batched-alone).max():.4f}  "
                f"max|batched-authors|={np.abs(batched[:16]-ref).max():.4f}  "
                f"mem={torch.cuda.max_memory_allocated()/2**30:.1f}GB")
        if dtype == "fp32":
            ref32 = batched
        elif ref32 is not None:
            d = batched - ref32
            line += f"  vs fp32: max|d|={np.abs(d).max():.4f} sd={d.std():.4f} mean={d.mean():+.4f}"
        print(line, flush=True)
        del sc
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    print(f"sequences={len(seqs)} tokens={n_tok} mean tokens/seq={n_tok/len(seqs):.0f}")


if __name__ == "__main__":
    main()
