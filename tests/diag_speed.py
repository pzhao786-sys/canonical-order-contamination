"""Where does the time go, and where does the batched-vs-authors difference come from?
  python -m tests.diag_speed --model gpt2
"""
import argparse
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


def timed(fn, n=3):
    fn(); torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        out = fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n, out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--B", type=int, default=6)
    ap.add_argument("--T", type=int, default=1000)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    m = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).cuda().eval()
    V = m.config.vocab_size
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, V, (args.B, args.T), generator=g).cuda()
    trg = torch.randint(0, V, (args.B, args.T), generator=g).cuda()
    n_tok = args.B * args.T

    dt, _ = timed(lambda: m.transformer(ids).last_hidden_state)
    print(f"body only        : {dt*1e3:8.1f} ms  ({n_tok/dt:9.0f} tok/s)")
    dt, logits = timed(lambda: m(input_ids=ids).logits)
    print(f"body + lm_head   : {dt*1e3:8.1f} ms  ({n_tok/dt:9.0f} tok/s)")
    dt, lp_ce = timed(lambda: -F.cross_entropy(logits.transpose(1, 2), trg, reduction="none"))
    print(f"cross_entropy    : {dt*1e3:8.1f} ms")
    dt, lp_ls = timed(lambda: torch.log_softmax(logits, -1).gather(-1, trg.unsqueeze(-1)).squeeze(-1))
    print(f"log_softmax+gather: {dt*1e3:7.1f} ms")
    print(f"CE vs log_softmax: max|d| per token = {(lp_ce-lp_ls).abs().max().item():.2e}, "
          f"sum diff = {(lp_ce.double().sum()-lp_ls.double().sum()).item():+.4f}")
    # batch-of-B vs one-at-a-time, per token
    single = torch.cat([m(input_ids=ids[i:i+1]).logits for i in range(args.B)])
    d = (torch.log_softmax(single, -1).gather(-1, trg.unsqueeze(-1)).squeeze(-1) - lp_ls)
    print(f"batched vs single: max|d| per token = {d.abs().max().item():.2e}, "
          f"per-seq sum diff = {[round(x, 4) for x in d.double().sum(1).tolist()]}")
    # same single sequence twice (run-to-run determinism)
    a = m(input_ids=ids[:1]).logits; b = m(input_ids=ids[:1]).logits
    print(f"same input twice : max|d| logits = {(a-b).abs().max().item():.2e}")
    print(f"fp32 sum vs fp64 sum of token log-probs: {(lp_ls.sum(1).double()-lp_ls.double().sum(1)).abs().max().item():.4f}")
    print(torch.cuda.get_device_name(0), f"mem={torch.cuda.max_memory_allocated()/2**30:.1f}GB")


if __name__ == "__main__":
    main()
