"""Batched log-probability scoring with a Hugging Face causal LM.

The per-sequence computation is the authors' strided sliding window
(compute_logprob_of_token_sequence): inputs = tokens[:-1], targets = tokens[1:],
windows of `context_len` tokens moved by `stride`; the first window scores all of
its positions, later windows only the positions not covered before. We only add
batching (right padding, which cannot affect earlier positions of a causal LM).
"""
from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def model_max_positions(config) -> int:
    for attr in ("n_positions", "max_position_embeddings"):
        v = getattr(config, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    return 2048


class LMScorer:
    def __init__(self, model_name: str, dtype: str = "fp32", context_len: int = 2048,
                 stride: int | None = None, max_batch_tokens: int = 6_000,
                 device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        torch_dtype = {"fp32": torch.float32, "tf32": torch.float32,
                       "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
        self.dtype = dtype
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype)
        self.model.to(device).eval()
        # fp32 = strict IEEE single precision; tf32 = fp32 weights with TF32 matmuls
        torch.backends.cuda.matmul.allow_tf32 = (dtype == "tf32")
        torch.backends.cudnn.allow_tf32 = (dtype == "tf32")
        self.context_len = min(context_len, model_max_positions(self.model.config))
        self.stride = stride if stride else self.context_len // 2
        self.max_batch_tokens = max_batch_tokens
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.backbone = self.model.base_model
        self.head = self.model.get_output_embeddings()
        self.head_chunk = 2048
        self.attn_budget = 2048 * 2048 * (1 if self.n_params > 1e9 else 4)
        self.self_check_err = self._self_check()

    def tokenize(self, examples: list[str]) -> list[list[int]]:
        # identical to the authors: tokenizer defaults, one example at a time.
        # (Tokenizers that add BOS by default, e.g. LLaMA/Mistral, therefore put
        # a BOS in front of every example, exactly as in the original code.)
        return [self.tokenizer.encode(ex) for ex in examples]

    def _windows(self, seqs):
        c, s = self.context_len, self.stride
        wins = []  # (seq_idx, start, end, rel_offs)
        for si, tokens in enumerate(seqs):
            t = len(tokens) - 1
            if t <= 0:
                continue
            k = math.ceil(max(0, t - c) / s)
            for j in range(k + 1):
                start = s * j
                end = min(s * j + c, t)
                rel = max(0, c - s) if j > 0 else 0
                if end - start > rel:
                    wins.append((si, start, end, rel))
        return wins

    @torch.no_grad()
    def _score_batch(self, seqs, batch):
        """All windows in `batch` have the same length and the same rel_offs:
        no padding and no attention mask are needed."""
        B = len(batch)
        T = batch[0][2] - batch[0][1]
        rel = batch[0][3]
        # (positions before rel_offs were scored by the previous window)
        ids = torch.tensor([seqs[si][s:e] for si, s, e, _ in batch], dtype=torch.long, device=self.device)
        trg = torch.tensor([seqs[si][s + 1:e + 1] for si, s, e, _ in batch], dtype=torch.long, device=self.device)
        # Same maths as the authors (log_softmax over the vocabulary, gather the
        # target), but the output layer is applied to a few thousand positions at
        # a time so the full B x T x V logits tensor never has to be in memory.
        hidden = self.backbone(input_ids=ids, use_cache=False).last_hidden_state  # B, T, H
        hidden = hidden[:, rel:].reshape(-1, hidden.shape[-1])
        flat_trg = trg[:, rel:].reshape(-1)
        lp = torch.empty(flat_trg.shape[0], dtype=torch.float64, device=self.device)
        for i in range(0, flat_trg.shape[0], self.head_chunk):
            logits = self.head(hidden[i:i + self.head_chunk]).float()
            logps = F.log_softmax(logits, dim=-1)
            lp[i:i + self.head_chunk] = logps.gather(-1, flat_trg[i:i + self.head_chunk, None]).squeeze(-1).double()
        return lp.view(B, T - rel).sum(dim=1).cpu().numpy()

    @torch.no_grad()
    def _self_check(self):
        """backbone + output layer must reproduce model(...).logits."""
        g = torch.Generator().manual_seed(0)
        ids = torch.randint(0, self.model.config.vocab_size, (2, 64), generator=g).to(self.device)
        ref = self.model(input_ids=ids).logits.float()
        got = self.head(self.backbone(input_ids=ids).last_hidden_state).float()
        err = (ref - got).abs().max().item()
        tol = 1e-3 if self.dtype in ("fp32", "tf32") else 0.5
        if not err <= tol:
            raise RuntimeError(f"backbone+head does not reproduce model logits (max err {err})")
        return err

    def score(self, seqs: list[list[int]]) -> np.ndarray:
        """Sum of log p(token | prefix) for each sequence (first token unscored,
        as in the authors' code).

        Windows are batched only with windows of exactly the same length (all
        orderings of a shard have the same length), in random order, so that the
        canonical / reversed / shuffled orderings are numerically treated alike."""
        out = np.zeros(len(seqs), dtype=np.float64)
        groups = {}
        for w in self._windows(seqs):
            groups.setdefault((w[2] - w[1], w[3]), []).append(w)
        order_rng = random.Random(12345)
        for (T, _), wins in sorted(groups.items(), reverse=True):
            order_rng.shuffle(wins)
            i = 0
            while i < len(wins):
                # token budget, and a T^2 budget for the attention maps of long windows
                bsz = max(1, min(self.max_batch_tokens // T, self.attn_budget // (T * T)))
                batch = wins[i:i + bsz]
                try:
                    vals = self._score_batch(seqs, batch)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if self.max_batch_tokens <= T:
                        raise
                    self.max_batch_tokens = max(T, self.max_batch_tokens // 2)
                    print(f"[scorer] OOM -> max_batch_tokens={self.max_batch_tokens}", flush=True)
                    continue
                for (si, _, _, _), v in zip(batch, vals):
                    out[si] += v
                i += len(batch)
        return out
