# Is a preference for the canonical order proof of contamination?

Course project code. Paper under examination: Oren et al., *Proving Test Set
Contamination in Black Box Language Models* (ICLR 2024). The report itself is
submitted separately; `make_tables.py` and `make_figures.py` write its tables and
figure to `../report/`.

## The question

The paper's test flags a model as contaminated when it assigns a higher likelihood
to a benchmark file in its canonical order than to shuffled orders. The guarantee
needs the file to be exchangeable, which cannot be verified for off-the-shelf
benchmarks; the paper guards against this with "negative control" models
(GPT-2 XL, BioMedLM) and then reads the remaining low p-values of LLaMA2-7B and
Mistral-7B on MMLU as (heuristic) evidence of contamination.

**Paper's explanation (memorisation):** the model prefers the canonical order
because it saw the file, in that order, during training.

**Alternative (order structure):** neighbouring list elements in the file are
related. A capable model exploits this in context, without ever having seen the
file. (We also conjectured that stronger models exploit it more.)

## Predictions written before the first GPU run

For every shard we score the canonical order, `m` random shuffles (both exactly as
in the paper) and, in addition, the **reversed** order.

| Measurement | If memorisation is right | If order structure is right |
|---|---|---|
| reversed vs. shuffled | no preference | preferred, about as much as canonical |
| canonical minus reversed | as large as canonical minus shuffled | about zero |
| canonical effect in provably clean models, as models get stronger | flat, near zero | present and growing |

## What we found (details and numbers: the report)

* The paper's Table 1 reproduces (authors' contaminated model, ten canary files),
  and so do its per-file MMLU p-values for GPT-2 XL and Pythia-1.4B (Spearman 0.99).
* Under true memorisation the reversed order is never preferred; fresh topic-sorted
  orderings are "detected" in clean models and the reversed order keeps most of the
  gain. So far as predicted.
* On MMLU, clean models reject strongly (GPT-2 small: p = 1e-31 on
  `professional_law`), but the reversed order gains nothing: our prediction failed.
  The cause is that the authors' `mmlu_json/*.json` lists are **mis-split**: 956 of
  14,042 list elements end in the middle of a CSV record (`mmlu_fragments.py`).
* Re-cutting the same text at record boundaries removes the effect in every clean
  model, including in the 19 files with only a few mis-split elements.
* Re-analysing the paper's own Table 4 (`reanalyze_table4.py`): its Fisher omnibus
  (0.014 / 0.011 for LLaMA2 / Mistral) becomes 0.22 / 0.24 without the 13 retained
  files that contain a mis-split element.
* No sign that stronger models exploit order structure more (124M to 1.5B parameters).

## Layout

* `contam/core.py` sharding, orderings, statistics (no torch; unit tested)
* `contam/scoring.py` batched strided log-probabilities (authors' window logic)
* `run_test.py` CLI, one JSON per (model, protocol, file, seed) with raw per-shard log-probs
* `make_orderings.py` fresh shuffled / topic-sorted control files
* `mmlu_fragments.py` counts mis-split MMLU elements and writes repaired files
* `reanalyze_table4.py` Fisher omnibus from the paper's Table 4, with and without mis-split files
* `analyze.py` -> `results/summary/*.csv`; `make_tables.py`, `make_figures.py` -> `../report/`
* `scripts/remote_env.sh`, `scripts/remote_setup.sh` environment and data on the GPU box
* `scripts/run_stage.sh`, `scripts/run_stage2.sh` the sweeps that produced `results/`
  (`scripts/run_stage3.sh`: Mistral-7B stages, not run: no GPU with enough memory was available)
* `results/` raw results (one JSON per run), `logs/` the queue logs. The control orderings
  and the repaired MMLU files are not stored here because they are copies of benchmark test
  sets; `make_orderings.py` (seeds 0-4, `--n 1000 --cluster_size 20`) and `mmlu_fragments.py`
  regenerate them from the authors' repository
* `tests/test_core.py` mock-scorer tests: `python -m tests.test_core`

## Reproducing the numbers in the report from `results/`

```
python analyze.py --results results --out results/summary
python reanalyze_table4.py
python make_tables.py --summary results/summary --out ../report/tables
python make_figures.py --summary results/summary --out ../report/figs
```

Tags in `results/<model>/<tag>/`: `paper50` = the paper's 50 shards; `k10` = at least
10 elements per shard; `*_fixed` = repaired MMLU files; `ctrl` = fresh orderings;
`paper50_fp32` = strict single precision.

## Differences from the authors' script

One thing is added (the reversed ordering). Tokenisation, shard construction,
shuffling primitive, strided window and the t-test (`np.std` with ddof=0,
df = r-1) follow `compute_sharded_comparison_test.py`. We use 50 shuffles per shard
(the paper: 250-1000). Sequences are scored in batches of windows of exactly equal
length (no padding), with fp32 weights and TF32 matrix multiplication; half precision
is not usable for GPT-2-family models (errors of tens of nats per sequence). A strict
fp32 rerun of five canary files gives the same p-values.
