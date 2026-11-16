#!/usr/bin/env bash
# Usage: bash scripts/run_stage.sh <stage> [<stage> ...]   (from /workspace/contam-order, inside tmux)
# Stages are resumable: finished (model, file, seed) results are skipped.
set -uo pipefail
source /workspace/env.sh
export TOKENIZERS_PARALLELISM=false
cd "$(dirname "$0")/.."
REPO=/workspace/data/test_set_contamination
CANARY=$REPO/detection_challenge_benchmarks/contam-1.4b-dupcount-higher/unzipped
BENCH=$REPO/benchmarks/unzipped
CTRL=/workspace/data/controls
OUT=results
CONTAM=yonatano/contam-1.4b-dupcount-higher
mkdir -p logs

rt() { python run_test.py --out $OUT "$@" 2>&1 | grep -vE "Loading weights|Token indices sequence length"; }
bench_files() { echo $BENCH/arc/ARC-Easy-Test.jsonl $BENCH/piqa/tests.jsonl $BENCH/openbookqa/test.jsonl \
                     $BENCH/gsm8k/test.jsonl $BENCH/naturalqa/NQ-open.efficientqa.test.1.1.jsonl; }

run_stage() {
  local stage="$1"
  case "$stage" in
    smoke)          # ~2 minutes: does the pipeline run end to end?
      rt --model gpt2 --files "$CANARY/*" --tag smoke --perms 5 --seeds 0 --max_examples 200 ;;
    canary)         # Reproduce Table 1 (sharded test) + reversed order under true memorisation
      rt --model $CONTAM --files "$CANARY/*" --tag paper50 --perms 50 --seeds 0 ;;
    canary_seeds)   # two more independent sets of 50 shuffles
      rt --model $CONTAM --files "$CANARY/*" --tag paper50 --perms 50 --seeds 0 1 2 ;;
    canary_fp32)    # precision check: strict fp32 on the five small canary files
      rt --model $CONTAM --tag paper50_fp32 --perms 50 --seeds 0 --dtype fp32 --files \
         $CANARY/piqa.jsonl $CANARY/truthfulqa.jsonl $CANARY/openbookqa.jsonl \
         $CANARY/mmlu_high_school_psychology.jsonl $CANARY/mmlu_professional_psychology.jsonl ;;
    canary_clean)   # the "dup count 0" row missing from Table 1: clean models on the canary files
      for M in gpt2-xl EleutherAI/pythia-1.4b; do
        rt --model $M --files "$CANARY/*" --tag paper50 --perms 50 --seeds 0
      done ;;
    controls_make)  # fresh orderings nobody can have trained on
      python make_orderings.py --src $BENCH/arc/ARC-Easy-Test.jsonl --out $CTRL --n 1000 --cluster_size 20 --seeds 0 1 2 3 4
      python make_orderings.py --src $BENCH/piqa/tests.jsonl --out $CTRL --n 1000 --cluster_size 20 --seeds 0 1 2 3 4 ;;
    controls_a)     # structure / null controls, clean models
      for M in gpt2 gpt2-medium gpt2-xl; do
        rt --model $M --files "$CTRL/*.txt" --tag ctrl --perms 50 --seeds 0
      done ;;
    controls_b)
      for M in gpt2-large EleutherAI/pythia-1.4b; do
        rt --model $M --files "$CTRL/*.txt" --tag ctrl --perms 50 --seeds 0
      done ;;
    mmlu_k10)       # MMLU, at least 10 examples per shard (so that reversal is informative)
      for M in gpt2 gpt2-medium gpt2-large gpt2-xl EleutherAI/pythia-1.4b; do
        rt --model $M --files "$BENCH/mmlu_json/*.json" --tag k10 --min_shard_size 10 --perms 50 --seeds 0
      done ;;
    mmlu_p50)       # MMLU with the paper's 50 shards, for the two models that are in Table 4
      for M in gpt2-xl EleutherAI/pythia-1.4b; do
        rt --model $M --files "$BENCH/mmlu_json/*.json" --tag paper50 --perms 50 --seeds 0
      done ;;
    bench)          # Table 2 files in their canonical order
      for M in gpt2-xl EleutherAI/pythia-1.4b gpt2 gpt2-medium gpt2-large; do
        rt --model $M --files $(bench_files) --tag paper50 --perms 50 --seeds 0
      done ;;
    *) echo "unknown stage '$stage'"; return 1 ;;
  esac
}

for s in "$@"; do
  echo "===== STAGE $s start $(date -u +%FT%TZ) ====="
  run_stage "$s"
  echo "===== STAGE $s end   $(date -u +%FT%TZ) rc=$? ====="
done
echo "===== QUEUE DONE $(date -u +%FT%TZ) ====="
