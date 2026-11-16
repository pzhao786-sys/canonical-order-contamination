#!/usr/bin/env bash
# Follow-up queue (added after the first results): the MMLU record-fragmentation experiment.
# Usage: bash scripts/run_stage2.sh <stage> [<stage> ...]   (from /workspace/contam-order, inside tmux)
set -uo pipefail
source /workspace/env.sh
export TOKENIZERS_PARALLELISM=false
cd "$(dirname "$0")/.."
REPO=/workspace/data/test_set_contamination
BENCH=$REPO/benchmarks/unzipped
CTRL=/workspace/data/controls
FIXED=$CTRL/mmlu_fixed
OUT=results
mkdir -p logs

rt() { python run_test.py --out $OUT "$@" 2>&1 | grep -vE "Loading weights|Token indices sequence length"; }

# the four subjects in which >30% of list elements are record fragments
MAIN="high_school_us_history high_school_world_history high_school_european_history professional_law"
# the other 19 subjects with at least one fragment
MINOR="professional_medicine logical_fallacies high_school_computer_science college_medicine security_studies
high_school_psychology moral_disputes philosophy machine_learning college_computer_science computer_security
high_school_statistics high_school_microeconomics professional_psychology prehistory
high_school_government_and_politics nutrition high_school_biology miscellaneous"
paths() { local dir="$1"; shift; for s in "$@"; do echo "$dir/$s.json"; done; }

run_stage() {
  local stage="$1"
  case "$stage" in
    fix_make)       # same text, re-split at true CSV record boundaries
      python mmlu_fragments.py --src $BENCH/mmlu_json --out $FIXED --csv $OUT/summary/mmlu_fragments.csv | head -n 1 ;;
    openbookqa)     # its result was overwritten by gsm8k (both files are called test.jsonl)
      cp $BENCH/openbookqa/test.jsonl $CTRL/openbookqa_test.jsonl
      for M in gpt2 gpt2-medium gpt2-large gpt2-xl EleutherAI/pythia-1.4b; do
        rt --model $M --files $CTRL/openbookqa_test.jsonl --tag paper50 --perms 50 --seeds 0
      done ;;
    fixed_main)     # the one-factor experiment: repaired boundaries, four heavily fragmented subjects, 3 seeds
      for M in gpt2 EleutherAI/pythia-1.4b gpt2-medium gpt2-large gpt2-xl; do
        rt --model $M --files $(paths $FIXED $MAIN) --tag k10_fixed --min_shard_size 10 --perms 50 --seeds 0 1 2
      done ;;
    orig_main_seeds) # two more shuffle seeds for the original (fragmented) files
      for M in gpt2 EleutherAI/pythia-1.4b gpt2-medium gpt2-large gpt2-xl; do
        rt --model $M --files $(paths $BENCH/mmlu_json $MAIN) --tag k10 --min_shard_size 10 --perms 50 --seeds 0 1 2
      done ;;
    # ---- reduced plan (the full stages above were too slow on the 10 GB slice) ----
    fixed_main_a)   # 3 shuffle seeds for the smallest and the strongest clean model
      for M in gpt2 EleutherAI/pythia-1.4b; do
        rt --model $M --files $(paths $FIXED $MAIN) --tag k10_fixed --min_shard_size 10 --perms 50 --seeds 0 1 2
      done ;;
    fixed_main_b)   # one seed for the rest of the ladder
      for M in gpt2-xl gpt2-medium gpt2-large; do
        rt --model $M --files $(paths $FIXED $MAIN) --tag k10_fixed --min_shard_size 10 --perms 50 --seeds 0
      done ;;
    orig_main_seeds_a)
      for M in gpt2 EleutherAI/pythia-1.4b; do
        rt --model $M --files $(paths $BENCH/mmlu_json $MAIN) --tag k10 --min_shard_size 10 --perms 50 --seeds 0 1 2
      done ;;
    fixed_minor)    # lightly fragmented subjects, repaired
      for M in gpt2 EleutherAI/pythia-1.4b gpt2-xl; do
        rt --model $M --files $(paths $FIXED $MINOR) --tag k10_fixed --min_shard_size 10 --perms 50 --seeds 0
      done ;;
    fixed_p50)      # paper's 50-shard protocol on the repaired main subjects
      for M in EleutherAI/pythia-1.4b gpt2-xl; do
        rt --model $M --files $(paths $FIXED $MAIN) --tag paper50_fixed --perms 50 --seeds 0
      done ;;
    *) echo "unknown stage '$stage'"; return 1 ;;
  esac
}

for s in "$@"; do
  echo "===== STAGE $s start $(date -u +%FT%TZ) ====="
  run_stage "$s"
  echo "===== STAGE $s end   $(date -u +%FT%TZ) rc=$? ====="
done
echo "===== QUEUE2 DONE $(date -u +%FT%TZ) ====="
