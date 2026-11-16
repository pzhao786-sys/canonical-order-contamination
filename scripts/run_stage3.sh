#!/usr/bin/env bash
# Mistral-7B queue (needs a full A100 / H100: 7.2B parameters in fp32 are 29 GB).
# Usage: bash scripts/run_stage3.sh <stage> [<stage> ...]   (from /workspace/contam-order, inside tmux)
set -uo pipefail
source /workspace/env.sh
export TOKENIZERS_PARALLELISM=false
cd "$(dirname "$0")/.."
REPO=/workspace/data/test_set_contamination
BENCH=$REPO/benchmarks/unzipped
CTRL=/workspace/data/controls
FIXED=$CTRL/mmlu_fixed
OUT=results
M=${MISTRAL:-mistralai/Mistral-7B-v0.1}
DT=${DTYPE:-tf32}
MBT=${MAX_BATCH_TOKENS:-8192}
mkdir -p logs $OUT/summary

rt() { python run_test.py --out $OUT --dtype $DT --max_batch_tokens $MBT "$@" 2>&1 | grep --line-buffered -vE "Loading weights|Token indices sequence length"; }
paths() { local dir="$1"; shift; for s in "$@"; do echo "$dir/$s.json"; done; }
FRAGCSV=$OUT/summary/mmlu_fragments.csv
sets() { python reanalyze_table4.py --fragments $FRAGCSV --print "$1"; }
HIST="high_school_us_history high_school_world_history high_school_european_history"

run_stage() {
  local stage="$1"
  case "$stage" in
    fix_make)        # repaired MMLU files + fragment statistics (CPU only)
      python mmlu_fragments.py --src $BENCH/mmlu_json --out $FIXED --csv $FRAGCSV | head -n 1 ;;
    m_smoke)         # does the 7B model load, and how fast is it?
      rt --model $M --files $(paths $BENCH/mmlu_json college_physics) --tag smoke --perms 4 --seeds 0 ;;
    m_frag13)        # the 13 files the paper kept that contain record fragments: authors' files
      rt --model $M --files $(paths $BENCH/mmlu_json $(sets kept_frag)) --tag paper50 --perms 50 --seeds 0 ;;
    m_frag13_fixed)  # same text, record boundaries repaired
      rt --model $M --files $(paths $FIXED $(sets kept_frag)) --tag paper50_fixed --perms 50 --seeds 0 ;;
    m_clean30)       # the 30 kept files without fragments (completes the paper's 43-file omnibus)
      rt --model $M --files $(paths $BENCH/mmlu_json $(sets kept_clean)) --tag paper50 --perms 50 --seeds 0 ;;
    m_arc)           # the paper's other positive: ARC-Easy on Mistral (p = 0.001)
      rt --model $M --files $BENCH/arc/ARC-Easy-Test.jsonl --tag paper50 --perms 50 --seeds 0 ;;
    m_seeds)         # two more shuffle seeds for the original-vs-repaired comparison
      rt --model $M --files $(paths $BENCH/mmlu_json $(sets kept_frag)) --tag paper50 --perms 50 --seeds 0 1 2
      rt --model $M --files $(paths $FIXED $(sets kept_frag)) --tag paper50_fixed --perms 50 --seeds 0 1 2 ;;
    m_clean30_seeds)
      rt --model $M --files $(paths $BENCH/mmlu_json $(sets kept_clean)) --tag paper50 --perms 50 --seeds 0 1 2 ;;
    m_hist)          # heavily fragmented history files, authors' vs repaired
      rt --model $M --files $(paths $BENCH/mmlu_json $HIST) --tag paper50 --perms 50 --seeds 0
      rt --model $M --files $(paths $FIXED $HIST) --tag paper50_fixed --perms 50 --seeds 0 ;;
    m_arc_ctrl)      # fresh shuffles of ARC-Easy (exchangeable by construction)
      [ -f $CTRL/ARC-Easy-Test.shuffled0.txt ] || python make_orderings.py --src $BENCH/arc/ARC-Easy-Test.jsonl --out $CTRL --n 1000 --cluster_size 20 --seeds 0 1 2 3 4
      rt --model $M --files "$CTRL/ARC-Easy-Test.shuffled*.txt" --tag ctrl --perms 50 --seeds 0 ;;
    *) echo "unknown stage '$stage'"; return 1 ;;
  esac
}

for s in "$@"; do
  echo "===== STAGE $s start $(date -u +%FT%TZ) ====="
  run_stage "$s"
  echo "===== STAGE $s end   $(date -u +%FT%TZ) rc=$? ====="
done
echo "===== QUEUE3 DONE $(date -u +%FT%TZ) ====="
