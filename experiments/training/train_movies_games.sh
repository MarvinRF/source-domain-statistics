#!/usr/bin/env bash
# experiments/training/train_movies_games.sh
# Sequentially trains the 24 learned alignment baselines on the movies and
# games domains (PW, NL-PW, MoE-PW x H3/H4 x with/without ZCA-book input),
# used for the learned-baseline comparison in the cross-domain results.
#
# OMP_NUM_THREADS=1 (etc.) avoids BLAS over-threading during the periodic
# evaluation, which otherwise makes training impractically slow.
#
# Usage: bash experiments/training/train_movies_games.sh
# Logs:  results/retrain_logs/<method>_<heuristic>_<domain>.log
set -e
cd "$(git rev-parse --show-toplevel)"
source venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1

run() {
  local name="$1"; shift
  echo "=== STARTING: $name ==="
  "$@" > "results/retrain_logs/${name}.log" 2>&1
  echo "=== DONE: $name ==="
}

for d in movies games; do
  for h in h3 h4; do
    run "pw_${h}_${d}"      python3 -m acw.baselines.parametric_whitening --heuristic "$h" --domain "$d"
    run "pw_zb_${h}_${d}"   python3 -m acw.baselines.parametric_whitening --heuristic "$h" --domain "$d" --input-zca-book
    run "nlpw_${h}_${d}"    python3 -m acw.baselines.nonlinear_whitening   --heuristic "$h" --domain "$d"
    run "nlpw_zb_${h}_${d}" python3 -m acw.baselines.nonlinear_whitening   --heuristic "$h" --domain "$d" --input-zca-book
    run "moe_${h}_${d}"     python3 -m acw.baselines.moe_whitening         --heuristic "$h" --domain "$d"
    run "moe_zb_${h}_${d}"  python3 -m acw.baselines.moe_whitening         --heuristic "$h" --domain "$d" --input-zca-book
  done
done
echo "ALL_24_TRAINING_RUNS_DONE"
