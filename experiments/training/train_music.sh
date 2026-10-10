#!/usr/bin/env bash
# experiments/training/train_music.sh
# ====================================
# Sequentially trains the 16 learned alignment baselines on the music domain
# (PW, NL-PW, MoE-PW x H3/H4 x with/without ZCA-book input, plus Infomax x
# H3/H4 x with/without ZCA) on the H3/H4 user profiles, used for the
# learned-baseline comparison.
#
# OMP_NUM_THREADS=1 (etc.) avoids BLAS over-threading in evaluate() of
# parametric_whitening.py; without it, thread-spawn overhead at every
# periodic evaluation (every 20 epochs) makes training impractically slow.
#
# Usage: bash experiments/training/train_music.sh
# Logs:  results/retrain_logs/<method>_<heuristic>[_zb].log

set -e
cd "$(dirname "$0")/../.."
source venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1

mkdir -p results/retrain_logs

run() {
  local name="$1"; shift
  echo "=== STARTING: $name ==="
  "$@" > "results/retrain_logs/${name}.log" 2>&1
  echo "=== DONE: $name ==="
  grep "Reported score =" "results/retrain_logs/${name}.log" || true
}

for h in h3 h4; do
  if [ -f "results/retrain_logs/pw_${h}.log" ]; then
    echo "=== ALREADY DONE (skipping): pw_${h} ==="
  else
    run "pw_${h}"      python3 -m acw.baselines.parametric_whitening --heuristic "$h" --domain music
  fi
  run "pw_zb_${h}"     python3 -m acw.baselines.parametric_whitening --heuristic "$h" --domain music --input-zca-book
  run "nlpw_${h}"      python3 -m acw.baselines.nonlinear_whitening   --heuristic "$h" --domain music
  run "nlpw_zb_${h}"   python3 -m acw.baselines.nonlinear_whitening   --heuristic "$h" --domain music --input-zca-book
  run "moe_${h}"       python3 -m acw.baselines.moe_whitening         --heuristic "$h" --domain music
  run "moe_zb_${h}"    python3 -m acw.baselines.moe_whitening         --heuristic "$h" --domain music --input-zca-book
  run "infomax_${h}"   python3 -m acw.baselines.infomax_alignment     --heuristic "$h"
  run "infomax_noZCA_${h}" python3 -m acw.baselines.infomax_alignment --heuristic "$h" --no-zca
done

echo "=== ALL 16 TRAINING RUNS DONE ==="
