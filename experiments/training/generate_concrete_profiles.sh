#!/bin/bash
# =============================================================================
# experiments/training/generate_concrete_profiles.sh
# Generates the H3_concrete/H4_concrete user profiles (closed vocabulary of
# genres/mood/context instead of free evocative language) with Llama 3.3 70B
# AWQ served by vLLM. Supports the prompt-register ablation: it tests whether
# a "concrete" (closed-vocabulary) prompt degrades retrieval quality relative
# to the evocative H3/H4 prompts.
#
# Differences from the standard H3/H4 text-only generation:
#   - --heuristic h3_concrete / h4_concrete instead of h3/h4.
#   - SCOPE fixed to "val" (canonical validation population, n=2,731, the
#     same population used for the main H3/H4 results). There is no "all"
#     option: these profiles exist only for this ablation.
#   - backup_now() only packs a local archive ($HOME/<stem>.tar.gz); it does
#     not try to copy it back (the GPU server may have no outbound route).
#     Pull it manually with scp from the local machine afterwards.
#
# Prerequisite: vLLM already serving Llama 3.3 70B AWQ on localhost:8000.
#
# Inputs:  data/processed/canonical_val_users.json
# Outputs: data/processed/user_profiles_<heuristic>__<run_tag>.json (+ metadata)
#          and $HOME/user_profiles_<heuristic>__<run_tag>.tar.gz
#
# Usage (on the GPU server, inside tmux):
#   tmux new -s dla_concrete
#   cd <repository root>
#   source venv/bin/activate
#   MODEL_LLM=<local llama-3.3-70b-awq weights> bash experiments/training/generate_concrete_profiles.sh <run_tag>
#
# Example:
#   bash experiments/training/generate_concrete_profiles.sh concrete_llama70b
# =============================================================================

set -e
PROJECT="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL_LLM="${MODEL_LLM:?set MODEL_LLM to the local llama-3.3-70b-awq weights}"
USER_LIST="${PROJECT}/data/processed/canonical_val_users.json"

RUN_TAG="${1:?Usage: bash experiments/training/generate_concrete_profiles.sh <run_tag>}"

cd "${PROJECT}"
mkdir -p logs

if [ ! -f "${USER_LIST}" ]; then
    echo "ERROR: ${USER_LIST} not found."
    echo "Sync data/processed/canonical_val_users.json before continuing."
    exit 1
fi

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "logs/h3h4_concrete_textonly_${RUN_TAG}.log"; }

log "=============================================="
log "H3_concrete/H4_concrete text-only — run_tag=${RUN_TAG}"
log "SCOPE=val — 2,731 canonical users (same population as the main H3/H4 results)"
log "=============================================="

log "Checking vLLM server..."
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    log "ERROR: vLLM is not running. Start the vLLM server first."
    exit 1
fi
log "vLLM OK ✓"

export LLM_BACKEND=vllm
export VLLM_BASE_URL="http://localhost:8000/v1"
export VLLM_MODEL_PATH="${MODEL_LLM}"
export ACW_EMBEDDING_MODEL="Alibaba-NLP/gte-large-en-v1.5"
export ACW_LLM_CONCURRENCY=24

backup_now() {
    local heuristic="$1"
    local stem="user_profiles_${heuristic}__${RUN_TAG}"

    log "  Packing artifacts for ${heuristic} (run_tag=${RUN_TAG})..."
    tar -czf "$HOME/${stem}.tar.gz" \
        "data/processed/${stem}.json" \
        "data/processed/${stem}_metadata.json" \
        "data/processed/run_manifest.jsonl" \
        "data/processed/debug_traces/${heuristic}/" \
        2>/dev/null || true
    log "  Packed: $HOME/${stem}.tar.gz"
    log "  Pull from the local machine:"
    log "    scp <user>@<server-ip>:$HOME/${stem}.tar.gz ."
}

# ── H3_concrete (closed vocabulary, raw text) ────────────────────────────────
log ""
log "Generating H3_concrete (val, n=2,731)..."
python -m acw.embeddings.profile_summarization.cli \
    --heuristic h3_concrete \
    --model "${MODEL_LLM}" \
    --text-only \
    --run-tag "${RUN_TAG}" \
    --user-list "${USER_LIST}" \
    2>&1 | tee -a "logs/h3h4_concrete_textonly_${RUN_TAG}.log"
log "H3_concrete done ✓"
backup_now h3_concrete

# ── H4_concrete (closed vocabulary, raw text) ────────────────────────────────
log ""
log "Generating H4_concrete (val, n=2,731)..."
python -m acw.embeddings.profile_summarization.cli \
    --heuristic h4_concrete \
    --model "${MODEL_LLM}" \
    --text-only \
    --run-tag "${RUN_TAG}" \
    --user-list "${USER_LIST}" \
    2>&1 | tee -a "logs/h3h4_concrete_textonly_${RUN_TAG}.log"
log "H4_concrete done ✓"
backup_now h4_concrete

log ""
log "=============================================="
log "H3_concrete/H4_concrete text-only COMPLETE (run_tag=${RUN_TAG}) ✓"
log "Next steps (LOCAL machine, not the GPU server):"
log "  1. scp pull both .tar.gz archives (see the 'Pull from the local machine' lines above)"
log "  2. Extract them into data/processed/ of this repository"
log "  3. Embed with GTE-large-en-v1.5 (GTEEncoder, NOT SentenceTransformer):"
log "     python -m acw.embeddings.profile_summarization.cli --heuristic h3_concrete \\"
log "         --embed-only --run-tag ${RUN_TAG} --embedding-model Alibaba-NLP/gte-large-en-v1.5"
log "     (repeat for h4_concrete)"
log "=============================================="
