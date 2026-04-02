#!/usr/bin/env bash
# Sweep over optimizer and LR combinations for GPT-2 d24.
# Runs back-to-back with a 10-minute sleep between runs.

set -euo pipefail

OPTIMS=("adamw" "muon-llm" "muon-jordan")
LRS=("1e-4" "3e-4" "1e-3" "3e-3")
SLEEP_SECS=300  # 5 minutes

RUNS=()
for OPTIM in "${OPTIMS[@]}"; do
    for LR in "${LRS[@]}"; do
        RUNS+=("${OPTIM}__${LR}")
    done
done

TOTAL=${#RUNS[@]}

for i in "${!RUNS[@]}"; do
    OPTIM="${RUNS[$i]%%__*}"
    LR="${RUNS[$i]##*__}"
    CKPT_DIR="d6-${OPTIM}-lr${LR}-epoch3"

    echo "========================================================"
    echo "Run $((i+1))/${TOTAL}: optim=${OPTIM}, lr=${LR}"
    echo "Checkpoint dir: ${CKPT_DIR}"
    echo "========================================================"

    torchrun --nproc_per_node=8 eval.py --checkpoint-dir "${CKPT_DIR}" --no-wandb --no-comet

    # Sleep between runs (skip after the last one)
    if [[ $((i+1)) -lt ${TOTAL} ]]; then
        echo ""
        echo "Run $((i+1))/${TOTAL} complete. Sleeping for $((SLEEP_SECS/60)) minutes..."
        sleep "${SLEEP_SECS}"
    fi
done

echo ""
echo "All ${TOTAL} runs complete."
